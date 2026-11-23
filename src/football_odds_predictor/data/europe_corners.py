"""Assemble European corner targets and each club's five preceding results."""

from bisect import bisect_left
from collections import Counter, defaultdict, deque
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .europe import CURRENT_LEAGUES, DOMESTIC, OPEN_MATCHDAY, read_europe_games
from .features import HISTORY_LENGTH

RAW = Path("data/raw")
FOOTY_PARTS = {
    "european-cups.champions-league": {"CL": "CL", "CLQ": "CLQ"},
    "european-cups.europa-league": {"EL": "EL", "ELQ": "ELQ"},
    "european-cups.conference-league": {"ECL": "UCOL", "ECLQ": "ECLQ"},
    "european-cups.super-cup": {"USC": "USC"},
    "england.premier-league": {"PL": "GB1"},
    "germany.bundesliga": {"BL": "L1"},
    "spain.la-liga": {"LL": "ES1"},
    "italy.serie-a": {"SA": "IT1"},
    "france.ligue-1": {"L1": "FR1"},
}
FOOTY_RELEASE = "11footy-2026-11-10"
COUNTS = ["home_corners", "away_corners"]
CLUB_INPUTS = ["corners_for", "corners_against", "observed_corners", "goal_margin", "days_since_match"]
CORNER_INPUTS = [f"{side}_{field}" for side in ("home", "away") for field in CLUB_INPUTS]
CUPS = {
    "ofb-cup": "OFB", "efl-cup": "CGB", "fa-cup": "FAC", "league-cup": "SLC",
    "scottish-cup": "SFA", "swiss-cup": "SCC",
}
QUALIFIERS = {"CLQ": "CL", "ELQ": "EL", "ECLQ": "UCOL"}


def read_footy():
    fixtures, team_names = [], defaultdict(set)
    for part, competitions in FOOTY_PARTS.items():
        path = RAW / f"{FOOTY_RELEASE}-{part}.csv.zip"
        with ZipFile(path) as archive:
            columns = ["match_id", "competition_code", "local_date", "status", "extra_time_played",
                       "home_team_id", "away_team_id", "home_goals", "away_goals"]
            with archive.open("matches.csv") as handle:
                games = pd.read_csv(handle, usecols=columns, parse_dates=["local_date"])
            with archive.open("teams.csv") as handle:
                names = pd.read_csv(handle, usecols=["team_id", "name", "short_name"])
            for row in names.itertuples(index=False):
                team_names[row.team_id].update(name for name in (row.name, row.short_name) if isinstance(name, str))
            with archive.open("match_team_statistics.csv") as handle:
                stats = pd.read_csv(handle, usecols=["match_id", "team_id", "layer", "corners", "fouls"])

        games = games.loc[games.status.eq("played") & games.local_date.ge("2020-07-01")].copy()
        games = games.dropna(subset=["home_goals", "away_goals"])
        games["competition"] = games.competition_code.map(competitions)
        # A feed with zero fouls and corners on both sides means unknown in 11footy.
        empty = stats.corners.eq(0) & stats.fouls.eq(0)
        empty_matches = stats.loc[stats.layer.eq("feed")].assign(empty=empty).groupby("match_id")["empty"].agg(
            ["all", "size"]
        )
        bogus = empty_matches.index[empty_matches["all"] & empty_matches["size"].eq(2)]
        stats = stats.loc[~(stats.layer.eq("feed") & stats.match_id.isin(bogus))]
        stats = stats.sort_values("layer").drop_duplicates(["match_id", "team_id"], keep="last")
        corners = stats.set_index(["match_id", "team_id"]).corners
        for side in ("home", "away"):
            keys = pd.MultiIndex.from_frame(games[["match_id", f"{side}_team_id"]])
            games[f"{side}_corners"] = corners.reindex(keys).to_numpy()
        # Extra time totals are incompatible with 90-minute corner markets.
        games.loc[~games.extra_time_played.eq(0), COUNTS] = float("nan")
        fixtures.extend(games.rename(columns={"local_date": "date"}).itertuples(index=False))
    return fixtures, team_names


def supplement_matches(games, names, club_key):
    """Read known current fixtures and corner counts from the local snapshots."""
    mapping = {"champions-league": "CL", "europa-league": "EL", "conference-league": "UCOL", **CUPS}
    if OPEN_MATCHDAY.is_file():
        with ZipFile(OPEN_MATCHDAY) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith("/results_2627.csv"):
                    continue
                country, tier, slug = member.split("/")[-2].split("_", maxsplit=2)
                competition = CURRENT_LEAGUES.get(country) if tier == "1" and country != "UEFA" else mapping.get(slug)
                if competition is None:
                    continue
                with archive.open(member) as handle:
                    rows = pd.read_csv(handle, parse_dates=["date"])
                rows = rows.dropna(subset=["date", "home", "away", "ft_home", "ft_away"])
                for row in rows.itertuples(index=False):
                    home, away = names.get(club_key(row.home)), names.get(club_key(row.away))
                    if home is None or away is None or home == away:
                        continue
                    key = (row.date, home, away)
                    # Keep the result already recorded in the main archive.
                    result = games.setdefault(key, [competition, row.ft_home, row.ft_away, None, None, False])
                    if result[1:3] == [row.ft_home, row.ft_away]:
                        # Cups can include extra time. Only take current domestic league stats here.
                        if tier == "1" and country != "UEFA":
                            result[3:5] = [getattr(row, name, None) for name in COUNTS]

    for folder in ("football-data", "football-data-2627", "football-data-warmup"):
        for path in sorted((RAW / folder).glob("*.csv")):
            columns = ["Date", "Div", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC"]
            rows = pd.read_csv(path, usecols=columns).dropna(subset=columns)
            rows["Date"] = pd.to_datetime(rows.Date, format="%d/%m/%Y")
            for row in rows.itertuples(index=False):
                key = (row.Date, names.get(club_key(row.HomeTeam)), names.get(club_key(row.AwayTeam)))
                result = games.get(key)
                if result and result[:3] == [DOMESTIC.get(row.Div), row.FTHG, row.FTAG]:
                    result[3:5] = [row.HC, row.AC]


def merge_footy(games, fixtures, team_names, names, club_key):
    """Resolve aliases using repeated agreement on date, score and the other club."""
    candidates = defaultdict(list)
    for (day, home, away), (competition, hg, ag, *_) in games.items():
        candidates[day, competition, hg, ag].append((home, away))
    direct = {}
    for team, aliases in team_names.items():
        ids = {names[club_key(alias)] for alias in aliases if club_key(alias) in names}
        if len(ids) == 1:
            direct[team] = next(iter(ids))
    evidence = defaultdict(Counter)
    for row in fixtures:
        pairs = candidates[(row.date, row.competition, row.home_goals, row.away_goals)]
        for own, opponent, side in ((row.home_team_id, row.away_team_id, 0),
                                    (row.away_team_id, row.home_team_id, 1)):
            known = direct.get(opponent)
            hits = {pair[side] for pair in pairs if known is not None and pair[1 - side] == known}
            if len(hits) == 1:
                evidence[own].update(hits)

    resolved = direct.copy()
    for team, counts in evidence.items():
        if len(counts) == 1 and counts.total() >= 2:
            resolved[team] = next(iter(counts))
    for row in fixtures:
        home, away = resolved.get(row.home_team_id), resolved.get(row.away_team_id)
        if home is None or away is None or home == away:
            continue
        key = row.date, home, away
        result = games.get(key)
        if result is None:
            # Include newer verified fixtures without inventing links for older unmatched records.
            if row.date < pd.Timestamp("2026-07-01"):
                continue
            result = games[key] = [row.competition, row.home_goals, row.away_goals, None, None, False]
        if result[1:3] != [row.home_goals, row.away_goals]:
            continue
        # Some recent fixture lists group qualifying rounds under the main UEFA cup.
        if result[0] != row.competition and QUALIFIERS.get(row.competition) != result[0]:
            continue
        if pd.notna(row.home_corners) and pd.notna(row.away_corners):
            result[3:5] = [row.home_corners, row.away_corners]
            result[5] = row.competition in {"CL", "EL"}


def match_inputs(timeline, home, away, day):
    """Read the same prior form for historic fixtures and future reports."""
    features = []
    for club in (home, away):
        entries = timeline.get(club, ())
        index = bisect_left(entries, day, key=lambda entry: entry[0])
        if index == 0:
            return None
        last_day, recent = entries[index - 1]
        age = (day - last_day).days
        if age > 60 or len(recent) < HISTORY_LENGTH:
            return None
        observed = [(won, conceded) for _, won, conceded in recent if pd.notna(won) and pd.notna(conceded)]
        if not observed:
            return None
        features.extend([sum(won for won, _ in observed) / len(observed),
                         sum(conceded for _, conceded in observed) / len(observed),
                         len(observed), sum(margin for margin, _, _ in recent) / len(recent), age])
    return features


def build_corner_dataset(club_key):
    rows, names = read_europe_games(club_key, include_domestic_cups=True)
    rows = rows.loc[rows.date.ge("2020-07-01")]
    games = {
        (row.date, row.home_club_id, row.away_club_id):
        [row.competition_id, row.home_club_goals, row.away_club_goals, None, None, False]
        for row in rows.itertuples(index=False)
    }
    supplement_matches(games, names, club_key)
    merge_footy(games, *read_footy(), names, club_key)
    timeline, history = defaultdict(list), defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    targets = []
    for (date, home, away), (competition, hg, ag, hc, ac, target) in sorted(
        games.items(), key=lambda item: item[0][0]
    ):
        # The date-only source cannot establish intra-day result availability.
        day = date.date()
        if target:
            features = match_inputs(timeline, home, away, day)
            if features is not None:
                season = date.year - (date.month < 7)
                targets.append([competition, date, season, hc, ac, hc + ac, *features])
        for club, entry in ((home, (hg - ag, hc, ac)), (away, (ag - hg, ac, hc))):
            history[club].append(entry)
            timeline[club].append((day, tuple(history[club])))
    columns = ["league", "source_kickoff", "season_start_year", *COUNTS, "match_corners", *CORNER_INPUTS]
    return pd.DataFrame(targets, columns=columns), dict(timeline), names
