"""Build a shared club Elo and pre-match form from domestic and UEFA results."""

from collections import defaultdict, deque
import json
import math
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .referees import ARCHIVE, CLUBS

EUROPE = {"CL", "EL"}
EUROPE_HISTORY = EUROPE | {"CLQ", "ELQ", "UCOL", "ECLQ"}
RECENT = Path("data/raw/football-data-2627")
OPEN_MATCHDAY = Path("data/raw/open-matchday/results-2627.zip")
SLOVAKIA = Path("data/raw/openfootball")
ELO_SEED = Path("data/raw/europe-elo-2020.csv")
START = pd.Timestamp("2020-08-01")
INPUTS = ("elo_difference", "home_margin", "away_margin", "home_opponents", "away_opponents",
          "home_history", "away_history", "home_domestic_share", "away_domestic_share")
DOMESTIC = {"E0": "GB1", "D1": "L1", "SP1": "ES1", "I1": "IT1"}
CURRENT_LEAGUES = {
    "AUT": "A1", "BEL": "BE1", "CRO": "KRO1", "CYP": "ZYP1", "CZE": "TS1",
    "DEN": "DK1", "ENG": "GB1", "ESP": "ES1", "FIN": "FI1", "FRA": "FR1",
    "GER": "L1", "GRE": "GR1", "ITA": "IT1", "KAZ": "KAS1", "NED": "NL1",
    "NOR": "NO1", "POL": "PL1", "POR": "PO1", "ROU": "RO1", "RUS": "RU1",
    "SCO": "SC1", "SUI": "C1", "SVK": "SK1", "SVN": "SL1", "SWE": "SE1",
    "TUR": "TR1", "UKR": "UKR1", "WAL": "WAL1",
}
CURRENT_CUPS = {"champions-league": "CL", "europa-league": "EL", "conference-league": "UCOL"}
ALIASES = {
    "Bayern München": "Bayern Munich", "FC Bayern München": "Bayern Munich",
    "Hull": "Hull City", "Newcastle": "Newcastle United", "Celta": "Celta de Vigo", "Celta Vigo": "Celta de Vigo",
    "Inter": "Inter Milan", "Sparta Praha": "Sparta Prague", "Lyon": "Olympique Lyon",
    "Bournemouth": "AFC Bournemouth", "Slovan": "Slovan Bratislava",
    "ŠK Slovan Bratislava": "Slovan Bratislava",
    "Atleti": "Atlético de Madrid", "B. Dortmund": "Borussia Dortmund",
    "Bodø/Glimt": "Bodo Glimt", "Leipzig": "RB Leipzig", "Lens": "RC Lens",
    "Lille": "LOSC Lille", "S. Bratislava": "Slovan Bratislava", "Sabah": "Sabah FK",
    "Shakhtar": "Shakhtar Donetsk", "Slavia Praha": "Slavia Prague",
    "Braga": "SC Braga", "Egnatia": "KF Egnatia", "Gent": "KAA Gent",
    "Iberia Tbilisi": "Iberia 1999 Tbilisi", "Inter Escaldes": "Inter Club d'Escaldes",
    "Kauno Žalgiris": "FK Kauno Zalgiris", "L. Red Imps": "Lincoln Red Imps",
    "Monaco": "AS Monaco", "Nordsjælland": "Nordsjaelland", "U. Craiova": "Univ. Craiova",
    "Benfica": "SL Benfica", "Ferencváros": "Ferencvarosi TC", "GNK Dinamo": "GNK Dinamo Zagreb",
    "H. Beer-Sheva": "Hapoel Be'er Sheva", "Lillestrøm": "Lillestrom",
    "Marseille": "Olympique Marseille", "N.E.C.": "NEC Nijmegen",
    "Rennes": "Stade Rennais", "Union SG": "Union Saint-Gilloise",
}


def pre_match_inputs(home, away, ratings, history):
    fields = []
    for team in (home, away):
        recent = history.get(team, ())
        count = len(recent)
        fields.append((
            sum(match[0] for match in recent) / count if count else 0,
            sum(match[1] - 1500 for match in recent) / count if count else 0,
            count,
            sum(match[2] not in EUROPE_HISTORY for match in recent) / count if count else 0,
        ))
    return [ratings.get(home, 1500) - ratings.get(away, 1500),
            *(value for pair in zip(*fields) for value in pair)]


def read_europe_games(club_key):
    columns = ["competition_id", "competition_type", "season", "round", "date",
               "home_club_id", "away_club_id", "home_club_goals", "away_club_goals",
               "home_club_name", "away_club_name", "referee"]
    games = pd.read_csv(ARCHIVE, usecols=columns, parse_dates=["date"])
    games = games.loc[
        games["competition_type"].eq("domestic_league") | games["competition_id"].isin(EUROPE_HISTORY)
    ].dropna(subset=["date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"])
    names = defaultdict(set)
    by_league = defaultdict(set)
    for row in games.itertuples(index=False):
        for club, name in ((row.home_club_id, row.home_club_name), (row.away_club_id, row.away_club_name)):
            if not isinstance(name, str):
                continue
            names[club_key(name)].add(club)
            if row.competition_type == "domestic_league":
                by_league[(row.competition_id, club_key(name))].add(club)
    for league, aliases in CLUBS.items():
        competition = DOMESTIC[league]
        for original, alias in aliases.items():
            source = by_league[(competition, club_key(original))]
            by_league[(competition, club_key(alias))].update(source)
            names[club_key(alias)].update(source)
    for alias, original in ALIASES.items():
        names[club_key(alias)].update(names[club_key(original)])
    extra = read_open_results(games, names, by_league, club_key)
    updated_competitions = {row["competition_id"] for row in extra}
    archive_end = games["date"].max()
    for path in sorted(RECENT.glob("*.csv")):
        rows = pd.read_csv(path, usecols=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        rows = rows.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        for row in rows.itertuples(index=False):
            competition = DOMESTIC.get(row.Div)
            if competition is None or competition in updated_competitions:
                continue
            date = pd.to_datetime(row.Date, format="%d/%m/%Y")
            if date <= archive_end:
                continue
            home = by_league[(competition, club_key(row.HomeTeam))]
            away = by_league[(competition, club_key(row.AwayTeam))]
            if len(home) != 1 or len(away) != 1:
                continue
            extra.append({
                "competition_id": competition, "competition_type": "domestic_league",
                "date": date, "season": date.year - (date.month < 7), "round": "",
                "home_club_id": next(iter(home)), "away_club_id": next(iter(away)),
                "home_club_goals": row.FTHG, "away_club_goals": row.FTAG,
            })
    if extra:
        games = pd.concat([games, pd.DataFrame(extra)], ignore_index=True)
    games = games.drop_duplicates(["date", "home_club_id", "away_club_id"])
    return games.sort_values("date"), {name: next(iter(ids)) for name, ids in names.items() if len(ids) == 1}


def read_open_results(games, names, by_league, club_key):
    """Load current open results, matching known IDs and retaining new local clubs."""
    result = []

    def club(competition, name, domestic):
        key = club_key(name)
        ids = by_league[(competition, key)] or names[key]
        if len(ids) == 1:
            return next(iter(ids))
        if ids:
            return None
        identifier = f"{competition if domestic else 'UEFA'}:{key}"
        names[key].add(identifier)
        if domestic:
            by_league[(competition, key)].add(identifier)
        return identifier

    def add(competition, date, home, away, home_goals, away_goals, domestic, group_stage=False):
        home_id, away_id = club(competition, home, domestic), club(competition, away, domestic)
        if home_id is None or away_id is None or home_id == away_id:
            return
        result.append({
            "competition_id": competition, "competition_type": "domestic_league" if domestic else "other",
            "date": date, "season": date.year - (date.month < 7), "round": "Group Stage" if group_stage else "",
            "home_club_id": home_id, "away_club_id": away_id, "home_club_name": home, "away_club_name": away,
            "home_club_goals": home_goals, "away_club_goals": away_goals,
        })

    for path in sorted(SLOVAKIA.glob("slovakia_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["matches"]:
            score = row.get("score")
            if isinstance(score, list) and len(score) == 2:
                add("SK1", pd.Timestamp(row["date"]), row["team1"], row["team2"], *score, True)
    archive_end = games["date"].max()
    if OPEN_MATCHDAY.is_file():
        with ZipFile(OPEN_MATCHDAY) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith("/results_2627.csv"):
                    continue
                folder = member.split("/")[-2]
                country, tier, slug = folder.split("_", maxsplit=2)
                if tier != "1":
                    continue
                domestic = country != "UEFA"
                competition = CURRENT_LEAGUES.get(country) if domestic else CURRENT_CUPS.get(slug)
                if competition is None:
                    continue
                with archive.open(member) as handle:
                    rows = pd.read_csv(handle, parse_dates=["date"])
                rows = rows.dropna(subset=["date", "home", "away", "ft_home", "ft_away"])
                for row in rows.loc[rows.date.gt(archive_end)].itertuples(index=False):
                    if domestic or row.date.month in (9, 10, 11, 12, 1):
                        add(competition, row.date, row.home, row.away,
                            row.ft_home, row.ft_away, domestic, not domestic)
    return result


def build_europe_history(games: pd.DataFrame, mixed: bool = True):
    """Emit fixture inputs before applying any results from that day."""
    seed = {}
    if mixed and ELO_SEED.is_file():
        rows = pd.read_csv(ELO_SEED, parse_dates=["as_of"])
        if (rows["as_of"] != START).any() or rows["club_id"].duplicated().any():
            raise ValueError("Expected one Elo per club at 2020-08-01")
        if not rows["elo"].map(math.isfinite).all():
            raise ValueError("Expected finite Elo values")
        seed = dict(zip(rows["club_id"], rows["elo"]))
        games = games.loc[games["date"].ge(START)]
    if not mixed:
        games = games.loc[games["competition_id"].isin(EUROPE)
                          & games["round"].str.startswith("Group", na=False)]
    ratings, history, timeline = seed.copy(), defaultdict(lambda: deque(maxlen=5)), defaultdict(list)
    observations = []
    for date, day in games.groupby("date", sort=True):
        changes = defaultdict(float)
        form_updates = []
        for row in day.itertuples(index=False):
            home, away = row.home_club_id, row.away_club_id
            margin = row.home_club_goals - row.away_club_goals
            if row.competition_id in EUROPE and str(row.round).startswith("Group") and row.season >= 2021:
                result = 0 if margin > 0 else (1 if margin == 0 else 2)
                observations.append((row.competition_id, row.season, result,
                                     row.home_club_goals + row.away_club_goals,
                                     *pre_match_inputs(home, away, ratings, history)))
            home_elo, away_elo = ratings.get(home, 1500), ratings.get(away, 1500)
            expected = 1 / (1 + 10 ** ((away_elo - home_elo - 60) / 400))
            actual = 1 if margin > 0 else (0.5 if margin == 0 else 0)
            change = 24 * (actual - expected)
            changes[home] += change
            changes[away] -= change
            form_updates.extend((
                (home, (margin, away_elo, row.competition_id)),
                (away, (-margin, home_elo, row.competition_id)),
            ))
        for team, match in form_updates:
            history[team].append(match)
        for team, change in changes.items():
            ratings[team] = ratings.get(team, 1500) + change
            timeline[team].append((date.date(), ratings[team], tuple(history[team])))
    rows = pd.DataFrame(observations, columns=["competition", "season", "result", "goals", *INPUTS])
    return rows, dict(timeline)
