"""Build the inputs and labels for independent event probabilities.

Each dataset returns a table and an explicit list of model inputs. The table
also carries identifiers and observed outcomes for training and evaluation.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .features import FIXTURE_COLUMNS, HISTORY_LENGTH, SIDES
from .player_archive import COMPETITIONS, SEASONS

TEAM_SOURCE = {
    "FTHG": "home_goals", "FTAG": "away_goals",
    "HTHG": "home_first_half_goals", "HTAG": "away_first_half_goals",
    "HY": "home_yellows", "AY": "away_yellows",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
}
TEAM_HISTORY = ("goals", "first_half_goals", "second_half_goals", "yellows", "corners", "fouls")


def _recent_means(
    rows: pd.DataFrame, groups: list[str], values: Sequence[str],
) -> pd.DataFrame:
    """For each row, average only the preceding five observations."""
    if rows.duplicated([*groups, "match_date"]).any():
        raise ValueError("Expected at most one observation per team or player and date")
    rows = rows.sort_values("match_date")
    # Shift by one fixture before rolling so the current result stays out of its inputs.
    return rows.groupby(groups)[list(values)].transform(
        lambda series: series.shift().rolling(HISTORY_LENGTH).mean()
    )


def team_dataset(features_path: Path, raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read target counts from raw match files and Elo from the feature table."""
    mapping = {"Div": "league", "HomeTeam": "home_team", "AwayTeam": "away_team", **TEAM_SOURCE}
    seasons = []
    for path in sorted(raw_dir.glob("*.csv")):
        rows = pd.read_csv(path, usecols=[*mapping, "Date", "Time"], encoding="utf-8-sig")
        rows = rows.dropna(subset=["HomeTeam", "AwayTeam"]).rename(columns=mapping)
        rows["season_start_year"] = 2000 + int(path.name[:2])
        rows["source_kickoff"] = pd.to_datetime(
            rows.pop("Date") + " " + rows.pop("Time"), format="%d/%m/%Y %H:%M"
        )
        seasons.append(rows)
    if not seasons:
        raise ValueError(f"No match CSVs in {raw_dir}")
    matches = pd.concat(seasons, ignore_index=True)
    matches["match_date"] = matches["source_kickoff"].dt.normalize()
    for side in SIDES:
        matches[f"{side}_second_half_goals"] = (
            matches[f"{side}_goals"] - matches[f"{side}_first_half_goals"]
        )

    # One row per team and fixture helps build the same history for either side.
    histories = []
    stats = [f"{stat}_{role}" for stat in TEAM_HISTORY for role in ("for", "against")]
    for side, opponent in (("home", "away"), ("away", "home")):
        renamed = {f"{side}_team": "team"}
        renamed.update({f"{side}_{stat}": f"{stat}_for" for stat in TEAM_HISTORY})
        renamed.update({f"{opponent}_{stat}": f"{stat}_against" for stat in TEAM_HISTORY})
        history = matches[["league", "match_date", *renamed]].rename(columns=renamed)
        histories.append(history.assign(side=side))
    histories = pd.concat(histories, ignore_index=True)
    means = _recent_means(histories, ["league", "team"], stats)
    for side in SIDES:
        recent = histories["side"].eq(side)
        # Restore the original fixture row order after computing means in date order.
        side_means = means.loc[histories.index[recent]].reset_index(drop=True)
        matches = matches.join(side_means.add_prefix(f"{side}_mean_"))

    key = list(FIXTURE_COLUMNS)
    elo_columns = ["home_elo", "away_elo"]
    # Reuse the pre-match ratings from the earlier experiment.
    elo = pd.read_csv(features_path, usecols=[*key, *elo_columns], parse_dates=["source_kickoff"])
    dataset = matches.merge(elo, on=key, validate="one_to_one")
    if len(dataset) != len(matches) or len(dataset) != len(elo):
        raise ValueError("The raw match and Elo files must contain the same fixtures")
    inputs = [*elo_columns, *(f"{side}_mean_{stat}" for side in SIDES for stat in stats)]
    return dataset.sort_values("source_kickoff", ignore_index=True), inputs


def player_dataset(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build player histories and labels for probabilities conditional on playing."""
    seasons = []
    for competition in COMPETITIONS:
        for season in SEASONS:
            folder = root / competition / str(season)
            fixtures = pd.read_csv(folder / "fixtures.csv", usecols=["fixture_id", "date"])
            appearances = pd.read_csv(
                folder / "appearances.csv",
                usecols=["fixture_id", "player_id", "player", "side", "minutes", "shots"],
            )
            data = appearances.merge(fixtures, on="fixture_id", validate="many_to_one")
            if len(data) != len(appearances):
                raise ValueError(f"Missing fixtures for player appearances in {folder}")
            seasons.append(data.assign(league=competition, season_start_year=season))
    data = pd.concat(seasons, ignore_index=True)
    data = data.loc[data["minutes"].gt(0) & data["shots"].ge(0)].copy()
    data["match_date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data["source_kickoff"] = data["match_date"]
    data["at_home"] = data["side"].eq("h").astype(int)
    # Current minutes establish which appearances are evaluated. Only prior means are inputs.
    history = _recent_means(data, ["league", "player_id"], ["minutes", "shots"]).add_prefix("mean_")
    data = data.join(history)
    # The shortlist is defined with prior minutes, before seeing the current outcome.
    data = data.loc[data["mean_minutes"].ge(45)]
    return data.sort_values("source_kickoff", ignore_index=True), ["at_home", "mean_minutes", "mean_shots"]
