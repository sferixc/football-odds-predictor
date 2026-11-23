"""Measure the effect of referee history on Premier League yellow-card forecasts."""

from pathlib import Path

import pandas as pd

from ..data.sporting import team_dataset
from ..models.cards import REFEREE_INPUT
from ..training.sporting import TEAM_THRESHOLDS, add_team_outcomes
from .sporting import assess_target

RAW = Path("data/raw/football-data")


def add_referee_history(rows: pd.DataFrame) -> pd.DataFrame:
    """Average observations from strictly earlier match days for each referee."""
    day = rows.groupby(["referee", "match_date"])["match_yellows"].agg(["sum", "count"])
    prior = day.groupby(level="referee").cumsum() - day
    days = prior["count"].where(prior["count"].gt(0))
    return rows.join((prior["sum"] / days).rename(REFEREE_INPUT), on=["referee", "match_date"])


def referee_dataset() -> tuple[pd.DataFrame, list[str]]:
    teams, inputs = team_dataset(Path("data/processed/features.csv"), RAW)
    leagues = teams.loc[teams["league"].eq("E0")].copy()
    officials = []
    for path in sorted(RAW.glob("*-E0.csv")):
        rows = pd.read_csv(path, usecols=["Date", "HomeTeam", "AwayTeam", "Referee", "HR", "AR"])
        rows = rows.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team", "Referee": "referee",
        })
        rows["match_date"] = pd.to_datetime(rows.pop("Date"), format="%d/%m/%Y")
        rows["referee"] = rows["referee"].str.strip().replace("", pd.NA)
        officials.append(rows)
    keys = ["match_date", "home_team", "away_team"]
    leagues = leagues.merge(pd.concat(officials), on=keys, how="left", validate="one_to_one")
    add_team_outcomes(leagues)
    leagues["home_card_points"] = leagues["home_yellows"] + 2 * leagues["HR"]
    leagues["away_card_points"] = leagues["away_yellows"] + 2 * leagues["AR"]
    leagues["match_card_points"] = leagues["home_card_points"] + leagues["away_card_points"]
    return add_referee_history(leagues), inputs


def main() -> None:
    rows, inputs = referee_dataset()
    # Compare both variants on fixtures with a previously seen referee.
    eligible = rows.dropna(subset=[REFEREE_INPUT])
    for target in ("home_yellows", "away_yellows", "match_yellows"):
        print("Without referee history")
        assess_target(eligible, inputs, target, "sgd", TEAM_THRESHOLDS[target], None)
        print("With referee history")
        assess_target(eligible, [*inputs, REFEREE_INPUT], target, "sgd", TEAM_THRESHOLDS[target], None)


if __name__ == "__main__":
    main()
