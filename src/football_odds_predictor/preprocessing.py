"""Normalize the Football-Data.co.uk season files for the V1 dataset."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_STATS = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
}
TARGET_ODDS = {
    "B365CH": "bet365_closing_home",
    "B365CD": "bet365_closing_draw",
    "B365CA": "bet365_closing_away",
}


def read_season(path: Path) -> pd.DataFrame:
    """Read one division and season, preserving missing statistics and quotes."""
    filename = re.fullmatch(r"(\d{2})(\d{2})-(E0|D1|SP1)\.csv", path.name)
    if filename is None:
        raise ValueError(f"{path}: expected a filename such as 2122-E0.csv")
    start, end, league = filename.groups()
    if (int(start) + 1) % 100 != int(end):
        raise ValueError(f"{path}: invalid season")

    source_columns = {
        "Div": "league",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        **TEAM_STATS,
        **TARGET_ODDS,
    }
    frame = pd.read_csv(
        path,
        usecols=[*source_columns, "Date", "Time"],
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    for column in frame:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    frame = frame.dropna(how="all").rename(columns=source_columns)
    identifiers = ["league", "Date", "Time", "home_team", "away_team"]
    if frame.empty or frame[identifiers].isna().any().any():
        raise ValueError(f"{path}: missing fixture identity or kickoff")
    if not frame["league"].eq(league).all():
        raise ValueError(f"{path}: division does not match the filename")

    kickoff = pd.to_datetime(
        frame.pop("Date") + " " + frame.pop("Time"), format="%d/%m/%Y %H:%M"
    )
    frame["season_start_year"] = 2000 + int(start)
    frame["source_kickoff"] = kickoff
    for column in TEAM_STATS.values():
        values = pd.to_numeric(frame[column], errors="raise")
        observed = values.dropna()
        if not (np.isfinite(observed) & observed.ge(0) & observed.mod(1).eq(0)).all():
            raise ValueError(f"{path}: invalid match statistics in {column}")
        frame[column] = values.astype("Int64")
    for column in TARGET_ODDS.values():
        values = pd.to_numeric(frame[column], errors="raise")
        observed = values.dropna()
        if not (np.isfinite(observed) & observed.gt(1)).all():
            raise ValueError(f"{path}: invalid decimal odds in {column}")
        frame[column] = values.astype("Float64")
    output_columns = [
        "league", "season_start_year", "source_kickoff", "home_team", "away_team",
        *TEAM_STATS.values(), *TARGET_ODDS.values(),
    ]
    return frame[output_columns]


def load_matches(raw_dir: Path) -> pd.DataFrame:
    """Combine all season CSVs and check the fixture key before sorting."""
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"{raw_dir}: no season CSVs found")

    matches = pd.concat([read_season(path) for path in paths], ignore_index=True)
    duplicate_keys = matches.assign(source_date=matches["source_kickoff"].dt.date)
    fixture_key = ["league", "source_date", "home_team", "away_team"]
    duplicates = duplicate_keys.duplicated(subset=fixture_key, keep=False)
    if duplicates.any():
        first = duplicate_keys.loc[duplicates, fixture_key].iloc[0].to_dict()
        raise ValueError(f"Duplicate fixture: {first}")
    return matches.sort_values(
        ["source_kickoff", "league", "home_team", "away_team"],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/football-data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/matches.csv"))
    args = parser.parse_args()
    matches = load_matches(args.raw_dir)
    if args.output.resolve() in {path.resolve() for path in args.raw_dir.glob("*.csv")}:
        parser.error("the output must not overwrite a raw CSV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Wrote {len(matches):,} fixtures to {args.output}")


if __name__ == "__main__":
    main()
