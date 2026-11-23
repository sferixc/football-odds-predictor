"""Build pre-match features from earlier source calendar dates."""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from .preprocessing import load_matches

FIXTURE_COLUMNS = (
    "league", "season_start_year", "source_kickoff", "home_team", "away_team",
)
SIDES = ("home", "away")
MATCH_STATS = ("goals", "shots", "shots_on_target", "corners")
HISTORY_FIELDS = (
    "was_home", "opponent_elo",
    *(f"{stat}_{direction}" for stat in MATCH_STATS for direction in ("for", "against")),
)
HISTORY_LENGTH = 5
INITIAL_ELO = 1500.0
ELO_K = 20.0


def build_features(
    matches: pd.DataFrame, *, elo_warmup: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Emit features for each fixture, including fixtures with unknown results."""
    stat_columns = [f"{side}_{stat}" for side in SIDES for stat in MATCH_STATS]
    ordered = matches.reindex(columns=[*FIXTURE_COLUMNS, *stat_columns]).assign(warmup=False)
    if elo_warmup is not None:
        if matches.empty or not elo_warmup["source_kickoff"].dt.normalize().lt(
            matches["source_kickoff"].min().normalize()
        ).all():
            raise ValueError("Elo warm-up fixtures must precede the first output fixture date")
        warmup = elo_warmup.reindex(columns=ordered.columns).assign(warmup=True)
        ordered = pd.concat([warmup, ordered], ignore_index=True)
    ordered = ordered.sort_values(["source_kickoff", "league", "home_team", "away_team"])
    days = ordered["source_kickoff"].dt.normalize()
    ratings = defaultdict(lambda: INITIAL_ELO)
    histories = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    daily_changes = defaultdict(float)
    columns = [
        *FIXTURE_COLUMNS, "home_elo", "away_elo", "history_complete",
        *(f"{side}_match_{index}_{field}"
          for side in SIDES
          for index in range(1, HISTORY_LENGTH + 1)
          for field in HISTORY_FIELDS),
    ]
    records = []

    for _, fixtures in ordered.groupby(days, sort=False):
        daily_changes.clear()
        for match in fixtures.itertuples(index=False):
            home = (match.league, match.home_team)
            away = (match.league, match.away_team)
            home_elo, away_elo = ratings[home], ratings[away]
            if not match.warmup:
                record = {column: getattr(match, column) for column in FIXTURE_COLUMNS}
                record.update(home_elo=home_elo, away_elo=away_elo)
                complete = True
                for side, team in zip(SIDES, (home, away)):
                    history = histories[team]
                    complete &= len(history) == HISTORY_LENGTH and all(
                        pd.notna(value) for entry in history for value in entry
                    )
                    # Keep the latest match in the final slot, even with short history.
                    for index, entry in enumerate(history, HISTORY_LENGTH + 1 - len(history)):
                        for field, value in zip(HISTORY_FIELDS, entry):
                            record[f"{side}_match_{index}_{field}"] = value
                record["history_complete"] = complete
                records.append(record)

            if pd.isna(match.home_goals) or pd.isna(match.away_goals):
                continue
            outcome = 1.0 if match.home_goals > match.away_goals else (
                0.5 if match.home_goals == match.away_goals else 0.0
            )
            expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            change = ELO_K * (outcome - expected)
            daily_changes[home] += change
            daily_changes[away] -= change

        # Apply results and history only after every fixture on this date is emitted.
        for match in fixtures.itertuples(index=False):
            if match.warmup or pd.isna(match.home_goals) or pd.isna(match.away_goals):
                continue
            for side, opponent in (("home", "away"), ("away", "home")):
                team = (match.league, getattr(match, f"{side}_team"))
                opponent_team = (match.league, getattr(match, f"{opponent}_team"))
                stats = (
                    getattr(match, f"{perspective}_{stat}")
                    for stat in MATCH_STATS
                    for perspective in (side, opponent)
                )
                history_stats = tuple(None if pd.isna(value) else value for value in stats)
                histories[team].append(
                    (int(side == "home"), ratings[opponent_team], *history_stats)
                )
        for team, change in daily_changes.items():
            ratings[team] += change

    return pd.DataFrame.from_records(records, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--elo-warmup-dir", type=Path, help="Earlier season CSVs used only for Elo")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("the output must be separate from the match input")
    matches = pd.read_csv(args.input, parse_dates=["source_kickoff"])
    warmup = load_matches(args.elo_warmup_dir) if args.elo_warmup_dir else None
    features = build_features(matches, elo_warmup=warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Wrote {len(features):,} fixtures to {args.output}")
    print(f"Complete histories: {features['history_complete'].sum():,}")


if __name__ == "__main__":
    main()
