"""Replay recently completed 2026/27 fixtures using the selected V1 model."""

from pathlib import Path

import pandas as pd

from ..data.dataset import TARGET_COLUMNS, final_test_split, join_dataset, load_dataset
from ..data.features import FIXTURE_COLUMNS, build_features
from ..data.preprocessing import load_matches
from ..models.baselines import xgboost_predictions
from .final_test import MAX_DEPTH, N_ESTIMATORS
from .metrics import odds_errors


def main() -> None:
    processed = Path("data/processed")
    historical = pd.read_csv(processed / "matches.csv", parse_dates=["source_kickoff"])
    recent = load_matches(Path("data/raw/football-data-2627"))
    warmup = load_matches(Path("data/raw/football-data-warmup"))
    if not recent["season_start_year"].eq(2026).all():
        raise ValueError("Expected only 2026/27 fixtures in the recent archive")
    if historical["source_kickoff"].max().date() >= recent["source_kickoff"].min().date():
        raise ValueError("Recent fixtures must follow the historical fixtures")

    # Replay earlier results to build form and Elo, including 2025/26.
    # XGBoost still fits only on 2021/22–2024/25.
    features = build_features(pd.concat([historical, recent], ignore_index=True), elo_warmup=warmup)
    recent_features = features.loc[features["season_start_year"].eq(2026)]
    evaluation = join_dataset(recent_features, recent)
    if evaluation.empty:
        raise ValueError("No recent fixtures have complete histories and valid closing odds")
    dataset = load_dataset(processed / "features.csv", processed / "matches.csv")
    train, _ = final_test_split(dataset)
    predictions = xgboost_predictions(
        train, evaluation, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
    )
    actual = evaluation[TARGET_COLUMNS].to_numpy(dtype=float)
    log_mae, decimal_mae = odds_errors(actual, predictions)
    print(f"2026/27 | scored {len(evaluation):,} of {len(recent):,} fixtures")
    print(f"Log MAE={log_mae:.4f} | decimal MAE={decimal_mae:.4f}")
    for league in sorted(recent["league"].unique()):
        mask = evaluation["league"].eq(league).to_numpy()
        print(f"{league} | scored {mask.sum()} of {recent['league'].eq(league).sum()}")

    output = evaluation.loc[:, [*FIXTURE_COLUMNS, *TARGET_COLUMNS]].copy()
    output[["predicted_home", "predicted_draw", "predicted_away"]] = predictions
    path = processed / "recent_predictions.csv"
    output.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
