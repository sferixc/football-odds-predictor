"""Evaluate the XGBoost configuration selected on validation on 2025/26."""

import argparse
from pathlib import Path

from ..data.dataset import TARGET_COLUMNS, final_test_split, load_dataset
from ..models.baselines import xgboost_predictions
from .metrics import odds_errors

MAX_DEPTH = 4
N_ESTIMATORS = 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--matches", type=Path, default=Path("data/processed/matches.csv"))
    args = parser.parse_args()

    train, test = final_test_split(load_dataset(args.features, args.matches))
    predicted = xgboost_predictions(
        train, test, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
    )
    # The test targets are used only after prediction to score the frozen configuration.
    actual = test[TARGET_COLUMNS].to_numpy()
    log_mae, decimal_mae = odds_errors(actual, predicted)
    print(f"XGBoost depth={MAX_DEPTH}, trees per outcome={N_ESTIMATORS}")
    print(f"Train 2021/22–2024/25: {len(train):,} | Test 2025/26: {len(test):,}")
    print(f"Overall | log MAE={log_mae:.4f} | decimal MAE={decimal_mae:.4f}")
    for league in sorted(test["league"].unique()):
        mask = test["league"].eq(league).to_numpy()
        log_mae, decimal_mae = odds_errors(actual[mask], predicted[mask])
        print(f"{league} | fixtures={mask.sum():,} | log MAE={log_mae:.4f} | decimal MAE={decimal_mae:.4f}")

    fixture = test.iloc[0]
    predicted_odds = " / ".join(f"{value:.2f}" for value in predicted[0])
    actual_odds = " / ".join(f"{value:.2f}" for value in actual[0])
    print(f"{fixture['home_team']}–{fixture['away_team']} ({fixture['source_kickoff'].date()})")
    print(f"1X2 predicted: {predicted_odds} | actual: {actual_odds}")


if __name__ == "__main__":
    main()
