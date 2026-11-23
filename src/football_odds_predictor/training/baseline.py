"""Evaluate closing-odds models on V1 validation folds.

Run with no options for the median baseline. Pass --model ridge for Elo and
form averages, or --model xgboost for Elo and each match in the history.
"""

import argparse
from pathlib import Path
from statistics import mean

from ..data.dataset import TARGET_COLUMNS, load_dataset, validation_folds
from ..evaluation.metrics import odds_errors
from ..models.baselines import median_predictions, ridge_predictions, xgboost_predictions


def main() -> None:
    """Read command-line options, run each fold, and print its prediction errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--matches", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--model", choices=("median", "ridge", "xgboost"), default="median")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum XGBoost tree depth")
    parser.add_argument("--n-estimators", type=int, default=200, help="XGBoost trees per outcome")
    # parse_args() turns options such as --model ridge into args.model == "ridge".
    args = parser.parse_args()
    if args.model == "xgboost" and min(args.max_depth, args.n_estimators) < 1:
        parser.error("--max-depth and --n-estimators must be positive")

    dataset = load_dataset(args.features, args.matches)
    log_errors = []
    decimal_errors = []
    for season, train, validation in validation_folds(dataset):
        prediction_summary = ""
        if args.model == "xgboost":
            predictions = xgboost_predictions(
                train, validation, max_depth=args.max_depth, n_estimators=args.n_estimators,
            )
        elif args.model == "ridge":
            predictions = ridge_predictions(train, validation)
        else:
            # Use the same three training medians for every validation fixture.
            predictions = median_predictions(train)
            odds = " / ".join(f"{value:.2f}" for value in predictions)
            prediction_summary = f" | odds={odds}"
        # Compare against the actual validation odds, averaging over matches and outcomes.
        actual = validation[TARGET_COLUMNS].to_numpy()
        log_mae, decimal_mae = odds_errors(actual, predictions)
        log_errors.append(log_mae)
        decimal_errors.append(decimal_mae)
        print(
            f"{season}/{(season + 1) % 100:02d} | train={len(train):,} "
            f"| validation={len(validation):,}{prediction_summary} "
            f"| log MAE={log_mae:.4f} | decimal MAE={decimal_mae:.4f}"
        )

    print(f"Fold mean | log MAE={mean(log_errors):.4f} | decimal MAE={mean(decimal_errors):.4f}")


if __name__ == "__main__":
    main()
