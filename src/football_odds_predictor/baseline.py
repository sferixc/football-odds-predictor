"""Evaluate closing-odds models on V1 validation folds.

Run with no options for the median baseline. Pass --model ridge for Elo and
form averages, or --model xgboost for Elo and each match in the history.
"""

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .dataset import TARGET_COLUMNS, load_dataset, validation_folds
from .features import HISTORY_LENGTH, MATCH_STATS, SIDES

HISTORY_FIELDS = (
    "was_home", "opponent_elo",
    *(f"{stat}_{direction}" for stat in MATCH_STATS for direction in ("for", "against")),
)


def summarize_form(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return 22 model inputs per match: two Elo ratings and 20 form averages.

    For example, home_mean_goals_for averages the goals the current home team
    scored across its five earlier matches.
    """
    summaries = {
        f"{side}_mean_{field}": fixtures[
            [f"{side}_match_{i}_{field}" for i in range(1, HISTORY_LENGTH + 1)]
        ].mean(axis=1, skipna=False)
        for side in SIDES for field in HISTORY_FIELDS
    }
    return fixtures[["home_elo", "away_elo"]].join(pd.DataFrame(summaries, index=fixtures.index))


def ridge_predictions(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    """Fit on training fixtures and return three decimal odds per validation match."""
    # Put Elo and the match stats on comparable scales. Ridge penalizes large weights.
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    # fit() learns the scaler and regression weights from the training fold.
    model.fit(summarize_form(train), np.log(train[TARGET_COLUMNS].to_numpy()))
    log_odds = model.predict(summarize_form(validation))
    # Convert from natural logs to decimal odds, with a minimum price of 1.01.
    return np.exp(np.maximum(log_odds, np.log(1.01)))


def xgboost_predictions(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    """Fit one tree ensemble per 1X2 outcome and return decimal odds."""
    inputs = [
        "home_elo", "away_elo",
        *(f"{side}_match_{i}_{field}"
          for side in SIDES for i in range(1, HISTORY_LENGTH + 1) for field in HISTORY_FIELDS),
    ]
    train_inputs = train[inputs]
    validation_inputs = validation[inputs]
    log_predictions = []
    # Each ensemble learns the log of one closing price from the same 102 inputs.
    for target in TARGET_COLUMNS:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
        )
        model.fit(train_inputs, np.log(train[target]))
        log_predictions.append(model.predict(validation_inputs))
    return np.exp(np.maximum(np.column_stack(log_predictions), np.log(1.01)))


def main() -> None:
    """Read command-line options, run each fold, and print its prediction errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--matches", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--model", choices=("median", "ridge", "xgboost"), default="median")
    # parse_args() turns options such as --model ridge into args.model == "ridge".
    args = parser.parse_args()

    dataset = load_dataset(args.features, args.matches)
    log_errors = []
    decimal_errors = []
    for season, train, validation in validation_folds(dataset):
        prediction_summary = ""
        if args.model == "xgboost":
            predictions = xgboost_predictions(train, validation)
        elif args.model == "ridge":
            predictions = ridge_predictions(train, validation)
        else:
            # Use the same three training medians for every validation fixture.
            predictions = train[TARGET_COLUMNS].median().to_numpy()
            odds = " / ".join(f"{value:.2f}" for value in predictions)
            prediction_summary = f" | odds={odds}"
        # Compare against the actual validation odds, averaging over matches and outcomes.
        actual = validation[TARGET_COLUMNS].to_numpy()
        log_mae = np.abs(np.log(actual) - np.log(predictions)).mean()
        decimal_mae = np.abs(actual - predictions).mean()
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
