"""Models for the median, Ridge, and XGBoost experiments."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ..data.dataset import TARGET_COLUMNS
from ..data.features import HISTORY_FIELDS, HISTORY_LENGTH, SIDES


def median_predictions(train: pd.DataFrame) -> np.ndarray:
    """Return the fixed triplet of training-set median odds."""
    return train[TARGET_COLUMNS].median().to_numpy()


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


def xgboost_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    max_depth: int = 3,
    n_estimators: int = 200,
) -> np.ndarray:
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
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
        )
        model.fit(train_inputs, np.log(train[target]))
        log_predictions.append(model.predict(validation_inputs))
    return np.exp(np.maximum(np.column_stack(log_predictions), np.log(1.01)))
