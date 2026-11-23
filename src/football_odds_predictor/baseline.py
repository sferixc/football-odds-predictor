"""Compatibility entry point for the baseline experiments."""

from .models.baselines import (
    HISTORY_FIELDS as HISTORY_FIELDS,
    ridge_predictions as ridge_predictions,
    summarize_form as summarize_form,
    xgboost_predictions as xgboost_predictions,
)
from .training.baseline import main as main


if __name__ == "__main__":
    main()
