"""Shared evaluation metrics for decimal odds."""

import numpy as np


def odds_errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return MAE on log odds and decimal odds, averaged across fixtures and outcomes."""
    log_mae = np.abs(np.log(actual) - np.log(predicted)).mean()
    decimal_mae = np.abs(actual - predicted).mean()
    return float(log_mae), float(decimal_mae)
