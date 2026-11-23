"""Inference with a previously fitted sporting classifier."""

from datetime import date

import numpy as np
import pandas as pd


def predict_saved(
    models: dict, current: pd.DataFrame, target: str, cutoff: date, threshold: float | None = None,
) -> np.ndarray:
    """Run a saved classifier using its recorded input column order."""
    model = models[(target, threshold)]
    if model["training_end"] >= cutoff:
        raise ValueError("The fixture date must follow the fitted model's training period")
    inputs = current[model["inputs"]].to_numpy()
    frequency = model["frequency"]
    if frequency is not None:
        return np.broadcast_to(frequency, (len(current), len(frequency)))
    return model["estimator"].predict_proba(inputs)
