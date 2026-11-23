"""Convert saved yellow-card count forecasts into probabilities at each line."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import poisson

CARD_MODELS = Path("data/processed/yellow-card-models.pkl")
CARD_LINES = np.arange(0.5, 10, 1)
REFEREE_INPUT = "referee_prior_yellows"


def load_card_models(models: dict) -> None:
    if CARD_MODELS.is_file():
        with CARD_MODELS.open("rb") as handle:
            models["yellow_card_counts"] = pickle.load(handle)


def card_probabilities(bundle, current, target, cutoff, referee):
    """Use referee history when a named pre-match referee has prior results."""
    if bundle is None or target not in bundle["models"] or cutoff <= bundle["training_end"]:
        return None, None
    history = bundle["referee_history"]
    prior = history.loc[
        history["referee"].str.casefold().eq(referee.strip().casefold())
        & history["match_date"].lt(pd.Timestamp(cutoff)), "match_yellows"
    ] if isinstance(referee, str) and referee.strip() else pd.Series(dtype=float)
    mean = prior.mean()
    with_referee = pd.notna(mean)
    variant = bundle["models"][target]["referee" if with_referee else "team"]
    inputs = current.assign(**{REFEREE_INPUT: mean})[variant["inputs"]]
    expected_count = variant["estimator"].predict(inputs)[0]
    return poisson.sf(CARD_LINES, expected_count), with_referee
