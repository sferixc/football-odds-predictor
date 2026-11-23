"""Load count models and turn expected corner counts into half-line odds."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..data.europe_corners import CORNER_INPUTS, match_inputs
from .ucl import club_key

CORNER_MODELS = Path("data/processed/corner-models.pkl")
EUROPE_CORNER_MODELS = Path("data/processed/europe-corner-models.pkl")
CORNER_LINES = {
    "home_corners": np.arange(0.5, 11, 1),
    "away_corners": np.arange(0.5, 11, 1),
    "match_corners": np.arange(0.5, 19, 1),
}


def load_corner_models(models: dict) -> None:
    """Load the optional corner bundles used by the dashboard and terminal."""
    for key, path in (("corner_counts", CORNER_MODELS), ("europe_corner_counts", EUROPE_CORNER_MODELS)):
        if path.is_file():
            with path.open("rb") as handle:
                models[key] = pickle.load(handle)


def european_corner_inputs(bundle, fixture, kickoff):
    if bundle is None:
        return None
    teams = fixture["event_name"].split("·")
    if len(teams) != 2:
        return None
    home, away = (bundle["names"].get(club_key(name.strip())) for name in teams)
    if home is None or away is None or home == away:
        return None
    inputs = match_inputs(bundle["timeline"], home, away, kickoff.date())
    return pd.DataFrame([inputs], columns=CORNER_INPUTS) if inputs is not None else None


def corner_probabilities(bundle, league: str, current, cutoff, lines=CORNER_LINES) -> dict[str, np.ndarray] | None:
    """Return Over probabilities for every supported half-line in one fixture."""
    league_bundle = (bundle or {}).get("leagues", {}).get(league)
    cutoff_day = cutoff.date() if callable(getattr(cutoff, "date", None)) else cutoff
    if current is None or league_bundle is None or cutoff_day <= league_bundle["training_end"]:
        return None

    results = {}
    for target, model in league_bundle["models"].items():
        expected_count = model["estimator"].predict(current[model["inputs"]])[0]
        results[target] = poisson.sf(lines[target], expected_count)
    return results
