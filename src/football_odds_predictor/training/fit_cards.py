"""Train and evaluate Premier League card count models."""

from pathlib import Path
from tempfile import NamedTemporaryFile
import pickle

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.dataset import final_test_split, validation_folds
from ..evaluation.referee_cards import referee_dataset
from ..models.cards import CARD_LINES, CARD_MODELS, REFEREE_INPUT


def main() -> None:
    rows, inputs = referee_dataset()
    targets = tuple(f"{scope}_{stat}" for stat in ("yellows", "card_points")
                    for scope in ("home", "away", "match"))
    history = rows[["match_date", "referee", "match_yellows"]]
    rows = rows.dropna(subset=[*inputs, *targets, REFEREE_INPUT])
    train, _ = final_test_split(rows)
    models = {}
    for target in targets:
        variants = {}
        for name, columns in (("team", inputs), ("referee", [*inputs, REFEREE_INPUT])):
            estimator = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0))
            scores, outcomes = [], []
            for _, earlier, later in validation_folds(rows):
                estimator.fit(earlier[columns], earlier[target])
                means = estimator.predict(later[columns])
                scores.extend(poisson.sf(CARD_LINES[:, None], means).T)
                outcomes.extend(later[target].to_numpy()[:, None] > CARD_LINES)
            actual, probabilities = np.asarray(outcomes), np.asarray(scores)
            for index, line in enumerate(CARD_LINES):
                brier = brier_score_loss(actual[:, index], probabilities[:, index])
                print(f"{target} > {line:g} | {name} | {actual[:, index].sum()}/{len(actual)} overs | Brier {brier:.4f}")
            estimator.fit(train[columns], train[target])
            variants[name] = {"estimator": estimator, "inputs": columns}
        models[target] = variants
    bundle = {
        "training_end": train["source_kickoff"].max().date(),
        "models": models,
        "referee_history": history,
    }
    with NamedTemporaryFile(dir=CARD_MODELS.parent, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            pickle.dump(bundle, temporary)
            temporary.close()
            path.replace(CARD_MODELS)
        finally:
            path.unlink(missing_ok=True)
    print(f"Saved count models to {CARD_MODELS}")


if __name__ == "__main__":
    main()
