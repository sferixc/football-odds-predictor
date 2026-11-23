# Working on football-odds-predictor

V1 predicts one bookmaker's pre-match closing decimal odds for home win, draw, and away win (1X2). The research workflow and open decisions are in [docs/project-plan.md](docs/project-plan.md).

## Workflow

- Work in small increments. Before implementing a change to the data contract, feature pipeline, split protocol, model, or inference interface, explain the proposed design and trade-offs.
- After each experiment, document the question, inputs, method, a small worked example, and what the result tells us in the project plan. Include the command to reproduce it.
- Use Python 3.12 and uv. Run project commands with `uv run` and keep `pyproject.toml` and `uv.lock` in sync when dependencies change.
- Put reusable loading, feature engineering, training, evaluation, and inference code in `src/football_odds_predictor/`. Use `notebooks/` for optional Jupyter exploration and have notebooks call the package when logic becomes reusable.
- Keep raw inputs immutable and processed outputs reproducible. Follow the existing ignore rules for `data/raw/` and `data/processed/`. Record source, coverage, dataset version, and transformation settings for each experiment.

## Research invariants

- Use the odds triplet the source labels as closing. Before building features, define when predictions are made and use only information available by then. Keep the closing odds and fields derived from them out of the feature set.
- Build each team's sequence from its five most recent eligible completed matches. Use historical Elo as of the cutoff and update Elo chronologically after results become available.
- Use chronological train, validation, and test windows across all leagues. Fit learned transformations on training data and use validation for model selection. Reserve the test window for final evaluation.
- Compare statistical baselines, XGBoost, and the PyTorch LSTM on the same target and evaluation fixtures. Record the dataset version, split boundaries, feature set, configuration, seed, and metrics.
- Verify that a historical fixture and an upcoming fixture can use the same feature-building logic. When changing time-sensitive logic, check cutoff boundaries, fixture ordering, and missing history with small deterministic examples.
