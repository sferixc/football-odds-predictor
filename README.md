# football-odds-predictor

Football research project using Python 3.12 and uv. The current goal is to estimate event probabilities from match and player statistics and compare them with pre-match Superbet Romania offers.

## What works today

| Stage | What it does |
| --- | --- |
| Historical match data | Normalizes five seasons from three leagues and builds pre-match Elo and five-match histories. |
| Bet365 closing-odds experiment | Compares median, Ridge, XGBoost, and LSTM. Includes a final-season test, a replay on recent fixtures, and result plots. |
| Sporting probabilities | Compares four approaches on 13 selections across 1X2, goals, yellow cards, corners, and player total shots. Models learn from sporting outcomes. |
| Superbet collection | Finds fixtures in the three leagues and attempts one capture around 15 minutes before kickoff. Saves the raw offer and extracts supported single-selection promotions. |
| Offer calculator | Takes supplied probabilities, price, stake, and budget. Shows value, loss probability, and a provisional color. |

The [Bet365 experiment notes](docs/project-plan.md) describe the first research track and its results. The [Superbet assessment plan](docs/superbet-assessment.md) explains the current track, its data sources, worked examples, and results.

## Run the pieces

From the repository root, install the locked dependencies:

```bash
uv sync --locked
```

The repository includes `data/processed/matches.csv` and `data/processed/features.csv`. With those files you can reproduce the earlier Bet365 results, including:

```bash
uv run python -m football_odds_predictor.training.baseline --model xgboost --max-depth 4 --n-estimators 400
uv run python -m football_odds_predictor.evaluation.final_test
```

The sporting experiment also needs the original match CSVs. Download the Premier League (`E0`), Bundesliga (`D1`), and La Liga (`SP1`) files for 2021/22 through 2025/26 and save them under `data/raw/football-data/` as `<season>-<division>.csv`, for example `2122-E0.csv`. To rebuild the processed tables, the Elo warm-up also uses the 2020/21 files in `data/raw/football-data-warmup/`. [Football-Data.co.uk download index](https://football-data.co.uk/downloadm.php)

```bash
uv run python -m football_odds_predictor.data.player_archive --download
uv run python -m football_odds_predictor.training.sporting
```

Preview upcoming capture times in UTC with `--list`. Start the collector without that flag and leave it running. It finds matches automatically and writes each successful snapshot to `data/raw/superbet/<event-id>.json`.

```bash
uv run python -m football_odds_predictor.data.superbet_watch --list
uv run python -m football_odds_predictor.data.superbet_watch
```

Try the offer calculator with a hypothetical Superbet quote at 1.90:

```bash
uv run python -m football_odds_predictor.evaluation.offer --odds 1.90 \
  --win-probability 0.60 --conservative-win-probability 0.55 --stake 2 --budget 100
```

## Where to start reading

| Code | What to look for |
| --- | --- |
| [Match preprocessing](src/football_odds_predictor/data/preprocessing.py) and [historical features](src/football_odds_predictor/data/features.py) | How raw match rows become the normalized table, then how earlier matches update history and Elo. |
| [Shared dataset and folds](src/football_odds_predictor/data/dataset.py) | How rows are matched and how training stays earlier than validation or test. |
| [Bet365 training runner](src/football_odds_predictor/training/baseline.py) | How the median, Ridge, XGBoost, and LSTM runs reuse those folds and metrics. |
| [Sporting inputs](src/football_odds_predictor/data/sporting.py) and [sporting training](src/football_odds_predictor/training/sporting.py) | What each model sees before the fixture, where observed labels enter, and how the SGD epoch is selected. |
| [Superbet watcher](src/football_odds_predictor/data/superbet_watch.py) and [snapshot capture](src/football_odds_predictor/data/superbet_snapshot.py) | How the collector discovers fixtures, decides when to capture, and keeps raw prices and simple promotions. |
| [Offer assessment](src/football_odds_predictor/evaluation/offer.py) | How the offered price and model probabilities turn into estimated return, risk, and a color. |

The remaining integration work is to check sporting probability calibration and the held-out season, save the selected models, and match predictions to exact selections in the terminal report. A virtual-budget simulation is a possible later experiment. Raw captures stay local.
