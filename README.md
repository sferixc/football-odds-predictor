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
| Match dashboard | Browse saved Superbet captures and compare model odds with Superbet prices for supported markets. |
| Virtual portfolio evaluator | Replays a CSV of simulated selections and reports profit, return, open stakes, and drawdown. |

The [Bet365 experiment notes](docs/project-plan.md) describe the first research track and its results. The [Superbet assessment plan](docs/superbet-assessment.md) explains the current track, its data sources, worked examples, and results.

## Getting started

Use the [contributor guide](CONTRIBUTING.md) to install dependencies, download the raw inputs, train the models locally, and open the local dashboard or run a terminal report. The clone includes `data/processed/matches.csv` and `data/processed/features.csv`, so you can reproduce the Bet365 experiments using the included tables.

Start the local dashboard from the project folder after fitting the models:

```bash
uv run python -m football_odds_predictor.evaluation.dashboard
```

Open [the dashboard](http://127.0.0.1:8765), choose a saved match, and switch between markets. Keep the command running while you use the dashboard.
