# Contributing to football-odds-predictor

Set up the project, reproduce the experiments, and view a Superbet match report. Start with a clone of the repository. The processed match and feature CSVs are included in the clone. The raw inputs, saved models, and offer snapshots are local to each contributor.

## 1. Clone and install

Clone the repository with the URL from GitHub's **Code** menu. Open a terminal in the cloned folder and install the locked dependencies:

```bash
uv sync --locked
```

## 2. Get the input data

The repository includes `data/processed/matches.csv` and `data/processed/features.csv`. These are enough to run the original Bet365 experiments. The Superbet workflow also needs the original CSVs for Premier League (`E0`), Bundesliga (`D1`), and La Liga (`SP1`). Each download is named by its division, so rename it locally to include the season. For example, save Premier League 2021/22 as `2122-E0.csv`. [Football-Data.co.uk download index](https://football-data.co.uk/downloadm.php)

```bash
mkdir -p data/raw/football-data data/raw/football-data-warmup data/raw/football-data-2627
```

| Source seasons | Local folder | Local names |
| --- | --- | --- |
| 2021/22–2025/26 | `data/raw/football-data/` | `<season>-<division>.csv`, with season codes `2122`, `2223`, `2324`, `2425`, `2526` |
| 2020/21, Elo warm-up | `data/raw/football-data-warmup/` | `2021-E0.csv`, `2021-D1.csv`, `2021-SP1.csv` |
| 2026/27, recent team results | `data/raw/football-data-2627/` | `2627-E0.csv`, `2627-D1.csv`, `2627-SP1.csv` |

Download the five-season player archive:

```bash
uv run python -m football_odds_predictor.data.player_archive --download
```

This saves 45 CSVs under `data/raw/xgtables-understat/` and prints counts by league. All raw files stay local.

## 3. Reproduce results and save the models

To regenerate the two processed CSVs from the original files:

```bash
uv run python -m football_odds_predictor.data.preprocessing
uv run python -m football_odds_predictor.data.features --elo-warmup-dir data/raw/football-data-warmup
```

These commands overwrite the generated CSVs locally. The original dataset yields **5,330 fixtures** and **5,075 complete histories**.

Run the research experiments when you want to reproduce their scores:

```bash
uv run python -m football_odds_predictor.training.baseline --model xgboost --max-depth 4 --n-estimators 400
uv run python -m football_odds_predictor.evaluation.final_test
uv run python -m football_odds_predictor.training.sporting
```

The first two commands print Bet365 odds errors for validation and the final test. The sporting command prints validation log loss and Brier scores. To reproduce the Bet365 replay on recent fixtures, run `uv run python -m football_odds_predictor.evaluation.recent` with the recent and warm-up folders populated.

To prepare the match reports, fit the selected sporting models once:

```bash
uv run python -m football_odds_predictor.training.fit_sporting
```

Expect `Saved 13 trained models to data/processed/sporting-models.pkl`. Every contributor generates this file locally. The dashboard loads it at startup. Refit when the training data or model settings change. Load only a model file you generated yourself.

The dashboard can also load locally generated count bundles: `fit_cards` adds domestic card forecasts, `fit_corners` adds domestic corner forecasts, and `fit_ucl` adds Champions League and Europa League forecasts, including referee-aware European card models. `fit_corners` uses the same Football-Data CSVs and processed Elo features already listed above. The extra card and European inputs are listed in the following sections. Run each fit command after downloading its inputs; you do not need to retrain models for every dashboard query.

To reproduce the [sporting calibration and final test](docs/superbet-assessment.md#calibration-check-and-final-sporting-test), run:

```bash
uv run python -m football_odds_predictor.evaluation.sporting validation
uv run python -m football_odds_predictor.evaluation.sporting test
```

The terminal prints log loss, Brier score, and the calibration gap for each selection. The detailed probability bands are saved in `data/processed/sporting-reliability-{validation,test}.csv`.

## 4. Capture an offer

Preview upcoming capture times in UTC. Then leave the watcher running in one terminal:

```bash
uv run python -m football_odds_predictor.data.superbet_watch --list
uv run python -m football_odds_predictor.data.superbet_watch
```

The watcher discovers fixtures in Premier League, Bundesliga, La Liga, Champions League, and Europa League. It refreshes the schedule hourly and attempts a capture about 15 minutes before kickoff. On success it prints `Saved … markets and … simple promotions to data/raw/superbet/<event-id>.json`. The accepted window is 10–20 minutes before kickoff. Stop the watcher with **Ctrl+C**. The 26 domestic leagues used for European team-history coverage are a separate historical-data scope; the live odds watcher does not currently cover all of them.

### Referee inputs

Football-Data supplies Premier League referees. To add referee variants for
Bundesliga and La Liga, save the [Transfermarkt games archive](https://github.com/dcaribou/transfermarkt-datasets)
locally and rebuild the card bundle:

```bash
mkdir -p data/raw/transfermarkt
curl -fL https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/games.csv.gz -o data/raw/transfermarkt/games.csv.gz
uv run python -m football_odds_predictor.training.fit_cards
```

If a timed offer omits the assigned referee, the report can read
`data/raw/referee-appointments.csv` with columns `event_id,referee,observed_at_utc`.
Use the name as written in the league archive and the UTC timestamp at which
the assignment was observed. The report uses the most recent assignment
recorded by the offer timestamp, with a team-only fallback.

The watcher also captures Champions League and Europa League offers. With the
same Transfermarkt archive, add recent results and rebuild the European models:

```bash
mkdir -p data/raw/open-matchday data/raw/openfootball
curl -fL https://codeload.github.com/oladeeayo/open-matchday/zip/refs/heads/master -o data/raw/open-matchday/results-2627.zip
curl -fL https://openfootball.github.io/europe/slovakia_2023-24_sk1.json -o data/raw/openfootball/slovakia_2023-24_sk1.json
curl -fL https://openfootball.github.io/europe/slovakia_2024-25_sk1.json -o data/raw/openfootball/slovakia_2024-25_sk1.json
curl -fL https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/game_events.csv.gz -o data/raw/transfermarkt/game_events.csv.gz
curl -fL https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/game_lineups.csv.gz -o data/raw/transfermarkt/game_lineups.csv.gz
uv run python -m football_odds_predictor.training.fit_ucl
```

Restart the dashboard and watcher. The European models cover 1X2, full-match goals and the home, away and match
card totals in both competitions. European card forecasts require the two
Transfermarkt event and lineup files. See the
[European model plan](docs/european-model-plan.md) for coverage and evaluation.

### European corner inputs

Download the pinned [11footy release](https://github.com/fm-devlab/11footy-dataset/releases/tag/11footy-2026-11-10)
alongside the match results above. Credit 11footy for these archives (CC BY-NC 4.0).

```bash
mkdir -p data/raw
release=11footy-2026-11-10
for part in european-cups.{champions-league,europa-league,conference-league,super-cup} \
    england.premier-league germany.bundesliga spain.la-liga italy.serie-a france.ligue-1; do
    curl -fL "https://github.com/fm-devlab/11footy-dataset/releases/download/$release/$part.csv.zip" \
        -o "data/raw/$release-$part.csv.zip" || break
done
uv run python -m football_odds_predictor.training.fit_europe_corners
```

This builds `data/processed/europe-corner-models.pkl`. Restart the dashboard
after fitting. The bundle stores the results available at build time. Update
the local results archives and rebuild it as new results become available.

## 5. Run the daily workflow and browse reports

After setup and model fitting, open two terminals in the project folder and leave both commands running:

**Terminal 1 — collect odds**

```bash
uv run python -m football_odds_predictor.data.superbet_watch
```

**Terminal 2 — open the dashboard**

```bash
uv run python -m football_odds_predictor.evaluation.dashboard
```

Open [the local dashboard](http://127.0.0.1:8765). It automatically lists today's upcoming fixtures from the Superbet schedule and combines them with local captures for today. The dashboard refreshes the schedule every two minutes; the browser refreshes its fixture list every 30 seconds. The watcher refreshes its schedule hourly and attempts one odds capture per fixture about 15 minutes before kickoff, in the 10–20 minute window. It does not keep updating a quote after that capture. The first fixture is selected automatically; click another fixture to open its report. A fixture without a saved price capture stays listed as pending. The displayed day uses Amsterdam time, while kickoff times are formatted in your browser's local time. The schedule endpoint lists upcoming fixtures, so already-started matches appear only when a local capture exists.

Keep both terminals open: the dashboard displays the schedule and saved data, while the watcher performs the odds capture. Stop each process with **Ctrl+C**. The watcher covers five competitions; the 26 domestic leagues in the European history archive are not all live-odds competitions.

The dashboard has tabs for 1X2, goals, cards, and corners. Player shots are not shown there; they remain available in the sporting experiments and optional terminal report. The sensitivity selector applies an optional absolute probability reduction (for example, 5 percentage points means 60% becomes 55%). The color signal is intentionally blank with no adjustment; it is provisional under a chosen adjustment and has not been validated as a betting strategy. The watcher must run separately for new odds to be captured. Restart the dashboard after regenerating model files or updating historical results.

### Optional terminal report

Open a second terminal in the project folder:

```bash
uv run python -m football_odds_predictor.evaluation.superbet_report
```

This opens the most recently saved capture. To choose a match, pass its JSON path. Add `--market 1x2`, `--market goals`, `--market cards`, `--market corners`, or `--market shots` to filter the output. Repeat `--market` for multiple sections. Use `--player "Dewsbury-Hall"` for a player name filter.

The output starts with kickoff and capture time, then groups selections by market. Player shots are still supported in this terminal report, even though the dashboard tab is hidden. For example, the Brentford–Everton probe with `--stress-pp 5` included:

```text
[Match goals (2.5)]
Selection  Model odds   Superbet    Promo  Loss risk  Est. return   Prudent    Signal
Over             1.67       1.85        —      40.2%       +10.6%     +1.4%    orange
```

- **Model odds** are the estimated break-even decimal price. **Loss risk** is the estimated chance of losing. **Superbet / Promo** show the standard and matching promotional prices.
- **Est. return** uses Promo when available and the standard Superbet price otherwise. **Prudent / Color** appear when you pass a sensitivity setting such as `--stress-pp 5`, which reduces the win probability by five percentage points. The colors are provisional.
- Signal labels have colored backgrounds in interactive terminals. Use `--color always` to force highlighting or `--color never` for plain text.
- The yellow-card section shows model estimates pending settlement mapping. Player shot model odds assume the player takes part. Check the latest team and player result dates printed by the report.

If you see **No saved captures yet**, leave the watcher running through an eligible fixture. If you see **Model file missing**, run the one-time fit in step 3. If a team cannot be matched or has incomplete history, check the results files and their team names.

To check the calculation while waiting for a capture, try the calculator with hypothetical inputs:

```bash
uv run python -m football_odds_predictor.evaluation.offer --odds 1.90 \
  --win-probability 0.60 --conservative-win-probability 0.55 --stake 2 --budget 100
```

Expect `Signal: orange`, `Estimated return: +14.0%`, `Conservative return: +4.5%`, and `Estimated loss probability: 40.0%`.

## Where to start reading

| Code | What to look for |
| --- | --- |
| [Match preprocessing](src/football_odds_predictor/data/preprocessing.py) and [historical features](src/football_odds_predictor/data/features.py) | How raw match rows become the normalized table, then how earlier matches update history and Elo. |
| [Shared dataset and folds](src/football_odds_predictor/data/dataset.py) | How rows are matched and how training stays earlier than validation or test. |
| [Bet365 training runner](src/football_odds_predictor/training/baseline.py) | How the median, Ridge, XGBoost, and LSTM runs reuse those folds and metrics. |
| [Sporting inputs](src/football_odds_predictor/data/sporting.py) and [sporting training](src/football_odds_predictor/training/sporting.py) | What each model sees before the fixture, where observed labels enter, and how the SGD epoch is selected. |
| [Superbet watcher](src/football_odds_predictor/data/superbet_watch.py) and [snapshot capture](src/football_odds_predictor/data/superbet_snapshot.py) | How the collector discovers fixtures, decides when to capture, and keeps raw prices and simple promotions. |
| [Offer assessment](src/football_odds_predictor/evaluation/offer.py) | How the offered price and model probabilities turn into estimated return, risk, and a color. |
| [Portfolio evaluation](src/football_odds_predictor/evaluation/portfolio.py) | How placements, results, and refunds affect a virtual budget over time. The [worked example](docs/superbet-assessment.md#virtual-portfolio-evaluation) includes a CSV and command. |

See the [terminal report notes](docs/superbet-assessment.md#terminal-report) for matching rules and remaining integration work.

## Before opening a pull request

Run `git diff --check` and rerun the command affected by your change. For model experiments, record the dataset version, split, settings, and results in the relevant project plan. Keep raw inputs, captured offers, and the locally fitted model file out of the commit. See [AGENTS.md](AGENTS.md) for the research workflow.
