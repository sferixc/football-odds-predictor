# Football odds prediction

V1 predicts one bookmaker's pre-match closing odds for home win, draw, and away win (1X2). Use the triplet the source labels as closing odds.

1. **Data and EDA.** Use the free Football-Data.co.uk CSVs for the student research experiment. The 15 local files in `data/raw/football-data/` cover 2021/22–2025/26 for Premier League (`E0`), Bundesliga (`D1`), and La Liga (`SP1`). Use the Bet365 closing columns `B365CH`, `B365CD`, and `B365CA`. [Download index](https://football-data.co.uk/downloadm.php) · [Data usage policy](https://football-data.co.uk/contact.php)
2. **Preprocessing and features.** Build a five-match history for each team, with past match statistics and historical Elo. Use only results, statistics, and other pre-match information available when the prediction is made. Keep the closing odds in the target.
3. **Training and evaluation.** Compare statistical baselines, XGBoost, and a PyTorch LSTM on the same fixtures. Use chronological train, validation, and test splits. Fit preprocessing on training data and use validation to choose models before evaluating on the test set.
4. **Upcoming matches.** Reuse the feature pipeline with newly collected statistics and start with batch predictions once the offline evaluation is complete.

## EDA results

Run `uv run python -m football_odds_predictor.preprocessing` to build `data/processed/matches.csv` from the raw season files. The output preserves missing statistics and keeps kickoff as the source clock time without assigning a time zone.

| League | Fixtures | Valid closing 1X2 | Complete match stats | Complete five-match histories |
| --- | ---: | ---: | ---: | ---: |
| Premier League | 1,900 | 1,900 | 1,900 | 1,815 |
| Bundesliga | 1,530 | 1,530 | 1,529 | 1,440 |
| La Liga | 1,900 | 1,900 | 1,900 | 1,820 |
| **Total** | **5,330** | **5,330** | **5,329** | **5,075** |

All kickoff strings parsed, and no fixture was duplicated on league, date, home team, and away team. Odds ranged from 1.04 to 29.00. Union Berlin–Bochum on 2024-12-14 was the only fixture missing shots, shots on target, and corners. Goals were present for all fixtures.

The history check uses each team's latest five league matches on earlier calendar dates and carries history across seasons. It requires goals, shots, shots on target, and corners for both sides in each of the five matches. A total of 245 fixtures had fewer than five earlier matches for at least one team. The missing Bundesliga stats affected another 10 fixtures, which explains the difference from the initial count of 5,085 that skipped incomplete prior matches. An illustrative three-season train, one-season validation, and one-season test split gives 2,988 / 1,036 / 1,051 eligible fixtures. This date-based count is an estimate of feature coverage until we define when predictions are made.

## V1 feature build

Run `uv run python -m football_odds_predictor.features --elo-warmup-dir data/raw/football-data-warmup` to generate `data/processed/features.csv` from the normalized match table. The warm-up directory contains the source's 2020/21 files, saved as `2021-E0.csv`, `2021-D1.csv`, and `2021-SP1.csv`. Those results update Elo before 2021/22. The output rows and five-match histories still start in 2021/22.

Each row has pre-match Elo and two five-match sequences, oldest first. History and Elo carry over between seasons within the same league. Each team's first rating is 1500, with K=20 and no home advantage adjustment. Incomplete histories stay in the output with `history_complete=false`. Closing odds stay in the normalized table, which can be joined on league, season, source kickoff, and both teams.

V1 assumes results and stats from earlier source dates are available when predicting on the fixture date. The archive lacks publication times, so this is an approximation. Fixtures from the prediction date never contribute results or stats to that day's predictions.

## Validation and median baseline

**Question.** How accurately can we estimate closing odds using only typical historical odds? This gives us a reference for measuring whether Elo and recent form improve predictions.

Run `uv run python -m football_odds_predictor.baseline` on the processed CSVs (dataset version `01551de`). Join on the fixture key and keep complete histories and valid 1X2 targets. For each fold, predict the three training-set median decimal odds for every validation fixture. This baseline uses no match features or random seed.

**Worked example.** For Arsenal–Chelsea on 2 May 2023, sort the home-win odds from the 919 eligible training matches in 2021/22 and take the median. Repeat for draws and away wins. This gives the same three predictions for every 2022/23 validation match. The actual Arsenal–Chelsea odds are used to score those predictions.

| Outcome | Predicted odds | Actual closing odds | Absolute error |
| --- | ---: | ---: | ---: |
| Arsenal win (1) | 2.15 | 1.57 | 0.58 |
| Draw (X) | 3.60 | 4.33 | 0.73 |
| Chelsea win (2) | 3.40 | 5.50 | 2.10 |

The decimal-odds MAE for this match is `(0.58 + 0.73 + 2.10) / 3 ≈ 1.14`. For the primary metric, log-odds MAE, take the natural log of each predicted and actual decimal odd before measuring the absolute difference. For example, changing 2 to 4 gives the same log error as changing 4 to 8. Squared error would give large misses more weight.

**Results.** Each row measures the error on the next season using only earlier seasons to calculate the medians.

| Validation | Training seasons | Train / validation fixtures | Log-odds MAE | Decimal-odds MAE |
| --- | --- | ---: | ---: | ---: |
| 2022/23 | 2021/22 | 919 / 1,028 | 0.3215 | 1.1842 |
| 2023/24 | 2021/22–2022/23 | 1,947 / 1,041 | 0.3495 | 1.3477 |
| 2024/25 | 2021/22–2023/24 | 2,988 / 1,036 | 0.3479 | 1.3325 |
| Fold mean | | | 0.3396 | 1.2881 |

Both metrics average absolute errors over fixtures and the three outcomes within each fold. Reserve 2025/26 for the final test after model selection.

The reference to beat is **0.3396 mean log-odds MAE** across the three folds. Compare the regression, XGBoost, and LSTM on those same fixtures and metric.
