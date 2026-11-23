# Football odds prediction

V1 predicts one bookmaker's pre-match closing odds for home win, draw, and away win (1X2). Use the triplet the source labels as closing odds.

1. **Data and EDA.** Use the free Football-Data.co.uk CSVs for the student research experiment. The 15 local files in `data/raw/football-data/` cover 2021/22–2025/26 for Premier League (`E0`), Bundesliga (`D1`), and La Liga (`SP1`). Use the Bet365 closing columns `B365CH`, `B365CD`, and `B365CA`. [Download index](https://football-data.co.uk/downloadm.php) · [Data usage policy](https://football-data.co.uk/contact.php)
2. **Preprocessing and features.** Build a five-match history for each team, with past match statistics and historical Elo. Use only results, statistics, and other pre-match information available when the prediction is made. Keep the closing odds in the target.
3. **Training and evaluation.** Compare statistical baselines, XGBoost, and a PyTorch LSTM on the same fixtures. Use chronological train, validation, and test splits. Fit preprocessing on training data and use validation to choose models before evaluating on the test set.
4. **Upcoming matches.** Reuse the feature pipeline with newly collected statistics and start with batch predictions once the offline evaluation is complete.

## Code layout

- `data/` reads the raw CSVs, builds features, joins odds, and defines validation folds.
- `models/` contains the median, Ridge, and XGBoost predictors.
- `training/` runs each experiment across the folds.
- `evaluation/` computes the shared error metrics.

These directories are inside `src/football_odds_predictor/`. The original module commands below also work.

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

## Ridge regression baseline

**Question.** How much does adding Elo and recent form improve on the median?

Run `uv run python -m football_odds_predictor.baseline --model ridge` on dataset `01551de`. Use 22 inputs per fixture, the two current Elo ratings and ten five-match averages per team. In each fold, fit the scaler and Ridge regression (`alpha=1.0`) on training fixtures. Fit three log-odds targets, convert predictions back to decimal odds, and apply a minimum of 1.01. This run is deterministic.

**Worked example.** Before Arsenal–Chelsea on 2 May 2023, Arsenal had Elo 1663 and averaged 2.4 goals scored in its five previous matches. Chelsea had Elo 1517 and averaged 0.2. With the full set of inputs, Ridge predicted **1.54 / 5.19 / 7.01**. The actual prices were **1.57 / 4.33 / 5.50**.

| Validation | Median log-odds MAE | Ridge log-odds MAE | Ridge decimal-odds MAE |
| --- | ---: | ---: | ---: |
| 2022/23 | 0.3215 | 0.1644 | 0.6414 |
| 2023/24 | 0.3495 | 0.1657 | 0.6821 |
| 2024/25 | 0.3479 | 0.1551 | 0.6520 |
| Fold mean | 0.3396 | **0.1617** | **0.6585** |

Ridge cut mean log-odds MAE by about 52% and improved every validation season. It is the current reference for XGBoost and LSTM. The 2025/26 test season remains reserved.

## XGBoost

**Question.** Do decision trees using the five matches individually improve on Ridge with averages?

Run `uv run python -m football_odds_predictor.baseline --model xgboost` on dataset `01551de`. Each fixture has two current Elo ratings and 100 historical values. Fit a separate model for each of the three log closing odds. Each model uses 200 trees, maximum depth 3, learning rate 0.05, and seed 42. Convert predictions back to decimal odds with a floor of 1.01.

**Worked example.** For Arsenal–Chelsea, XGBoost sees Arsenal's goals scored as five separate values, `4, 2, 2, 3, 1`. Ridge summarizes them as `2.4`. XGBoost predicted **1.47 / 5.67 / 8.02** against actual odds **1.57 / 4.33 / 5.50**. It missed this particular match by more than Ridge on average across the three odds.

| Validation | Ridge log-odds MAE | XGBoost log-odds MAE | XGBoost decimal-odds MAE |
| --- | ---: | ---: | ---: |
| 2022/23 | 0.1644 | 0.1559 | 0.6389 |
| 2023/24 | 0.1657 | 0.1619 | 0.6866 |
| 2024/25 | 0.1551 | 0.1356 | 0.5801 |
| Fold mean | 0.1617 | **0.1512** | **0.6352** |

XGBoost improved the primary metric in every fold and lowered its fold mean by about 6.5%. The experiment changes both model and input representation, so it measures their combined effect. The 2025/26 test season remains reserved.

### XGBoost parameter search

**Question.** Does changing tree depth or the number of trees improve validation error? Try all nine pairs on the same dataset and folds as above. Keep the inputs, targets, learning rate, seed, and thread count fixed.

```bash
for depth in 2 3 4; do
  for trees in 100 200 400; do
    echo "Depth $depth, trees $trees"
    uv run python -m football_odds_predictor.baseline --model xgboost \
      --max-depth "$depth" --n-estimators "$trees"
  done
done
```

| Maximum depth | 100 trees | 200 trees | 400 trees |
| ---: | ---: | ---: | ---: |
| 2 | 0.1629 | 0.1552 | 0.1532 |
| 3 | 0.1554 | 0.1512 | 0.1506 |
| 4 | 0.1526 | 0.1505 | **0.1502** |

**Worked example.** Depth 4 with 400 trees scored 0.1569, 0.1613, and 0.1324 on the three validation seasons. Their average was 0.1502 log-odds MAE, about 0.6% lower than the original depth 3 with 200 trees. It scored slightly worse than the original in 2022/23. Use the winning settings as the tuned comparison and keep the original command defaults for reproducing the first run. The 2025/26 season was held out.

## Small LSTM

**Question.** Does learning a representation of each five-match sequence improve on the tuned XGBoost?

Run `uv run python -m football_odds_predictor.training.baseline --model lstm` on dataset `01551de`. A shared one-layer LSTM with 16 hidden units reads each team's five matches in order. A linear head uses the two resulting vectors and current Elo to predict three log closing odds. Fit feature scaling on each training fold. Start each fold from seed 42 and train on CPU with shuffled batches of 32 fixtures, mean squared error on log odds, and AdamW with learning rate 0.001 and weight decay 0.01. Train up to 100 epochs and retain predictions from the epoch with the lowest validation log-odds MAE. Stop after 12 epochs without improvement. Decimal predictions have a minimum of 1.01.

**Worked example.** For Arsenal–Chelsea, the model reads Arsenal's sequence starting with the 4–1 and ending with the 1–4, plus Chelsea's five matches and the two current Elo ratings. It predicted **1.52 / 3.94 / 4.79** against actual odds **1.57 / 4.33 / 5.50**.

| Validation | Best epoch | LSTM train log-odds MAE | LSTM validation log-odds MAE | Tuned XGBoost validation log-odds MAE |
| --- | ---: | ---: | ---: | ---: |
| 2022/23 | 32 | 0.1596 | 0.1620 | 0.1569 |
| 2023/24 | 29 | 0.1499 | 0.1617 | 0.1613 |
| 2024/25 | 23 | 0.1508 | 0.1497 | 0.1324 |
| Fold mean | | | **0.1578** | **0.1502** |

The LSTM's mean decimal-odds MAE was 0.6489. It improved on Ridge on average, and tuned XGBoost led on the primary metric in all three folds. This comparison covers one LSTM setup and seed. The 2025/26 test season remains reserved.
