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
