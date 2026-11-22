# Football odds prediction

V1 predicts one bookmaker's pre-match closing odds for home win, draw, and away win (1X2). Use the triplet the source labels as closing odds.

1. **Data and EDA.** Find a source permitted for model training with about five years of closing odds, results, and match statistics across multiple leagues. An initial audit found complete Bet365 closing triplets for 5,330 matches across the Premier League, Bundesliga, and La Liga. The source and bookmaker are still open pending permitted use and coverage.
2. **Preprocessing and features.** Build a five-match history for each team, with past match statistics and historical Elo. Use only results, statistics, and other pre-match information available when the prediction is made. Keep the closing odds in the target.
3. **Training and evaluation.** Compare statistical baselines, XGBoost, and a PyTorch LSTM on the same fixtures. Use chronological train, validation, and test splits. Fit preprocessing on training data and use validation to choose models before evaluating on the test set.
4. **Upcoming matches.** Reuse the feature pipeline with newly collected statistics and start with batch predictions once the offline evaluation is complete.
