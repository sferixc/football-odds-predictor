# European match models

## Results and club identity

- Use Transfermarkt club IDs across domestic leagues, Champions League and Europa League. Qualifiers and Conference League also contribute to team history. Keep domestic, UCL and Europa League targets separate when comparing models.
- The local inputs are `data/raw/transfermarkt/games.csv.gz`, `game_events.csv.gz`, and `game_lineups.csv.gz`; current results come from `data/raw/open-matchday/results-2627.zip` and the available Slovak seasons are in `data/raw/openfootball/`. Raw files are ignored by Git, so each contributor downloads them locally. Log the latest known result per club and audit unmatched clubs before expanding market coverage.
- Give each club one Elo state. Update both clubs after the match day. Early results warm up ratings before the training window. An optional dated Elo seed can initialize that same state. Evaluate an external seed only once a historical snapshot and club ID join have been verified.
- Maintain one Elo state per club across domestic and UEFA matches, rather than separate domestic and European ratings. For each fixture, use the last five available matches across competitions. Store the opponent's pre-match Elo and the source competition with each observation. Use the latest state strictly before the incoming fixture date. If either team has fewer than five matches or last played more than 60 days ago in the archive, the European forecast is unavailable and the report falls back to displayed bookmaker prices.

## Training and evaluation

- Fit 1X2 and full-time goals separately for UCL and Europa League, using league/group-stage matches as targets so the score corresponds to regulation time. Fit on 2021/22–2024/25. Use 2024/25 to choose the goal model and keep 2025/26 for final evaluation.
- Compare the combined history with the Europe-only Elo and form baseline. The current combined 1X2 model is experimental. It scored 0.9513 versus 0.9408 log loss for UCL and 0.9615 versus 0.9449 for Europa League on 144 fixtures each. Lower is better.
- Audit fixture and referee coverage each time the artifact is rebuilt. The current snapshot has domestic results for 114 of 127 identified clubs since 2024/25, across 26 domestic leagues. For each European competition, the 377 historical fixtures have referee names. The 72 current-season fixtures in the result export lack referee assignments.
- Next evaluate which inputs help using expanding validation folds. Compare shared Elo alone with shared Elo plus recent form and opponent strength. Stratify by tournament, domestic coverage, and the age of each team's latest result. Keep subsequent seasons for a fresh final comparison.

## European cards and remaining markets

- European 1X2 and full-time goals are implemented for Champions League and Europa League. The saved bundle from `training.fit_ucl` also contains referee-aware, Poisson-style card-count forecasts for home, away, and match totals. It selects between referee and team-only variants using validation; when referee history is unavailable, it uses the team-only model.
- The card inputs are reconstructed from Transfermarkt events and lineups by game ID. Active-player lineups are inferred from starters and substitutions; incomplete or ambiguous records are excluded. The feed records 90th-minute events at minute 90, so stoppage time cannot always be distinguished from the final whistle. These labels have not been reconciled with actual bookmaker settlements. [Superbet settlement rules](https://superbet.ro/wiki/regulament-de-joc-pariuri-cap-7)
- Obtain dated, match-level UEFA corner counts and first-half results for multiple seasons before fitting those models. The current Open Matchday European files lack those fields.
- Player total-shot models remain in the sporting code and terminal report, but the dashboard's Player shots tab has been removed. Participation and settlement still need to be modeled before using those estimates as value signals.
- Domestic models still use their existing trained features. Migrate them to shared cross-competition features only after retraining and checking chronological metrics for those markets.

## Reproduce

See [the contributor guide](../CONTRIBUTING.md) for downloads. Rebuild after updating local results:

```bash
uv run python -m football_odds_predictor.training.fit_ucl
```

The command writes a model bundle and `data/processed/europe-coverage.csv` for auditing. An optional historical seed goes at `data/raw/europe-elo-2020.csv` with columns `club_id,as_of,elo`, keyed by Transfermarkt IDs and dated `2020-08-01`.

The local run used the Transfermarkt games archive with SHA-256
`142561989017d379bcf9e72bad0b87e0996bae0e7b710ebb3def9255f464c741`
and the Open Matchday export with SHA-256
`ea1365fb53523133fceb0f70046f60e31261dfcc02159f0240ad231f559af41e`.
The two Slovak inputs are the 2023/24 and 2024/25 archives linked in the
contributor guide. [Transfermarkt](https://github.com/dcaribou/transfermarkt-datasets) ·
[Open Matchday](https://github.com/oladeeayo/open-matchday) ·
[OpenFootball](https://openfootball.github.io/europe/)

The card run reconstructed 1,110 of 1,152 historical fixtures. In 2024/25
validation, mean Brier for match totals moved from 0.1305 to 0.1302 in UCL and
from 0.1466 to 0.1385 in Europa League when adding referee history. The paired
sample sizes were 140 and 101, respectively. In the held-out 2025/26 season, the
selected models with team-only fallbacks scored 0.1237 across 141 UCL fixtures
and 0.1223 across 139 Europa League fixtures for match totals. Lower is better.

The archive files used were `game_events.csv.gz` with SHA-256
`80eb3b2a85c780b2caaf1e8850c3d16de7f8b1240b53aaf415bdc56b641d7d68`
and `game_lineups.csv.gz` with SHA-256
`5e83d6fad28364aadfea608013700cf09990062eede1ad9ef189189b0af47ba2`.
The saved Arsenal–Dortmund probe had open total-card quotes. The Celta
Vigo–Bournemouth probe had none, so the report shows model odds for that match
and leaves the signal blank.
