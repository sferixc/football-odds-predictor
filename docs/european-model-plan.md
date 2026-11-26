# European match models

## Results and club identity

- Use Transfermarkt club IDs across domestic leagues, Champions League and Europa League. Qualifiers and Conference League also contribute to team history. Keep domestic, UCL and Europa League targets separate when comparing models.
- The local inputs are `data/raw/transfermarkt/games.csv.gz`, `game_events.csv.gz`, and `game_lineups.csv.gz`; current results come from `data/raw/open-matchday/results-2627.zip` and the available Slovak seasons are in `data/raw/openfootball/`. Raw files are ignored by Git, so each contributor downloads them locally. Log the latest known result per club and audit unmatched clubs before expanding market coverage.
- Give each club one Elo state. Start from the most recent verified public rating before the club's first match in the shared timeline from August 2020 or its later debut. Update both clubs after the match day.
- Maintain one Elo state per club across domestic and UEFA matches. For each fixture, use the last five available matches across competitions. Store the opponent's pre-match Elo and the source competition with each observation. Use the latest state strictly before the earlier of the capture and kickoff dates. If either team has fewer than five matches or last played more than 60 days before that cutoff in the archive, the European forecast is unavailable and the report falls back to displayed bookmaker prices.

## Training and evaluation

- Fit 1X2 and full-time goals separately for UCL and Europa League, using league/group-stage matches as targets so the score corresponds to regulation time. Fit on 2021/22–2024/25. Use 2024/25 to choose the goal model and keep 2025/26 for final evaluation.
- Compare the combined history with the Europe-only Elo and form baseline. The current combined 1X2 model scored 0.9358 versus 0.9408 log loss for UCL and 0.9587 versus 0.9449 for Europa League on 144 fixtures each. Lower is better.
- Audit fixture and referee coverage each time the artifact is rebuilt. For each European competition, the 377 historical fixtures have referee names. The 72 current-season fixtures in the result export lack referee assignments.
- Next evaluate which inputs help using expanding validation folds. Compare shared Elo alone with shared Elo plus recent form and opponent strength. Stratify by tournament, domestic coverage, and the age of each team's latest result. Keep subsequent seasons for a fresh final comparison.

## European cards and remaining markets

- European 1X2 and full-time goals are implemented for Champions League and Europa League. The saved bundle from `training.fit_ucl` also contains Poisson card-count forecasts for home, away, and match totals. The dashboard uses the referee variant whenever the assigned referee has at least three measured matches before kickoff day. It uses the team variant if the assignment or history is unavailable. The validation comparison below describes the original model selection experiment.
- The card inputs are reconstructed from Transfermarkt events and lineups by game ID. Active-player lineups are inferred from starters and substitutions; incomplete or ambiguous records are excluded. The feed records 90th-minute events at minute 90, so stoppage time cannot always be distinguished from the final whistle. These labels have not been reconciled with actual bookmaker settlements. [Superbet settlement rules](https://superbet.ro/wiki/regulament-de-joc-pariuri-cap-7)
- `training.fit_europe_corners` fits Poisson regressions for home, away, and match corners in UCL and UEL on regulation-time counts from [11footy 2026-11-10](https://github.com/fm-devlab/11footy-dataset/releases/tag/11footy-2026-11-10). Team form uses the five most recent recorded results strictly before kickoff day across domestic leagues, known cups, and UEFA competitions. The model uses mean goal margin, mean corners for and against where known, the number of observed corner records, and days since the last match. It requires all five results and at least one measured corner total in that window for each club. The dashboard shows that coverage beside the model and Superbet odds. Coverage varies by club and competition.
- Corner evaluation uses expanding validation folds and a held-out 2025/26 season. After matching additional club aliases, average match-total Brier was 0.1040 vs 0.1042 for the mean-count baseline on 130 UCL fixtures, and 0.0922 vs 0.0902 on 111 UEL fixtures. Lower is better. The colors are experimental.
- First-half models still need regulation-time, half-specific results before training.
- Player total-shot models remain in the sporting code and terminal report, but the dashboard's Player shots tab has been removed. Participation and settlement still need to be modeled before using those estimates as value signals.
- Domestic models still use their existing trained features. Migrate them to shared cross-competition features only after retraining and checking chronological metrics for those markets.

## Reproduce

See [the contributor guide](../CONTRIBUTING.md) for downloads. Rebuild after updating local results:

```bash
uv run python -m football_odds_predictor.training.fit_ucl
```

The command writes a model bundle and `data/processed/europe-coverage.csv` for auditing.

### Dated Elo initialization

**Question.** Do club-specific starting ratings improve pre-match 1X2 estimates over the former shared 1500 start?

**Inputs and method.** Use the [Sport Eden Elo](https://sporteden.com/rankings/elo/football/download) historical CSVs for 2019–2026, retrieved on 2026-11-26 under CC BY 4.0, and the match archives listed below. Match provider club names to project IDs and reject ambiguous joins. Use the latest club rating at most 180 days old and strictly before the first match day in the shared timeline. If there is no such rating, use the club's pre-August 2020 replay when it has five results and is at most 365 days old. Otherwise use the latest country median when the club enters through a known domestic league, or the European median. The model bundle records the date and source of each initial rating. Start the main replay on 2020-08-01.

**Worked example.** Manchester City starts at 1948 from 2020-07-31 before its first recorded match on 2020-08-07. OFI Creta enters later, at 1547 from 2026-05-31 before its first recorded match on 2026-08-23. Both ratings update after matches against the opponents on the same scale.

**Coverage.** Of the 117 clubs with UCL or UEL results since 2024/25, 86 start from a published club rating and 31 start from their own historical replay. Across the full timeline, including clubs outside the recent UEFA set, 490 use club ratings, 122 use historical replay, 244 use country medians, and 318 use the European median.

**Result.** Refit the same standardized logistic regression on expanding windows starting in 2021/22. Compare the former and new Elo histories on identical fixtures. Also refit the European goal and card models with the new inputs. Run `uv run python -m football_odds_predictor.training.fit_ucl`.

| 1X2 log loss, lower is better | Former UCL | Dated UCL | Former UEL | Dated UEL |
| --- | ---: | ---: | ---: | ---: |
| Validation 2022/23 | 0.9048 | 0.9106 | 0.9796 | 0.9351 |
| Validation 2023/24 | 0.9742 | 0.9808 | 0.9600 | 0.9631 |
| Validation 2024/25 | 0.8758 | 0.8859 | 1.0062 | 0.9977 |
| Holdout 2025/26 | 0.9513 | 0.9358 | 0.9615 | 0.9587 |

The dated start addresses differences in club strength at entry. Prediction scores are mixed across validation seasons, so the holdout gains do not establish that the model will improve in every season.

### Check Elo updates and forecast timing

Run `uv run python -m unittest discover -s tests -v`. The checks cover dated
ratings, win/draw/loss, rating conservation, and the history available before a
capture. Elo changes by `24 × (actual − expected)`, with an actual score of 1
for a win, 0.5 for a draw, and 0 for a loss. A home team rated 1500 against a
visitor rated 1500 gains 9.95 points for a win, loses 2.05 for a draw, or
loses 14.05 for a loss because of the 60-point home advantage. The visitor
changes by the opposite amount. Ratings and form from results on the cutoff
day are excluded because the archives use match dates.

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
