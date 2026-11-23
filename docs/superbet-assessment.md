# Independent football probabilities and Superbet offers

Estimate event probabilities from sporting data, compare them with **Superbet Romania pre-match offers**, and report the estimated advantage and budget exposure for a user-supplied stake. The earlier Bet365 model estimates a bookmaker price. The new models learn from observed sporting outcomes and exclude bookmaker prices and wagering volumes from training, prediction, and calibration.

## Markets and data

| Market | Target and initial data |
| --- | --- |
| 1X2 | Home win, draw, away win at full time. Match scores are available for 5,330 fixtures. |
| Goals | Match total and first- or second-half total. Full-time and halftime scores are available for 5,329 fixtures. Derive second-half goals from their difference. |
| Yellow cards | Team or match total. Counts are present for 5,329 fixtures. Premier League has referee names and cards for all 1,900 fixtures. Add the referee's prior match count and yellow-card average when the assignment was known at prediction time. |
| Corners | Team or match total. Counts are present for 5,329 fixtures. |
| Player shots | Total shots for selected players. Use player fixture IDs, total shots, minutes, starting appearances, and pre-match lineup availability. Define the player shortlist before evaluation, using prior minutes, starting frequency, or a user watchlist. |

### Historical player data

The free [xGTables Understat archive](https://www.xgtables.io/understat-data/) supplies fixtures, player appearances with minutes, and shot events. Download the 45 published files for 2021/22–2025/26 and audit their coverage locally.

```bash
uv run python -m football_odds_predictor.data.player_archive --download
```

| League | Fixtures | Player appearances | Shot-event rows | Fixtures with no shot map |
| --- | ---: | ---: | ---: | ---: |
| Premier League | 1,900 | 56,271 | 49,345 | 0 |
| Bundesliga | 1,530 | 47,447 | 40,338 | 1 |
| La Liga | 1,900 | 59,383 | 46,218 | 0 |
| **Total** | **5,330** | **163,101** | **135,901** | **1** |

The ordinary shot events match the reported appearance shot counts for fixtures with shot maps. The missing map is Holstein Kiel–Bochum on 9 February 2025. Defer shots on target until we can verify goal-line clearances against Superbet's count. For total shots, verify the selection's counting and participation rules before classifying real offers. A historical appearance tells us who played and for how long. For prediction, obtain lineup and player-availability information that was known before kickoff. The current xGTables downloads use a different provider, so verify their fields and comparability separately. [Superbet settlement rules](https://superbet.ro/wiki/regulament-de-joc-pariuri-cap-7)

## Build and validate

1. Build sporting datasets for all listed markets. Use chronological folds, fit preprocessing on training data, and retain a later test window. Start with simple probability and count baselines, then compare candidate models using log loss, Brier score, and calibration by market. For team and player counts, estimate probabilities for the exact offered thresholds. For players, account for likely minutes and the settlement rule if they do not play or start.
2. The current collector reads Superbet's public Romanian offer endpoint. It targets a capture 15 minutes before kickoff and accepts captures 10–20 minutes before kickoff. Each successful snapshot keeps its observation time, fixture, raw markets, and supported single-selection promotions. Untimed public probes have been saved for development; a real timed capture from the running watcher still needs verification. Market settlement rules must be checked before treating a quoted market as the model target. [Superbet market definitions](https://superbet.ro/wiki/regulament-de-joc-pariuri-cap-5)
3. Join predictions with exact matching offers and report fair odds, break-even probability, estimated downside, and signal color. Record the prediction and quote before kickoff. Replay timestamped offers with a simulated budget to measure returns and drawdown by color. Use future fixtures to check whether the color thresholds generalize.

### Virtual portfolio evaluation

**Question.** How would a set of simulated selections affect a fixed budget as results arrive? The offline evaluator replays a supplied CSV by placement and settlement time. It reserves each stake until settlement, credits `stake × odds` on a win, refunds a void, and credits zero on a loss. It rejects placements that would exceed available funds.

Use ISO timestamps with explicit UTC offsets. Leave both `settled_at` and `outcome` blank for open positions. Outcomes are `win`, `loss`, or `void`. Record placements and price before the match, then fill in the outcome under the applicable settlement rule. Use a separate ledger for each selection strategy when comparing strategies. Store market, threshold, probabilities, model version, and quote ID alongside the ledger when they are available.

**Worked example.** Save the following fictional ledger as `/tmp/football-portfolio-example.csv` and run the command below.

```csv
bet_id,fixture_id,selection,placed_at,kickoff,settled_at,stake,odds,outcome,signal
b1,f1,Home win,2026-08-01T12:45:00Z,2026-08-01T13:00:00Z,2026-08-01T15:00:00Z,20,1.90,loss,green
b2,f2,Over 2.5 goals,2026-08-01T13:45:00Z,2026-08-01T14:00:00Z,2026-08-01T16:00:00Z,25,2.00,win,purple
b3,f3,Over 1.5 player shots,2026-08-02T12:45:00Z,2026-08-02T13:00:00Z,2026-08-02T15:00:00Z,10,1.90,void,orange
b4,f4,Under 9.5 corners,2026-08-03T12:45:00Z,2026-08-03T13:00:00Z,,5,1.90,,yellow
```

```bash
uv run python -m football_odds_predictor.evaluation.portfolio /tmp/football-portfolio-example.csv --budget 100
```

The final equity is **105 units**, including **5 units** in the open bet. The three settled bets give **+5 units** on **55 units staked** for **+9.09% ROI**. The largest drop from an earlier equity peak is **20%**, and at most **45 units** were tied up at once. Drawdown uses available cash plus open stakes valued at cost. ROI includes refunded stakes in the turnover and excludes open stakes. The report breaks down settled outcomes, stakes, net results, and ROI by color.

The evaluator uses supplied outcomes. Automatic settlement, sufficient historical Superbet captures, and validation of color thresholds remain open. For that validation, compare colors at a fixed one-unit stake, group selections from the same fixture when measuring uncertainty, and test the chosen thresholds on later captures.

### Superbet snapshot pilot

A single unauthenticated GET to the Romanian site's public offer service returned 443 markets for Arsenal–Borussia Dortmund, event 14790413, kickoff 24 November 2026 at 20:00 UTC. The response included 1X2, totals for goals in the match and in each half, match and team corners, and 217 selections in the total player shots market. For example, the player-shot records include player ID, name, threshold, and price. Card markets were labeled total cards. Verify their settlement rules before mapping them to yellow-card targets. [Superbet public football offer](https://superbet.ro/pariuri-sportive/fotbal)

Start the collector once. It discovers Premier League, Bundesliga, La Liga, Champions League, and Europa League fixtures from the public schedule, refreshes that schedule hourly, and attempts one detailed offer capture per fixture at kickoff minus 15 minutes. If it starts late, it can still capture within the 10–20 minute window. The collector must remain running. Failures appear in the terminal. It saves the observation time and one raw offer to `data/raw/superbet/<event-id>.json` and keeps existing snapshots. Pass `--list` to preview upcoming capture times in UTC. This live odds scope is five competitions; the 26 domestic leagues in the European history archive provide team context and are not all collected by the watcher.

```bash
uv run python -m football_odds_predictor.data.superbet_watch
```

The network probes were outside the target window. The capture-window checks were verified locally with a real response, and an actual timed capture remains to be verified. OddsPapi separately documents historical snapshots for `superbet.ro`. Verify access under the available account and match the historical observation time to the same pre-match window before using them for a replay. [OddsPapi historical example](https://oddspapi.io/blog/superbet-api-odds-access/#step-4-check-the-closing-line-on-played-fixtures)

### SuperCote and boosted combinations

The raw capture already preserves single-selection SuperCote under `markets`, combinations under `superbets`, and price boosts under `priceboosts`. In the Brentford–Everton probe, Superbet listed a boosted combination of at least nine match shots on target, three match goals, and ten match corners. It listed 4.25 as the offered price and 3.75 as the original price. The payload identifies all three legs individually.

New snapshots extract open single-selection promotions into `simple_promotions`. The extractor pairs 1X2 SuperCote with the corresponding standard prices. It also keeps single-leg boosts on identified goal, corner, player total-shot, and 1X2 markets, with the original price reference if supplied. The Brentford–Everton response yielded three simple entries, one per 1X2 outcome. Its boosted combinations stayed in the raw response.

Display combinations separately when we add a report view for them. Scoring a same-match combination requires a validated joint probability that accounts for dependence and matches the settlement rules. Card promotions remain in the raw data until their market definition can be mapped to the sporting targets.

## Offer report prototype

The main use case is to check a selection the user already picked. Find the exact selection in the Superbet snapshot, obtain an independent model probability for it, and show the standard and promotional prices where available. The primary metric is estimated return per unit staked, with its conservative counterpart next to it. Display the chance of losing separately. With no refund, a 60% chance at odds of 1.90 gives `0.60 × 1.90 − 1 = +14%` estimated return, a 52.6% break-even probability, and a 40% estimated chance of loss. The return is an average implied by the estimate across comparable bets.

The calculator supports one win/loss selection, with an optional full refund. It takes the two probability estimates and the odds as inputs. It reports an expected return per unit staked, expected profit for the chosen stake, chance of loss, and maximum loss as a share of the budget. Partial settlements and combinations are outside this prototype.

| Signal | Provisional rule |
| --- | --- |
| Red | Central return is zero or negative |
| Yellow | Central return is positive and conservative return is zero or negative |
| Orange | Conservative return is positive and below 5% |
| Green | Conservative return is at least 5% and below 10% |
| Purple | Conservative return is at least 10% |

The conservative probability must eventually come from out-of-sample evidence for the relevant market. Until then, it is a sensitivity scenario supplied by the caller. If there is a possible full refund, expected return is `p_win × odds + p_void − 1`. The signal measures estimated offer value and leaves the probability of losing visible separately.

**Worked example.** A hypothetical Superbet offer at 1.90, central probability 60%, conservative probability 55%, and stake 2 out of 100 gives an orange signal, +14% central return, +4.5% conservative return, and 40% estimated loss probability. The maximum loss is 2% of the budget.

```bash
uv run python -m football_odds_predictor.evaluation.offer --odds 1.90 \
  --win-probability 0.60 --conservative-win-probability 0.55 --stake 2 --budget 100
```

### Terminal report

Train the selected models once on 2021/22–2024/25 and save them locally:

```bash
uv run python -m football_odds_predictor.training.fit_sporting
```

This writes `data/processed/sporting-models.pkl`, including fitted classifiers, scalers, input columns, and historical frequencies where that baseline won. Keep the file local and load only an artifact you generated yourself. Refit when the training data or model settings change.

The report reads the saved models and recreates Elo and form using earlier match dates. For matching team markets it prints model odds, the standard price and promotion, estimated return at the promotional price when available or the standard price otherwise, and the chance of losing. It shows the most recent result date used for each team and player.

```bash
uv run python -m football_odds_predictor.evaluation.superbet_report
```

The command opens the most recently saved capture in `data/raw/superbet/`. Pass a path to choose another capture. Options `--market 1x2`, `--market goals`, `--market cards`, `--market corners`, and `--market shots` filter sections. Use `--player 'Dewsbury-Hall'` to see one player's selections. Add `--stress-pp 5` to see provisional colors on comparable team offers using a five-percentage-point reduction in probability. Without a sensitivity setting, colors are unset; the sensitivity amount is not yet estimated from market-specific calibration data.

The team sections need the original training files, the 2020/21 Elo warm-up files, and 2026/27 results in `data/raw/football-data-2627/`. Team names must match the archive. The player section uses the xGTables archive and shows model odds conditional on playing, alongside available prices. It leaves value and colors unset until participation and settlement are modeled. The yellow-card section shows model odds. Superbet's total-cards prices remain separate pending rule mapping. Supported lines are the fixed thresholds evaluated below.

**Worked example.** An untimed Brentford–Everton probe gave a home-win probability of 45.1%, with a SuperCota of 2.40 and an estimated return of +8.2%. A five-point reduction gives −3.8%, or yellow. It gave Kiernan Dewsbury-Hall a 29.6% chance of more than 1.5 shots if he played, with an observed price of 1.48. The latest inputs were from November for the teams and May for the player. Treat this as a reproduction example.

```bash
uv run python -m football_odds_predictor.evaluation.superbet_report \
  data/raw/superbet-probes/14656294.json --probe --stress-pp 5
```

The report runs independently of the watcher. Fresh player results remain open. The evaluation below measures probability calibration. The report's optional color still uses a user-supplied sensitivity scenario.

### Local dashboard

Fit the saved models after downloading or updating their inputs. The sporting bundle is required; domestic and European card bundles add their respective reports:

```bash
uv run python -m football_odds_predictor.training.fit_sporting
uv run python -m football_odds_predictor.training.fit_cards
uv run python -m football_odds_predictor.training.fit_ucl
```

These models are loaded when the dashboard starts. Do not refit for each match lookup. After fitting, use two terminals:

**Terminal 1 — keep the odds watcher running**

```bash
uv run python -m football_odds_predictor.data.superbet_watch
```

**Terminal 2 — keep the dashboard running**

```bash
uv run python -m football_odds_predictor.evaluation.dashboard
```

The dashboard requests the upcoming public Superbet schedule, filters it to the current day in Europe/Amsterdam, and merges it with local captures for that day. It refreshes the live schedule every two minutes and the browser's fixture list every 30 seconds. The watcher refreshes its schedule hourly and attempts one odds capture per fixture about 15 minutes before kickoff (10–20 minutes before kickoff is accepted); it does not continuously refresh captured prices. The first fixture is selected automatically; click a listed fixture to open its report. A listed fixture without a timed offer shows that its capture is pending. The schedule feed only lists upcoming matches, so a match that has started appears only if a local capture exists. If the schedule request fails, the page can still show saved captures and reports the schedule error. Keep both terminals open and stop each process with Ctrl+C.

The dashboard displays 1X2, goals, cards, and corners. Player shots remain in the model and optional terminal report, but the dashboard tab and player filter have been removed. Color signals are deliberately absent under “No sensitivity adjustment”; choosing a five- or ten-percentage-point reduction lowers the predicted win probability by that many absolute points (for example, 60% to 55%) and calculates a provisional conservative return. These colors are not validated betting recommendations.

The domestic corner bundle is generated separately with `uv run python -m football_odds_predictor.training.fit_corners`. It fits Poisson count regressions for home, away, and match corners in Premier League, Bundesliga, and La Liga, using the same chronological folds as the sporting experiments. It derives Over and Under prices at every half-line from 0.5 to 10.5 for a team and 0.5 to 18.5 for the match. The dashboard and terminal report show Superbet's exact quote beside each model line when that quote is available; an absent quote stays blank. For UCL and UEL, use the separate European corner bundle from `training.fit_europe_corners`. It uses the five most recent recorded club matches across competitions and displays how many had corner statistics. See the [European model notes](european-model-plan.md#european-cards-and-remaining-markets).

## First sporting model comparison

**Question.** Can sporting data improve probability estimates over the historical frequency for each selection? Run:

```bash
uv run python -m football_odds_predictor.training.sporting
```

Use Football-Data.co.uk outcomes from 2021/22–2025/26, the Elo columns from dataset `01551de`, and the xGTables Understat player appearances described above. Team inputs are pre-match Elo and five prior league matches of goals, goals by half, yellows, corners, and fouls for and against. Referee assignments are pending a pre-match availability source. Player inputs are venue and the mean shots and minutes over five earlier appearances in the same league. The player evaluation covers positive-minute appearances for players averaging at least 45 minutes over those prior appearances. Its probabilities are conditional on playing. Current-match minutes and results are excluded from model inputs. Earlier fixture dates serve as the availability approximation.

Compare historical frequency, standardized logistic regression, epoch-trained SGD, and XGBoost (150 trees, depth 2, learning rate 0.05). Reuse the three expanding validation folds ending in 2022/23, 2023/24, and 2024/25. For SGD, use the last fifth of dates in each training fold to choose the best epoch by log loss, up to 30 epochs with patience 5. Refit on the full training fold for that many epochs. Fit scaling inside each training window. SGD uses averaged weights, a constant learning rate of 0.01, and alpha 0.001. SGD and XGBoost use seed 42. The command prints validation log loss and Brier score. The 2025/26 holdout was not scored for these targets.

**Worked example.** In the first fold, logistic regression gave Arsenal–Chelsea probabilities of **62.16% / 20.18% / 17.66%** for home win, draw, and away win. The model saw Arsenal's initial Elo and its mean of 2.4 goals in the prior five matches. The eventual result is used as the training or evaluation label for the applicable fold.

| Selection | Frequency log loss | Best candidate | Candidate log loss |
| --- | ---: | --- | ---: |
| 1X2 | 1.0686 | SGD | 0.9980 |
| Match goals > 2.5 | 0.6887 | SGD | 0.6803 |
| First-half goals > 0.5 | 0.5808 | Frequency | 0.5808 |
| Second-half goals > 1.5 | 0.6938 | SGD | 0.6911 |
| Home yellows > 2.5 | 0.6390 | SGD | 0.6305 |
| Away yellows > 2.5 | 0.6712 | SGD | 0.6672 |
| Match yellows > 4.5 | 0.6848 | SGD | 0.6732 |
| Home corners > 4.5 | 0.6806 | Logistic | 0.6490 |
| Away corners > 4.5 | 0.6788 | XGBoost | 0.6514 |
| Match corners > 9.5 | 0.6924 | SGD | 0.6915 |
| Player shots > 0.5 | 0.6918 | XGBoost | 0.5849 |
| Player shots > 1.5 | 0.5341 | XGBoost | 0.4415 |
| Player shots > 2.5 | 0.3328 | XGBoost | 0.2679 |

Values are equal-weight means across the three validation seasons. The biggest improvements are for player shots and 1X2. Some goals and match-corner selections are close to the frequency baseline. The comparison applies to the fixed thresholds listed above.

## Calibration check and final sporting test

**Question.** Do the probabilities correspond to observed frequencies, and how well do the selected models predict 2025/26?

```bash
uv run python -m football_odds_predictor.evaluation.sporting validation
uv run python -m football_odds_predictor.evaluation.sporting test
```

Use the same team and player data as the validation comparison. The first command pools predictions from the three validation folds. The second loads the 13 saved models fitted on 2021/22–2024/25 and scores 2025/26. The models and probability outputs were kept fixed for this test. Metrics pool eligible rows, so the validation log loss can differ slightly from the equal-weight fold means above.

The commands write `data/processed/sporting-reliability-validation.csv` and `sporting-reliability-test.csv` in the same folder. Each row shows the sample count, mean predicted probability, and realized frequency in one 10-percentage-point band. The calibration gap averages the absolute differences across occupied bands, weighted by their sample counts. For 1X2 it includes all three outcomes. For binary selections it uses the over probability.

**Worked example.** In the 2025/26 test, 489 fixtures had an estimate between 50% and 60% for over 2.5 goals. Their mean prediction was **54.9%** and **55.8%** actually went over.

| Selection | Test rows | Test log loss | Validation calibration gap | Test calibration gap |
| --- | ---: | ---: | ---: | ---: |
| 1X2 | 1,051 | 1.0001 | 2.1 pp | 2.7 pp |
| Match goals > 2.5 | 1,051 | 0.6803 | 2.6 pp | 1.9 pp |
| Home yellows > 2.5 | 1,051 | 0.6003 | 3.4 pp | 5.2 pp |
| Home corners > 4.5 | 1,051 | 0.6547 | 2.3 pp | 4.3 pp |
| Player shots > 1.5, conditional on playing | 23,952 | 0.4442 | 0.9 pp | 0.6 pp |

These gaps describe groups of predictions, with sample counts in the CSVs. They are insufficient to set a conservative probability for an individual selection. Validating the color signals still requires matching actual Superbet offers and settlement rules. Player results do not cover the chance of a player appearing.

### Referee history and card settlement

Superbet's full-time total cards market counts one point for a yellow and two for a red, discards the second yellow, and excludes cards shown to bench players and staff or after the match. The saved Brentford–Everton offer has total cards for the match and each team, with no equivalent yellow-only totals. The Football-Data archive has aggregate yellows and reds. Those columns cannot identify individual second yellows or bench incidents. The local Understat appearance archive also lacks card timing and a separate second-yellow field. A settlement-compatible target requires event-level card types and recipient status. [Superbet settlement rules](https://superbet.ro/wiki/regulament-de-joc-pariuri-cap-7)

The 2021/22–2025/26 Football-Data archive has a referee for all 1,900 Premier League fixtures and no referee field in the Bundesliga or La Liga files. The yellow-card experiment adds one input, the referee's average match yellow total across strictly earlier dates. Both variants use the same 1,096 eligible fixtures across expanding validation folds and train SGD with the same protocol. The 2025/26 final test stays reserved.

| Yellow-card line | Log loss without referee | Log loss with referee |
| --- | ---: | ---: |
| Home over 2.5 | 0.6220 | 0.6237 |
| Away over 2.5 | 0.6524 | 0.6554 |
| Match over 4.5 | 0.6604 | 0.6652 |

Lower is better. This feature worsened all three validation scores, so the saved models and dashboard are unchanged for cards. A compatible model also needs the referee assignment as it was known before kickoff, with a fallback when the assignment or history is missing.

```bash
uv run python -m football_odds_predictor.evaluation.referee_cards
```

### Expanded yellow-card lines

Fit the league-specific count models with `uv run python -m football_odds_predictor.training.fit_cards`.
This writes `data/processed/yellow-card-models.pkl`. Restart the dashboard to load it.
Each of Premier League, Bundesliga, and La Liga has separate Poisson models
for home, away, and match yellow totals. Each league can fit a variant that
uses the referee's historical mean. A single
count distribution per target gives ordered Over probabilities across 0.5–9.5.
The expanded models use a named referee from the pre-match fixture when prior
results are available and fall back to the team-only version otherwise. The
Brentford–Everton capture has no referee field. For Bundesliga and La Liga,
Transfermarkt's games snapshot (through 2026-07-06) supplies referee names.
We join on date, home and away team, and score. Coverage in the 2021/22–2025/26
sample was 1,530 of 1,530 Bundesliga fixtures and 1,897 of 1,900 La Liga fixtures.
[Transfermarkt dataset](https://github.com/dcaribou/transfermarkt-datasets)

The training command compares both variants on the same expanding validation
folds and prints Brier scores and event counts for every line. Only six of
1,096 validation matches went over 9.5 total yellows. The 2025/26 season is
held out from model fitting. Superbet total-card quotes are shown in their
own tables for all open lines in the capture.

### Experimental total-card signals

The same fit command also trains separate models for aggregate yellows plus
twice aggregate reds in each of the three leagues. The dashboard labels this target as a proxy and
applies the standard five-color scale only to matching open lines in the
snapshot. The Brier scores measure the aggregate proxy, whose labels may
differ from Superbet settlement because the source omits the circumstances
of individual cards. Exact settlement validation still needs event-level
data. Refresh the model file with the fit command and restart the dashboard.

### European competition offers

The watcher includes Champions League and Europa League. The dashboard uses
combined domestic and European form for 1X2 and full-match goal signals.
See the [European model plan](european-model-plan.md) for source coverage,
evaluations, and the remaining market work.
