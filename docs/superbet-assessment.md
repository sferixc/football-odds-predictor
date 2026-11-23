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
2. Capture one pre-match snapshot per selected fixture. Target 15 minutes before kickoff and accept captures 10–20 minutes before kickoff. Record the actual observation time, scheduled kickoff, fixture and player IDs, market, period, threshold, selection, decimal odds, and settlement rule. Report missing markets for that snapshot. Check a candidate feed on one real fixture before integrating it. Superbet's published clause 3.3.4 restricts direct scraping. [Odds-API.io advertises Superbet coverage](https://odds-api.io/integrations/superbet), but free access, player-market coverage, and exact Romanian prices are unverified. A genuine pre-match quote entered manually can exercise the workflow while automatic access is being verified. [Superbet terms](https://superbet.ro/wiki/termeni-si-conditii) · [Superbet market definitions](https://superbet.ro/wiki/regulament-de-joc-pariuri-cap-5)
3. Join predictions with exact matching offers and report fair odds, break-even probability, estimated downside, and signal color. Record the prediction and quote before kickoff. Replay timestamped offers with a simulated budget to measure returns and drawdown by color. Use future fixtures to check whether the color thresholds generalize.

### Superbet snapshot pilot

A single unauthenticated GET to the Romanian site's public offer service returned 443 markets for Arsenal–Borussia Dortmund, event 14790413, kickoff 24 November 2026 at 20:00 UTC. The response included 1X2, totals for goals in the match and in each half, match and team corners, and 217 selections in the total player shots market. For example, the player-shot records include player ID, name, threshold, and price. Card markets were labeled total cards. Verify their settlement rules before mapping them to yellow-card targets. [Superbet public football offer](https://superbet.ro/pariuri-sportive/fotbal)

Start the collector once. It discovers Premier League, Bundesliga, and La Liga fixtures from the public schedule, refreshes that schedule hourly, and attempts one detailed offer capture per fixture at kickoff minus 15 minutes. If it starts late, it can still capture within the 10–20 minute window. The collector must remain running. Failures appear in the terminal. It saves the observation time and one raw offer to `data/raw/superbet/<event-id>.json` and keeps existing snapshots. Pass `--list` to preview scheduled captures in UTC.

```bash
uv run python -m football_odds_predictor.data.superbet_watch
```

The network probe was outside the target window. The capture-window checks were verified locally with the real response, and an actual timed capture remains to be verified. OddsPapi separately documents historical snapshots for `superbet.ro`. Verify access under the available account and match the historical observation time to the same pre-match window before using them for a replay. [OddsPapi historical example](https://oddspapi.io/blog/superbet-api-odds-access/#step-4-check-the-closing-line-on-played-fixtures)

### SuperCote and boosted combinations

The raw capture already preserves single-selection SuperCote under `markets`, combinations under `superbets`, and price boosts under `priceboosts`. In the Brentford–Everton probe, Superbet listed a boosted combination of at least nine match shots on target, three match goals, and ten match corners. It listed 4.25 as the offered price and 3.75 as the original price. The payload identifies all three legs individually.

New snapshots extract open single-selection promotions into `simple_promotions`. The extractor pairs 1X2 SuperCote with the corresponding standard prices. It also keeps single-leg boosts on identified goal, corner, player total-shot, and 1X2 markets, with the original price reference if supplied. The Brentford–Everton response yielded three simple entries, one per 1X2 outcome. Its boosted combinations stayed in the raw response.

Display combinations separately when we add a report view for them. Scoring a same-match combination requires a validated joint probability that accounts for dependence and matches the settlement rules. Card promotions remain in the raw data until their market definition can be mapped to the sporting targets.

## Offer report prototype

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

The prototype currently scores supplied inputs. The first probability models are evaluated below. Saving selected models and linking their predictions to captured offers are pending.

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

Values are equal-weight means across the three validation seasons. The biggest improvements are for player shots and 1X2. Some goals and match-corner selections are close to the frequency baseline. This experiment evaluates fixed thresholds and does not establish calibration for the full set of Superbet offers. Check reliability by probability band and score the held-out season before interpreting offer colors.
