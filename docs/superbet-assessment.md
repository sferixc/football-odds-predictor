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

The prototype currently scores supplied inputs. The probability models and the link between the snapshot and the report are pending.
