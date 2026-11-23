"""Replay a virtual betting ledger and report realized results and budget risk."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

ZERO = Decimal(0)
SIGNALS = ("red", "yellow", "orange", "green", "purple")
LEDGER_COLUMNS = {
    "bet_id", "fixture_id", "selection", "placed_at", "kickoff",
    "settled_at", "stake", "odds", "outcome", "signal",
}


@dataclass(frozen=True)
class Bet:
    bet_id: str
    fixture_id: str
    selection: str
    placed_at: datetime
    kickoff: datetime
    settled_at: datetime | None
    stake: Decimal
    odds: Decimal
    outcome: str
    signal: str

    def payout(self) -> Decimal:
        """Return the amount credited at settlement, including any stake returned."""
        if self.outcome == "win":
            return self.stake * self.odds
        if self.outcome == "void":
            return self.stake
        return ZERO


@dataclass(frozen=True)
class Results:
    settled: int
    wins: int
    losses: int
    voids: int
    stake: Decimal
    profit: Decimal

    @property
    def roi(self) -> Decimal | None:
        return self.profit / self.stake if self.stake else None


@dataclass(frozen=True)
class Portfolio:
    overall: Results
    by_signal: dict[str, Results]
    available: Decimal
    open_stake: Decimal
    peak_open_stake: Decimal
    maximum_drawdown: Decimal
    pending: int


def parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamps need an explicit UTC offset")
    return timestamp.astimezone(UTC)


def read_ledger(path: Path) -> list[Bet]:
    """Read a CSV ledger. Blank outcome and settlement time mark an open bet."""
    bets = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = LEDGER_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing ledger columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                fields = {key: (row[key] or "").strip() for key in LEDGER_COLUMNS}
                for key in ("placed_at", "kickoff"):
                    fields[key] = parse_timestamp(fields[key])
                settled_at = fields["settled_at"]
                fields["settled_at"] = parse_timestamp(settled_at) if settled_at else None
                for key in ("stake", "odds"):
                    fields[key] = Decimal(fields[key])
                fields["outcome"] = fields["outcome"].lower()
                fields["signal"] = fields["signal"].lower()
                bets.append(Bet(**fields))
            except (InvalidOperation, ValueError) as error:
                raise ValueError(f"Ledger line {reader.line_num}: {error}") from error
    return bets


def validate_bet(bet: Bet) -> None:
    if not all((bet.bet_id, bet.fixture_id, bet.selection)) or bet.signal not in SIGNALS:
        raise ValueError(f"Missing identifiers or unknown signal for bet {bet.bet_id!r}")
    if not bet.stake.is_finite() or not bet.odds.is_finite() or bet.stake <= 0 or bet.odds <= 1:
        raise ValueError(f"Bet {bet.bet_id}: stake must be positive and decimal odds must exceed 1")
    if bet.placed_at >= bet.kickoff:
        raise ValueError(f"Bet {bet.bet_id}: placement must precede kickoff")
    if bet.settled_at is None:
        if bet.outcome:
            raise ValueError(f"Bet {bet.bet_id}: an outcome requires a settlement time")
    elif (
        bet.outcome not in ("win", "loss", "void")
        or bet.settled_at <= bet.placed_at
        or (bet.outcome != "void" and bet.settled_at < bet.kickoff)
    ):
        raise ValueError(f"Bet {bet.bet_id}: invalid outcome or settlement time")


def summarize(bets: list[Bet]) -> Results:
    return Results(
        settled=len(bets),
        wins=sum(bet.outcome == "win" for bet in bets),
        losses=sum(bet.outcome == "loss" for bet in bets),
        voids=sum(bet.outcome == "void" for bet in bets),
        stake=sum((bet.stake for bet in bets), ZERO),
        profit=sum((bet.payout() - bet.stake for bet in bets), ZERO),
    )


def evaluate_portfolio(bets: list[Bet], budget: Decimal) -> Portfolio:
    """Replay placements and settlements in time order without reusing open stakes."""
    if not budget.is_finite() or budget <= 0:
        raise ValueError("The starting budget must be positive and finite")
    if len({bet.bet_id for bet in bets}) != len(bets):
        raise ValueError("The ledger contains duplicate bet IDs")
    placements, settlements = defaultdict(list), defaultdict(list)
    settled = []
    for bet in bets:
        validate_bet(bet)
        placements[bet.placed_at].append(bet)
        if bet.settled_at is not None:
            settlements[bet.settled_at].append(bet)
            settled.append(bet)

    cash = peak_equity = budget
    open_stake = peak_open_stake = maximum_drawdown = ZERO
    for timestamp in sorted(placements.keys() | settlements.keys()):
        # Credit simultaneous settlements together before funding new placements.
        for bet in settlements[timestamp]:
            cash += bet.payout()
            open_stake -= bet.stake
        new_stake = sum((bet.stake for bet in placements[timestamp]), ZERO)
        if new_stake > cash:
            raise ValueError(f"Insufficient available budget for placements at {timestamp.isoformat()}")
        cash -= new_stake
        open_stake += new_stake
        peak_open_stake = max(peak_open_stake, open_stake)
        # Open stakes are carried at cost. Drawdown tracks realized changes in equity.
        equity = cash + open_stake
        peak_equity = max(peak_equity, equity)
        maximum_drawdown = max(maximum_drawdown, (peak_equity - equity) / peak_equity)

    return Portfolio(
        overall=summarize(settled),
        by_signal={signal: summarize([bet for bet in settled if bet.signal == signal])
                   for signal in SIGNALS},
        available=cash,
        open_stake=open_stake,
        peak_open_stake=peak_open_stake,
        maximum_drawdown=maximum_drawdown,
        pending=len(bets) - len(settled),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="CSV of simulated placements and outcomes")
    parser.add_argument("--budget", required=True, type=Decimal, help="Initial virtual budget")
    args = parser.parse_args()
    try:
        report = evaluate_portfolio(read_ledger(args.ledger), args.budget)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Final equity: {report.available + report.open_stake:.2f} "
          f"| available: {report.available:.2f} | open stakes: {report.open_stake:.2f}")
    print(f"Pending: {report.pending} | peak open stakes: {report.peak_open_stake:.2f} "
          f"| maximum drawdown: {report.maximum_drawdown:.2%}")
    print(f"{'Signal':<8} {'Settled':>7} {'W/L/V':>10} {'Staked':>10} {'Net':>10} {'ROI':>9}")
    for signal, results in (("overall", report.overall), *report.by_signal.items()):
        outcomes = f"{results.wins}/{results.losses}/{results.voids}"
        roi = f"{results.roi:+.2%}" if results.roi is not None else "—"
        print(f"{signal:<8} {results.settled:>7} {outcomes:>10} "
              f"{results.stake:>10.2f} {results.profit:>+10.2f} {roi:>9}")


if __name__ == "__main__":
    main()
