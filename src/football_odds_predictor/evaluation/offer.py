"""Assess one pre-match Superbet quote against independent probabilities."""

import argparse
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class OfferAssessment:
    fair_odds: Decimal
    break_even_probability: Decimal
    loss_probability: Decimal
    expected_return: Decimal
    conservative_return: Decimal
    expected_profit: Decimal
    maximum_loss: Decimal
    budget_at_risk: Decimal
    signal: str


def assess_offer(
    *,
    odds: Decimal,
    win_probability: Decimal,
    conservative_win_probability: Decimal,
    stake: Decimal,
    budget: Decimal,
    void_probability: Decimal = Decimal(0),
) -> OfferAssessment:
    """Score a single selection with win, loss, or a full stake refund."""
    values = (odds, win_probability, conservative_win_probability, stake, budget, void_probability)
    if not all(value.is_finite() for value in values):
        raise ValueError("All inputs must be finite")
    if odds <= 1 or not 0 < stake <= budget:
        raise ValueError("Decimal odds must exceed 1 and the stake must be within the budget")
    if not 0 <= void_probability < 1 or not (
        0 <= conservative_win_probability <= win_probability <= 1 - void_probability
    ) or win_probability == 0:
        raise ValueError("Expected a positive win probability and a valid conservative and void probability")

    # A void refunds the stake. The conservative case holds void probability
    # fixed and assigns the reduction in win probability to losses.
    settled_probability = 1 - void_probability
    expected_return = win_probability * odds - settled_probability
    conservative_return = conservative_win_probability * odds - settled_probability
    if expected_return <= 0:
        signal = "red"
    elif conservative_return <= 0:
        signal = "yellow"
    elif conservative_return < Decimal("0.05"):
        signal = "orange"
    elif conservative_return < Decimal("0.10"):
        signal = "green"
    else:
        signal = "purple"

    return OfferAssessment(
        fair_odds=settled_probability / win_probability,
        break_even_probability=settled_probability / odds,
        loss_probability=settled_probability - win_probability,
        expected_return=expected_return,
        conservative_return=conservative_return,
        expected_profit=stake * expected_return,
        maximum_loss=stake,
        budget_at_risk=stake / budget,
        signal=signal,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odds", type=Decimal, required=True, help="Superbet decimal odds")
    parser.add_argument("--win-probability", type=Decimal, required=True, help="Model probability, e.g. 0.60")
    parser.add_argument("--conservative-win-probability", type=Decimal, required=True)
    parser.add_argument("--void-probability", type=Decimal, default=Decimal(0))
    parser.add_argument("--stake", type=Decimal, required=True)
    parser.add_argument("--budget", type=Decimal, required=True)
    args = parser.parse_args()
    try:
        report = assess_offer(**vars(args))
    except ValueError as error:
        parser.error(str(error))

    print(f"Signal: {report.signal}")
    print(f"Fair odds: {report.fair_odds:.2f}")
    print(f"Break-even win probability: {report.break_even_probability:.1%}")
    print(f"Estimated return: {report.expected_return:+.1%}")
    print(f"Conservative return: {report.conservative_return:+.1%}")
    print(f"Estimated profit for this stake: {report.expected_profit:+.2f}")
    print(f"Estimated loss probability: {report.loss_probability:.1%}")
    print(f"Maximum loss: {report.maximum_loss:.2f} ({report.budget_at_risk:.1%} of budget)")


if __name__ == "__main__":
    main()
