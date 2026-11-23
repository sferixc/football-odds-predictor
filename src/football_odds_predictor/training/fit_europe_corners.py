"""Fit UEFA corner models using recent matches from all available competitions."""

from ..data.europe_corners import CORNER_INPUTS, build_corner_dataset
from ..models.corners import EUROPE_CORNER_MODELS
from ..models.ucl import club_key
from .fit_corners import fit_league, save_bundle


def main():
    rows, timeline, names = build_corner_dataset(club_key)
    print("Eligible fixtures with measured corner totals:")
    print(rows.groupby(["league", "season_start_year"]).size().unstack(fill_value=0).to_string())
    bundle = {
        "leagues": {league: fit_league(league, rows, CORNER_INPUTS) for league in ("CL", "EL")},
        "timeline": timeline,
        "names": names,
    }
    save_bundle(EUROPE_CORNER_MODELS, bundle)


if __name__ == "__main__":
    main()
