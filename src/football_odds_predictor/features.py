"""Compatibility entry point for features."""

from .data.features import (
    FIXTURE_COLUMNS as FIXTURE_COLUMNS,
    SIDES as SIDES,
    MATCH_STATS as MATCH_STATS,
    HISTORY_LENGTH as HISTORY_LENGTH,
    INITIAL_ELO as INITIAL_ELO,
    ELO_K as ELO_K,
    build_features as build_features,
    main as main,
)


if __name__ == "__main__":
    main()
