"""Compatibility entry point for preprocessing."""

from .data.preprocessing import (
    TEAM_STATS as TEAM_STATS,
    TARGET_ODDS as TARGET_ODDS,
    read_season as read_season,
    load_matches as load_matches,
    main as main,
)


if __name__ == "__main__":
    main()
