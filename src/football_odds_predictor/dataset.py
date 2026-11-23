"""Compatibility entry point for dataset."""

from .data.dataset import (
    TARGET_COLUMNS as TARGET_COLUMNS,
    FIRST_SEASON as FIRST_SEASON,
    TEST_SEASON as TEST_SEASON,
    load_dataset as load_dataset,
    validation_folds as validation_folds,
)
