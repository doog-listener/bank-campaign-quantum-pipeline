"""Shared fixtures: the released workbook is parsed once per test session."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data import Dataset, load_dataset
from src.utils import load_config


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    return load_config()


@pytest.fixture(scope="session")
def ds(config: dict[str, Any]) -> Dataset:
    """Verified dataset (assert_key_integrity already ran inside load_dataset)."""
    return load_dataset(config=config)


@pytest.fixture(scope="session")
def constraints_map(ds: Dataset) -> dict[str, int]:
    """Symbol -> integer value; the 'Decision variable' row ('0 or 1') is skipped."""
    numeric = ds.constraints[
        ds.constraints["value"].astype(str).str.fullmatch(r"-?\d+")
    ]
    return {str(r["symbol"]): int(r["value"]) for _, r in numeric.iterrows()}


@pytest.fixture()
def train_frame(ds: Dataset) -> pd.DataFrame:
    return ds.train.copy()
