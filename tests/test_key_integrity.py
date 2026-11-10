"""The id contract: unique, disjoint, joinable. `customer_id` is the only key.

There is no id_map in this project; these tests are what replaces it. They also
prove the guard actually fires -- a passing assertion is worthless if the check
silently accepts a corrupted workbook.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.data import ID, Dataset, KeyIntegrityError, assert_key_integrity, load_dataset

EXPECTED_TEST_IDS = 100


def test_test_ids_are_100_unique(ds: Dataset) -> None:
    assert len(ds.test) == EXPECTED_TEST_IDS
    assert ds.test[ID].nunique() == EXPECTED_TEST_IDS
    assert ds.test[ID].is_unique
    assert not ds.test[ID].isna().any()


def test_test_ids_disjoint_from_train(ds: Dataset) -> None:
    assert set(ds.train[ID]).isdisjoint(set(ds.test[ID]))
    assert ds.train[ID].is_unique and len(ds.train) == 300


def test_every_test_id_present_in_stage2(ds: Dataset) -> None:
    missing = set(ds.test[ID]) - set(ds.stage2[ID])
    assert missing == set(), f"test ids without Stage 2 attributes: {sorted(missing)}"


def test_stage2_joins_one_to_one_on_customer_id(ds: Dataset) -> None:
    """Stage2_Data must be exactly one row per (train U test) customer."""
    assert len(ds.stage2) == 400
    assert ds.stage2[ID].is_unique
    union = set(ds.train[ID]) | set(ds.test[ID])
    assert len(union) == 400
    assert union == set(ds.stage2[ID])

    # The 1:1 promise, stated as an actual merge: row count is preserved.
    merged = ds.test[[ID]].merge(ds.stage2, on=ID, how="left", validate="one_to_one")
    assert len(merged) == len(ds.test)
    assert not merged.drop(columns=[ID]).isna().any().any()


def test_key_integrity_passes_on_release(ds: Dataset) -> None:
    assert_key_integrity(ds)  # must not raise


def test_guard_rejects_duplicate_test_id(ds: Dataset) -> None:
    broken = replace(ds, test=pd.concat([ds.test, ds.test.iloc[[0]]], ignore_index=True))
    with pytest.raises(KeyIntegrityError, match="duplicate"):
        assert_key_integrity(broken)


def test_guard_rejects_train_test_overlap(ds: Dataset) -> None:
    leaked = ds.test.copy()
    leaked.loc[leaked.index[0], ID] = ds.train[ID].iloc[0]
    with pytest.raises(KeyIntegrityError, match="overlap"):
        assert_key_integrity(replace(ds, test=leaked))


def test_guard_rejects_test_id_missing_from_stage2(ds: Dataset) -> None:
    trimmed = ds.stage2[ds.stage2[ID] != ds.test[ID].iloc[0]]
    with pytest.raises(KeyIntegrityError):
        assert_key_integrity(replace(ds, stage2=trimmed))


def test_guard_rejects_wrong_row_count(ds: Dataset) -> None:
    with pytest.raises(KeyIntegrityError, match="expected 100 rows"):
        assert_key_integrity(replace(ds, test=ds.test.iloc[:99]))


def test_loader_runs_integrity_check(config) -> None:
    load_dataset(config=config)  # verify=True by default
