"""Audit invariants: ranges, balance, leakage, constraint envelope, fold-local prep.

Each assertion here encodes a fact the modelling stages are allowed to rely on.
If the release changes, these fail loudly instead of quietly shifting results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.data import ID, TARGET, Dataset, audit_dataset
from src.features import (
    FORBIDDEN_AS_FEATURE,
    add_derived_features,
    assert_feature_manifest_clean,
    assert_no_leakage,
    feature_manifest,
    make_preprocessor,
    prepare_xy,
)


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_duration_removed_from_all_sheets(ds: Dataset) -> None:
    for frame in (ds.train, ds.test, ds.stage2):
        assert "duration" not in frame.columns


def test_target_exists_only_on_train(ds: Dataset) -> None:
    assert TARGET in ds.train.columns
    assert TARGET not in ds.test.columns
    assert TARGET not in ds.stage2.columns


def test_no_identical_feature_rows_between_train_and_test(ds: Dataset) -> None:
    report = audit_dataset(ds)
    assert report["leakage"]["identical_feature_rows_train_test"]["count"] == 0


def test_feature_manifest_excludes_key_target_and_duration() -> None:
    assert set(feature_manifest()) & FORBIDDEN_AS_FEATURE == set()
    assert_feature_manifest_clean()
    with pytest.raises(ValueError):
        assert_no_leakage(pd.DataFrame({ID: ["C1"], "age": [30]}))


# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #
def test_no_missing_values(ds: Dataset) -> None:
    assert int(ds.train.isna().sum().sum()) == 0
    assert int(ds.test.isna().sum().sum()) == 0
    assert int(ds.stage2.isna().sum().sum()) == 0


def test_no_duplicate_rows(ds: Dataset) -> None:
    assert int(ds.train.duplicated().sum()) == 0
    assert int(ds.test.duplicated().sum()) == 0
    assert int(ds.stage2.duplicated().sum()) == 0


def test_numeric_ranges_are_plausible(ds: Dataset) -> None:
    violations = audit_dataset(ds)["range_violations"]
    assert all(v == [] for v in violations.values()), violations


def test_class_balance_matches_release(ds: Dataset) -> None:
    bal = audit_dataset(ds)["class_balance"]
    assert bal["positives"] == 35
    assert bal["negatives"] == 265
    assert bal["positive_rate_pct"] == pytest.approx(11.67, abs=0.01)


# --------------------------------------------------------------------------- #
# derived features
# --------------------------------------------------------------------------- #
def test_prev_contacted_flag_marks_pdays_sentinel(train_frame: pd.DataFrame) -> None:
    out = add_derived_features(train_frame)
    never = train_frame["pdays"] == -1
    assert (out.loc[never, "prev_contacted"] == 0).all()
    assert (out.loc[~never, "prev_contacted"] == 1).all()
    assert out.loc[never, "pdays"].isna().all()
    assert out["prev_contacted"].isin([0, 1]).all()


def test_derived_features_do_not_mutate_input(train_frame: pd.DataFrame) -> None:
    add_derived_features(train_frame)
    assert "prev_contacted" not in train_frame.columns
    assert (train_frame["pdays"] == -1).any()


def test_prepare_xy_shapes_and_label_encoding(ds: Dataset) -> None:
    X_train, y_train = prepare_xy(ds.train, labelled=True)
    X_test, y_test = prepare_xy(ds.test, labelled=False)

    assert list(X_train.columns) == feature_manifest()
    assert len(X_train) == 300 and len(X_test) == 100
    assert y_train is not None and set(np.unique(y_train)) == {0, 1}
    assert int(y_train.sum()) == 35
    assert y_test is None


# --------------------------------------------------------------------------- #
# fold-local preprocessing (hard rule: nothing fitted on test or outside folds)
# --------------------------------------------------------------------------- #
def test_preprocessor_is_unfitted_until_fitted() -> None:
    prep = make_preprocessor()
    with pytest.raises(NotFittedError):
        check_is_fitted(prep)


def test_preprocessor_fits_on_train_fold_only_and_tolerates_unseen_levels(ds: Dataset) -> None:
    X_train, _ = prepare_xy(ds.train, labelled=True)
    X_test, _ = prepare_xy(ds.test, labelled=False)

    fold = X_train.iloc[:240]  # inner-training slice of one fold
    prep = make_preprocessor().fit(fold, np.zeros(len(fold)))
    check_is_fitted(prep)

    transformed = prep.transform(X_test)
    assert transformed.shape[0] == len(X_test)
    assert np.isfinite(transformed).all()

    # A category never seen in the whole dataset degrades to one-hot all-zeros.
    novel = X_test.iloc[[0]].copy()
    novel["job"] = "astronaut"
    assert prep.transform(novel).shape == (1, transformed.shape[1])


def test_two_preprocessors_are_independent_instances(ds: Dataset) -> None:
    X_train, _ = prepare_xy(ds.train, labelled=True)
    first = make_preprocessor().fit(X_train, np.zeros(len(X_train)))
    second = make_preprocessor()
    assert first is not second
    with pytest.raises(NotFittedError):
        check_is_fitted(second)


# --------------------------------------------------------------------------- #
# Stage 2 constraint envelope
# --------------------------------------------------------------------------- #
def test_constraint_constants_match_release(ds: Dataset, constraints_map: dict[str, int]) -> None:
    assert constraints_map["N"] == 10
    assert constraints_map["K"] == 6
    assert constraints_map["C"] == 25
    assert constraints_map["H"] == 22


def test_budget_never_binds_but_capacity_can(ds: Dataset, constraints_map: dict[str, int]) -> None:
    worst_cost = int(ds.stage2["contact_cost"].max()) * constraints_map["K"]
    worst_capacity = int(ds.stage2["capacity_units"].max()) * constraints_map["K"]
    assert worst_cost <= constraints_map["C"], "budget is supposed to be slack"
    assert worst_capacity > constraints_map["H"], "capacity must be able to bind"


def test_stage2_ranges_and_segments(ds: Dataset) -> None:
    assert ds.stage2["contact_cost"].between(0, 25).all()
    assert ds.stage2["capacity_units"].between(0, 22).all()
    assert (ds.stage2["expected_deposit_value"] > 0).all()
    assert ds.stage2["customer_segment"].nunique() <= 6
