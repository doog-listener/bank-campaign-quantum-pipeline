"""Feature contract + fold-local preprocessing factory.

Two hard rules are enforced structurally here:

1. **Nothing is fitted outside a CV fold.** ``make_preprocessor()`` returns an
   *unfitted* transformer; only a fold's training rows may ever be passed to
   ``.fit``. There is no module-level fitted object to accidentally reuse.
2. **The test frame never contributes to feature construction.** Derived
   features are row-local pure functions (no target, no cross-row statistics),
   and every learned statistic (median, mode, category vocabulary, mean/std)
   lives inside the fold-fitted transformer.

``pdays == -1`` is a sentinel for "never contacted before", not a duration, so
it is split into a ``prev_contacted`` flag and a masked ``pdays`` column that is
imputed in-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID: str = "customer_id"
TARGET: str = "y"
PDays_SENTINEL: int = -1

# Raw columns used as features (post `duration` removal by the organisers).
NUMERIC_FEATURES: tuple[str, ...] = ("age", "balance", "day", "campaign", "pdays", "previous")
BINARY_FEATURES: tuple[str, ...] = ("default", "housing", "loan")  # yes/no -> one-hot
CATEGORICAL_FEATURES: tuple[str, ...] = ("job", "marital", "education", "contact", "month", "poutcome")
DERIVED_FEATURES: tuple[str, ...] = ("prev_contacted",)

FORBIDDEN_AS_FEATURE: frozenset[str] = frozenset({ID, TARGET, "duration"})


def feature_manifest() -> list[str]:
    """Ordered list of model-input columns; frozen into the Stage 0 audit."""
    return [*NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES, *DERIVED_FEATURES]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local (leak-free) derivations. Safe to apply to train or test alone."""
    out = df.copy()
    pdays = pd.to_numeric(out["pdays"], errors="coerce")
    out["prev_contacted"] = (pdays != PDays_SENTINEL).astype(int)
    # Mask the sentinel so the in-fold median is computed over real durations.
    out["pdays"] = pdays.where(pdays != PDays_SENTINEL)
    return out


def assert_no_leakage(df: pd.DataFrame, *, allow_target: bool = False) -> None:
    """Fail fast if a forbidden or target column is in a frame headed for a model."""
    bad = sorted({"customer_id", "duration"} & set(df.columns))
    if bad:
        raise ValueError(f"forbidden feature columns present: {bad}")
    if TARGET in df.columns and not allow_target:
        raise ValueError(f"'{TARGET}' must not be present in an unlabelled frame")


def assert_feature_manifest_clean() -> None:
    """The manifest itself must never name the key, the target or `duration`."""
    overlap = sorted(set(feature_manifest()) & FORBIDDEN_AS_FEATURE)
    if overlap:
        raise ValueError(f"feature manifest contains forbidden columns: {overlap}")


def make_preprocessor() -> ColumnTransformer:
    """Unfitted, fold-local preprocessing. Fit inside each CV fold on train only.

    * numeric: median impute (learned in-fold) + missing indicator + standardise
    * binary/categorical: most-frequent impute + one-hot with
      ``handle_unknown='ignore'`` so a test-only level degrades to all-zeros
      instead of crashing or leaking a vocabulary from test.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipe, [*BINARY_FEATURES, *CATEGORICAL_FEATURES]),
            ("flag", "passthrough", list(DERIVED_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_xy(df: pd.DataFrame, *, labelled: bool) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Split a frame into (features, y). Labels exist on train only."""
    assert_feature_manifest_clean()
    if labelled and TARGET not in df.columns:
        raise ValueError(f"labelled frame must carry '{TARGET}'")
    if not labelled and TARGET in df.columns:
        raise ValueError(f"unlabelled frame must not carry '{TARGET}'")
    base_cols = [c for c in feature_manifest() if c not in DERIVED_FEATURES]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    # Derive from raw inputs only, then freeze the canonical column order.
    X = add_derived_features(df[base_cols])[list(feature_manifest())]
    assert_no_leakage(X)  # belt and braces: no key/target/duration survived selection
    if not labelled:
        return X, None
    return X, (df[TARGET] == "yes").astype(int).to_numpy()
