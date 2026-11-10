"""Stage 1 invariants: CV geometry, metric definitions, determinism, no leakage."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

from src.classical import CANDIDATES, CANDIDATES_BY_LABEL, _pipeline, score_fold
from src.eval import PREREGISTERED_SELECTION
from src.features import prepare_xy
from src.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def train_xy():
    from src.data import load_dataset

    ds = load_dataset(config=load_config())
    X, y = prepare_xy(ds.train, labelled=True)
    return X, y


# --------------------------------------------------------------------------- #
# geometry: 50 folds, 60-row validation slices, train-only indices
# --------------------------------------------------------------------------- #
def test_fold_geometry_is_5x10_on_train_rows(cfg, train_xy):
    X, y = train_xy
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        n_repeats=cfg["cv"]["n_repeats"],
        random_state=cfg["cv"]["shuffle_seed"],
    )
    folds = list(splitter.split(X, y))
    assert len(folds) == 50, "5 x 10 repeated CV must yield 50 validation folds"
    for tr, va in folds:
        assert len(tr) == 240 and len(va) == 60
        # No index may point outside the 300 training rows: test rows are unreachable.
        assert tr.max() < 300 and va.max() < 300
        assert np.intersect1d(tr, va).size == 0
        # Stratification: both classes present in the validation slice.
        assert set(np.unique(y[va])) == {0, 1}


def test_repeats_partition_the_train_set(cfg, train_xy):
    X, y = train_xy
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=1, random_state=cfg["cv"]["shuffle_seed"]
    )
    for _, va in splitter.split(X, y):
        assert va.size == 60
    covered = np.sort(np.concatenate([va for _, va in splitter.split(X, y)]))
    assert np.array_equal(covered, np.arange(300)), "one repeat must cover every train row once"


# --------------------------------------------------------------------------- #
# metric definitions
# --------------------------------------------------------------------------- #
def test_perfect_ranking_scores_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = score_fold(y, s)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["k_top"] == 1  # round(0.1 * 6) = 1
    assert m["precision_at_10pct"] == pytest.approx(1.0)
    assert m["recall_at_10pct"] == pytest.approx(1 / 3)


def test_precision_and_recall_use_the_same_top_slice():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    m = score_fold(y, s)
    assert m["k_top"] == 1
    assert m["precision_at_10pct"] == pytest.approx(1.0)
    assert m["recall_at_10pct"] == pytest.approx(0.5)


def test_brier_is_mean_squared_error_of_probability():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.2, 0.6, 0.1])
    m = score_fold(y, s)
    assert m["brier"] == pytest.approx(np.mean((s - y) ** 2))


# --------------------------------------------------------------------------- #
# candidate registry <-> pre-registration
# --------------------------------------------------------------------------- #
def test_every_candidate_maps_to_a_preregistered_model():
    registered = set(PREREGISTERED_SELECTION["candidate_models"])
    for cand in CANDIDATES:
        assert cand.model in registered, f"{cand.label} is not a registered candidate"


def test_all_three_registered_models_are_represented():
    models = {c.model for c in CANDIDATES}
    assert models == set(PREREGISTERED_SELECTION["candidate_models"])


def test_gb_configs_share_one_registered_model():
    assert CANDIDATES_BY_LABEL["gradient_boosting"].model == "gradient_boosting"
    assert CANDIDATES_BY_LABEL["gradient_boosting_tuned"].model == "gradient_boosting"


# --------------------------------------------------------------------------- #
# fold-local fitting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(CANDIDATES_BY_LABEL))
def test_pipelines_start_unfitted(label, cfg):
    pipe = _pipeline(label, cfg)
    assert not hasattr(pipe.named_steps["prep"], "transformers_") or (
        pipe.named_steps["prep"].transformers_ is None
    )
    with pytest.raises(Exception):
        pipe.predict_proba(np.zeros((1, 17)))


def test_fitting_one_pipeline_does_not_affect_the_next(cfg, train_xy):
    X, y = train_xy
    first = _pipeline("gradient_boosting_tuned", cfg).fit(X.iloc[:120], y[:120])
    second = _pipeline("gradient_boosting_tuned", cfg)
    assert first.named_steps["prep"].transformers_ is not None
    assert getattr(second.named_steps["prep"], "transformers_", None) is None


def test_two_fits_with_the_same_seed_are_identical(cfg, train_xy):
    X, y = train_xy
    a = _pipeline("gradient_boosting_tuned", cfg).fit(X.iloc[:200], y[:200])
    b = _pipeline("gradient_boosting_tuned", cfg).fit(X.iloc[:200], y[:200])
    np.testing.assert_allclose(
        a.predict_proba(X.iloc[200:])[:, 1], b.predict_proba(X.iloc[200:])[:, 1], rtol=0, atol=1e-12
    )
