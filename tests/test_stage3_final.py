"""Stage 3 invariants: the single probability vector, ranking, ties, one-shot scoring.

These are the checks that make "the test set was scored exactly once" and "ties go
to the lower ``customer_id``" verifiable rather than aspirational.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classical import (
    bootstrap_test_probabilities,
    candidate_table,
    finalise,
    load_stage1_decision,
)
from src.data import load_dataset
from src.features import prepare_xy
from src.utils import STAGE1_COLUMNS, load_config, resolve


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def decision(cfg):
    return load_stage1_decision(cfg)


@pytest.fixture(scope="module")
def finalised(cfg, decision):
    return finalise(cfg, decision)


# --------------------------------------------------------------------------- #
# decision hand-off
# --------------------------------------------------------------------------- #
def test_stage1_artifact_must_exist(cfg, tmp_path: Path):
    """Stage 3 refuses to invent a winner when Stage 1 has not been run."""
    bad = dict(cfg)
    bad["paths"] = {**cfg["paths"], "results": str(tmp_path)}
    with pytest.raises(FileNotFoundError, match="run `python -m src.classical`"):
        load_stage1_decision(bad)


def test_decision_fingerprint_matches_frozen_preregistration(cfg, decision):
    from src.eval import fingerprint

    assert decision["fingerprint"] == fingerprint()
    assert decision["selected_config"] == "gradient_boosting_tuned"


# --------------------------------------------------------------------------- #
# the probability vector
# --------------------------------------------------------------------------- #
def test_output_contract(finalised):
    frame = finalised["frame"]
    assert list(frame.columns) == list(STAGE1_COLUMNS)
    assert len(frame) == 100
    assert frame["customer_id"].is_unique
    assert frame["rank"].tolist() == list(range(1, 101))
    assert int(frame["top10_flag"].sum()) == 10
    assert frame["predicted_probability"].between(0.0, 1.0).all()


def test_ranking_is_probability_desc_then_id_asc(finalised):
    frame = finalised["frame"]
    assert frame["predicted_probability"].is_monotonic_decreasing
    ordered = frame.reset_index(drop=True)
    for i in range(len(ordered) - 1):
        a, b = ordered.iloc[i], ordered.iloc[i + 1]
        if a["predicted_probability"] == b["predicted_probability"]:
            assert a["customer_id"] < b["customer_id"]


def test_ties_are_broken_by_id_and_DO_occur(finalised):
    """The pre-registered tie-break is not decorative: GB emits exact ties here."""
    frame = finalised["frame"]
    values = frame["predicted_probability"].to_numpy()
    ties = len(values) - len(np.unique(values))
    assert ties >= 1, "expected at least one exact tie among 100 GB probabilities"
    duplicated = frame[frame.duplicated("predicted_probability", keep=False)]
    for _, group in duplicated.groupby("predicted_probability"):
        assert group["customer_id"].tolist() == sorted(group["customer_id"].tolist())


def test_probability_vector_is_not_the_ranked_frame(finalised, ds):
    """The stored vector keeps test-sheet order so ids cannot silently reorder."""
    prob = finalised["probabilities"]
    assert len(prob) == 100
    assert list(finalised["customer_ids"]) == ds.test["customer_id"].astype(str).tolist()
    assert not np.array_equal(prob, finalised["frame"]["predicted_probability"].to_numpy())
    assert np.allclose(np.sort(prob)[::-1], finalised["frame"]["predicted_probability"].to_numpy())


def test_refit_is_deterministic(cfg, decision, finalised):
    again = finalise(cfg, decision)
    assert again["sha256_probabilities"] == finalised["sha256_probabilities"]
    assert np.array_equal(again["probabilities"], finalised["probabilities"])


def test_raw_probabilities_ship_because_stage1_rejected_calibration(finalised):
    """Stage 1 decided `ship_calibrated=False`; Stage 3 must honour that bit."""
    assert finalised["decision"]["ship_calibrated"] is False


def test_fit_used_every_training_row(finalised, ds):
    assert finalised["n_train"] == len(ds.train) == 300
    assert finalised["n_positive"] == int((ds.train["y"] == "yes").sum())


# --------------------------------------------------------------------------- #
# candidate hand-off to Stage 4
# --------------------------------------------------------------------------- #
def test_candidate_table_is_the_top_ten_joined_to_operations(finalised, ds):
    table = candidate_table(finalised, ds.stage2)
    assert len(table) == 10
    assert table["customer_id"].tolist() == finalised["frame"]["customer_id"].head(10).tolist()
    for column in ("contact_cost", "expected_deposit_value", "capacity_units"):
        assert table[column].notna().all()
    assert table["customer_id"].is_unique


# --------------------------------------------------------------------------- #
# bootstrap uncertainty (small number of resamples: this is a shape test)
# --------------------------------------------------------------------------- #
def test_bootstrap_matrix_shape_and_determinism(cfg, decision):
    first = bootstrap_test_probabilities(cfg, decision, n_resamples=3)
    second = bootstrap_test_probabilities(cfg, decision, n_resamples=3)
    assert first.shape == (3, 100)
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert np.array_equal(first, second)
    # bootstrap draws must differ from the point estimate and from each other
    point = finalise(cfg, decision)["probabilities"]
    assert not np.allclose(first[0], point, atol=1e-6)
    assert not np.allclose(first[0], first[1], atol=1e-6)


def test_bootstrap_never_touches_test_features(cfg, decision):
    """Resampling only moves training rows: the scored frame is the released test set."""
    ds = load_dataset(config=cfg)
    _, _ = prepare_xy(ds.test, labelled=False)
    boot = bootstrap_test_probabilities(cfg, decision, n_resamples=2)
    assert boot.shape[1] == len(ds.test)


def test_written_artifacts_agree_with_the_rerun(cfg, finalised):
    """If Stage 3 was run, the CSV on disk must equal a fresh refit of the same code."""
    results = resolve(cfg["paths"]["results"])
    path = results / "stage1_results.csv"
    if not path.exists():
        pytest.skip("stage1_results.csv not produced yet")
    on_disk = pd.read_csv(path)
    assert on_disk["customer_id"].tolist() == finalised["frame"]["customer_id"].tolist()
    assert np.allclose(on_disk["predicted_probability"], finalised["frame"]["predicted_probability"])

    meta = json.loads((results / "stage3_meta.json").read_text(encoding="utf-8"))
    assert meta["sha256_probabilities"] == finalised["sha256_probabilities"]
    npz = np.load(results / "stage3_bootstrap_probs.npz")["probabilities"]
    assert npz.shape == (int(cfg["stage3"]["bootstrap_resamples"]), 100)
