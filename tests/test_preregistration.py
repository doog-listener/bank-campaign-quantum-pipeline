"""The selection rule must be provably frozen and internally well-formed.

These tests are the tripwire against silent post-hoc tuning: if anyone edits the
rule after looking at results, ``fingerprint()`` changes and the Stage 0 record
no longer matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import PREREGISTERED_SELECTION, as_text, fingerprint

STAGE0_RECORD = Path(__file__).resolve().parents[1] / "results" / "stage0_preregistration.json"


def test_rule_is_frozen_at_stage_zero() -> None:
    assert PREREGISTERED_SELECTION["frozen_at_stage"] == 0
    assert len(fingerprint()) == 64


def test_primary_metric_is_cv_only() -> None:
    assert PREREGISTERED_SELECTION["primary_metric"] == "mean_roc_auc_over_folds"
    cv = PREREGISTERED_SELECTION["cv"]
    assert (cv["n_splits"], cv["n_repeats"]) == (5, 10)


def test_tie_break_is_ascending_customer_id() -> None:
    assert "customer_id ascending" in PREREGISTERED_SELECTION["ranking"]["tie_break"]


def test_quantum_admission_bar_is_explicit() -> None:
    bar = PREREGISTERED_SELECTION["quantum_admission"]["admitted_if"]
    assert any("+0.01" in c for c in bar)
    assert any("60%" in c for c in bar)
    assert "stacked" in PREREGISTERED_SELECTION["quantum_admission"]["allowed_form"]


def test_test_set_is_predicted_exactly_once() -> None:
    final = PREREGISTERED_SELECTION["final_fit"]
    assert "once" in final["test_usage"]
    assert final["fit_on"].startswith("all 300")


def test_fingerprint_matches_the_stage0_record() -> None:
    """Tripwire: the rule must be byte-identical to what Stage 0 froze."""
    if not STAGE0_RECORD.exists():
        pytest.skip("run `python -m src.eval` to write the Stage 0 record")
    record = json.loads(STAGE0_RECORD.read_text(encoding="utf-8"))
    assert record["fingerprint"] == fingerprint(), "preregistration was edited after Stage 0"
    assert record["rule"] == PREREGISTERED_SELECTION


def test_fingerprint_is_stable_and_renders() -> None:
    assert fingerprint() == fingerprint()
    text = as_text()
    assert "Pre-registered" in text
    assert fingerprint() in text
    # Canonical serialisation must be byte-stable.
    a = json.dumps(PREREGISTERED_SELECTION, sort_keys=True, separators=(",", ":"))
    b = json.dumps(PREREGISTERED_SELECTION, sort_keys=True, separators=(",", ":"))
    assert a == b
