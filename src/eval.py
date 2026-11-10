"""Pre-registered model-selection rule + metric plumbing.

FROZEN AT STAGE 0, BEFORE ANY MODEL WAS FITTED. The rule below is the single
source of truth for how Stage 3 picks a model and emits probabilities. Its
SHA-256 is written to ``results/stage0_preregistration.json`` at Stage 0 and
quoted in ``PREREGISTRATION.md``, so any later edit is detectable.

Deliberate design constraints:
  * every criterion is evaluated on **training data only** via repeated
    stratified CV -- the test set is never a selection signal;
  * ties resolve to the simpler model, then to the earliest candidate, so the
    outcome is a deterministic function of the data;
  * the quantum model may only enter through a stacked score, and only if it
    clears a pre-registered bar. Failing that, it is reported honestly as not
    helping.
"""

from __future__ import annotations

from typing import Any

from src.utils import sha256_text

PREREGISTERED_SELECTION: dict[str, Any] = {
    "version": "1.0",
    "frozen_at_stage": 0,
    "candidate_models": ["logistic_regression_l2", "random_forest", "gradient_boosting"],
    "complexity_order": ["logistic_regression_l2", "gradient_boosting", "random_forest"],
    "cv": {"scheme": "repeated_stratified_kfold", "n_splits": 5, "n_repeats": 10, "seed": 42},
    "primary_metric": "mean_roc_auc_over_folds",
    "criteria_in_order": [
        {"rank": 1, "criterion": "maximise mean CV ROC-AUC across the 50 folds"},
        {"rank": 2, "criterion": "tie within 1e-6 -> maximise mean CV PR-AUC"},
        {"rank": 3, "criterion": "tie within 1e-6 -> minimise mean CV Brier (calibrated)"},
        {"rank": 4, "criterion": "tie within 1e-6 -> earlier in complexity_order (simpler wins)"},
        {"rank": 5, "criterion": "still tied -> earlier in candidate_models (deterministic)"},
    ],
    "calibration": {
        "method": "sigmoid (Platt)",
        "where": "fitted on out-of-fold predictions of training rows only (nested CV)",
        "shipped_probability_rule": (
            "ship calibrated probability if CV Brier(calibrated) <= CV Brier(raw), "
            "else ship raw; decided once on train CV and logged"
        ),
    },
    "quantum_admission": {
        "allowed_form": "stacked/blended score with the selected classical model only",
        "admitted_if": [
            "delta mean CV ROC-AUC >= +0.01 vs the selected classical baseline",
            "strict improvement in at least 60% of the 10 CV repeats (paired per repeat)",
            "computed on identical folds and identical in-fold PCA features",
        ],
        "otherwise": "reject the quantum score and report 'did not help' with the measured delta",
    },
    "final_fit": {
        "fit_on": "all 300 Stage1_Train rows",
        "hyperparameters": "those selected by CV above; no re-tuning after seeing test",
        "test_usage": "predict exactly once, at Stage 3",
    },
    "ranking": {
        "order": "predicted_probability descending",
        "tie_break": "customer_id ascending (alphanumeric)",
        "top10_flag": "first 10 rows of the ranked test set; never edited afterwards",
    },
    "seeds": {"global": 42, "note": "all stochastic components derive from this seed"},
}


def fingerprint() -> str:
    """SHA-256 of the canonical preregistration text (change detector)."""
    import json

    return sha256_text(json.dumps(PREREGISTERED_SELECTION, sort_keys=True, separators=(",", ":")))


def as_text() -> str:
    """Markdown rendering used by PREREGISTRATION.md and the Stage 0 report."""
    rule = PREREGISTERED_SELECTION
    lines = [
        f"# Pre-registered model-selection rule (v{rule['version']})",
        "",
        f"Frozen at Stage {rule['frozen_at_stage']}, before any fit. Fingerprint: `{fingerprint()}`",
        "",
        "## Selection order",
        "",
    ]
    lines += [f"{c['rank']}. {c['criterion']}" for c in rule["criteria_in_order"]]
    lines += [
        "",
        f"Primary metric: `{rule['primary_metric']}` over "
        f"{rule['cv']['n_splits']}x{rule['cv']['n_repeats']} repeated stratified CV on train only.",
        "",
        "## Calibration",
        "",
        f"- method: {rule['calibration']['method']}",
        f"- {rule['calibration']['where']}",
        f"- shipped probability: {rule['calibration']['shipped_probability_rule']}",
        "",
        "## Quantum admission",
        "",
        f"- allowed form: {rule['quantum_admission']['allowed_form']}",
    ]
    lines += [f"- admitted only if: {c}" for c in rule["quantum_admission"]["admitted_if"]]
    lines += [
        f"- else: {rule['quantum_admission']['otherwise']}",
        "",
        "## Final fit and ranking",
        "",
        f"- fit on: {rule['final_fit']['fit_on']}",
        f"- test usage: {rule['final_fit']['test_usage']}",
        f"- rank by {rule['ranking']['order']}, ties by {rule['ranking']['tie_break']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """`python -m src.eval` -> freeze the rule to results/ and PREREGISTRATION.md.

    Run at Stage 0, before any model is fitted. The written fingerprint is what
    tests/test_preregistration.py compares against later.
    """
    from src.utils import get_logger, load_config, resolve, write_json, write_text

    log = get_logger("stage0")
    cfg = load_config()
    results = resolve(cfg["paths"]["results"])

    payload = {
        "version": PREREGISTERED_SELECTION["version"],
        "frozen_at_stage": PREREGISTERED_SELECTION["frozen_at_stage"],
        "fingerprint": fingerprint(),
        "rule": PREREGISTERED_SELECTION,
    }
    write_json(results / "stage0_preregistration.json", payload)
    write_text(
        resolve("PREREGISTRATION.md"),
        as_text()
        + "\nThe fingerprint above is a SHA-256 of the canonical JSON form of "
        "`src/eval.py::PREREGISTERED_SELECTION`. It is recorded in "
        "`results/stage0_preregistration.json` at Stage 0 and asserted by "
        "`tests/test_preregistration.py`, so any post-hoc edit of the rule is detectable.\n",
    )
    log.info("pre-registered selection rule v%s fingerprint=%s", payload["version"], payload["fingerprint"])
    log.info("wrote %s and PREREGISTRATION.md", results / "stage0_preregistration.json")


if __name__ == "__main__":
    main()
