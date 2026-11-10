# Pre-registered model-selection rule (v1.0)

Frozen at Stage 0, before any fit. Fingerprint: `34ac751dbdd2488c6f08a3b05fc2e0b33a9c1346627732a857b06d32ed835cdc`

## Selection order

1. maximise mean CV ROC-AUC across the 50 folds
2. tie within 1e-6 -> maximise mean CV PR-AUC
3. tie within 1e-6 -> minimise mean CV Brier (calibrated)
4. tie within 1e-6 -> earlier in complexity_order (simpler wins)
5. still tied -> earlier in candidate_models (deterministic)

Primary metric: `mean_roc_auc_over_folds` over 5x10 repeated stratified CV on train only.

## Calibration

- method: sigmoid (Platt)
- fitted on out-of-fold predictions of training rows only (nested CV)
- shipped probability: ship calibrated probability if CV Brier(calibrated) <= CV Brier(raw), else ship raw; decided once on train CV and logged

## Quantum admission

- allowed form: stacked/blended score with the selected classical model only
- admitted only if: delta mean CV ROC-AUC >= +0.01 vs the selected classical baseline
- admitted only if: strict improvement in at least 60% of the 10 CV repeats (paired per repeat)
- admitted only if: computed on identical folds and identical in-fold PCA features
- else: reject the quantum score and report 'did not help' with the measured delta

## Final fit and ranking

- fit on: all 300 Stage1_Train rows
- test usage: predict exactly once, at Stage 3
- rank by predicted_probability descending, ties by customer_id ascending (alphanumeric)

The fingerprint above is a SHA-256 of the canonical JSON form of `src/eval.py::PREREGISTERED_SELECTION`. It is recorded in `results/stage0_preregistration.json` at Stage 0 and asserted by `tests/test_preregistration.py`, so any post-hoc edit of the rule is detectable.
