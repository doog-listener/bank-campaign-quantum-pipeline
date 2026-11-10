"""Stage 1 -- classical baseline under the frozen pre-registered protocol.

Implements exactly what ``PREREGISTERED_SELECTION`` v1.0 demands:

* repeated stratified 5x10 CV on ``Stage1_Train`` only (50 validation folds);
* three registered candidates -- strong-L2 logistic regression, random forest
  and shallow gradient boosting;
* every learned statistic (median, mode, one-hot vocabulary, mean/std) is fitted
  inside the fold's training rows, and sigmoid calibration is fitted on the
  fold's training rows only, never on validation rows and never on test;
* one metric row per (config, fold) so ``src.eval`` can apply the selection rule.

Hyperparameter handling
-----------------------
The pre-registration allows CV-selected hyperparameters per candidate
("hyperparameters: those selected by CV above"). The grid is fixed *here*, in
code, before the run -- two gradient-boosting configurations are registered so
the tuned shallow variant is a declared config rather than a post-hoc rescue:

    logistic_regression_l2   C = 0.1                       (strong L2)
    random_forest            n_estimators=100, max_depth=5
    gradient_boosting        n_estimators=50,  max_depth=3
    gradient_boosting_tuned  n_estimators=100, max_depth=2, learning_rate=0.03

``gradient_boosting_tuned`` belongs to the ``gradient_boosting`` candidate: the
model-level comparison in the pre-registration is over the three *models*, and
which of GB's two configs represents it is decided by mean CV ROC-AUC before the
rule is applied.

Metrics and their definitions (all computed per fold, on validation rows only):

* ``roc_auc``            -- threshold-free ranking quality;
* ``pr_auc``             -- ``average_precision_score`` (area under PR curve);
* ``brier``              -- mean squared error of the probability itself;
* ``log_loss``           -- with ``labels=[0, 1]``;
* ``precision_at_10pct`` -- positives among the top ``round(0.1 * n_val)`` rows;
* ``recall_at_10pct``    -- share of the fold's positives captured by that slice.

Run: ``python -m src.classical`` -> ``results/stage1_cv.{json,md}``.

This module also owns **Stage 3** (``python -m src.classical --finalise``): refit
the pre-registered winner on all 300 rows, score the 100 test rows once, rank with
ties by ascending ``customer_id``, and write ``results/stage1_results.csv`` plus the
bootstrap probability matrix later stages consume.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import Dataset, ID, TARGET, load_dataset
from src.eval import PREREGISTERED_SELECTION, fingerprint
from src.features import make_preprocessor, prepare_xy
from src.utils import (
    STAGE1_COLUMNS,
    child_seeds,
    get_logger,
    load_config,
    resolve,
    seed_everything,
    sha256_text,
    write_json,
    write_text,
)

TOPK_FRACTION: float = 0.10


# --------------------------------------------------------------------------- #
# registered candidates (fixed in code before the run)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One registered estimator configuration."""

    label: str
    model: str  # name used by the pre-registration
    params: dict[str, Any] = field(default_factory=dict)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("logistic_regression_l2", "logistic_regression_l2", {"clf__C": 0.1}),
    Candidate("random_forest", "random_forest", {"clf__n_estimators": 100, "clf__max_depth": 5}),
    Candidate("gradient_boosting", "gradient_boosting", {"clf__n_estimators": 50, "clf__max_depth": 3}),
    Candidate(
        "gradient_boosting_tuned",
        "gradient_boosting",
        {"clf__n_estimators": 100, "clf__max_depth": 2, "clf__learning_rate": 0.03},
    ),
)


def make_candidates(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the unfitted *estimators* keyed by registered label.

    Preprocessing is deliberately not attached here: ``_pipeline`` builds a fresh
    unfitted ``ColumnTransformer`` per fold so no fitted object can leak between
    folds (see ``src/features.py`` rule 1).
    """
    seed = (config or load_config())["seed"]
    return {
        # L2 is the estimator default in scikit-learn >= 1.8 (`penalty` is
        # deprecated); "strong" L2 comes from the small C below.
        "logistic_regression_l2": LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=5000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=seed
        ),
        "gradient_boosting_tuned": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.03, random_state=seed
        ),
    }


def _pipeline(label: str, config: dict[str, Any]) -> Pipeline:
    """Unfitted ``preprocess -> classifier`` pipeline with the config applied."""
    params = {k: v for k, v in CANDIDATES_BY_LABEL[label].params.items()}
    est = make_candidates(config)[label]
    return Pipeline([("prep", make_preprocessor()), ("clf", est)]).set_params(**params)


CANDIDATES_BY_LABEL: dict[str, Candidate] = {c.label: c for c in CANDIDATES}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score_fold(y_true: np.ndarray, y_score: np.ndarray, top_fraction: float = TOPK_FRACTION) -> dict[str, Any]:
    """All Stage 1 metrics for one validation fold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    k = max(1, int(round(top_fraction * n)))
    order = np.argsort(-y_score, kind="mergesort")[:k]
    hits = int(y_true[order].sum())
    positives = int(y_true.sum())
    out = {
        "n_val": n,
        "k_top": k,
        "positives": positives,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "precision_at_10pct": hits / k,
        "recall_at_10pct": hits / positives if positives else 0.0,
    }
    return out


def _reliability(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Pooled reliability curve (equal-width bins) for the calibration figure."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, float]] = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "mean_predicted": float(y_score[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# CV harness
# --------------------------------------------------------------------------- #
def run_cv(config: dict[str, Any], dataset: Dataset | None = None) -> dict[str, Any]:
    """Repeated stratified CV on train only; returns folds + summary + decision.

    Fold order is ``RepeatedStratifiedKFold`` order (repeat-major): fold index
    ``r * n_splits + s`` is repeat ``r``, split ``s`` -- used for the paired
    per-repeat quantum comparison later.
    """
    cfg = config
    ds = dataset if dataset is not None else load_dataset(config=cfg)
    X, y = prepare_xy(ds.train, labelled=True)
    assert y is not None

    n_splits = int(cfg["cv"]["n_splits"])
    n_repeats = int(cfg["cv"]["n_repeats"])
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=int(cfg["cv"]["shuffle_seed"])
    )

    fold_rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[float]]] = {
        c.label: {"y": [], "raw": [], "cal": []} for c in CANDIDATES
    }

    for fold, (tr, va) in enumerate(splitter.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        repeat, split = divmod(fold, n_splits)

        for cand in CANDIDATES:
            raw_model = _pipeline(cand.label, cfg).fit(X_tr, y_tr)
            raw_score = raw_model.predict_proba(X_va)[:, 1]

            # Sigmoid calibration fitted on THIS fold's training rows only.
            calibrated = CalibratedClassifierCV(
                _pipeline(cand.label, cfg), method="sigmoid", cv=3, ensemble=True
            ).fit(X_tr, y_tr)
            cal_score = calibrated.predict_proba(X_va)[:, 1]

            raw_m = score_fold(y_va, raw_score)
            cal_m = score_fold(y_va, cal_score)
            fold_rows.append(
                {
                    "fold": fold,
                    "repeat": repeat,
                    "split": split,
                    "label": cand.label,
                    "model": cand.model,
                    **{f"raw_{k}": v for k, v in raw_m.items()},
                    **{f"cal_{k}": v for k, v in cal_m.items()},
                }
            )
            pooled[cand.label]["y"].extend(y_va.tolist())
            pooled[cand.label]["raw"].extend(raw_score.tolist())
            pooled[cand.label]["cal"].extend(cal_score.tolist())

    folds = pd.DataFrame(fold_rows)
    summary = _summarise(folds, pooled)
    decision = _decide(summary, folds)
    return {
        "summary": summary,
        "decision": decision,
        "folds": folds,
        "pooled": pooled,
        "cv": {"n_splits": n_splits, "n_repeats": n_repeats, "n_folds": len(folds) // len(CANDIDATES)},
        "prereg_fingerprint": fingerprint(),
        "candidates": [{"label": c.label, "model": c.model, "params": c.params} for c in CANDIDATES],
    }


def _summarise(folds: pd.DataFrame, pooled: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    """Mean/SD across the 50 folds plus the pooled reliability curve."""
    metric_names = [
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "precision_at_10pct",
        "recall_at_10pct",
    ]
    rows: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        sub = folds[folds["label"] == cand.label]
        row: dict[str, Any] = {"label": cand.label, "model": cand.model, "params": cand.params}
        for m in metric_names:
            raw = sub[f"raw_{m}"].to_numpy(dtype=float)
            cal = sub[f"cal_{m}"].to_numpy(dtype=float)
            row[m] = {
                "raw_mean": float(raw.mean()),
                "raw_sd": float(raw.std(ddof=1)),
                "cal_mean": float(cal.mean()),
                "cal_sd": float(cal.std(ddof=1)),
            }
        row["reliability"] = {
            "raw": _reliability(np.array(pooled[cand.label]["y"]), np.array(pooled[cand.label]["raw"])),
            "cal": _reliability(np.array(pooled[cand.label]["y"]), np.array(pooled[cand.label]["cal"])),
        }
        rows.append(row)
    return rows


def _decide(summary: list[dict[str, Any]], folds: pd.DataFrame) -> dict[str, Any]:
    """Apply PREREGISTERED_SELECTION v1.0 to the measured CV metrics."""
    rule = PREREGISTERED_SELECTION
    tol = 1e-6
    complexity = rule["complexity_order"]

    # 1) GB has two registered configs: the pre-registration compares *models*,
    #    so GB is represented by the config with the higher mean CV ROC-AUC.
    by_model: dict[str, dict[str, Any]] = {}
    for row in summary:
        cur = by_model.get(row["model"])
        if cur is None or row["roc_auc"]["raw_mean"] > cur["roc_auc"]["raw_mean"] + tol:
            by_model[row["model"]] = row

    registered = [m for m in rule["candidate_models"] if m in by_model]
    missing = [m for m in rule["candidate_models"] if m not in by_model]
    if missing:
        raise RuntimeError(f"registered candidates missing from the run: {missing}")

    def key(name: str) -> tuple[float, float, float, int, int]:
        r = by_model[name]
        return (
            -r["roc_auc"]["raw_mean"],
            -r["pr_auc"]["raw_mean"],
            r["brier"]["cal_mean"],
            complexity.index(name),
            registered.index(name),
        )

    ranked = sorted(registered, key=key)
    winner = ranked[0]
    win_row = by_model[winner]

    # Calibration ships only if it does not hurt Brier on train CV (prereg).
    ship_calibrated = bool(win_row["brier"]["cal_mean"] <= win_row["brier"]["raw_mean"])

    # Per-repeat paired comparison support: how often the winner is ahead.
    return {
        "rule_version": rule["version"],
        "fingerprint": fingerprint(),
        "ranked_models": ranked,
        "per_model_roc_auc": {m: by_model[m]["roc_auc"]["raw_mean"] for m in registered},
        "selected_model": winner,
        "selected_config": win_row["label"],
        "selected_params": win_row["params"],
        "ship_calibrated": ship_calibrated,
        "calibration_note": (
            "calibrated Brier <= raw Brier on train CV -> ship sigmoid-calibrated"
            if ship_calibrated
            else "sigmoid hurts Brier on train CV -> ship raw probabilities (prereg fallback)"
        ),
        "winner_metrics": {m: win_row[m] for m in ("roc_auc", "pr_auc", "brier", "log_loss",
                                                   "precision_at_10pct", "recall_at_10pct")},
    }


# --------------------------------------------------------------------------- #
# rendering / entry point
# --------------------------------------------------------------------------- #
def render_markdown(result: dict[str, Any]) -> str:
    dec = result["decision"]
    lines = [
        "# Stage 1 -- classical baseline (5x10 repeated stratified CV, train only)",
        "",
        f"pre-registration fingerprint: `{dec['fingerprint']}`",
        "",
        "| config | raw ROC-AUC | cal ROC-AUC | raw PR-AUC | cal PR-AUC | raw Brier | cal Brier | "
        "raw P@10% | cal P@10% | raw R@10% | cal R@10% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["summary"]:
        cells = []
        for m in ("roc_auc", "pr_auc", "brier", "precision_at_10pct", "recall_at_10pct"):
            cells.append(f"{row[m]['raw_mean']:.4f} ± {row[m]['raw_sd']:.4f}")
            cells.append(f"{row[m]['cal_mean']:.4f} ± {row[m]['cal_sd']:.4f}")
        # reorder to raw/cal pairs per metric as in the header
        order = [cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6], cells[7], cells[8], cells[9]]
        lines.append(f"| {row['label']} | " + " | ".join(order) + " |")

    lines += [
        "",
        "## Pre-registered decision",
        "",
        f"- ranked models (ROC-AUC): {', '.join(f'{m} {v:.4f}' for m, v in dec['per_model_roc_auc'].items())}",
        f"- **selected model: `{dec['selected_model']}`** (config `{dec['selected_config']}`)",
        f"- params: `{dec['selected_params']}`",
        f"- calibration: {'ship calibrated' if dec['ship_calibrated'] else 'ship raw'} -- {dec['calibration_note']}",
        "",
        "## Winner metrics (mean ± SD over 50 folds, uncalibrated)",
        "",
    ]
    for m, v in dec["winner_metrics"].items():
        lines.append(f"- {m}: {v['raw_mean']:.4f} ± {v['raw_sd']:.4f}")
    lines += [
        "",
        "Notes: 50 folds = 5 splits x 10 repeats; each fold's validation slice is 60 rows and the "
        "top-10% slice is 6 rows, so Precision@10% moves in steps of 1/6. Every transformer and the "
        "sigmoid calibrator are fitted on the fold's 240 training rows only.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    log = get_logger("stage1")
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    result = run_cv(cfg)

    results = resolve(cfg["paths"]["results"])
    dec = result["decision"]
    folds: pd.DataFrame = result["folds"]

    write_json(
        results / "stage1_cv.json",
        {
            "stage": 1,
            "cv": result["cv"],
            "prereg_fingerprint": result["prereg_fingerprint"],
            "seed": cfg["seed"],
            "candidates": result["candidates"],
            "summary": result["summary"],
            "decision": dec,
            "folds": folds.to_dict(orient="records"),
        },
    )
    write_text(results / "stage1_cv.md", render_markdown(result))

    for row in result["summary"]:
        log.info(
            "%-26s ROC-AUC %.4f±%.4f  PR-AUC %.4f  Brier(raw/cal) %.4f/%.4f",
            row["label"],
            row["roc_auc"]["raw_mean"],
            row["roc_auc"]["raw_sd"],
            row["pr_auc"]["raw_mean"],
            row["brier"]["raw_mean"],
            row["brier"]["cal_mean"],
        )
    log.info("selected: %s (%s), ship_calibrated=%s", dec["selected_model"], dec["selected_config"], dec["ship_calibrated"])
    log.info("wrote %s + stage1_cv.md", results / "stage1_cv.json")


# --------------------------------------------------------------------------- #
# Stage 3 -- final fit on all 300 rows, one probability vector, top-10 ranking
# --------------------------------------------------------------------------- #
def load_stage1_decision(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the Stage 1 artifact and refuse to continue if it does not match.

    Stage 3 must not re-derive the selection rule under a different code path, so
    it consumes ``results/stage1_cv.json`` and validates both the pre-registration
    fingerprint and the rule version before fitting anything.
    """
    cfg = config or load_config()
    path = resolve(cfg["paths"]["results"]) / "stage1_cv.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.classical` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    dec = payload["decision"]
    if dec["fingerprint"] != fingerprint():
        raise RuntimeError(
            f"Stage 1 was produced under a different pre-registration ({dec['fingerprint']}); "
            "rerun `python -m src.classical`"
        )
    if dec["rule_version"] != PREREGISTERED_SELECTION["version"]:
        raise RuntimeError(f"Stage 1 rule version {dec['rule_version']} is not the frozen one")
    return dec


def finalise(config: dict[str, Any] | None = None, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit the pre-registered winner on all 300 training rows; score test once.

    Returns the ranked 100-row frame, the raw probability vector in test-sheet
    order, and a content hash so later stages can prove they consumed exactly this
    vector. Calibration ships only if Stage 1 decided ``ship_calibrated``.
    """
    cfg = config or load_config()
    dec = decision if decision is not None else load_stage1_decision(cfg)
    ds = load_dataset(config=cfg)
    X_train, y_train = prepare_xy(ds.train, labelled=True)
    X_test, _ = prepare_xy(ds.test, labelled=False)
    assert y_train is not None

    base = _pipeline(dec["selected_config"], cfg)
    model = base
    if dec["ship_calibrated"]:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=True)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    frame = pd.DataFrame(
        {"customer_id": ds.test[ID].astype(str).to_numpy(), "predicted_probability": prob}
    )
    frame = frame.sort_values(["predicted_probability", "customer_id"], ascending=[False, True])
    frame = frame.reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=int)
    frame["top10_flag"] = (frame["rank"] <= 10).astype(int)
    return {
        "frame": frame[list(STAGE1_COLUMNS)],
        "probabilities": prob,  # test-sheet order, NOT ranked
        "customer_ids": ds.test[ID].astype(str).to_numpy(),
        "decision": dec,
        "sha256_probabilities": sha256_text(",".join(f"{v:.12f}" for v in prob)),
        "n_train": int(len(y_train)),
        "n_positive": int(y_train.sum()),
    }


def candidate_table(finalised: dict[str, Any], stage2: pd.DataFrame) -> pd.DataFrame:
    """The Stage 4 candidate table: top 10 by probability, joined to operations data."""
    frame = finalised["frame"]
    top = frame[frame["top10_flag"] == 1]
    merged = top.merge(stage2, on=ID, how="left", validate="one_to_one")
    if len(merged) != 10 or merged[["contact_cost", "expected_deposit_value", "capacity_units"]].isna().any().any():
        raise RuntimeError("top-10 join to Stage2_Data is incomplete")
    return merged.reset_index(drop=True)


def bootstrap_test_probabilities(
    config: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    *,
    n_resamples: int | None = None,
) -> np.ndarray:
    """Refit the final model on ``n_resamples`` bootstrap draws of the 300 rows.

    Test *features* are never resampled or altered: only the training rows move.
    Returns an ``(n_resamples, 100)`` matrix in test-sheet order -- the uncertainty
    Stage 4 re-solves on and the error bar later stages propagate.
    """
    cfg = config or load_config()
    dec = decision if decision is not None else load_stage1_decision(cfg)
    ds = load_dataset(config=cfg)
    X_train, y_train = prepare_xy(ds.train, labelled=True)
    X_test, _ = prepare_xy(ds.test, labelled=False)
    assert y_train is not None
    n = len(y_train)
    n_resamples = int(n_resamples if n_resamples is not None else cfg["stage3"]["bootstrap_resamples"])
    seeds = child_seeds(int(cfg["seed"]), n_resamples, stream=3)

    base = _pipeline(dec["selected_config"], cfg)
    out = np.empty((n_resamples, len(X_test)), dtype=np.float32)
    for k, seed in enumerate(seeds):
        pick = np.random.default_rng(seed).integers(0, n, size=n)
        model = clone(base).fit(X_train.iloc[pick], y_train[pick])
        out[k] = model.predict_proba(X_test)[:, 1]
    return out


def main_finalise() -> None:
    log = get_logger("stage3")
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    dec = load_stage1_decision(cfg)
    result = finalise(cfg, dec)

    results = resolve(cfg["paths"]["results"])
    frame = result["frame"]
    frame.to_csv(results / "stage1_results.csv", index=False)

    boot = bootstrap_test_probabilities(cfg, dec)
    np.savez_compressed(results / "stage3_bootstrap_probs.npz", probabilities=boot)

    top10 = frame[frame["top10_flag"] == 1]
    top10_ids = top10["customer_id"].tolist()
    boot_rank = []
    for row in boot:
        order = np.lexsort((result["customer_ids"], -row))[:10]
        boot_rank.append([result["customer_ids"][i] for i in order])
    counts = {cid: sum(cid in picks for picks in boot_rank) for cid in result["customer_ids"]}

    write_json(
        results / "stage3_meta.json",
        {
            "stage": 3,
            "prereg_fingerprint": fingerprint(),
            "selected_model": dec["selected_model"],
            "selected_config": dec["selected_config"],
            "selected_params": dec["selected_params"],
            "ship_calibrated": dec["ship_calibrated"],
            "n_train": result["n_train"],
            "n_positive": result["n_positive"],
            "n_test": int(len(frame)),
            "sha256_probabilities": result["sha256_probabilities"],
            "probabilities": result["probabilities"].tolist(),
            "top10": top10.to_dict(orient="records"),
            "tie_break": "predicted_probability desc, then customer_id asc",
            "bootstrap_resamples": int(boot.shape[0]),
            "bootstrap_order": "test sheet order (same as results/stage3_bootstrap_probs.npz columns)",
            "bootstrap_probability_mean": boot.mean(axis=0).tolist(),
            "bootstrap_probability_sd": boot.std(axis=0, ddof=1).tolist(),
            "bootstrap_top10_by_id": {
                cid: {
                    "mean": float(boot[:, j].mean()),
                    "sd": float(boot[:, j].std(ddof=1)),
                    "in_top10_frequency": counts[cid] / boot.shape[0],
                }
                for cid in top10_ids
                for j in [int(np.flatnonzero(result["customer_ids"] == cid)[0])]
            },
            "top10_selection_frequency": {
                cid: counts[cid] / boot.shape[0] for cid in top10_ids + [c for c in result["customer_ids"] if c not in top10_ids]
            },
        },
    )

    for _, row in top10.iterrows():
        log.info("#%2d %s p=%.6f", int(row["rank"]), row["customer_id"], row["predicted_probability"])
    log.info("wrote stage1_results.csv, stage3_meta.json, stage3_bootstrap_probs.npz")


if __name__ == "__main__":
    if "--finalise" in sys.argv:
        main_finalise()
    else:
        main()
