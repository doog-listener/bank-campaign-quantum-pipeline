"""Stage 0 data layer: load the release, assert key integrity, audit the data.

Key-integrity policy (replaces any id-mapping indirection): ``customer_id`` is
the only join key in the project. There is no id_map module and no positional
index arithmetic anywhere -- every stage joins on this string, so a silent
reorder of rows can never change a result.

Integrity guarantees asserted here and re-asserted in tests/:
  * 300 train / 100 test / 400 Stage2 rows, ids unique within each frame;
  * train and test id sets are disjoint;
  * every test id is present in Stage2_Data;
  * Stage2_Data joins 1:1 with train U test (no dups, no orphans).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    PROJECT_ROOT,
    get_logger,
    load_config,
    resolve,
    sha256_file,
    write_json,
    write_text,
)

ID: str = "customer_id"
TARGET: str = "y"
FORBIDDEN_COLUMNS: tuple[str, ...] = ("duration",)  # removed by the organisers

# Physically/domain plausible ranges for the numeric fields we model.
RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "balance": (-20_000, 500_000),
    "day": (1, 31),
    "campaign": (1, 100),
    "pdays": (-1, 1000),
    "previous": (0, 100),
    "contact_cost": (0, 25),
    "expected_deposit_value": (0, 100_000),
    "capacity_units": (0, 22),
}


class KeyIntegrityError(RuntimeError):
    """Raised when the id contract between sheets is broken."""


@dataclass(frozen=True)
class Dataset:
    """Immutable container for the released workbook.

    ``test`` deliberately has no target column -- Stage 1-3 never see test labels
    (none are released) and never fit anything on test features.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    stage2: pd.DataFrame
    constraints: pd.DataFrame
    source: Path
    sha256: str

    @property
    def train_ids(self) -> pd.Series:
        return self.train[ID]

    @property
    def test_ids(self) -> pd.Series:
        return self.test[ID]


def load_dataset(
    workbook: Path | str | None = None,
    config: dict[str, Any] | None = None,
    *,
    verify: bool = True,
) -> Dataset:
    """Read every sheet in one pass and (optionally) verify the id contract."""
    cfg = config or load_config()
    path = resolve(workbook if workbook is not None else cfg["data"]["workbook"])
    if not path.exists():
        raise FileNotFoundError(f"workbook not found: {path}")

    sheets = cfg["data"]["sheets"]
    frames = pd.read_excel(path, sheet_name=list(sheets.values()))
    ds = Dataset(
        train=frames[sheets["train"]].copy(),
        test=frames[sheets["test"]].copy(),
        stage2=frames[sheets["stage2"]].copy(),
        constraints=frames[sheets["constraints"]].copy(),
        source=path,
        sha256=sha256_file(path),
    )
    if verify:
        assert_key_integrity(ds, expected=cfg["data"]["expected_rows"])
    return ds


def assert_key_integrity(ds: Dataset, *, expected: dict[str, int] | None = None) -> None:
    """Assert the four-part key contract; raise with *all* violations listed."""
    expected = expected or {"train": 300, "test": 100, "stage2": 400}
    problems: list[str] = []

    for name, frame, want in (
        ("train", ds.train, expected["train"]),
        ("test", ds.test, expected["test"]),
        ("stage2", ds.stage2, expected["stage2"]),
    ):
        if ID not in frame.columns:
            problems.append(f"{name}: missing key column '{ID}'")
            continue
        if len(frame) != want:
            problems.append(f"{name}: expected {want} rows, found {len(frame)}")
        if frame[ID].isna().any():
            problems.append(f"{name}: {int(frame[ID].isna().sum())} null {ID} values")
        if not frame[ID].is_unique:
            dupes = frame.loc[frame[ID].duplicated(), ID].head(5).tolist()
            problems.append(f"{name}: duplicate {ID} e.g. {dupes}")
        if not pd.api.types.is_string_dtype(frame[ID]) and frame[ID].dtype != object:
            problems.append(f"{name}: {ID} is not string-typed ({frame[ID].dtype})")

    if problems:  # cannot compare id sets safely below
        raise KeyIntegrityError(" | ".join(problems))

    train_ids, test_ids, s2_ids = set(ds.train[ID]), set(ds.test[ID]), set(ds.stage2[ID])

    overlap = train_ids & test_ids
    if overlap:
        problems.append(f"train/test overlap: {len(overlap)} ids e.g. {sorted(overlap)[:5]}")

    missing_from_stage2 = test_ids - s2_ids
    if missing_from_stage2:
        problems.append(f"test ids absent from Stage2_Data: {len(missing_from_stage2)} e.g. {sorted(missing_from_stage2)[:5]}")

    orphans = s2_ids - (train_ids | test_ids)
    if orphans:
        problems.append(f"Stage2_Data ids not in train U test: {len(orphans)} e.g. {sorted(orphans)[:5]}")

    union = train_ids | test_ids
    if len(union) != len(ds.stage2):
        problems.append(
            f"train U test has {len(union)} ids vs {len(ds.stage2)} Stage2 rows -> not a 1:1 join"
        )

    if problems:
        raise KeyIntegrityError(" | ".join(problems))


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _numeric_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            out[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 4),
                "n_unique": int(s.nunique()),
            }
    return out


def _leakage_checks(ds: Dataset) -> dict[str, Any]:
    """Structural leakage audit. All checks must be clean before any fitting."""
    checks: dict[str, Any] = {}

    checks["duration_absent_everywhere"] = {
        sheet: (FORBIDDEN_COLUMNS[0] not in frame.columns)
        for sheet, frame in (("train", ds.train), ("test", ds.test), ("stage2", ds.stage2))
    }

    checks["target_only_in_train"] = {
        "train_has_y": TARGET in ds.train.columns,
        "test_has_y": TARGET in ds.test.columns,
        "stage2_has_y": TARGET in ds.stage2.columns,
    }

    # Identical feature rows in train and test would leak labels via duplication.
    feature_cols = [c for c in ds.test.columns if c != ID]
    shared = [c for c in feature_cols if c in ds.train.columns]
    train_keys = ds.train[shared].astype(str).agg("|".join, axis=1)
    test_keys = ds.test[shared].astype(str).agg("|".join, axis=1)
    dup = set(train_keys) & set(test_keys)
    checks["identical_feature_rows_train_test"] = {"count": len(dup), "columns_compared": len(shared)}

    # Single-valued columns carry no signal and usually indicate a broken export.
    checks["constant_columns"] = {
        sheet: [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
        for sheet, frame in (("train", ds.train), ("test", ds.test), ("stage2", ds.stage2))
    }

    # Categorical levels present in test but never seen in train: the in-fold
    # encoder must tolerate them (handle_unknown='ignore'), reported for honesty.
    cat_cols = [c for c in ds.train.columns if ds.train[c].dtype == object and c not in (ID, TARGET)]
    checks["test_levels_unseen_in_train"] = {
        c: sorted(set(ds.test[c].dropna()) - set(ds.train[c].dropna())) for c in cat_cols
    }
    return checks


def audit_dataset(ds: Dataset) -> dict[str, Any]:
    """Full Stage 0 audit: shapes, integrity, ranges, balance, leakage."""
    numeric_cols = [
        "age", "balance", "day", "campaign", "pdays", "previous",
        "contact_cost", "expected_deposit_value", "capacity_units",
    ]

    y = ds.train[TARGET]
    positives = int((y == "yes").sum())
    negatives = int((y == "no").sum())

    range_violations: dict[str, list[str]] = {}
    for sheet, frame in (("train", ds.train), ("test", ds.test), ("stage2", ds.stage2)):
        bad: list[str] = []
        for col, (lo, hi) in RANGE_CHECKS.items():
            if col in frame.columns:
                s = frame[col]
                if s.min() < lo or s.max() > hi:
                    bad.append(f"{col} in [{s.min()}, {s.max()}] outside [{lo}, {hi}]")
        range_violations[sheet] = bad

    # Budget head-room: worst case cost if the K=6 priciest candidates are chosen.
    max_cost = int(ds.stage2["contact_cost"].max())
    k = int(ds.constraints.loc[ds.constraints["symbol"] == "K", "value"].iloc[0])
    budget = int(ds.constraints.loc[ds.constraints["symbol"] == "C", "value"].iloc[0])
    capacity = int(ds.constraints.loc[ds.constraints["symbol"] == "H", "value"].iloc[0])
    max_capacity = int(ds.stage2["capacity_units"].max())

    prev = ds.train["pdays"] == -1
    return {
        "workbook": {"file": ds.source.name, "sha256": ds.sha256},
        "shapes": {
            "train": list(ds.train.shape),
            "test": list(ds.test.shape),
            "stage2": list(ds.stage2.shape),
            "constraints": list(ds.constraints.shape),
        },
        "columns": {s: list(f.columns) for s, f in (("train", ds.train), ("test", ds.test), ("stage2", ds.stage2))},
        "key_integrity": {
            "status": "ok (asserted)",
            "train_ids_unique": bool(ds.train[ID].is_unique),
            "test_ids_unique": bool(ds.test[ID].is_unique),
            "test_id_count": int(ds.test[ID].nunique()),
            "train_test_disjoint": bool(set(ds.train[ID]).isdisjoint(set(ds.test[ID]))),
            "test_ids_in_stage2": int(len(set(ds.test[ID]) & set(ds.stage2[ID]))),
            "stage2_join": "1:1 (train U test == Stage2 ids)",
        },
        "duplicates": {
            "duplicate_rows_train": int(ds.train.duplicated().sum()),
            "duplicate_rows_test": int(ds.test.duplicated().sum()),
            "duplicate_rows_stage2": int(ds.stage2.duplicated().sum()),
            "duplicate_ids": 0,
        },
        "missing_values": {
            "train": int(ds.train.isna().sum().sum()),
            "test": int(ds.test.isna().sum().sum()),
            "stage2": int(ds.stage2.isna().sum().sum()),
        },
        "class_balance": {
            "positives": positives,
            "negatives": negatives,
            "positive_rate": round(positives / (positives + negatives), 6),
            "positive_rate_pct": round(100 * positives / (positives + negatives), 2),
        },
        "numeric_summary": {
            "train": _numeric_summary(ds.train, numeric_cols),
            "test": _numeric_summary(ds.test, numeric_cols),
            "stage2": _numeric_summary(ds.stage2, numeric_cols),
        },
        "categorical_cardinality": {
            c: int(ds.train[c].nunique())
            for c in ds.train.columns
            if ds.train[c].dtype == object and c not in (ID, TARGET)
        },
        "range_violations": range_violations,
        "derived_features": {
            "prev_contacted_rule": "prev_contacted = (pdays != -1)",
            "never_contacted_frac_train": round(float(prev.mean()), 4),
            "pdays_values_when_not_contacted": sorted(map(int, ds.train.loc[prev, "pdays"].unique())),
        },
        "stage2_feasibility": {
            "max_contact_cost": max_cost,
            "max_capacity_units": max_capacity,
            "k": k,
            "budget": budget,
            "capacity": capacity,
            "worst_case_cost_k_selected": max_cost * k,
            "budget_binds": bool(max_cost * k > budget),
            "capacity_binds": bool(max_capacity * k > capacity),
            "note": "budget is slack by construction; only capacity can be tight",
        },
        "leakage": _leakage_checks(ds),
        "environment": _environment(),
    }


def _freeze() -> list[str]:
    """Pinned `pip freeze`-equivalent from installed distribution metadata."""
    from importlib.metadata import distributions

    rows = {
        f"{d.metadata['Name']}=={d.version}"
        for d in distributions()
        if d.metadata["Name"]
    }
    return sorted(rows, key=str.lower)


def render_env(cfg: dict[str, Any]) -> str:
    """Provenance block: interpreter, freeze, simulator availability, seed/shots config."""
    import platform
    import sys

    lines = [
        "# Stage 0 environment and reproducibility record",
        f"interpreter: {sys.version.split()[0]} ({platform.platform()})",
        f"global seed: {cfg['seed']}",
        "",
        "## pip freeze (pinned)",
    ]
    lines += _freeze()
    lines += ["", "## quantum backend / shots configuration (used from Stage 2 on)"]
    try:
        from qiskit_aer import AerSimulator

        sims = AerSimulator().available_devices()
        lines.append(f"qiskit-aer available devices: {list(sims)}")
        lines.append("training backend: exact statevector (method='statevector', shots=None)")
    except Exception as exc:  # pragma: no cover - reported, never silently swallowed
        lines.append(f"qiskit-aer unavailable: {exc!r}")
    lines += [
        f"finite-shot checks: {cfg['stage3']['shots']}",
        f"monte-carlo reference: {cfg['stage3']['mc_samples']} samples",
        f"stage 2 penalties A/B/C: {cfg['stage2']['penalty_a']}/{cfg['stage2']['penalty_b']}/{cfg['stage2']['penalty_c']}",
        f"cv scheme: {cfg['cv']['n_splits']}x{cfg['cv']['n_repeats']} stratified, seed {cfg['cv']['shuffle_seed']}",
        "transpiler seed: derived via utils.qiskit_seed(seed)",
        "",
    ]
    return "\n".join(lines)


def _environment() -> dict[str, str]:
    import importlib

    out: dict[str, str] = {}
    for mod in ("sys", "numpy", "pandas", "scipy", "sklearn", "qiskit", "qiskit_aer", "openpyxl"):
        try:
            m = importlib.import_module(mod)
            if mod == "sys":
                out["python"] = m.version.split()[0]
            else:
                out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = "MISSING"
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_report(a: dict[str, Any]) -> str:
    """Human-readable Stage 0 audit, written to results/stage0_audit.md."""
    ci, bal, s2f, leak = a["key_integrity"], a["class_balance"], a["stage2_feasibility"], a["leakage"]
    lines = [
        "# Stage 0 audit",
        "",
        f"- workbook: `{a['workbook']['file']}`",
        f"- sha256: `{a['workbook']['sha256']}`",
        f"- rows: train {a['shapes']['train']}, test {a['shapes']['test']}, stage2 {a['shapes']['stage2']}",
        "",
        "## Key integrity (join on `customer_id` only, no id_map)",
        "",
        f"- test ids: {ci['test_id_count']} unique = {ci['test_ids_unique']}",
        f"- disjoint from train: {ci['train_test_disjoint']}",
        f"- present in Stage2_Data: {ci['test_ids_in_stage2']}/100",
        f"- Stage2 join: {ci['stage2_join']}",
        "",
        "## Class balance",
        "",
        f"- {bal['positives']} yes / {bal['negatives']} no = **{bal['positive_rate_pct']}%** positive (train only)",
        "",
        "## Data quality",
        "",
        f"- duplicate rows: train {a['duplicates']['duplicate_rows_train']}, "
        f"test {a['duplicates']['duplicate_rows_test']}, stage2 {a['duplicates']['duplicate_rows_stage2']}",
        f"- missing values: train {a['missing_values']['train']}, test {a['missing_values']['test']}, "
        f"stage2 {a['missing_values']['stage2']}",
        f"- range violations: {a['range_violations']}",
        "",
        "## Leakage checks",
        "",
        f"- `duration` absent everywhere: {leak['duration_absent_everywhere']}",
        f"- target column placement: {leak['target_only_in_train']}",
        f"- identical feature rows shared train/test: {leak['identical_feature_rows_train_test']['count']}",
        f"- constant columns: {leak['constant_columns']}",
        f"- unseen test levels: {leak['test_levels_unseen_in_train']}",
        "",
        "## Stage 2 feasibility envelope",
        "",
        f"- max cost {s2f['max_contact_cost']} x K {s2f['k']} = {s2f['worst_case_cost_k_selected']} "
        f"vs budget {s2f['budget']} -> budget_binds={s2f['budget_binds']}",
        f"- max capacity {s2f['max_capacity_units']} x K {s2f['k']} = {s2f['max_capacity_units'] * s2f['k']} "
        f"vs capacity {s2f['capacity']} -> capacity_binds={s2f['capacity_binds']}",
        "",
        f"## Environment\n\n- {a['environment']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """`python -m src.data` -> audit printed + written to results/."""
    log = get_logger("stage0")
    cfg = load_config()
    ds = load_dataset(config=cfg)
    assert_key_integrity(ds, expected=cfg["data"]["expected_rows"])
    report = audit_dataset(ds)

    results = resolve(cfg["paths"]["results"])
    write_json(results / "stage0_audit.json", report)
    write_text(results / "stage0_audit.md", render_report(report))
    write_text(results / "stage0_env.txt", render_env(cfg))

    ci, bal = report["key_integrity"], report["class_balance"]
    log.info("rows: train=%s test=%s stage2=%s", ds.train.shape, ds.test.shape, ds.stage2.shape)
    log.info("test ids: %d unique=%s disjoint=%s in_stage2=%d/100",
             ci["test_id_count"], ci["test_ids_unique"], ci["train_test_disjoint"], ci["test_ids_in_stage2"])
    log.info("class balance: %d/%d positive = %.2f%%", bal["positives"], bal["positives"] + bal["negatives"], bal["positive_rate_pct"])
    log.info("duplicates(train) = %d, missing(train) = %d", report["duplicates"]["duplicate_rows_train"], report["missing_values"]["train"])
    log.info("duration removed: %s", report["leakage"]["duration_absent_everywhere"])
    log.info("budget binds: %s | capacity binds: %s", report["stage2_feasibility"]["budget_binds"], report["stage2_feasibility"]["capacity_binds"])
    log.info("env: %s", report["environment"])
    log.info("wrote stage0_audit.{json,md} + stage0_env.txt to %s", results)


if __name__ == "__main__":
    main()
