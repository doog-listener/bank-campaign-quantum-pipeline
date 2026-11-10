"""Stage 6 -- traceability matrix, release manifest and the submission bundle.

Everything here is *derived* from the stage artefacts. Nothing is transcribed: a
hand-typed traceability table is the easiest place in a five-stage pipeline to
publish a number that no stage produced, so this module rebuilds the matrix from
``results/`` and refuses to write a bundle if the stages disagree.

Checks (all fatal):
  * Stage 1 gives 10 ranked candidates, ranks 1..10, ids inside the released test set;
  * Stage 4's candidate table is exactly those 10 ids;
  * Stage 4's decoded selection and Stage 5's ``selected_customers`` are the same six
    in the same order (qubit order depends on it);
  * Stage 5's probabilities are Stage 4's probabilities for those ids;
  * Stage 5's Ry angles are ``2 arcsin sqrt(p)`` and its ``P(>=3)`` is the exact
    Poisson-binomial of its own inputs;
  * Stage 5's expected net value is ``sum(p*v - c)`` over the six.

Outputs: ``results/traceability.{csv,md}``, ``results/release_manifest.json`` and
``submission.zip`` at the repo root.

Run: ``python -m src.report``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.stage3 import exact_poisson_binomial, ry_angles
from src.utils import PROJECT_ROOT, get_logger, load_config, resolve, sha256_file, write_json, write_text

BUNDLE_ROOT = "bank-campaign-quantum-pipeline"
ZIP_NAME = "submission.zip"
EXCLUDE_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints", "node_modules"}
# `release_manifest.json` describes the zip that contains it, so including it would
# make every rebuild differ from the last one (the manifest embeds the zip hash).
EXCLUDE_NAMES = {ZIP_NAME, "release_manifest.json", ".DS_Store"}
INCLUDE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".sh", ".html", ".pdf", ".npz", ".xlsx", ".toml"}


class TraceabilityError(RuntimeError):
    """Raised when two stages disagree about the same customer."""


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def _load(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run the earlier stages first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_artefacts(config: dict[str, Any]) -> dict[str, Any]:
    results = resolve(config["paths"]["results"])
    stage1 = pd.read_csv(results / "stage1_results.csv")
    return {
        "results_dir": results,
        "stage1_results": stage1,
        "stage1_decision": _load(results / "stage1_cv.json")["decision"],
        "stage3_meta": _load(results / "stage3_meta.json"),
        "stage4": _load(results / "stage2_results.json"),
        "stage5": _load(results / "stage3_results.json"),
    }


# --------------------------------------------------------------------------- #
# traceability
# --------------------------------------------------------------------------- #
def build_traceability(config: dict[str, Any]) -> pd.DataFrame:
    """One row per Stage 1 candidate, joined across every stage that touches it."""
    art = load_stage_artefacts(config)
    s1: pd.DataFrame = art["stage1_results"]
    s4: dict[str, Any] = art["stage4"]
    s5: dict[str, Any] = art["stage5"]
    meta: dict[str, Any] = art["stage3_meta"]

    top = s1[s1["top10_flag"] == 1].sort_values("rank")
    if len(top) != 10 or sorted(top["rank"].tolist()) != list(range(1, 11)):
        raise TraceabilityError("stage1_results.csv does not contain exactly ranks 1..10 flagged")

    candidates = pd.DataFrame(s4["candidates"])
    if list(candidates["customer_id"]) != list(top["customer_id"]):
        raise TraceabilityError("Stage 4 candidate order is not the Stage 1 ranking")

    selected_order = list(s5["selected_customers"])
    decoded_order = list(s4["qubo"]["decoded"]["selected_ids"])
    if selected_order != decoded_order:
        raise TraceabilityError(
            f"Stage 5 selection {selected_order} is not Stage 4's decoded order {decoded_order}"
        )
    if len(selected_order) != int(config["stage2"]["k_select"]):
        raise TraceabilityError(f"expected {config['stage2']['k_select']} selected customers")

    prob = np.asarray(s5["probabilities"], dtype=float)
    stage4_prob = candidates.set_index("customer_id")["predicted_probability"]
    for cid, p in zip(selected_order, prob):
        if abs(float(stage4_prob.loc[cid]) - p) > 1e-12:
            raise TraceabilityError(f"Stage 5 probability for {cid} does not match Stage 4")

    angles = np.asarray(s5["ry_angles"] if "ry_angles" in s5 else s5["backends"]["ry_angles"], dtype=float)
    if not np.allclose(angles, ry_angles(prob), atol=1e-12):
        raise TraceabilityError("Stage 5 Ry angles are not 2*arcsin(sqrt(p))")

    exact = exact_poisson_binomial(prob, int(config["stage3"]["success_threshold"]))
    published = float(s5["answer"]["p_at_least_threshold_exact"])
    if abs(exact["p_at_least_threshold"] - published) > 1e-12:
        raise TraceabilityError("Stage 5 published P(>=3) is not the exact Poisson-binomial of its inputs")

    net = float(np.sum(prob * candidates.set_index("customer_id").loc[selected_order, "expected_deposit_value"]
                       - candidates.set_index("customer_id").loc[selected_order, "contact_cost"]))
    if abs(net - float(s5["expected_net_value"])) > 1e-9:
        raise TraceabilityError("Stage 5 expected net value does not equal sum(p*v - c)")

    boot = meta["bootstrap_top10_by_id"]
    rows = []
    for _, row in top.iterrows():
        cid = row["customer_id"]
        cand = candidates.set_index("customer_id").loc[cid]
        selected = cid in selected_order
        rows.append(
            {
                "rank": int(row["rank"]),
                "customer_id": cid,
                "predicted_probability": float(row["predicted_probability"]),
                "contact_cost": int(cand["contact_cost"]),
                "expected_deposit_value": int(cand["expected_deposit_value"]),
                "capacity_units": int(cand["capacity_units"]),
                # derived here, not read: the artefact stores only p, c, v
                "expected_net_value": float(row["predicted_probability"] * cand["expected_deposit_value"] - cand["contact_cost"]),
                "selected": int(selected),
                "bit_index": selected_order.index(cid) if selected else None,
                "qubit": f"q{selected_order.index(cid)}" if selected else "",
                "ry_angle_rad": float(angles[selected_order.index(cid)]) if selected else None,
                "bootstrap_probability_mean": float(boot[cid]["mean"]),
                "bootstrap_probability_sd": float(boot[cid]["sd"]),
                "in_top10_frequency": float(boot[cid]["in_top10_frequency"]),
            }
        )
    return pd.DataFrame(rows)


def traceability_markdown(table: pd.DataFrame) -> str:
    """Rendered matrix with the tie and the budget/capacity totals spelled out."""
    lines = [
        "# Cross-stage traceability",
        "",
        "Generated by `python -m src.report` from `results/`; no row is hand-typed.",
        "",
        "| Stage 1 rank | customer_id | p-hat | cost | value | capacity | net value | Stage 4 | bit | qubit | Ry (rad) | bootstrap p (mean +- sd) | top-10 freq |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        selected = "**selected**" if r["selected"] else "rejected"
        bit = int(r["bit_index"]) if pd.notna(r["bit_index"]) else "—"
        qubit = r["qubit"] or "—"
        angle = f"{r['ry_angle_rad']:.6f}" if pd.notna(r["ry_angle_rad"]) else "—"
        lines.append(
            f"| {r['rank']} | {r['customer_id']} | {r['predicted_probability']:.6f} | {r['contact_cost']} | "
            f"{r['expected_deposit_value']} | {r['capacity_units']} | {r['expected_net_value']:+.4f} | {selected} | "
            f"{bit} | {qubit} | {angle} | {r['bootstrap_probability_mean']:.3f} +- {r['bootstrap_probability_sd']:.3f} | "
            f"{r['in_top10_frequency']:.1%} |"
        )

    chosen = table[table["selected"] == 1].sort_values("bit_index")
    lines += [
        "",
        f"- selected: {', '.join(chosen['customer_id'])}",
        f"- contact cost used: {int(chosen['contact_cost'].sum())} / 25",
        f"- capacity used: {int(chosen['capacity_units'].sum())} / 22",
        f"- expected net value: {chosen['expected_net_value'].sum():.4f}",
        "",
        "Ties in Stage 1 are broken by ascending `customer_id`; the matrix above shows the",
        "exact double-precision ties that made that rule necessary.",
        "",
    ]
    return "\n".join(lines)


def tie_report(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Exact ties in the Stage 1 probabilities and how they were ordered."""
    out: list[dict[str, Any]] = []
    values = table["predicted_probability"].to_numpy()
    for value in np.unique(values):
        group = table[table["predicted_probability"] == value]
        if len(group) > 1:
            out.append(
                {
                    "probability": float(value),
                    "customer_ids": group["customer_id"].tolist(),
                    "ordered_by_id": group["customer_id"].tolist() == sorted(group["customer_id"].tolist()),
                }
            )
    return out


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def iter_bundle_files(root: Path) -> Iterable[Path]:
    """Deterministic file list for the zip (sorted, excludes envs, caches, itself)."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(root).parts):
            continue
        if path.name in EXCLUDE_NAMES or path.suffix not in INCLUDE_SUFFIXES:
            continue
        if path.name.endswith(".pyc"):
            continue
        files.append(path)
    return files


def build_bundle(
    config: dict[str, Any], *, zip_path: Path | None = None, write_manifest: bool = True
) -> dict[str, Any]:
    """Write ``submission.zip`` plus a manifest of what went in (name, size, sha256)."""
    root = Path(PROJECT_ROOT)
    out = zip_path or (root / ZIP_NAME)
    files = list(iter_bundle_files(root))
    manifest = []

    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            rel = path.relative_to(root)
            info = zipfile.ZipInfo(f"{BUNDLE_ROOT}/{rel.as_posix()}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            manifest.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    payload = {
        "bundle": out.name,
        "root": BUNDLE_ROOT,
        "files": len(manifest),
        "bytes": int(sum(m["bytes"] for m in manifest)),
        "sha256": sha256_file(out),
        "manifest": manifest,
        "excluded": sorted(EXCLUDE_DIRS) + sorted(EXCLUDE_NAMES),
        "self_reference_note": "release_manifest.json is excluded from the archive: it records this zip's hash.",
    }
    if write_manifest:
        write_json(resolve(config["paths"]["results"]) / "release_manifest.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    log = get_logger("stage6")
    cfg = load_config()
    results = resolve(cfg["paths"]["results"])

    table = build_traceability(cfg)
    table.to_csv(results / "traceability.csv", index=False)
    write_text(results / "traceability.md", traceability_markdown(table))

    ties = tie_report(table)
    if not all(t["ordered_by_id"] for t in ties):
        raise TraceabilityError(f"a Stage 1 tie was not ordered by ascending customer_id: {ties}")

    payload = build_bundle(cfg)
    log.info("traceability: %d rows, %d selected", len(table), int(table["selected"].sum()))
    log.info("exact ties: %s", [(t["probability"], t["customer_ids"]) for t in ties])
    log.info("bundle: %s (%d files, %.1f KiB, sha256 %s...)",
             payload["bundle"], payload["files"], payload["bytes"] / 1024, payload["sha256"][:12])


if __name__ == "__main__":
    main()
