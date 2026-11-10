"""Stage 6 invariants: the traceability matrix and the release bundle.

The matrix exists so that a published number can be traced to the stage that
produced it. These tests check the matrix against the artefacts rather than against
itself, and check that the zip contains exactly the files the manifest claims.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.report import (
    BUNDLE_ROOT,
    TraceabilityError,
    build_bundle,
    build_traceability,
    iter_bundle_files,
    tie_report,
    traceability_markdown,
)
from src.stage3 import exact_poisson_binomial, ry_angles
from src.utils import PROJECT_ROOT, load_config, resolve


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def artefacts(cfg):
    results = resolve(cfg["paths"]["results"])
    required = ["stage1_results.csv", "stage1_cv.json", "stage3_meta.json",
                "stage2_results.json", "stage3_results.json"]
    missing = [f for f in required if not (results / f).exists()]
    if missing:
        pytest.skip(f"stage artefacts missing: {missing}")
    return results


@pytest.fixture(scope="module")
def table(cfg, artefacts):
    return build_traceability(cfg)


# --------------------------------------------------------------------------- #
# matrix
# --------------------------------------------------------------------------- #
def test_matrix_covers_the_ten_candidates_in_rank_order(table):
    assert len(table) == 10
    assert table["rank"].tolist() == list(range(1, 11))
    assert table["customer_id"].is_unique


def test_exactly_six_selected_with_qubits_in_order(table):
    chosen = table[table["selected"] == 1].sort_values("bit_index")
    assert len(chosen) == 6
    assert chosen["bit_index"].tolist() == list(range(6))
    assert chosen["qubit"].tolist() == [f"q{i}" for i in range(6)]
    assert table[table["selected"] == 0]["bit_index"].isna().all()


def test_angles_in_the_matrix_are_the_circuit_angles(table):
    chosen = table[table["selected"] == 1].sort_values("bit_index")
    assert np.allclose(chosen["ry_angle_rad"].to_numpy(), ry_angles(chosen["predicted_probability"].to_numpy()), atol=1e-12)


def test_matrix_probabilities_come_from_stage_one(table, artefacts):
    published = pd.read_csv(artefacts / "stage1_results.csv")
    merged = table.merge(published, on="customer_id", suffixes=("", "_s1"))
    assert np.allclose(merged["predicted_probability"], merged["predicted_probability_s1"])
    assert (merged["rank"] == merged["rank_s1"]).all()


def test_net_value_is_derived_not_transcribed(table):
    recomputed = table["predicted_probability"] * table["expected_deposit_value"] - table["contact_cost"]
    assert np.allclose(table["expected_net_value"], recomputed)
    chosen = table[table["selected"] == 1]
    assert chosen["expected_net_value"].sum() == pytest.approx(39.784885, abs=1e-6)


def test_published_answer_is_reproducible_from_the_matrix(table, cfg, artefacts):
    chosen = table[table["selected"] == 1].sort_values("bit_index")
    published = json.loads((artefacts / "stage3_results.json").read_text(encoding="utf-8"))
    exact = exact_poisson_binomial(chosen["predicted_probability"].to_numpy(), int(cfg["stage3"]["success_threshold"]))
    assert exact["p_at_least_threshold"] == pytest.approx(
        float(published["answer"]["p_at_least_threshold_exact"]), abs=1e-12
    )


def test_one_exact_tie_and_it_is_ordered_by_id(table):
    ties = tie_report(table)
    assert len(ties) == 1
    tie = ties[0]
    assert tie["customer_ids"] == ["C0349", "C0359"]
    assert tie["ordered_by_id"] is True
    assert tie["probability"] == pytest.approx(0.1691564, abs=1e-6)


def test_matrix_disagreement_is_fatal(cfg, monkeypatch):
    """If two stages disagree the builder must raise, not print a plausible table."""
    import src.report as report_module

    real = report_module.load_stage_artefacts

    def corrupt(config):
        art = real(config)
        art["stage5"] = dict(art["stage5"])
        art["stage5"]["selected_customers"] = list(reversed(art["stage5"]["selected_customers"]))
        return art

    monkeypatch.setattr(report_module, "load_stage_artefacts", corrupt)
    with pytest.raises(TraceabilityError, match="decoded order"):
        build_traceability(cfg)


def test_markdown_totals_match_the_table(table):
    text = traceability_markdown(table)
    chosen = table[table["selected"] == 1]
    assert f"contact cost used: {int(chosen['contact_cost'].sum())} / 25" in text
    assert f"capacity used: {int(chosen['capacity_units'].sum())} / 22" in text
    assert text.count("\n| ") >= 10


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def test_bundle_file_list_excludes_the_environment_and_caches():
    files = [p.relative_to(PROJECT_ROOT).as_posix() for p in iter_bundle_files(Path(PROJECT_ROOT))]
    assert all(".venv" not in f for f in files)
    assert all("__pycache__" not in f for f in files)
    assert "submission.zip" not in files
    assert {"README.md", "reproduce.sh", "src/qubo.py", "tests/conftest.py"} <= set(files)


def test_bundle_is_byte_identical_across_builds(cfg, tmp_path):
    first = build_bundle(cfg, zip_path=tmp_path / "a.zip", write_manifest=False)
    second = build_bundle(cfg, zip_path=tmp_path / "b.zip", write_manifest=False)
    assert first["sha256"] == second["sha256"]
    assert first["files"] == second["files"]


def test_bundle_manifest_matches_the_zip_contents(cfg, tmp_path):
    payload = build_bundle(cfg, zip_path=tmp_path / "c.zip", write_manifest=False)
    with zipfile.ZipFile(tmp_path / "c.zip") as archive:
        names = archive.namelist()
        assert len(names) == payload["files"]
        assert all(n.startswith(f"{BUNDLE_ROOT}/") for n in names)
        assert payload["manifest"][0]["path"] in {n[len(BUNDLE_ROOT) + 1:] for n in names}
        # every declared file is readable and non-empty
        for entry in payload["manifest"]:
            data = archive.read(f"{BUNDLE_ROOT}/{entry['path']}")
            assert len(data) == entry["bytes"] > 0


def test_bundle_never_contains_its_own_manifest(cfg, tmp_path):
    """The manifest records the zip's hash; bundling it makes every rebuild differ."""
    import copy as _copy

    config = _copy.deepcopy(cfg)
    config["paths"]["results"] = str(tmp_path)
    first = build_bundle(config, zip_path=tmp_path / "one.zip", write_manifest=True)
    second = build_bundle(config, zip_path=tmp_path / "two.zip", write_manifest=True)
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(tmp_path / "two.zip") as archive:
        assert not any(n.endswith("release_manifest.json") for n in archive.namelist())
