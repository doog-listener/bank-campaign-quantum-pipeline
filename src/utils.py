"""Shared helpers: seeding, config, provenance fingerprints, deterministic IO.

Design rule for the whole project: no stochastic call may invent its own seed.
Everything derives from ``configs/config.yaml`` -> ``seed`` via ``rng(seed)``
or ``numpy.random.SeedSequence`` so a run is bit-reproducible from one integer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML config and return it as a plain dict."""
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    if "seed" not in cfg:
        raise KeyError(f"{cfg_path} is missing a top-level 'seed'")
    return cfg


def resolve(path_like: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed every global RNG we rely on (stdlib, numpy).

    Qiskit/Aer primitives are seeded per-backend in later stages; torch is not
    a dependency of this project.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int) -> np.random.Generator:
    """Independent numpy Generator; prefer this over global state."""
    return np.random.default_rng(seed)


def child_seeds(seed: int, n: int, *, stream: int = 0) -> list[int]:
    """Deterministic, non-overlapping integer seeds (e.g. one per CV repeat).

    ``stream`` namespaces independent consumers that share the run seed.
    """
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(stream,))
    return [int(s) for s in ss.generate_state(n)]


def qiskit_seed(seed: int) -> int:
    """Aer/transpiler seeds must be < 2**32; derive stably from the run seed."""
    return int(np.random.SeedSequence(entropy=seed, spawn_key=(99,)).generate_state(1)[0] % (2**32 - 1))


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def sha256_file(path: Path | str) -> str:
    """Content fingerprint of an input file (provenance / leakage audit trail)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# deterministic IO
# --------------------------------------------------------------------------- #
def write_json(path: Path | str, payload: Any) -> Path:
    """Write JSON with a trailing newline and no NaN surprises."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    return out


def write_text(path: Path | str, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return out


def get_logger(name: str = "bank_campaign") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# result row contracts (shared by stages so schemas cannot drift)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage1Row:
    """One row of results/stage1_results.csv -- the single probability vector."""

    customer_id: str
    predicted_probability: float
    rank: int
    top10_flag: int


STAGE1_COLUMNS: tuple[str, ...] = ("customer_id", "predicted_probability", "rank", "top10_flag")
