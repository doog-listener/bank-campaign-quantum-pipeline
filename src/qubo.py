"""Stage 4 -- QUBO construction, decoding and brute-force ground truth.

Objective (maximise) for the 10 Stage 1 candidates::

    w_i = p_i * expected_deposit_value_i - contact_cost_i
    minimise  H(x, s) = -sum_i w_i x_i
                        + A     (sum_i x_i - K)^2
                        + B     (sum_i c_i x_i + s_b - C)^2
                        + C_pen (sum_i h_i x_i + s_c - H)^2

with ``s_b``, ``s_c`` binary-expanded slacks so the inequalities become equalities
(``s_b`` in ``[0, 31]``, ``s_c`` in ``[0, 31]``; values above the bound are
reachable but strictly increase the residual, which is asserted in tests).

Vectors
-------
``U`` is upper triangular, ``H = offset + sum_{i<=j} U[i, j] x_i x_j``. The
diagonal form is materialised once over all ``2**20`` states and is what the
simulator and the brute force both consume, so the algebra cannot drift from the
thing being optimised.

Contract
--------
* penalties come from ``configs/config.yaml`` (frozen at Stage 0) and are then
  *verified*: the QUBO ground state must be feasible and must equal the
  brute-force optimum over all ``C(10, 6) = 210`` 6-subsets;
* decoding maps bitstrings back to ``customer_id``;
* the final pick is the best *feasible decoded* bitstring, cross-checked against
  brute force -- never a raw argmin of an unverified Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# instance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Instance:
    """The 10-candidate sub-problem handed over by Stage 3."""

    customer_ids: tuple[str, ...]
    probabilities: np.ndarray  # p_i, candidate order
    costs: np.ndarray  # contact_cost_i        (budget weight)
    values: np.ndarray  # expected_deposit_value_i
    capacities: np.ndarray  # capacity_units_i      (capacity weight)

    @property
    def n(self) -> int:
        return len(self.customer_ids)

    @property
    def weights(self) -> np.ndarray:
        """``w_i = p_i * v_i - c_i`` -- the Stage 4 objective coefficients."""
        return self.probabilities * self.values - self.costs

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "customer_id": list(self.customer_ids),
                "predicted_probability": self.probabilities,
                "contact_cost": self.costs.astype(int),
                "expected_deposit_value": self.values.astype(int),
                "capacity_units": self.capacities.astype(int),
                "expected_net_value": self.weights,
            }
        )


def net_values(probabilities: Any, stage2: pd.DataFrame) -> np.ndarray:
    """``w_i = p_i * expected_deposit_value_i - contact_cost_i``, in candidate order.

    ``stage2`` must already be in the same row order as ``probabilities``.
    """
    p = np.asarray(probabilities, dtype=float)
    value = stage2["expected_deposit_value"].to_numpy(dtype=float)
    cost = stage2["contact_cost"].to_numpy(dtype=float)
    if not (len(p) == len(value) == len(cost)):
        raise ValueError("probabilities and Stage2_Data rows are not aligned")
    return p * value - cost


def build_instance(
    candidates: pd.DataFrame, *, customer_id: str = "customer_id", probability: str = "predicted_probability"
) -> Instance:
    """Freeze one Stage 4 instance from a candidate table (order is preserved)."""
    required = {customer_id, probability, "contact_cost", "expected_deposit_value", "capacity_units"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table missing columns: {sorted(missing)}")
    return Instance(
        customer_ids=tuple(candidates[customer_id].astype(str)),
        probabilities=candidates[probability].to_numpy(dtype=float),
        costs=candidates["contact_cost"].to_numpy(dtype=int),
        values=candidates["expected_deposit_value"].to_numpy(dtype=int),
        capacities=candidates["capacity_units"].to_numpy(dtype=int),
    )


# --------------------------------------------------------------------------- #
# QUBO assembly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layout:
    """Register layout: ``x_0..x_{n-1}``, then budget slack bits, then capacity."""

    n_x: int
    slack_b_bits: int
    slack_c_bits: int

    @property
    def n_qubits(self) -> int:
        return self.n_x + self.slack_b_bits + self.slack_c_bits

    @property
    def b_bits(self) -> tuple[int, ...]:
        return tuple(range(self.n_x, self.n_x + self.slack_b_bits))

    @property
    def c_bits(self) -> tuple[int, ...]:
        return tuple(range(self.n_x + self.slack_b_bits, self.n_qubits))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_x": self.n_x,
            "n_qubits": self.n_qubits,
            "slack_b_bits": self.slack_b_bits,
            "slack_c_bits": self.slack_c_bits,
            "slack_b_max": 2**self.slack_b_bits - 1,
            "slack_c_max": 2**self.slack_c_bits - 1,
            "qubit_order": [f"x{i}" for i in range(self.n_x)]
            + [f"sb{k}" for k in range(self.slack_b_bits)]
            + [f"sc{k}" for k in range(self.slack_c_bits)],
        }


def _bits_needed(value: int) -> int:
    """Smallest ``b`` with ``2**b - 1 >= value`` (at least 1)."""
    return max(1, int(np.ceil(np.log2(value + 1))))


def layout_for(instance: Instance, config: dict[str, Any]) -> Layout:
    """Register layout for this instance; ``n_x`` is the candidate count, not a constant.

    ``config['stage2']['n_candidates']`` is the problem size the run is *supposed*
    to have; the driver asserts they agree, so a toy instance in tests cannot
    silently inherit the production layout.
    """
    cfg = config["stage2"]
    return Layout(
        n_x=int(instance.n),
        slack_b_bits=_bits_needed(int(cfg["budget"])),
        slack_c_bits=_bits_needed(int(cfg["capacity"])),
    )


def _quadratic_residual(linear: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients of ``(sum_i a_i b_i - target)**2`` for *binary* ``b``.

    Returns ``(linear_coeffs, pair_coeffs_upper, offset)`` where the value is
    ``offset + sum_i lin_i b_i + sum_{i<j} pair_ij b_i b_j``. Uses ``b_i**2 = b_i``
    with ``a_i`` folded in as ``a_i**2 - 2*target*a_i`` -- the ``a_i**2`` matters
    for the slack registers, where the binary weight is ``2**k`` and not 1.
    """
    k = len(linear)
    lin = linear**2 - 2.0 * target * linear
    pair = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            pair[i, j] = 2.0 * linear[i] * linear[j]
    return lin, pair, float(target**2)


def build_qubo(instance: Instance, config: dict[str, Any]) -> dict[str, Any]:
    """Assemble the binary-expanded QUBO.

    Returns the upper-triangular coefficient matrix ``U`` (layout above), the
    constant ``offset``, the exact diagonal over all ``2**n_qubits`` states, its
    minimiser, and the Ising ``(h, J)`` form used to build circuits.
    """
    cfg = config["stage2"]
    lay = layout_for(instance, config)
    n = lay.n_qubits
    A = float(cfg["penalty_a"])
    B = float(cfg["penalty_b"])
    Cpen = float(cfg["penalty_c"])
    K = int(cfg["k_select"])
    budget = int(cfg["budget"])
    capacity = int(cfg["capacity"])

    U = np.zeros((n, n), dtype=float)
    offset = 0.0

    # -sum w_i x_i
    U[np.arange(instance.n), np.arange(instance.n)] -= instance.weights

    # A (sum x - K)^2 over the x register only
    lin, pair, off = _quadratic_residual(np.ones(instance.n), K)
    U[: instance.n, : instance.n] += A * pair
    U[np.arange(instance.n), np.arange(instance.n)] += A * lin
    offset += A * off

    # B (sum c_i x_i + s_b - C)^2 : x register, slack bits, x-slack cross terms
    a_b = np.concatenate([instance.costs.astype(float), 2.0 ** np.arange(lay.slack_b_bits)])
    idx_b = np.arange(instance.n + lay.slack_b_bits)
    lin_b, pair_b, off_b = _quadratic_residual(a_b, budget)
    U[np.ix_(idx_b, idx_b)] += B * pair_b
    U[idx_b, idx_b] += B * lin_b
    offset += B * off_b

    # C_pen (sum h_i x_i + s_c - H)^2
    a_c = np.concatenate([instance.capacities.astype(float), 2.0 ** np.arange(lay.slack_c_bits)])
    idx_c = np.concatenate([np.arange(instance.n), np.array(lay.c_bits)])
    lin_c, pair_c, off_c = _quadratic_residual(a_c, capacity)
    U[np.ix_(idx_c, idx_c)] += Cpen * pair_c
    U[idx_c, idx_c] += Cpen * lin_c
    offset += Cpen * off_c

    diag = qubo_diagonal(U, offset, n)
    best_idx = int(np.argmin(diag))

    h, J, ising_offset = to_ising(U, offset, n)
    return {
        "layout": lay,
        "U": U,
        "offset": float(offset),
        "diag": diag,
        "ground_state": {
            "bitstring": format(best_idx, f"0{n}b"),
            "energy": float(diag[best_idx]),
        },
        "ising": {"h": h, "J": J, "offset": float(ising_offset)},
        "penalties": {"A": A, "B": B, "C": Cpen, "K": K, "budget": budget, "capacity": capacity},
    }


def _popcount_matrix(n: int, chunk: int = 1 << 16) -> np.ndarray:
    """Binary matrix ``X`` with ``X[s, i] = bit i of s`` for every state ``s``."""
    states = np.arange(2**n, dtype=np.uint32)
    return ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float64)


def qubo_diagonal(U: np.ndarray, offset: float, n: int, chunk: int = 1 << 16) -> np.ndarray:
    """Exact energy of every one of the ``2**n`` basis states, in chunks.

    ``sum_{i<=j} U[i,j] x_i x_j == sum_i x_i (U x)_i`` because ``U`` is upper
    triangular, so one matrix product per chunk is enough.
    """
    out = np.empty(2**n, dtype=np.float64)
    X = _popcount_matrix(n, chunk)
    for start in range(0, 2**n, chunk):
        block = X[start : start + chunk]
        out[start : start + chunk] = np.einsum("ij,ij->i", block, block @ U)
    return out + offset


def evaluate(instance: Instance, config: dict[str, Any], x_bits: int, s_b: int, s_c: int, *, U: np.ndarray | None = None) -> float:
    """Direct (non-matrix) QUBO energy -- the reference the diagonal is checked against."""
    cfg = config["stage2"]
    x = np.array([(x_bits >> i) & 1 for i in range(instance.n)], dtype=float)
    w = instance.weights
    A, B, Cpen = float(cfg["penalty_a"]), float(cfg["penalty_b"]), float(cfg["penalty_c"])
    K, budget, capacity = int(cfg["k_select"]), int(cfg["budget"]), int(cfg["capacity"])
    cost = float(instance.costs @ x)
    cap = float(instance.capacities @ x)
    return (
        -float(w @ x)
        + A * (x.sum() - K) ** 2
        + B * (cost + s_b - budget) ** 2
        + Cpen * (cap + s_c - capacity) ** 2
    )


def to_ising(U: np.ndarray, offset: float, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """``x = (1 - z) / 2`` conversion: ``H = offset' + sum h_i z_i + sum_{i<j} J_ij z_i z_j``."""
    diag = np.diag(U).copy()
    # x_i = (1 - z_i)/2  =>  U_ii x_i contributes -U_ii/2 to h_i, +U_ii/2 to const
    h = -diag / 2.0
    J = np.zeros((n, n), dtype=float)
    const = offset + diag.sum() / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            q = U[i, j]
            if q == 0.0:
                continue
            J[i, j] = J[j, i] = q / 4.0
            h[i] -= q / 4.0
            h[j] -= q / 4.0
            const += q / 4.0
    return h, J, float(const)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode(state_index: int, instance: Instance, layout: Layout) -> dict[str, Any]:
    """Map a full-register basis state back to customers and resource usage."""
    x_bits = state_index & ((1 << layout.n_x) - 1)
    s_b = (state_index >> layout.n_x) & ((1 << layout.slack_b_bits) - 1)
    s_c = (state_index >> (layout.n_x + layout.slack_b_bits)) & ((1 << layout.slack_c_bits) - 1)
    chosen = [i for i in range(layout.n_x) if (x_bits >> i) & 1]
    cost = int(instance.costs[chosen].sum()) if chosen else 0
    cap = int(instance.capacities[chosen].sum()) if chosen else 0
    return {
        # index i of the printed string is x_i, i.e. left-to-right candidate order
        "bitstring": "".join("1" if (x_bits >> i) & 1 else "0" for i in range(layout.n_x)),
        "x_bits": int(x_bits),
        "selected_indices": chosen,
        "selected_ids": [instance.customer_ids[i] for i in chosen],
        "n_selected": len(chosen),
        "cost": cost,
        "capacity": cap,
        "expected_net_value": float(instance.weights[chosen].sum()) if chosen else 0.0,
        "slack_b": int(s_b),
        "slack_c": int(s_c),
    }


def is_feasible(selected_indices: list[int] | tuple[int, ...], instance: Instance, config: dict[str, Any]) -> bool:
    """Cardinality, budget and capacity -- the three declared constraints."""
    cfg = config["stage2"]
    if len(set(selected_indices)) != len(selected_indices):
        return False
    if len(selected_indices) != int(cfg["k_select"]):
        return False
    cost = sum(int(instance.costs[i]) for i in selected_indices)
    cap = sum(int(instance.capacities[i]) for i in selected_indices)
    return cost <= int(cfg["budget"]) and cap <= int(cfg["capacity"])


# --------------------------------------------------------------------------- #
# brute force
# --------------------------------------------------------------------------- #
def brute_force(instance: Instance, config: dict[str, Any]) -> dict[str, Any]:
    """Exhaustive ground truth over all feasible 6-subsets of the 10 candidates."""
    cfg = config["stage2"]
    k = int(cfg["k_select"])
    budget, capacity = int(cfg["budget"]), int(cfg["capacity"])
    weights = instance.weights

    rows: list[dict[str, Any]] = []
    for combo in combinations(range(instance.n), k):
        cost = int(instance.costs[list(combo)].sum())
        cap = int(instance.capacities[list(combo)].sum())
        rows.append(
            {
                "indices": combo,
                "customer_ids": [instance.customer_ids[i] for i in combo],
                "bitstring": "".join("1" if i in combo else "0" for i in range(instance.n)),
                "cost": cost,
                "capacity": cap,
                "expected_net_value": float(weights[list(combo)].sum()),
                "feasible": cost <= budget and cap <= capacity,
            }
        )

    feasible = [r for r in rows if r["feasible"]]
    feasible.sort(key=lambda r: (-r["expected_net_value"], r["customer_ids"]))
    if not feasible:
        raise RuntimeError(f"no feasible {k}-subset exists -- the problem is infeasible")

    best = feasible[0]
    runner = feasible[1] if len(feasible) > 1 else None
    all_costs = [r["cost"] for r in rows]
    all_caps = [r["capacity"] for r in rows]

    # Is the constraint set actually load-bearing? If every subset is feasible the
    # optimiser reduces to "sort by w and take the top K", which must be reported.
    binding = {
        "all_subsets_feasible": len(feasible) == len(rows),
        "max_cost_any_subset": int(max(all_costs)),
        "max_capacity_any_subset": int(max(all_caps)),
        "budget_slack_at_optimum": budget - best["cost"],
        "capacity_slack_at_optimum": capacity - best["capacity"],
        "budget_ever_binds": bool(max(all_costs) > budget),
        "capacity_ever_binds": bool(max(all_caps) > capacity),
        "min_budget_slack_any_subset": int(budget - max(all_costs)),
        "min_capacity_slack_any_subset": int(capacity - max(all_caps)),
    }
    by_weight = sorted(range(instance.n), key=lambda i: (-weights[i], instance.customer_ids[i]))[:k]
    binding["top_k_by_weight"] = [instance.customer_ids[i] for i in by_weight]
    binding["optimum_equals_top_k_by_weight"] = set(binding["top_k_by_weight"]) == set(best["customer_ids"])

    return {
        "n_combinations": len(rows),
        "n_feasible": len(feasible),
        "optimum": best,
        "runner_up": runner,
        "gap_to_runner_up": None if runner is None else best["expected_net_value"] - runner["expected_net_value"],
        "top_5_feasible": feasible[:5],
        "binding_diagnostics": binding,
        "n_candidates": instance.n,
        "k": k,
    }
