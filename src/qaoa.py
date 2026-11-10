"""Stage 4 -- QAOA variants on the Stage 4 QUBO.

Two circuit families, both simulated exactly on the statevector:

(a) **standard QAOA** on all 20 register qubits (10 decision + 2x5 slack). The
    cardinality, budget and capacity rows are soft penalties, so the optimiser
    can and does spend amplitude on infeasible bitstrings -- that is the point of
    measuring a feasibility rate.
(b) **Dicke-state QAOA** on the 10 decision qubits: the initial state is the
    uniform superposition over all ``C(10,6)=210`` Hamming-weight-6 strings, and
    the mixer is the ring XY operator
    ``exp(-i beta sum_i (X_i X_{i+1} + Y_i Y_{i+1})/2)``, which preserves Hamming
    weight, so ``sum_i x_i = 6`` holds by construction and no slack register is
    needed for cardinality. Budget and capacity are *not* enforced by this mixer;
    they are verified inactive on all 210 weight-6 states for this instance
    (``src.qubo.brute_force`` diagnostics) and that is reported, not assumed.

Simulator
---------
The 20-qubit register is 2**20 amplitudes, so the phase separator is applied as
an exact diagonal multiply and the mixer as per-qubit / per-pair rotations -- the
same gate sequence the Qiskit circuits below contain, without per-gate overhead.
The equivalence is asserted, not assumed: ``cross_check_qiskit`` rebuilds each
family as a Qiskit circuit and compares outcome distributions.

Optimisation: COBYLA, seeded multi-start, ``(gamma, beta)`` shared across qubits
per layer. Nothing here reads test labels (there are none) or the Stage 1/2 CV
folds; it consumes only the Stage 3 probability vector.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.qubo import Instance, brute_force, is_feasible

DICKE_PREPARATION_NOTE = (
    "The uniform Dicke(6,10) start state is prepared with an arbitrary "
    "state-preparation primitive (QuantumCircuit.initialize on the 210 support "
    "states in Qiskit, exact vector in the numpy engine). Its cost is reported separately "
    "and is NOT part of the p-scaling depth; a hardware-realizable replacement is "
    "the linear-depth Dicke preparation of Bartschi & Eidenbenz (2022), which this "
    "project does not implement."
)


# --------------------------------------------------------------------------- #
# bit helpers
# --------------------------------------------------------------------------- #
def _bits(states: np.ndarray, n: int) -> np.ndarray:
    return ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)


def _apply_x_rotation(state: np.ndarray, n: int, angle: float) -> np.ndarray:
    """``exp(-i angle/2 sum_q X_q)`` = ``prod_q Ry(angle)`` on every qubit.

    Applied two qubits at a time as a 4x4 batched matmul: half the memory passes
    of the one-qubit loop for identical arithmetic (verified to 1e-15 in tests).
    """
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    ry = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    ry2 = np.kron(ry, ry)
    out = state
    q = 0
    while q + 1 < n:
        v = out.reshape(2 ** (n - q - 2), 4, 2**q)
        out = np.matmul(ry2, v).reshape(-1)
        q += 2
    if q < n:  # odd n: one qubit left over
        v = out.reshape(2 ** (n - q - 1), 2, 2**q)
        out = np.matmul(ry, v).reshape(-1)
    return out


def _xy_pairs(n: int, qi: int, qj: int) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays of the |01> and |10> subspaces for qubit pair (qi, qj)."""
    states = np.arange(2**n, dtype=np.uint32)
    bi = (states >> qi) & 1
    bj = (states >> qj) & 1
    return states[(bi == 0) & (bj == 1)], states[(bi == 1) & (bj == 0)]


def _apply_xy_pair(state: np.ndarray, idx: tuple[np.ndarray, np.ndarray], theta: float) -> np.ndarray:
    """``exp(-i theta (X_i X_j + Y_i Y_j) / 2)``: identity on |00>, |11>.

    On the ``{|01>, |10>}`` block ``(XX + YY) = 2 sigma_x``, so the block is
    ``exp(-i theta sigma_x) = cos(theta) I - i sin(theta) sigma_x`` -- the factor 2
    is why this is ``cos(theta)`` and not ``cos(theta/2)``. Getting that wrong is
    invisible on a uniform superposition (rotating two equal amplitudes changes
    only phases), which is exactly why the mixer test uses unequal amplitudes.
    """
    c, s = np.cos(theta), np.sin(theta)
    a_idx, b_idx = idx
    a, b = state[a_idx].copy(), state[b_idx].copy()
    state[a_idx] = c * a - 1j * s * b
    state[b_idx] = c * b - 1j * s * a
    return state


# --------------------------------------------------------------------------- #
# state evolution
# --------------------------------------------------------------------------- #
def standard_state(diag: np.ndarray, gammas: np.ndarray, betas: np.ndarray, n: int) -> np.ndarray:
    """|+>^n, then p layers of ``exp(-i gamma H) exp(-i beta sum X)``."""
    state = np.full(2**n, 1.0 / np.sqrt(2**n), dtype=np.complex128)
    d = diag - diag.min()  # constant shift is a global phase only
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * d)
        state = _apply_x_rotation(state, n, 2.0 * beta)
    return state


def dicke_state(weights: np.ndarray, gammas: np.ndarray, betas: np.ndarray, k: int) -> np.ndarray:
    """Uniform Dicke(k, n) start, then p layers of ``exp(-i gamma H_P)`` and ring XY."""
    n = len(weights)
    states = np.arange(2**n, dtype=np.uint32)
    support = states[_bits(states, n).sum(axis=1) == k]
    state = np.zeros(2**n, dtype=np.complex128)
    state[support] = 1.0 / np.sqrt(len(support))

    diag = -(_bits(states, n).astype(np.float64) @ weights)
    ring = [(q, (q + 1) % n) for q in range(n)]
    pairs = {pair: _xy_pairs(n, *pair) for pair in ring}
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * diag)
        for pair in ring:
            # rxx(2 beta) . ryy(2 beta) = exp(-i beta (XX + YY)) = _apply_xy_pair(.., 2 beta)
            state = _apply_xy_pair(state, pairs[pair], 2.0 * beta)
    return state


def energy(state: np.ndarray, diag: np.ndarray) -> float:
    return float(np.sum(np.abs(state) ** 2 * diag))


def marginalise(state: np.ndarray, n_x: int) -> np.ndarray:
    """``P(x)`` over the decision register, summing out the slack registers."""
    p = np.abs(state) ** 2
    p = p / p.sum()
    return p.reshape(2**n_x, -1).sum(axis=1)


# --------------------------------------------------------------------------- #
# Qiskit circuits (resource accounting + simulator cross-check)
# --------------------------------------------------------------------------- #
def standard_circuit(qubo: dict[str, Any], p: int, gammas: np.ndarray, betas: np.ndarray) -> Any:
    from qiskit import QuantumCircuit

    n = qubo["layout"].n_qubits
    h, J = qubo["ising"]["h"], qubo["ising"]["J"]
    qc = QuantumCircuit(n, name=f"qaoa_soft_p{p}")
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2.0 * gamma * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.rzz(2.0 * gamma * J[i, j], i, j)
        qc.rx(2.0 * beta, range(n))
    return qc


def dicke_circuit(weights: np.ndarray, k: int, p: int, gammas: np.ndarray, betas: np.ndarray) -> Any:
    from qiskit import QuantumCircuit

    n = len(weights)
    states = np.arange(2**n, dtype=np.uint32)
    vector = np.zeros(2**n, dtype=complex)
    support = states[_bits(states, n).sum(axis=1) == k]
    vector[support] = 1.0 / np.sqrt(len(support))

    qc = QuantumCircuit(n, name=f"dicke_xy_p{p}")
    # `initialize` (not StatePreparation) so Aer can compile the circuit for the
    # statevector cross-check; both prepare the same vector.
    qc.initialize(vector, range(n))
    ring = [(q, (q + 1) % n) for q in range(n)]
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            # H_P = -sum w_i x_i, x = (1-z)/2  =>  h_i = w_i/2, RZ(2 gamma h_i) = RZ(gamma w_i)
            if weights[i] != 0.0:
                qc.rz(gamma * float(weights[i]), i)
        for qi, qj in ring:
            qc.rxx(2.0 * beta, qi, qj)
            qc.ryy(2.0 * beta, qi, qj)
    return qc


def circuit_report(qc: Any, seed: int, *, strip: str | None = None) -> dict[str, Any]:
    """Logical + transpiled resources on the fixed ``{rz,sx,x,cx}`` chain basis."""
    from qiskit import transpile

    from src.utils import qiskit_seed

    body = qc.copy_empty_like()
    for inst in qc.data:
        if strip is not None and inst.operation.name == strip:
            continue
        body.append(inst)
    tqc = transpile(body, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                    seed_transpiler=qiskit_seed(seed))

    def counts(c: Any) -> dict[str, int]:
        ops: dict[str, int] = {}
        for inst in c.data:
            name = inst.operation.name
            ops[name] = ops.get(name, 0) + 1
        return ops

    return {
        "name": qc.name,
        "qubits": int(qc.num_qubits),
        "logical_depth": int(body.depth()),
        "transpiled_depth": int(tqc.depth()),
        "logical_gates": counts(body),
        "transpiled_gates": counts(tqc),
        "swaps": int(counts(tqc).get("swap", 0)),
        "stripped_gate": strip,
    }


def cross_check_qiskit(reference_probs: np.ndarray, qc: Any, seed: int) -> dict[str, float]:
    """Compare the numpy engine against Aer's exact statevector (phase-invariant)."""
    from qiskit_aer import AerSimulator

    from src.utils import qiskit_seed

    sim = AerSimulator(method="statevector", seed_simulator=qiskit_seed(seed))
    qc2 = qc.copy()
    qc2.save_statevector()
    sv = np.asarray(sim.run(qc2).result().get_statevector(qc2))
    probs = np.abs(sv) ** 2
    return {
        "max_abs_probability_difference": float(np.max(np.abs(probs - reference_probs))),
        "total_variation_distance": float(0.5 * np.abs(probs - reference_probs).sum()),
    }


# --------------------------------------------------------------------------- #
# optimisation
# --------------------------------------------------------------------------- #
def optimise(objective: Any, p: int, config: dict[str, Any], *, stream: int) -> dict[str, Any]:
    """Seeded multi-start COBYLA on ``(gamma_1..p, beta_1..p)``; keep the best."""
    from src.utils import child_seeds

    solver = config["stage4"]
    seeds = child_seeds(int(config["seed"]), int(solver["restarts"]), stream=stream)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x0 = np.concatenate([rng.uniform(0.0, np.pi, size=p), rng.uniform(0.0, np.pi / 2.0, size=p)])
        res = minimize(
            objective, x0, method="COBYLA",
            options={"maxiter": int(solver["maxiter"]), "rhobeg": 0.5, "tol": 1e-4},
        )
        history.append({"restart": k, "energy": float(res.fun), "nfev": int(res.nfev)})
        if best is None or float(res.fun) < best["energy"]:
            best = {"energy": float(res.fun), "params": [float(v) for v in res.x], "nfev": int(res.nfev)}
    assert best is not None
    return {**best, "restarts": history, "p": p}


def split_params(params: list[float], p: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(params[:p]), np.asarray(params[p:])


# --------------------------------------------------------------------------- #
# distribution summaries
# --------------------------------------------------------------------------- #
def feasible_mask(instance: Instance, config: dict[str, Any]) -> np.ndarray:
    n = instance.n
    mask = np.zeros(2**n, dtype=bool)
    for x in range(2**n):
        if is_feasible([i for i in range(n) if (x >> i) & 1], instance, config):
            mask[x] = True
    return mask


def state_values(instance: Instance) -> np.ndarray:
    """Expected net value of every decision-register state."""
    return _bits(np.arange(2**instance.n, dtype=np.uint32), instance.n).astype(float) @ instance.weights


def optimum_index(instance: Instance, config: dict[str, Any]) -> int:
    bf = brute_force(instance, config)
    return sum(1 << i for i in bf["optimum"]["indices"])


def summarise(probs_x: np.ndarray, instance: Instance, config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """P(optimal), feasibility rate, top decoded selections, plus a shot cross-check."""
    shots = int(config["stage4"]["shots"])
    n = instance.n
    bf = brute_force(instance, config)
    opt = optimum_index(instance, config)
    feas = feasible_mask(instance, config)
    values = state_values(instance)
    e_opt = float(values[opt])

    sampled = rng.choice(2**n, size=shots, p=probs_x / probs_x.sum())
    order = np.argsort(-probs_x)[:5]
    top = []
    for x in order:
        chosen = [i for i in range(n) if (int(x) >> i) & 1]
        top.append(
            {
                "bitstring": "".join("1" if (int(x) >> i) & 1 else "0" for i in range(n)),
                "probability": float(probs_x[x]),
                "selected_ids": [instance.customer_ids[i] for i in chosen],
                "feasible": bool(feas[x]),
                "expected_net_value": float(values[x]),
            }
        )
    mean_value = float(probs_x @ values)
    return {
        "optimal_bitstring": "".join("1" if (opt >> i) & 1 else "0" for i in range(n)),
        "optimal_expected_net_value": e_opt,
        "p_optimal": float(probs_x[opt]),
        "approximation_ratio": float(mean_value / e_opt) if e_opt else float("nan"),
        "mean_expected_net_value": mean_value,
        "feasibility_rate": float(probs_x[feas].sum()),
        "expected_value_over_feasible_only": float((probs_x[feas] @ values[feas]) / probs_x[feas].sum())
        if probs_x[feas].sum() > 0
        else float("nan"),
        "shots": shots,
        "sampled_optimal_counts": int((sampled == opt).sum()),
        "sampled_feasible_fraction": float(feas[sampled].mean()),
        "top_bitstrings": top,
        "brute_force_optimum_value": float(bf["optimum"]["expected_net_value"]),
    }


# --------------------------------------------------------------------------- #
# Stage 4 driver
# --------------------------------------------------------------------------- #
def full_candidate_pool(config: dict[str, Any], stage2: pd.DataFrame, test_ids: list[str]) -> pd.DataFrame:
    """All 100 test rows in *sheet* order, with Stage 3 probability + operations data.

    Sheet order matters: ``results/stage3_bootstrap_probs.npz`` stores its columns in
    the released test-sheet order (``src.classical.bootstrap_test_probabilities``),
    so any frame that gets multiplied by those rows has to be aligned to it. The
    written CSV is ranked, which is why the pool is reindexed through ``test_ids``
    here instead of trusting file order.
    """
    from pathlib import Path

    from src.utils import resolve

    path = Path(resolve(config["paths"]["results"])) / "stage1_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.classical --finalise` first")
    frame = pd.read_csv(path)
    if len(frame) != len(test_ids) or set(frame["customer_id"]) != set(test_ids):
        raise RuntimeError("stage1_results.csv does not cover the released test ids exactly once")
    merged = frame.merge(stage2, on="customer_id", how="left", validate="one_to_one")
    if merged[["contact_cost", "expected_deposit_value", "capacity_units"]].isna().any().any():
        raise RuntimeError("stage1_results.csv did not join 1:1 onto Stage2_Data")
    pool = merged.set_index("customer_id").loc[list(test_ids)].reset_index()
    assert pool["customer_id"].tolist() == list(test_ids)
    return pool


def robustness(pool: pd.DataFrame, config: dict[str, Any], *, primary_value: float, primary_set: tuple[str, ...]) -> dict[str, Any]:
    """Re-solve the whole Stage 4 problem (candidate set, then selection) on bootstrap draws."""
    from pathlib import Path

    from src.utils import resolve

    path = Path(resolve(config["paths"]["results"])) / "stage3_bootstrap_probs.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.classical --finalise` first")
    boot = np.load(path)["probabilities"]
    n_draws = min(int(config["stage4"]["robustness_resamples"]), len(boot))
    # Alignment guard. A per-customer tolerance would be wrong here: bootstrap
    # means shrink hard on the extreme predictions (C0081 is 0.718 point vs
    # 0.498 +/- 0.242 bootstrap mean), so the check has to be about *ordering*
    # agreement, not magnitude. Misaligned columns fall to ~0 correlation.
    point = pool["predicted_probability"].to_numpy(dtype=float)
    if boot.shape[1] != len(point):
        raise RuntimeError("bootstrap probability matrix has the wrong width for the pool")
    alignment = float(np.corrcoef(point, boot.mean(axis=0))[0, 1])
    if alignment < 0.9:
        raise RuntimeError(
            f"bootstrap matrix does not follow the candidate pool (corr={alignment:.3f})"
        )
    k = int(config["stage2"]["k_select"])
    budget, capacity = int(config["stage2"]["budget"]), int(config["stage2"]["capacity"])
    ids = pool["customer_id"].astype(str).to_numpy()
    cost = pool["contact_cost"].to_numpy(int)
    value = pool["expected_deposit_value"].to_numpy(int)
    cap = pool["capacity_units"].to_numpy(int)

    index_of = {cid: i for i, cid in enumerate(ids)}
    primary_idx = [index_of[cid] for cid in primary_set]

    same_set, regrets, values, infeasible, candidate_overlap = 0, [], [], 0, []
    for row in boot[:n_draws]:
        order = np.lexsort((ids, -row))[:10]  # top 10 by probability, ties by id
        assert len(set(order)) == 10
        w = row[order] * value[order] - cost[order]
        candidate_overlap.append(len(set(ids[order]) & set(primary_set)) / k)
        best: tuple[float, tuple[str, ...]] | None = None
        for combo in combinations(range(10), k):
            if cost[order][list(combo)].sum() > budget or cap[order][list(combo)].sum() > capacity:
                continue
            v = float(w[list(combo)].sum())
            if best is None or v > best[0]:
                best = (v, tuple(sorted(ids[order][list(combo)])))
        if best is None:
            infeasible += 1
            continue
        # Regret is measured on the *resampled* probabilities: how much value the
        # frozen primary selection loses when the model is re-estimated.
        primary_on_row = float(
            np.sum(row[primary_idx] * value[primary_idx] - cost[primary_idx])
        )
        values.append(best[0])
        regrets.append(best[0] - primary_on_row)
        same_set += best[1] == tuple(sorted(primary_set))

    return {
        "resamples": len(values),
        "infeasible_resamples": infeasible,
        "same_optimum_fraction": same_set / len(values) if values else float("nan"),
        "candidate_set_overlap_mean": float(np.mean(candidate_overlap)),
        "candidate_set_identical_fraction": float(np.mean([o == 1.0 for o in candidate_overlap])),
        "regret_mean": float(np.mean(regrets)) if regrets else float("nan"),
        "regret_p95": float(np.percentile(regrets, 95)) if regrets else float("nan"),
        "resample_optimum_value_mean": float(np.mean(values)) if values else float("nan"),
        "resample_optimum_value_p05": float(np.percentile(values, 5)) if values else float("nan"),
        "resample_optimum_value_p95": float(np.percentile(values, 95)) if values else float("nan"),
        "pool_alignment_correlation": alignment,
    }


def main() -> None:
    from src.data import load_dataset
    from src.eval import fingerprint
    from src.qubo import build_instance, build_qubo, brute_force, decode
    from src.utils import get_logger, load_config, resolve, seed_everything, write_json

    log = get_logger("stage4")
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    results = resolve(cfg["paths"]["results"])
    ds = load_dataset(config=cfg)
    test_ids = ds.test["customer_id"].astype(str).tolist()
    pool = full_candidate_pool(cfg, ds.stage2, test_ids)
    top = pool[pool["rank"] <= 10].sort_values("rank")

    instance = build_instance(top)
    qubo = build_qubo(instance, cfg)
    lay, n, diag = qubo["layout"], qubo["layout"].n_qubits, qubo["diag"]
    bf = brute_force(instance, cfg)

    ground = decode(int(np.argmin(diag)), instance, lay)
    matches = set(ground["selected_ids"]) == set(bf["optimum"]["customer_ids"])
    if not matches:
        raise RuntimeError(
            "QUBO ground state is not the brute-force optimum -- the penalty set is "
            f"unverified ({ground['selected_ids']} vs {bf['optimum']['customer_ids']})"
        )
    log.info(
        "instance: %d candidates, %d qubits, %d subsets (%d feasible), brute-force optimum %.6f",
        instance.n, n, bf["n_combinations"], bf["n_feasible"], bf["optimum"]["expected_net_value"],
    )

    rng = np.random.default_rng(np.random.SeedSequence(int(cfg["seed"]), spawn_key=(41,)))
    w = instance.weights
    k = int(cfg["stage2"]["k_select"])
    p_values = [int(v) for v in cfg["stage4"]["p_values"]]
    rows: list[dict[str, Any]] = []

    def record(solver: str, p: int, opt: dict[str, Any], summary: dict[str, Any], rep: dict[str, Any]) -> None:
        gamma, beta = split_params(opt["params"], p)
        rows.append(
            {
                "solver": solver,
                "p": p,
                "qubits": rep["qubits"],
                "transpiled_depth": rep["transpiled_depth"],
                "transpiled_cx": int(rep["transpiled_gates"].get("cx", 0)),
                "energy": opt["energy"],
                "p_optimal": summary["p_optimal"],
                "approximation_ratio": summary["approximation_ratio"],
                "feasibility_rate": summary["feasibility_rate"],
                "mean_expected_net_value": summary["mean_expected_net_value"],
                "most_probable_bitstring": summary["top_bitstrings"][0]["bitstring"],
                "most_probable_selection": " ".join(summary["top_bitstrings"][0]["selected_ids"]),
                "sampled_optimal_counts": summary["sampled_optimal_counts"],
                "nfev": opt["nfev"],
                "gammas": [round(float(v), 6) for v in gamma],
                "betas": [round(float(v), 6) for v in beta],
            }
        )
        log.info(
            "%-14s p=%d energy=%9.4f P(opt)=%.4f feas=%.3f alpha=%.4f depth=%d",
            solver, p, opt["energy"], summary["p_optimal"], summary["feasibility_rate"],
            summary["approximation_ratio"], rep["transpiled_depth"],
        )

    # (a) standard QAOA on the full 20-qubit register, soft penalties
    standard: dict[str, Any] = {}
    for p in p_values:
        def objective(x: np.ndarray, p: int = p) -> float:
            return energy(standard_state(diag, x[:p], x[p:], n), diag)

        opt = optimise(objective, p, cfg, stream=100 + p)
        g, b = split_params(opt["params"], p)
        state = standard_state(diag, g, b, n)
        summary = summarise(marginalise(state, lay.n_x), instance, cfg, rng)
        qc = standard_circuit(qubo, p, g, b)
        rep = circuit_report(qc, int(cfg["seed"]))
        standard[str(p)] = {"opt": opt, "summary": summary, "circuit": rep}
        record("standard_soft", p, opt, summary, rep)
        if p in set(int(v) for v in cfg["stage4"]["cross_check_p"]):
            standard[str(p)]["qiskit_cross_check"] = cross_check_qiskit(np.abs(state) ** 2, qc, int(cfg["seed"]))
            log.info("  cross-check vs Aer p=%d ok: %s", p, standard[str(p)]["qiskit_cross_check"])

    # (b) Dicke-state QAOA on the 10 decision qubits with the ring XY mixer
    states10 = np.arange(2**instance.n, dtype=np.uint32)
    bits10 = _bits(states10, instance.n)
    support = bits10.sum(axis=1) == k
    dicke_diag = -(bits10.astype(float) @ w)
    dicke: dict[str, Any] = {}
    for p in p_values:
        def objective_d(x: np.ndarray, p: int = p) -> float:
            return energy(dicke_state(w, x[:p], x[p:], k), dicke_diag)

        opt = optimise(objective_d, p, cfg, stream=200 + p)
        g, b = split_params(opt["params"], p)
        state = dicke_state(w, g, b, k)
        probs = np.abs(state) ** 2
        summary = summarise(probs, instance, cfg, rng)
        qc = dicke_circuit(w, k, p, g, b)
        rep = circuit_report(qc, int(cfg["seed"]), strip="initialize")
        dicke[str(p)] = {
            "opt": opt,
            "summary": summary,
            "circuit": rep,
            "support_within_constraint": bool(np.all(probs[~support] < 1e-12)),
        }
        record("dicke_xy", p, opt, summary, rep)
        if p in set(int(v) for v in cfg["stage4"]["cross_check_p"]):
            dicke[str(p)]["qiskit_cross_check"] = cross_check_qiskit(probs, qc, int(cfg["seed"]))
            log.info("  cross-check vs Aer p=%d ok: %s", p, dicke[str(p)]["qiskit_cross_check"])

    rob = robustness(pool, cfg, primary_value=bf["optimum"]["expected_net_value"],
                     primary_set=tuple(bf["optimum"]["customer_ids"]))
    log.info("bootstrap: same optimum in %.3f of %d re-solves; mean candidate overlap %.2f",
             rob["same_optimum_fraction"], rob["resamples"], rob["candidate_set_overlap_mean"])

    write_json(
        results / "stage2_results.json",
        {
            "stage": 4,
            "prereg_fingerprint": fingerprint(),
            "seed": cfg["seed"],
            "problem": {
                "k": k,
                "budget": int(cfg["stage2"]["budget"]),
                "capacity": int(cfg["stage2"]["capacity"]),
                "penalties": qubo["penalties"],
                "layout": lay.as_dict(),
                "n_nonzero_couplings": int((qubo["ising"]["J"] != 0).sum() // 2),
            },
            "candidates": top.to_dict(orient="records"),
            "brute_force": bf,
            "qubo": {
                "offset": qubo["offset"],
                "ground_state_energy": qubo["ground_state"]["energy"],
                "decoded": ground,
                "matches_brute_force": matches,
                "penalty_verification": {
                    "ground_state_cardinality_ok": ground["n_selected"] == k,
                    "ground_state_within_budget": ground["cost"] <= int(cfg["stage2"]["budget"]),
                    "ground_state_within_capacity": ground["capacity"] <= int(cfg["stage2"]["capacity"]),
                    "budget_slack_exact": ground["slack_b"] == int(cfg["stage2"]["budget"]) - ground["cost"],
                    "capacity_slack_exact": ground["slack_c"] == int(cfg["stage2"]["capacity"]) - ground["capacity"],
                },
            },
            "qaoa": {
                "standard_soft_penalty": standard,
                "dicke_xy": dicke,
                "dicke_preparation_note": DICKE_PREPARATION_NOTE,
            },
            "robustness": rob,
        },
    )
    pd.DataFrame(rows).to_csv(results / "stage2_results.csv", index=False)
    log.info("wrote stage2_results.json + stage2_results.csv")


if __name__ == "__main__":
    main()
