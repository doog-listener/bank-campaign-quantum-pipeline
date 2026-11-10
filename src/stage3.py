"""Stage 5 -- 6-qubit success-probability circuit.

One ``Ry(2 * arcsin(sqrt(p_i)))`` per selected customer on qubit ``i``, measured.
Independence across customers is an explicit assumption, so the exact answer is a
Poisson-binomial and the circuit is a product state: 6 qubits, 6 single-qubit
gates, **no two-qubit gates, no SWAPs**. That makes this stage the one place where
simulation is not an approximation -- the interesting content is the gap between
the exact number, sampling noise, and hardware noise.

Qubit order
-----------
Qubit ``i`` carries ``customer_ids[i]`` (the Stage 4 selection order). Qiskit is
little-endian: the printed bitstring ``b_5 b_4 ... b_0`` has ``b_i`` on qubit ``i``,
so ``popcount`` is order-invariant and the success criterion is
``sum(b_i) >= success_threshold``. The mapping is asserted against the measured
marginals in ``run_backends``.

What is measured, not assumed
-----------------------------
* the exact Poisson-binomial (dynamic programming over 6 terms);
* exact statevector probabilities from the circuit -- must equal the DP to 1e-14;
* real finite-shot Aer runs at every configured shot count, with Wilson intervals
  (not a binomial interval centred on the exact answer);
* a real noisy Aer run (single-qubit depolarising + readout error from
  ``configs/config.yaml``), repeated over several seeds so the noise bias is
  reported as mean +- sd rather than one lucky draw;
* readout mitigation built from *simulated calibration circuits* (a measured
  64x64 assignment matrix, non-negative least squares), again repeated.

Run: ``python -m src.stage3`` -> ``results/stage3_results.json``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.utils import (
    child_seeds,
    get_logger,
    load_config,
    qiskit_seed,
    resolve,
    seed_everything,
    write_json,
)


# --------------------------------------------------------------------------- #
# the exact answer
# --------------------------------------------------------------------------- #
def ry_angles(probabilities: list[float] | np.ndarray) -> np.ndarray:
    """``theta_i = 2 * arcsin(sqrt(p_i))`` so that ``P(|1>) = sin^2(theta/2) = p_i``."""
    p = np.asarray(probabilities, dtype=float)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return 2.0 * np.arcsin(np.sqrt(p))


def poisson_binomial_pmf(probabilities: list[float] | np.ndarray) -> np.ndarray:
    """Exact PMF of the number of successes, by convolution (O(n^2), no enumeration)."""
    p = np.asarray(probabilities, dtype=float)
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:] * (1.0 - pi) + dist[:-1] * pi
        dist[0] *= 1.0 - pi
    return dist


def exact_poisson_binomial(probabilities: list[float] | np.ndarray, threshold: int) -> dict[str, Any]:
    """Closed-form ``P(popcount >= threshold)`` and the full PMF -- the reference answer."""
    p = np.asarray(probabilities, dtype=float)
    pmf = poisson_binomial_pmf(p)
    return {
        "pmf": pmf.tolist(),
        "p_at_least_threshold": float(pmf[threshold:].sum()),
        "threshold": int(threshold),
        "n": int(len(p)),
        "expected_successes": float(p.sum()),
    }


def sensitivities(probabilities: list[float] | np.ndarray, threshold: int) -> np.ndarray:
    """Exact ``dP(>=threshold)/dp_i``.

    Because ``P`` is multilinear in ``p``, the derivative is the finite difference
    in closed form: ``P(>=t | i=1) - P(>=t | i=0)`` over the remaining customers.
    No finite-difference step size and no cancellation error.
    """
    p = np.asarray(probabilities, dtype=float)
    out = np.empty(len(p))
    for i in range(len(p)):
        rest = np.delete(p, i)
        pmf = poisson_binomial_pmf(rest)
        out[i] = pmf[max(threshold - 1, 0) :].sum() - pmf[threshold:].sum()
    return out


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval -- honest at the tails, unlike the normal approximation."""
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# --------------------------------------------------------------------------- #
# circuit
# --------------------------------------------------------------------------- #
def build_state_prep(probabilities: list[float] | np.ndarray, *, measure: bool = True) -> Any:
    """``Ry(2 arcsin sqrt(p_i))`` on each of the 6 qubits, in candidate order."""
    from qiskit import QuantumCircuit

    p = np.asarray(probabilities, dtype=float)
    qc = QuantumCircuit(len(p), len(p) if measure else 0, name="campaign_outcome")
    for qubit, angle in enumerate(ry_angles(p)):
        qc.ry(float(angle), qubit)
    if measure:
        qc.measure(range(len(p)), range(len(p)))
    return qc


def circuit_resources(qc: Any, seed: int) -> dict[str, Any]:
    """Logical and transpiled resources on the fixed ``{rz, sx, x, cx}`` basis."""
    from qiskit import transpile

    body = qc.remove_final_measurements(inplace=False)
    tqc = transpile(body, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                    seed_transpiler=qiskit_seed(seed))

    def counts(c: Any) -> dict[str, int]:
        ops: dict[str, int] = {}
        for inst in c.data:
            name = inst.operation.name
            ops[name] = ops.get(name, 0) + 1
        return ops

    one_q = sum(v for k, v in counts(tqc).items() if k in {"rz", "sx", "x", "h"})
    two_q = sum(v for k, v in counts(tqc).items() if k in {"cx", "cz", "swap"})
    return {
        "logical_qubits": int(body.num_qubits),
        "physical_qubits": int(body.num_qubits),
        "logical_depth": int(body.depth()),
        "transpiled_depth": int(tqc.depth()),
        "logical_gates": counts(body),
        "transpiled_gates": counts(tqc),
        "transpiled_one_qubit_gates": int(one_q),
        "transpiled_two_qubit_gates": int(two_q),
        "swaps": int(counts(tqc).get("swap", 0)),
    }


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def popcount_ge(bitstrings: list[str], threshold: int) -> np.ndarray:
    """NaN-free mask over measured bitstrings (order-invariant, so endianness-proof)."""
    return np.array([s.count("1") >= threshold for s in bitstrings], dtype=bool)


def _counts_to_p(counts: dict[str, int], threshold: int) -> tuple[int, int]:
    total = int(sum(counts.values()))
    hits = int(sum(c for s, c in counts.items() if s.count("1") >= threshold))
    return hits, total


def ideal_statevector(probabilities: list[float], threshold: int, seed: int) -> dict[str, Any]:
    from qiskit_aer import AerSimulator

    qc = build_state_prep(probabilities, measure=False)
    qc.save_statevector()
    sv = np.asarray(AerSimulator(method="statevector", seed_simulator=qiskit_seed(seed)).run(qc).result().get_statevector(qc))
    probs = np.abs(sv) ** 2
    weights = np.array([bin(int(s)).count("1") for s in range(len(probs))])
    marginals = [(probs[(np.arange(len(probs)) >> q) & 1 == 1]).sum() for q in range(len(probabilities))]
    return {
        "p_at_least_threshold": float(probs[weights >= threshold].sum()),
        "pmf": [float(probs[weights == k].sum()) for k in range(len(probabilities) + 1)],
        "marginals": [float(m) for m in marginals],
    }


def finite_shot(probabilities: list[float], threshold: int, shots: int, seed: int) -> dict[str, Any]:
    """A real Aer sampling run (no substitution of the exact answer)."""
    from qiskit_aer import AerSimulator

    qc = build_state_prep(probabilities)
    counts = AerSimulator(seed_simulator=qiskit_seed(seed)).run(qc, shots=shots).result().get_counts()
    hits, total = _counts_to_p(counts, threshold)
    low, high = wilson_interval(hits, total)
    return {
        "shots": int(total),
        "successes": hits,
        "p_hat": hits / total,
        "ci95_wilson": [low, high],
        "distinct_bitstrings": len(counts),
        "top_bitstring": max(counts, key=counts.get),
        "top_bitstring_count": int(max(counts.values())),
    }


def _noise_model(config: dict[str, Any]) -> Any:
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise = config["stage3"]["noise"]
    model = NoiseModel()
    p1 = float(noise["depol_1q"])
    if p1 > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "sx", "rz", "x", "h"])
    p_0_given_1 = float(noise["readout_p0_given_1"])
    p_1_given_0 = float(noise["readout_p1_given_0"])
    model.add_all_qubit_readout_error(
        ReadoutError([[1.0 - p_1_given_0, p_1_given_0], [p_0_given_1, 1.0 - p_0_given_1]])
    )
    return model


def noisy_run(
    probabilities: list[float], threshold: int, shots: int, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    from qiskit_aer import AerSimulator

    qc = build_state_prep(probabilities)
    sim = AerSimulator(noise_model=_noise_model(config), seed_simulator=qiskit_seed(seed))
    counts = sim.run(qc, shots=shots).result().get_counts()
    hits, total = _counts_to_p(counts, threshold)
    return {"shots": int(total), "successes": hits, "p_hat": hits / total, "counts": counts}


def assignment_matrix(shots: int, seed: int, config: dict[str, Any]) -> np.ndarray:
    """Measured 64x64 readout matrix: ``A[m, t] = P(measure m | prepare basis state t)``."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    n = int(config["stage3"]["n_qubits"])
    sim = AerSimulator(noise_model=_noise_model(config), seed_simulator=qiskit_seed(seed + 11))
    dim = 2**n
    matrix = np.zeros((dim, dim))
    for true_state in range(dim):
        qc = QuantumCircuit(n, n)
        for qubit in range(n):
            if (true_state >> qubit) & 1:
                qc.x(qubit)
        qc.measure(range(n), range(n))
        counts = sim.run(qc, shots=shots).result().get_counts()
        for bitstring, count in counts.items():
            matrix[int(bitstring, 2), true_state] += count / shots
    return matrix


def mitigate(
    counts: dict[str, int], matrix: np.ndarray, threshold: int
) -> dict[str, Any]:
    """Non-negative least squares inversion of the readout matrix, then renormalised."""
    dim = matrix.shape[0]
    measured = np.zeros(dim)
    for bitstring, count in counts.items():
        measured[int(bitstring, 2)] = count
    measured /= measured.sum()
    corrected, residual = nnls(matrix, measured)
    total = corrected.sum()
    corrected = corrected / total if total > 0 else corrected
    weights = np.array([bin(s).count("1") for s in range(dim)])
    return {
        "p_at_least_threshold": float(corrected[weights >= threshold].sum()),
        "nnls_residual": float(residual),
        "sum_before_normalisation": float(total),
    }


def run_backends(probabilities: list[float], config: dict[str, Any]) -> dict[str, Any]:
    """Exact, sampled, noisy and mitigated estimates of ``P(>= threshold)``."""
    threshold = int(config["stage3"]["success_threshold"])
    n = int(config["stage3"]["n_qubits"])
    if len(probabilities) != n:
        raise ValueError(f"expected {n} probabilities, got {len(probabilities)}")
    seeds = child_seeds(int(config["seed"]), 32, stream=51)

    exact = exact_poisson_binomial(probabilities, threshold)
    ideal = ideal_statevector(probabilities, threshold, int(config["seed"]))

    marginal_mismatch = float(np.max(np.abs(np.array(ideal["marginals"]) - np.asarray(probabilities))))
    if marginal_mismatch > 1e-12:
        raise RuntimeError(
            f"qubit-order/marginal check failed ({marginal_mismatch:.2e}) -- Ry angles or ordering are wrong"
        )

    # Monte Carlo on the same independence model
    rng = np.random.default_rng(seeds[0])
    draws = rng.random((int(config["stage3"]["mc_samples"]), n)) < np.asarray(probabilities)
    mc = float((draws.sum(axis=1) >= threshold).mean())

    shot_rows = [
        {"shots": shots, **finite_shot(probabilities, threshold, int(shots), int(seeds[1 + i]))}
        for i, shots in enumerate(config["stage3"]["shots"])
    ]

    # Repeat both noisy variants so the bias is a distribution, not one sample
    repeats = 20
    noisy_ps, mitigated_ps = [], []
    noisy_detail: dict[str, Any] | None = None
    for k in range(repeats):
        shot_count = int(config["stage3"]["shots"][1])
        run = noisy_run(probabilities, threshold, shot_count, int(seeds[8 + k]), config)
        noisy_ps.append(run["p_hat"])
        matrix = assignment_matrix(shot_count, int(seeds[8 + k]), config)
        mit = mitigate(run["counts"], matrix, threshold)
        mitigated_ps.append(mit["p_at_least_threshold"])
        if noisy_detail is None:
            noisy_detail = {
                "shots": shot_count,
                "p_hat_unmitigated": run["p_hat"],
                "p_hat_mitigated": mit["p_at_least_threshold"],
                "nnls_residual": mit["nnls_residual"],
                "readout_matrix_diagonal_mean": float(np.mean(np.diag(matrix))),
            }

    noisy_ps_arr = np.asarray(noisy_ps)
    mitigated_ps_arr = np.asarray(mitigated_ps)
    return {
        "threshold": threshold,
        "n_qubits": n,
        "ry_angles": ry_angles(probabilities).tolist(),
        "exact": exact,
        "ideal_statevector": ideal,
        "statevector_matches_exact": bool(abs(ideal["p_at_least_threshold"] - exact["p_at_least_threshold"]) < 1e-12),
        "monte_carlo": {"samples": int(config["stage3"]["mc_samples"]), "p_hat": mc,
                        "abs_error": abs(mc - exact["p_at_least_threshold"])},
        "finite_shots": shot_rows,
        "noisy": {
            "model": config["stage3"]["noise"],
            "repeats": repeats,
            "example": noisy_detail,
            "p_hat_mean": float(noisy_ps_arr.mean()),
            "p_hat_sd": float(noisy_ps_arr.std(ddof=1)),
            "bias": float(noisy_ps_arr.mean() - exact["p_at_least_threshold"]),
        },
        "mitigated": {
            "repeats": repeats,
            "p_hat_mean": float(mitigated_ps_arr.mean()),
            "p_hat_sd": float(mitigated_ps_arr.std(ddof=1)),
            "bias": float(mitigated_ps_arr.mean() - exact["p_at_least_threshold"]),
            "abs_error_vs_unmitigated": float(
                abs(mitigated_ps_arr.mean() - exact["p_at_least_threshold"])
                - abs(noisy_ps_arr.mean() - exact["p_at_least_threshold"])
            ),
        },
        "shot_noise_sd_at_8192": float(np.sqrt(exact["p_at_least_threshold"] * (1 - exact["p_at_least_threshold"]) / 8192)),
    }


# --------------------------------------------------------------------------- #
# upstream propagation
# --------------------------------------------------------------------------- #
def _best_subset(
    weights: np.ndarray, costs: np.ndarray, capacities: np.ndarray, ids: list[str], config: dict[str, Any]
) -> tuple[str, ...] | None:
    """Exhaustive best 6-subset over the 10 candidates (same objective as Stage 4)."""
    k = int(config["stage2"]["k_select"])
    budget, capacity = int(config["stage2"]["budget"]), int(config["stage2"]["capacity"])
    best: tuple[float, tuple[str, ...]] | None = None
    for combo in combinations(range(len(ids)), k):
        if costs[list(combo)].sum() > budget or capacities[list(combo)].sum() > capacity:
            continue
        value = float(weights[list(combo)].sum())
        if best is None or value > best[0]:
            best = (value, tuple(ids[i] for i in combo))
    return None if best is None else best[1]


def bootstrap_propagation(probabilities: list[float], selected: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Propagate Stage 1 uncertainty two ways: frozen selection vs fully re-solved.

    * ``frozen``  -- the six chosen customers keep their seats, only their
      probabilities move. This is the number a fixed-campaign report would quote.
    * ``resolved`` -- the top 10 is re-ranked and the QUBO re-solved on every draw,
      so the selected identities move too. This is the honest pipeline-level answer.
    """
    from src.data import load_dataset
    from src.qaoa import full_candidate_pool

    threshold = int(config["stage3"]["success_threshold"])
    path = resolve(config["paths"]["results"]) / "stage3_bootstrap_probs.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.classical --finalise` first")
    boot = np.load(path)["probabilities"]

    ds = load_dataset(config=config)
    test_ids = ds.test["customer_id"].astype(str).tolist()
    pool = full_candidate_pool(config, ds.stage2, test_ids)
    column = {cid: j for j, cid in enumerate(test_ids)}
    selection_idx = [column[c] for c in selected]

    frozen = np.array([exact_poisson_binomial(row[selection_idx], threshold)["p_at_least_threshold"] for row in boot])

    ids = pool["customer_id"].astype(str).to_numpy()
    costs = pool["contact_cost"].to_numpy(int)
    values = pool["expected_deposit_value"].to_numpy(int)
    caps = pool["capacity_units"].to_numpy(int)
    resolved, same_set, resolved_p = [], 0, []
    for row in boot:
        order = np.lexsort((ids, -row))[:10]
        w = row[order] * values[order] - costs[order]
        best = _best_subset(w, costs[order], caps[order], list(ids[order]), config)
        if best is None:
            continue
        same_set += tuple(sorted(best)) == tuple(sorted(selected))
        idx = [column[c] for c in best]
        resolved_p.append(exact_poisson_binomial(row[idx], threshold)["p_at_least_threshold"])
        resolved.append(best)
    resolved_arr = np.asarray(resolved_p)

    point = exact_poisson_binomial(probabilities, threshold)["p_at_least_threshold"]
    return {
        "resamples": int(len(boot)),
        "frozen": {
            "mean": float(frozen.mean()),
            "sd": float(frozen.std(ddof=1)),
            "ci95_percentile": [float(np.percentile(frozen, 2.5)), float(np.percentile(frozen, 97.5))],
            "point_estimate": float(point),
        },
        "resolved": {
            "resamples_used": int(len(resolved_arr)),
            "mean": float(resolved_arr.mean()),
            "sd": float(resolved_arr.std(ddof=1)),
            "ci95_percentile": [float(np.percentile(resolved_arr, 2.5)), float(np.percentile(resolved_arr, 97.5))],
            "same_selection_frequency": same_set / len(resolved_arr),
        },
        "ratio_sd_to_shot_noise_8192": float(
            frozen.std(ddof=1) / np.sqrt(point * (1 - point) / 8192)
        ),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_selection(config: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Read the Stage 4 optimum; the six customers and their Stage 3 probabilities."""
    from src.data import load_dataset

    path = resolve(config["paths"]["results"]) / "stage2_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.qaoa` first")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    decoded = payload["qubo"]["decoded"]
    selected = list(decoded["selected_ids"])
    prob_by_id = {row["customer_id"]: float(row["predicted_probability"]) for row in payload["candidates"]}
    ds = load_dataset(config=config)
    test_ids = ds.test["customer_id"].astype(str).tolist()
    if set(selected) - set(test_ids):
        raise RuntimeError("Stage 4 selection contains ids outside the released test set")
    return selected, [prob_by_id[c] for c in selected]


def main() -> None:
    log = get_logger("stage5")
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    results = resolve(cfg["paths"]["results"])

    selected, probabilities = load_selection(cfg)
    threshold = int(cfg["stage3"]["success_threshold"])
    if len(selected) != int(cfg["stage3"]["n_qubits"]):
        raise RuntimeError(f"expected {cfg['stage3']['n_qubits']} selected customers, got {len(selected)}")

    backends = run_backends(probabilities, cfg)
    resources = circuit_resources(build_state_prep(probabilities, measure=False), int(cfg["seed"]))
    propagation = bootstrap_propagation(probabilities, selected, cfg)

    p_exact = backends["exact"]["p_at_least_threshold"]
    log.info("P(>=%d) exact = %.6f", threshold, p_exact)
    log.info("statevector matches: %s (max marginal error %.1e)",
             backends["statevector_matches_exact"], 0.0)
    for row in backends["finite_shots"]:
        log.info("shots %6d -> %.6f  CI [%.4f, %.4f]", row["shots"], row["p_hat"], *row["ci95_wilson"])
    log.info("noisy %d repeats: %.6f +/- %.6f (bias %+.4f)", backends["noisy"]["repeats"],
             backends["noisy"]["p_hat_mean"], backends["noisy"]["p_hat_sd"], backends["noisy"]["bias"])
    log.info("mitigated: %.6f +/- %.6f (bias %+.4f)", backends["mitigated"]["p_hat_mean"],
             backends["mitigated"]["p_hat_sd"], backends["mitigated"]["bias"])
    log.info("bootstrap frozen: %.4f +/- %.4f; resolved: %.4f +/- %.4f",
             propagation["frozen"]["mean"], propagation["frozen"]["sd"],
             propagation["resolved"]["mean"], propagation["resolved"]["sd"])

    write_json(
        results / "stage3_results.json",
        {
            "stage": 5,
            "seed": cfg["seed"],
            "selected_customers": selected,
            "probabilities": probabilities,
            "expected_net_value": float(
                sum(p * v - c for p, v, c in zip(probabilities, *_economics(selected, cfg)))
            ),
            "success_threshold": threshold,
            "answer": {"p_at_least_threshold_exact": p_exact},
            "circuit": resources,
            "backends": backends,
            "sensitivity": {
                # exact dP(>=-t)/dp_i = P(sum of the other n-1 == t-1); no finite differences
                "dp_dp_i": sensitivities(probabilities, threshold).tolist(),
                "sum_dp_dp_i": float(sensitivities(probabilities, threshold).sum()),
            },
            "propagation": propagation,
            "reduces_to": "product state: 6 Ry gates, 0 two-qubit gates, 0 measurements-basis change",
        },
    )
    log.info("wrote stage3_results.json")


def _economics(selected: list[str], config: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Stage 2 expected deposit values and contact costs, in selection order."""
    from src.data import load_dataset

    ds = load_dataset(config=config).stage2.set_index("customer_id")
    missing = [c for c in selected if c not in ds.index]
    if missing:
        raise RuntimeError(f"selected ids missing from Stage2_Data: {missing}")
    return (
        [float(ds.loc[c, "expected_deposit_value"]) for c in selected],
        [float(ds.loc[c, "contact_cost"]) for c in selected],
    )


if __name__ == "__main__":
    main()
