"""Stage 5 invariants: Ry convention, bit ordering, exact-vs-simulated, mitigation.

The claims this stage makes are small but easy to get subtly wrong (angle factor,
endianness, popcount, interval type), so each one is pinned here. The exact answer
and the ideal statevector are required to agree to 1e-12 -- if they do not, either
the angle formula or the bit mapping is wrong.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.stage3 import (
    _best_subset,
    build_state_prep,
    circuit_resources,
    exact_poisson_binomial,
    ideal_statevector,
    mitigate,
    poisson_binomial_pmf,
    popcount_ge,
    run_backends,
    ry_angles,
    sensitivities,
    wilson_interval,
)
from src.utils import load_config, rng


@pytest.fixture(scope="module")
def cfg():
    return load_config()


P6 = [0.718054, 0.703851, 0.654671, 0.208464, 0.203974, 0.186305]


# --------------------------------------------------------------------------- #
# angle convention
# --------------------------------------------------------------------------- #
def test_ry_angle_gives_the_requested_probability():
    p = np.array([0.0, 0.01, 0.186305, 0.5, 0.718054, 1.0])
    theta = ry_angles(p)
    assert np.allclose(np.sin(theta / 2) ** 2, p)


def test_ry_angles_are_bounded_and_monotone():
    p = np.linspace(0, 1, 11)
    theta = ry_angles(p)
    assert np.all(theta >= 0) and np.all(theta <= np.pi + 1e-15)
    assert np.all(np.diff(theta) > 0)


def test_ry_rejects_probabilities_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ry_angles([0.5, 1.2])


# --------------------------------------------------------------------------- #
# exact distribution
# --------------------------------------------------------------------------- #
def test_pmf_matches_enumeration_over_all_bitstrings():
    p = rng(3).uniform(0.05, 0.95, 6)
    pmf = poisson_binomial_pmf(p)
    brute = np.zeros(7)
    for bits in product([0, 1], repeat=6):
        prob = 1.0
        for bi, pi in zip(bits, p):
            prob *= pi if bi else 1.0 - pi
        brute[sum(bits)] += prob
    assert np.allclose(pmf, brute, atol=1e-14)
    assert pmf.sum() == pytest.approx(1.0)


def test_pmf_edges():
    assert np.allclose(poisson_binomial_pmf([0.0, 0.0]), [1.0, 0.0, 0.0])
    assert np.allclose(poisson_binomial_pmf([1.0, 1.0]), [0.0, 0.0, 1.0])
    assert poisson_binomial_pmf([]).tolist() == [1.0]


def test_threshold_edges():
    assert exact_poisson_binomial(P6, 0)["p_at_least_threshold"] == pytest.approx(1.0)
    assert exact_poisson_binomial(P6, 7)["p_at_least_threshold"] == pytest.approx(0.0, abs=1e-15)
    assert exact_poisson_binomial(P6, 6)["pmf"][6] == pytest.approx(0.002621, abs=1e-6)


def test_exact_answer_for_the_published_selection():
    # P6 here is rounded to 6 dp; the artefact uses full precision (0.5676553417)
    out = exact_poisson_binomial(P6, 3)
    assert out["p_at_least_threshold"] == pytest.approx(0.567655, abs=1e-6)
    assert out["expected_successes"] == pytest.approx(sum(P6))
    # the tail must be the sum of the last four PMF entries, by construction
    assert out["p_at_least_threshold"] == pytest.approx(sum(out["pmf"][3:]))


def test_pmf_is_not_symmetric_when_probabilities_are_skewed():
    """A Gaussian/binomial approximation would be wrong here; the PMF must be skewed."""
    pmf = poisson_binomial_pmf(P6)
    assert pmf[3] > pmf[5]
    assert not np.isclose(pmf[1], pmf[5])


# --------------------------------------------------------------------------- #
# sensitivities
# --------------------------------------------------------------------------- #
def test_sensitivity_equals_the_conditional_probability_identity():
    """dP(>=t)/dp_i = P(sum of the other n-1 == t-1), exactly."""
    p = np.array(P6)
    sens = sensitivities(p, 3)
    for i in range(6):
        rest = np.delete(p, i)
        assert sens[i] == pytest.approx(poisson_binomial_pmf(rest)[2], abs=1e-15)


def test_sensitivities_match_central_differences():
    p = np.array(P6)
    sens = sensitivities(p, 3)
    eps = 1e-6
    for i in range(6):
        up, down = p.copy(), p.copy()
        up[i] += eps
        down[i] -= eps
        fd = (poisson_binomial_pmf(up)[3:].sum() - poisson_binomial_pmf(down)[3:].sum()) / (2 * eps)
        assert sens[i] == pytest.approx(fd, abs=1e-6)


def test_sensitivities_are_positive_and_bounded():
    sens = sensitivities(np.array(P6), 3)
    assert np.all(sens > 0)
    assert np.all(sens < 1.0)


# --------------------------------------------------------------------------- #
# circuit
# --------------------------------------------------------------------------- #
def test_circuit_is_the_declared_product_state():
    qc = build_state_prep(P6)
    assert qc.num_qubits == 6
    assert qc.count_ops()["ry"] == 6
    assert sum(v for k, v in qc.count_ops().items() if k in {"cx", "cz", "swap"}) == 0


def test_statevector_marginals_are_the_probabilities():
    """Endianness/order check: qubit i must carry customer i."""
    qc = build_state_prep(P6, measure=False)
    probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    states = np.arange(2**6)
    for q, pi in enumerate(P6):
        marginal = probs[((states >> q) & 1) == 1].sum()
        assert marginal == pytest.approx(pi, abs=1e-12)


def test_ideal_statevector_matches_the_exact_distribution(cfg):
    ideal = ideal_statevector(P6, 3, int(cfg["seed"]))
    exact = exact_poisson_binomial(P6, 3)
    assert ideal["p_at_least_threshold"] == pytest.approx(exact["p_at_least_threshold"], abs=1e-12)
    assert np.allclose(ideal["pmf"], exact["pmf"], atol=1e-12)


def test_popcount_is_order_invariant():
    assert popcount_ge(["111000", "000111", "111111", "100000"], 3).tolist() == [True, True, True, False]


def test_resource_accounting_has_no_two_qubit_gates(cfg):
    report = circuit_resources(build_state_prep(P6, measure=False), int(cfg["seed"]))
    assert report["logical_qubits"] == 6
    assert report["physical_qubits"] == 6
    assert report["logical_depth"] == 1
    assert report["transpiled_two_qubit_gates"] == 0
    assert report["swaps"] == 0
    assert report["transpiled_depth"] >= 1


# --------------------------------------------------------------------------- #
# sampling and readout
# --------------------------------------------------------------------------- #
def test_wilson_interval_is_sane():
    low, high = wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0.0) and high < 0.05
    low, high = wilson_interval(100, 100)
    assert high == pytest.approx(1.0) and low > 0.95


def test_backend_table_is_internally_consistent(cfg):
    """Every estimate must sit near the exact answer at its own shot count."""
    out = run_backends(P6, cfg)
    exact = out["exact"]["p_at_least_threshold"]
    assert out["statevector_matches_exact"] is True
    assert abs(out["monte_carlo"]["p_hat"] - exact) < 0.005
    for row in out["finite_shots"]:
        # 5 sigma of binomial noise, plus a margin for the interval being Wilson
        sigma = np.sqrt(exact * (1 - exact) / row["shots"])
        assert abs(row["p_hat"] - exact) < 5 * sigma + 1e-12
        low, high = row["ci95_wilson"]
        assert 0.0 <= low <= row["p_hat"] <= high <= 1.0
    assert out["mitigated"]["repeats"] == out["noisy"]["repeats"] >= 5


def test_mitigation_reduces_a_known_readout_bias():
    """Synthetic 3-qubit readout matrix: invert it and recover the true distribution."""
    n = 3
    single = np.array([[0.9, 0.1], [0.15, 0.85]])
    matrix = single
    for _ in range(n - 1):
        matrix = np.kron(matrix, single)

    # truth: two single-success states, so P(popcount >= 2) is exactly 0
    true_p = np.zeros(2**n)
    true_p[0b001] = 0.6
    true_p[0b100] = 0.4
    measured = matrix @ true_p
    counts = {format(s, f"0{n}b"): int(round(1e6 * p)) for s, p in enumerate(measured)}

    out = mitigate(counts, matrix, threshold=2)
    raw = sum(c for s, c in counts.items() if s.count("1") >= 2) / sum(counts.values())
    assert raw > 0.01  # readout error invents multi-success outcomes
    assert out["p_at_least_threshold"] < 0.1 * raw  # and the inversion removes most of it


def test_mitigation_on_an_identity_matrix_is_a_no_op():
    n = 3
    matrix = np.eye(2**n)
    counts = {"001": 600000, "100": 400000}  # both popcount 1
    out = mitigate(counts, matrix, threshold=2)
    assert abs(out["p_at_least_threshold"]) < 1e-9
    assert mitigate(counts, matrix, 1)["p_at_least_threshold"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# cross-stage checks
# --------------------------------------------------------------------------- #
def test_best_subset_helper_agrees_with_stage4_brute_force(cfg):
    from src.data import load_dataset
    from src.qubo import build_instance, brute_force

    ds = load_dataset(config=cfg)
    import json
    from pathlib import Path

    from src.utils import resolve

    payload = json.loads((Path(resolve(cfg["paths"]["results"])) / "stage2_results.json").read_text(encoding="utf-8"))
    import pandas as pd

    table = pd.DataFrame(payload["candidates"])
    inst = build_instance(table)
    bf = brute_force(inst, cfg)
    ids = list(inst.customer_ids)
    best = _best_subset(inst.weights, inst.costs, inst.capacities, ids, cfg)
    assert best is not None
    assert set(best) == set(bf["optimum"]["customer_ids"])
