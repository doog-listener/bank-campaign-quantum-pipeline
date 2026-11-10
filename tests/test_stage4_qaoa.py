"""Stage 4 QAOA invariants: mixer algebra, constraint preservation, Aer agreement.

The numpy simulator is the engine the reported numbers come from, so the point of
these tests is to pin it to Qiskit for small instances, and to pin the two
structural claims: the XY mixer preserves Hamming weight (cardinality holds by
construction) and the soft-penalty variant does not.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.qaoa import (
    _apply_x_rotation,
    _bits,
    dicke_circuit,
    dicke_state,
    energy,
    feasible_mask,
    marginalise,
    optimise,
    split_params,
    standard_circuit,
    standard_state,
    state_values,
    summarise,
)
from src.qubo import Instance, build_qubo, brute_force
from src.utils import load_config, rng


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def toy(n: int = 6, k: int = 3, budget: int = 8, capacity: int = 9, seed: int = 11) -> tuple[Instance, dict]:
    g = rng(seed)
    inst = Instance(
        customer_ids=tuple(f"T{i:02d}" for i in range(n)),
        probabilities=g.uniform(0.05, 0.9, n).round(4),
        costs=g.integers(1, 3, n),
        values=g.integers(8, 13, n),
        capacities=g.integers(1, 5, n),
    )
    config = copy.deepcopy(load_config())
    config["stage2"].update({"n_candidates": n, "k_select": k, "budget": budget, "capacity": capacity})
    config["stage4"].update({"p_values": [1], "restarts": 1, "maxiter": 40, "shots": 512})
    return inst, config


# --------------------------------------------------------------------------- #
# mixer algebra
# --------------------------------------------------------------------------- #
def test_paired_mixer_equals_single_qubit_mixer():
    for n in (1, 3, 4, 7):
        state = rng(n).normal(size=2**n) + 1j * rng(n + 1).normal(size=2**n)
        paired = _apply_x_rotation(state.copy(), n, 0.37)
        c, s = np.cos(0.37 / 2), np.sin(0.37 / 2)
        sequential = state.copy()
        for q in range(n):
            low, high = 1 << q, 1 << (n - 1 - q)
            v = sequential.reshape(high, 2, low)
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = c * a0 - 1j * s * a1
            v[:, 1, :] = c * a1 - 1j * s * a0
        assert np.allclose(paired, sequential, atol=1e-14)


def test_zero_angle_is_the_identity():
    state = rng(5).normal(size=2**4) + 0j
    assert np.allclose(_apply_x_rotation(state.copy(), 4, 0.0), state)


def test_standard_state_starts_uniform():
    diag = rng(0).random(2**4)
    state = standard_state(diag, np.empty(0), np.empty(0), 4)
    assert np.allclose(np.abs(state), 1.0 / np.sqrt(2**4))


def test_standard_state_is_normalised_at_every_depth():
    inst, config = toy(n=4, k=2)
    qubo = build_qubo(inst, config)
    for p in (1, 2):
        g = np.full(p, 0.4)
        b = np.full(p, 0.3)
        state = standard_state(qubo["diag"], g, b, qubo["layout"].n_qubits)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)


# --------------------------------------------------------------------------- #
# Dicke / XY structure
# --------------------------------------------------------------------------- #
def test_dicke_start_is_uniform_on_the_weight_k_shell():
    w = rng(3).normal(size=6)
    state = dicke_state(w, np.empty(0), np.empty(0), 3)
    support = np.abs(state) > 0
    weights = [bin(int(s)).count("1") for s in np.flatnonzero(support)]
    assert set(weights) == {3}
    assert np.allclose(np.abs(state[support]), 1.0 / np.sqrt(20))


def test_xy_ring_mixer_matches_qiskit_on_unequal_amplitudes():
    """Rotating *equal* amplitudes changes only phases, so this needs unequal ones."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    from src.qaoa import _apply_xy_pair, _xy_pairs

    n, k, beta = 6, 3, 0.3
    states = np.arange(2**n, dtype=np.uint32)
    support = states[_bits(states, n).sum(axis=1) == k]
    g = rng(5)
    vector = np.zeros(2**n, dtype=complex)
    vector[support] = g.normal(size=len(support)) + 1j * g.normal(size=len(support))
    vector /= np.linalg.norm(vector)

    ring = [(q, (q + 1) % n) for q in range(n)]
    ours = vector.copy()
    for pair in ring:
        ours = _apply_xy_pair(ours, _xy_pairs(n, *pair), 2.0 * beta)

    qc = QuantumCircuit(n)
    qc.initialize(vector, range(n))
    for qi, qj in ring:
        qc.rxx(2 * beta, qi, qj)
        qc.ryy(2 * beta, qi, qj)
    ref = Statevector.from_instruction(qc).data
    assert np.max(np.abs(np.abs(ref) ** 2 - np.abs(ours) ** 2)) < 1e-14


def test_xy_mixer_preserves_hamming_weight():
    """This is the whole promise of the Dicke construction: cardinality by design."""
    w = rng(4).normal(size=6)
    state = dicke_state(w, np.array([0.7, 1.1]), np.array([0.45, 0.9]), 3)
    probs = np.abs(state) ** 2
    weights = np.array([bin(int(s)).count("1") for s in range(2**6)])
    assert probs[weights != 3].max() < 1e-14
    assert np.isclose(probs.sum(), 1.0)


def test_soft_penalty_state_is_not_constrained():
    """The comparison is only meaningful if the soft variant *does* leak amplitude."""
    inst, config = toy(n=6, k=3)
    qubo = build_qubo(inst, config)
    state = standard_state(qubo["diag"], np.array([0.3]), np.array([0.3]), qubo["layout"].n_qubits)
    weights = np.array([bin(int(s)).count("1") for s in range(2 ** qubo["layout"].n_qubits)])
    x_weight = weights & ((1 << qubo["layout"].n_x) - 1)
    assert (x_weight != 3).any()
    probs = np.abs(state) ** 2
    assert probs[x_weight != 3].sum() > 1e-6


# --------------------------------------------------------------------------- #
# agreement with Qiskit / Aer
# --------------------------------------------------------------------------- #
def test_standard_circuit_matches_the_numpy_engine():
    from qiskit_aer import AerSimulator

    inst, config = toy(n=4, k=2)
    qubo = build_qubo(inst, config)
    p = 2
    g, b = np.array([0.33, 0.61]), np.array([0.22, 0.47])
    state = standard_state(qubo["diag"], g, b, qubo["layout"].n_qubits)
    qc = standard_circuit(qubo, p, g, b).copy()
    qc.save_statevector()
    sv = np.asarray(AerSimulator(method="statevector").run(qc).result().get_statevector(qc))
    assert np.max(np.abs(np.abs(sv) ** 2 - np.abs(state) ** 2)) < 1e-12


def test_dicke_circuit_matches_the_numpy_engine():
    from qiskit_aer import AerSimulator

    w = np.array([5.0, -1.0, 3.0, 0.5, 2.0, 4.0])
    p = 1
    g, b = np.array([0.4]), np.array([0.3])
    state = dicke_state(w, g, b, 3)
    qc = dicke_circuit(w, 3, p, g, b).copy()
    qc.save_statevector()
    sv = np.asarray(AerSimulator(method="statevector").run(qc).result().get_statevector(qc))
    assert np.max(np.abs(np.abs(sv) ** 2 - np.abs(state) ** 2)) < 1e-12


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def test_marginalise_sums_the_slack_registers():
    inst, config = toy(n=4, k=2)
    qubo = build_qubo(inst, config)
    probs = np.abs(standard_state(qubo["diag"], np.array([0.2]), np.array([0.4]), qubo["layout"].n_qubits)) ** 2
    marg = marginalise(np.sqrt(probs) + 0j, 4)
    assert marg.shape == (16,)
    assert np.isclose(marg.sum(), 1.0)
    np.testing.assert_allclose(marg, probs.reshape(16, -1).sum(axis=1))


def test_feasible_mask_matches_the_declared_constraints():
    inst, config = toy(n=4, k=2)
    mask = feasible_mask(inst, config)
    assert mask.sum() == len([c for c in __import__("itertools").combinations(range(4), 2)
                              if inst.costs[list(c)].sum() <= 8 and inst.capacities[list(c)].sum() <= 9])
    assert not mask[0]  # cardinality 0 is infeasible


def test_state_values_are_the_expected_net_values():
    inst, config = toy(n=4, k=2)
    values = state_values(inst)
    assert values[0] == 0.0
    assert values[0b0011] == pytest.approx(inst.weights[0] + inst.weights[1])


def test_summarise_on_a_point_mass_is_perfect():
    inst, config = toy(n=4, k=2)
    bf = brute_force(inst, config)
    opt = sum(1 << i for i in bf["optimum"]["indices"])
    probs = np.zeros(2**4)
    probs[opt] = 1.0
    out = summarise(probs, inst, config, rng(0))
    assert out["p_optimal"] == pytest.approx(1.0)
    assert out["feasibility_rate"] == pytest.approx(1.0)
    assert out["approximation_ratio"] == pytest.approx(1.0)
    assert out["sampled_optimal_counts"] == out["shots"]


def test_summarise_on_the_uniform_distribution_is_not_optimal():
    inst, config = toy(n=4, k=2)
    out = summarise(np.full(2**4, 1 / 2**4), inst, config, rng(1))
    assert 0.0 < out["p_optimal"] < 0.5
    assert 0.0 < out["approximation_ratio"] < 1.0


def test_optimise_is_seeded_and_reports_restarts():
    inst, config = toy(n=4, k=2)
    qubo = build_qubo(inst, config)
    n = qubo["layout"].n_qubits

    def objective(x):
        return energy(standard_state(qubo["diag"], x[:1], x[1:], n), qubo["diag"])

    first = optimise(objective, 1, config, stream=7)
    second = optimise(objective, 1, config, stream=7)
    assert first["params"] == second["params"]
    assert first["energy"] == second["energy"]
    assert len(first["restarts"]) == int(config["stage4"]["restarts"])
    gam, bet = split_params(first["params"], 1)
    assert gam.shape == bet.shape == (1,)


def test_split_params_round_trip():
    g, b = split_params([1.0, 2.0, 3.0, 4.0], 2)
    assert list(g) == [1.0, 2.0] and list(b) == [3.0, 4.0]
