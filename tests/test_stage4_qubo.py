"""Stage 4 QUBO invariants: algebra, slack encoding, decoding, brute force, penalties.

The important assertions are the ones that make "the penalties were verified, not
assumed" checkable: the materialised diagonal must equal the written Hamiltonian,
the Ising form must equal the QUBO, and the ground state must be the feasible
brute-force optimum.
"""

from __future__ import annotations

import copy
from itertools import combinations

import numpy as np
import pytest

from src.qubo import (
    Instance,
    Layout,
    _bits_needed,
    build_instance,
    build_qubo,
    brute_force,
    decode,
    evaluate,
    layout_for,
    net_values,
    qubo_diagonal,
    to_ising,
)
from src.utils import load_config, rng


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def toy(n: int = 6, k: int = 3, budget: int = 6, capacity: int = 9, seed: int = 7) -> tuple[Instance, dict]:
    """Small instance with the same schema, solved by exhaustive enumeration."""
    g = rng(seed)
    # Weights are kept below the smallest penalty step (12), so the registered
    # penalty set is strong enough for *this* instance by construction. That is
    # exactly the condition the production run has to verify rather than assume.
    inst = Instance(
        customer_ids=tuple(f"T{i:02d}" for i in range(n)),
        probabilities=g.uniform(0.05, 0.9, n).round(4),
        costs=g.integers(1, 3, n),
        values=g.integers(8, 13, n),
        capacities=g.integers(1, 5, n),
    )
    config = copy.deepcopy(load_config())
    config["stage2"].update({"n_candidates": n, "k_select": k, "budget": budget, "capacity": capacity})
    return inst, config


# --------------------------------------------------------------------------- #
# instance
# --------------------------------------------------------------------------- #
def test_net_value_definition():
    import pandas as pd

    stage2 = pd.DataFrame({"expected_deposit_value": [10, 20], "contact_cost": [1, 5]})
    w = net_values([0.5, 0.25], stage2)
    assert np.allclose(w, [0.5 * 10 - 1, 0.25 * 20 - 5])


def test_instance_rejects_misaligned_input():
    import pandas as pd

    with pytest.raises(ValueError, match="missing columns"):
        build_instance(pd.DataFrame({"customer_id": ["a"], "predicted_probability": [0.5]}))


def test_slack_bit_count_is_minimal_but_sufficient():
    assert _bits_needed(0) == 1
    assert _bits_needed(25) == 5
    assert _bits_needed(22) == 5
    assert _bits_needed(32) == 6
    for value in (1, 7, 22, 25, 31):
        assert 2 ** _bits_needed(value) - 1 >= value


def test_layout_is_ten_plus_two_five_bit_slacks(cfg):
    inst, _ = toy(n=10)
    lay = layout_for(inst, cfg)
    assert (lay.n_x, lay.slack_b_bits, lay.slack_c_bits) == (10, 5, 5)
    assert lay.n_qubits == 20
    assert lay.b_bits + lay.c_bits == tuple(range(10, 20))


# --------------------------------------------------------------------------- #
# algebra: the matrix, the diagonal and the Ising form must agree
# --------------------------------------------------------------------------- #
def test_diagonal_equals_direct_hamiltonian():
    inst, config = toy()
    qubo = build_qubo(inst, config)
    lay = qubo["layout"]
    g = rng(99)
    for state in g.integers(0, 2**lay.n_qubits, 300):
        decoded = decode(int(state), inst, lay)
        direct = evaluate(inst, config, decoded["x_bits"], decoded["slack_b"], decoded["slack_c"])
        assert direct == pytest.approx(qubo["diag"][state], abs=1e-9)


def test_ising_conversion_is_exact():
    inst, config = toy()
    qubo = build_qubo(inst, config)
    h, J, offset = qubo["ising"]["h"], qubo["ising"]["J"], qubo["ising"]["offset"]
    lay = qubo["layout"]
    g = rng(123)
    for state in g.integers(0, 2**lay.n_qubits, 300):
        z = 1.0 - 2.0 * np.array([(int(state) >> i) & 1 for i in range(lay.n_qubits)])
        assert float(offset + h @ z + 0.5 * z @ J @ z) == pytest.approx(qubo["diag"][state], abs=1e-9)


def test_quadratic_coefficients_reproduce_the_objective():
    """Penalty expansion is the classic place to lose a factor of two."""
    inst, config = toy()
    qubo = build_qubo(inst, config)
    lay = qubo["layout"]
    x = np.zeros(lay.n_x, dtype=int)
    x[[0, 2, 3]] = 1  # exactly k=3
    s_b, s_c = 1, 2
    state = int(sum(1 << i for i in range(lay.n_x) if x[i]))
    state |= s_b << lay.n_x
    state |= s_c << (lay.n_x + lay.slack_b_bits)
    chosen = [i for i in range(lay.n_x) if x[i]]
    expected = -inst.weights[chosen].sum()
    expected += float(config["stage2"]["penalty_a"]) * (3 - config["stage2"]["k_select"]) ** 2
    expected += float(config["stage2"]["penalty_b"]) * (inst.costs[chosen].sum() + s_b - config["stage2"]["budget"]) ** 2
    expected += float(config["stage2"]["penalty_c"]) * (
        inst.capacities[chosen].sum() + s_c - config["stage2"]["capacity"]
    ) ** 2
    assert qubo["diag"][state] == pytest.approx(expected, abs=1e-9)


def test_diagonal_chunking_is_chunk_invariant():
    U = np.array([[1.0, 2.0], [0.0, -1.0]])
    assert np.allclose(qubo_diagonal(U, 0.5, 2, chunk=2), qubo_diagonal(U, 0.5, 2, chunk=1))
    assert np.allclose(qubo_diagonal(U, 0.5, 2), [0.5, 1.5, -0.5, 2.5])


def test_ising_constant_matches_at_the_all_zero_state():
    inst, config = toy()
    qubo = build_qubo(inst, config)
    h, J, offset = qubo["ising"]["h"], qubo["ising"]["J"], qubo["ising"]["offset"]
    z = np.ones(qubo["layout"].n_qubits)
    assert float(offset + h @ z + 0.5 * z @ J @ z) == pytest.approx(qubo["diag"][0], abs=1e-9)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def test_decode_round_trip():
    inst, config = toy()
    lay = layout_for(inst, config)
    decoded = decode(0b101101, inst, lay)
    assert decoded["selected_ids"] == [inst.customer_ids[i] for i in (0, 2, 3, 5)]
    assert decoded["bitstring"] == "101101"
    assert decoded["cost"] == int(sum(inst.costs[[0, 2, 3, 5]]))
    assert decoded["expected_net_value"] == pytest.approx(float(inst.weights[[0, 2, 3, 5]].sum()))


def test_slack_registers_decode_independently():
    inst, config = toy()
    lay = layout_for(inst, config)
    state = 0b011 | (5 << lay.n_x) | (3 << (lay.n_x + lay.slack_b_bits))
    decoded = decode(state, inst, lay)
    assert (decoded["slack_b"], decoded["slack_c"]) == (5, 3)
    assert decoded["x_bits"] == 0b011


# --------------------------------------------------------------------------- #
# brute force
# --------------------------------------------------------------------------- #
def test_brute_force_enumerates_every_subset():
    inst, config = toy(n=8, k=4, budget=20, capacity=20)
    result = brute_force(inst, config)
    assert result["n_combinations"] == len(list(combinations(range(8), 4))) == 70
    assert result["n_feasible"] <= result["n_combinations"]


def test_brute_force_optimum_is_optimal_and_feasible():
    inst, config = toy()
    result = brute_force(inst, config)
    best = result["optimum"]
    chosen = [int(inst.customer_ids.index(cid)) for cid in best["customer_ids"]]
    assert len(chosen) == config["stage2"]["k_select"]
    assert inst.costs[chosen].sum() <= config["stage2"]["budget"]
    assert inst.capacities[chosen].sum() <= config["stage2"]["capacity"]
    feasible_values = [
        float(inst.weights[list(c)].sum())
        for c in combinations(range(inst.n), config["stage2"]["k_select"])
        if inst.costs[list(c)].sum() <= config["stage2"]["budget"]
        and inst.capacities[list(c)].sum() <= config["stage2"]["capacity"]
    ]
    assert best["expected_net_value"] == pytest.approx(max(feasible_values))


def test_binding_diagnostics_detect_a_degenerate_problem(cfg):
    """When every subset is feasible the optimum is just the top-k by weight."""
    inst, _ = toy(n=10, k=6, budget=200, capacity=200)
    result = brute_force(inst, cfg)
    diag = result["binding_diagnostics"]
    assert diag["all_subsets_feasible"] is True
    assert diag["budget_ever_binds"] is False
    assert diag["optimum_equals_top_k_by_weight"] is True


# --------------------------------------------------------------------------- #
# the QUBO must actually solve the problem
# --------------------------------------------------------------------------- #
def test_ground_state_is_the_feasible_brute_force_optimum():
    inst, config = toy()
    qubo = build_qubo(inst, config)
    ground = decode(int(np.argmin(qubo["diag"])), inst, qubo["layout"])
    bf = brute_force(inst, config)
    assert set(ground["selected_ids"]) == set(bf["optimum"]["customer_ids"])


def test_ground_state_slacks_are_the_exact_complement():
    inst, config = toy()
    qubo = build_qubo(inst, config)
    ground = decode(int(np.argmin(qubo["diag"])), inst, qubo["layout"])
    assert ground["slack_b"] == config["stage2"]["budget"] - ground["cost"]
    assert ground["slack_c"] == config["stage2"]["capacity"] - ground["capacity"]


def test_slacks_above_the_bound_can_be_reached_but_never_help():
    """5 bits encode 0..31; values above the target are reachable and strictly worse."""
    inst, config = toy()
    qubo = build_qubo(inst, config)
    lay = qubo["layout"]
    x = [i for i in range(lay.n_x)][: config["stage2"]["k_select"]]
    x_bits = sum(1 << i for i in x)
    cost = int(inst.costs[x].sum())
    good = evaluate(inst, config, x_bits, config["stage2"]["budget"] - cost, config["stage2"]["capacity"] - int(inst.capacities[x].sum()))
    over = evaluate(inst, config, x_bits, 31, 31)
    assert over > good


def test_layout_matches_the_declared_problem_size(cfg):
    """Guard against a toy layout leaking into the production run."""
    inst = Instance(
        customer_ids=tuple(f"C{i:02d}" for i in range(10)),
        probabilities=np.full(10, 0.2),
        costs=np.ones(10, dtype=int),
        values=np.full(10, 10, dtype=int),
        capacities=np.ones(10, dtype=int),
    )
    lay = layout_for(inst, cfg)
    assert lay.n_qubits == 20 == Layout(10, 5, 5).n_qubits
