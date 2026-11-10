"""Stage 2 invariants: simulator fidelity, kernel validity, projection leakage, gate logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qiskit.quantum_info import Statevector

from src.quantum_clf import (
    _weighted_bce,
    circuit_report,
    evaluate_gate,
    kernel_matrix,
    optimise_vqc,
    project,
    unentangled_circuit,
    vqc_circuit,
    vqc_probabilities,
    zz_circuit,
)
from src.utils import load_config, rng


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# the batch simulator must agree with Qiskit exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 4])
def test_batch_simulator_matches_qiskit(n):
    g = rng(1234 + n)
    x = g.uniform(0, np.pi, n)
    params = g.uniform(0, 2 * np.pi, 2 * 2 * n)
    probs = np.abs(Statevector.from_instruction(vqc_circuit(params, x)).data) ** 2
    expected = probs[(np.arange(2**n) & 1).astype(bool)].sum()
    assert vqc_probabilities(params, x[None, :])[0] == pytest.approx(expected, abs=1e-12)


def test_batch_simulator_matches_qiskit_row_by_row():
    """Regression: a batch > 1 must broadcast identically to per-row Qiskit states."""
    g = rng(99)
    n = 4
    X = g.uniform(0, np.pi, (5, n))
    params = g.uniform(0, 2 * np.pi, 2 * 2 * n)
    batched = vqc_probabilities(params, X)
    single = np.array([vqc_probabilities(params, X[i : i + 1])[0] for i in range(len(X))])
    np.testing.assert_allclose(batched, single, rtol=0, atol=1e-14)
    for i in range(len(X)):
        probs = np.abs(Statevector.from_instruction(vqc_circuit(params, X[i])).data) ** 2
        assert batched[i] == pytest.approx(probs[(np.arange(2**n) & 1).astype(bool)].sum(), abs=1e-12)


def test_vqc_probabilities_are_bounded_and_batched():
    g = rng(7)
    X = g.uniform(0, np.pi, (13, 4))
    p = vqc_probabilities(np.zeros(16), X)
    assert p.shape == (13,)
    assert np.all((p >= 0) & (p <= 1))


# --------------------------------------------------------------------------- #
# kernel validity
# --------------------------------------------------------------------------- #
def test_zz_kernel_is_a_proper_gram_matrix():
    g = rng(11)
    A = g.uniform(0, np.pi, (9, 4))
    K = kernel_matrix(A, A, zz_circuit)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9, "fidelity kernel must be PSD"
    assert (K >= -1e-12).all() and (K <= 1 + 1e-12).all()


def test_unentangled_kernel_differs_from_entangled():
    A = np.array([[0.3, 1.2, 2.5, 0.7], [2.0, 0.4, 1.1, 2.9]])
    K_zz = kernel_matrix(A, A, zz_circuit)
    K_un = kernel_matrix(A, A, unentangled_circuit)
    assert K_zz[0, 1] != pytest.approx(K_un[0, 1]), "entangling gates must change the kernel"


def test_cross_kernel_is_not_square_when_sizes_differ():
    g = rng(3)
    A, B = g.uniform(0, np.pi, (5, 4)), g.uniform(0, np.pi, (7, 4))
    assert kernel_matrix(A, B, zz_circuit).shape == (5, 7)


# --------------------------------------------------------------------------- #
# projection: fitted on train only
# --------------------------------------------------------------------------- #
def test_projection_bounds_come_from_train_only():
    g = rng(5)
    Z_tr, Z_va = g.normal(size=(60, 11)), g.normal(size=(20, 11))
    A_tr, A_va, proj = project(Z_tr, Z_va, 4, seed=0)
    assert A_tr.shape == (60, 4) and A_va.shape == (20, 4)
    assert np.all(A_tr >= 0) and np.all(A_tr <= np.pi + 1e-9)
    assert np.all(A_va >= 0) and np.all(A_va <= np.pi + 1e-9)

    # Same train slice, wildly different val slice -> identical fitted bounds.
    _, _, proj2 = project(Z_tr, Z_va * 50 + 7, 4, seed=0)
    np.testing.assert_allclose(proj.lo, proj2.lo)
    np.testing.assert_allclose(proj.hi, proj2.hi)


def test_projection_clips_out_of_range_validation_rows():
    Z_tr = np.linspace(-1, 1, 200).reshape(50, 4)
    Z_va = np.array([[100.0, -100.0, 0.0, 0.0]])
    _, A_va, _ = project(Z_tr, Z_va, 4, seed=0)
    assert np.all(A_va >= 0) and np.all(A_va <= np.pi + 1e-9)


# --------------------------------------------------------------------------- #
# training and gate logic
# --------------------------------------------------------------------------- #
def test_optimiser_reduces_the_loss():
    g = rng(21)
    X = g.uniform(0, np.pi, (40, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    start = _weighted_bce(y, vqc_probabilities(np.zeros(8), X))
    out = optimise_vqc(X, y, layers=2, seed=0, restarts=1, maxiter=60)
    assert out["loss"] <= start, "training must not make the weighted BCE worse"
    assert out["n_evals"] > 0


def test_class_weights_balance_imbalanced_bce():
    y = np.array([1] + [0] * 9)
    p = np.full(10, 0.5)
    # weights: pos -> 10/(2*1)=5, neg -> 10/(2*9)=0.5556
    expected = -(5 * np.log(0.5) + 9 * (10 / 18) * np.log(0.5)) / 10
    assert _weighted_bce(y, p) == pytest.approx(expected)


def _folds(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"fold": np.arange(len(values)), "zz_kernel_roc_auc": values})


def test_gate_admits_only_a_clear_winner():
    base = np.full(50, 0.60)
    baseline = pd.DataFrame({"fold": np.arange(50), "raw_roc_auc": base})

    big = evaluate_gate([], _folds(np.full(50, 0.65)), baseline, "zz_kernel")
    assert big["admitted"] and big["delta_mean_roc_auc"] == pytest.approx(0.05)

    marginal = evaluate_gate([], _folds(np.full(50, 0.605)), baseline, "zz_kernel")
    assert not marginal["admitted"], "Δ below +0.01 must be rejected"

    # Big mean gain but only 4 of 10 repeats improving -> rejected on the repeat rule.
    mixed = np.full(50, 0.70)
    for r in range(4):
        mixed[r * 5 : (r + 1) * 5] = 0.50
    gate = evaluate_gate([], _folds(mixed), baseline, "zz_kernel")
    assert gate["repeats_improved"] == 6 and gate["admitted"], "6/10 repeats must pass"
    mixed2 = mixed.copy()
    mixed2[4 * 5 : 5 * 5] = 0.50
    assert not evaluate_gate([], _folds(mixed2), baseline, "zz_kernel")["admitted"]


def test_circuit_report_counts_and_no_swaps(cfg):
    rep = circuit_report(np.full(4, 0.5), cfg)
    assert rep["zz_feature_map"]["qubits"] == 4
    assert rep["zz_feature_map"]["swaps"] == 0
    assert rep["vqc_parameters"] == 16
    assert rep["vqc"]["logical_2q"] == 6  # 2 layers x 3 CNOTs on a 4-qubit ladder
    assert rep["zz_feature_map"]["logical_ops"]["rzz"] == 3  # linear entanglement, 4 qubits
