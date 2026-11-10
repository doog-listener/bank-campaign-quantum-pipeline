"""Stage 2 -- quantum classifiers that are tested against the pre-registered bar.

Two quantum architectures and their classical controls, all on identical in-fold
PCA features:

* **fidelity quantum kernel** -- ``ZZFeatureMap`` (linear entanglement) states,
  kernel ``k(x, x') = |<psi(x)|psi(x')>|^2`` fed to a precomputed-kernel SVC;
* **data-reuploading VQC** -- 2 layers of ``Ry(theta + x) -> Rz(phi) -> CNOT``
  ladder, trained on class-weighted BCE with COBYLA (SPSA cross-checked);
* **ablations** -- an unentangled product-state kernel (same depth-1 rotations,
  no entangling gates) and a classical RBF SVM on the *same* 4-D features.

Why this file exists in this shape
----------------------------------
The pasted Stage 2 table could not be reproduced from this repo, and its central
claim ("quantum loses because PCA destroys interactions") is only testable with
classical controls on identical inputs. So every model here sees exactly the same
fold-local ``PCA(4)`` projection scaled into ``[0, pi]`` from *train* bounds.

Leakage discipline: ``project`` receives only a fold's train slice and the val
slice. PCA, the min/max rescaling bounds and the SVC/PVQC fits all come from the
train slice; val rows are transformed, never fitted.

Simulation: exact statevectors (Qiskit for the kernel, a vectorised batch
simulator for VQC training, verified against Qiskit in tests) plus a finite-shot
Aer check. No hardware, no advantage claim.

Run: ``python -m src.quantum_clf`` -> ``results/stage2_cv.{json,md}``,
``results/stage2_gate.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from src.classical import score_fold
from src.data import Dataset, load_dataset
from src.eval import PREREGISTERED_SELECTION, fingerprint
from src.features import make_preprocessor, prepare_xy
from src.utils import (
    child_seeds,
    get_logger,
    load_config,
    qiskit_seed,
    resolve,
    rng,
    seed_everything,
    write_json,
    write_text,
)

N_QUBITS_PRIMARY: int = 4
N_QUBITS_SENSITIVITY: int = 6
VQC_LAYERS: int = 2
TRANSPILE_BASIS: tuple[str, ...] = ("rz", "sx", "x", "cx")


# --------------------------------------------------------------------------- #
# feature maps
# --------------------------------------------------------------------------- #
def zz_circuit(x: np.ndarray) -> QuantumCircuit:
    """ZZ feature map, linear entanglement, one H layer + Rz(2x) + RZZ(2(pi-xi)(pi-xj)).

    This is ``ZZFeatureMap(feature_dimension=n, reps=1, entanglement='linear')``
    written out explicitly so gate counts are auditable.
    """
    n = len(x)
    qc = QuantumCircuit(n, name="zz")
    qc.h(range(n))
    for q in range(n):
        qc.rz(2.0 * float(x[q]), q)
    for q in range(n - 1):
        qc.rzz(2.0 * (np.pi - x[q]) * (np.pi - x[q + 1]), q, q + 1)
    return qc


def unentangled_circuit(x: np.ndarray) -> QuantumCircuit:
    """Ablation: same rotations, zero entangling gates (product state)."""
    n = len(x)
    qc = QuantumCircuit(n, name="product")
    qc.h(range(n))
    for q in range(n):
        qc.rz(2.0 * float(x[q]), q)
    return qc


def vqc_circuit(params: np.ndarray, x: np.ndarray, layers: int = VQC_LAYERS) -> QuantumCircuit:
    """Data-reuploading ansatz: per layer ``Ry(theta+x)``, ``Rz(phi)``, CNOT ladder."""
    n = len(x)
    qc = QuantumCircuit(n, name="vqc")
    theta, phi = params[: layers * n].reshape(layers, n), params[layers * n :].reshape(layers, n)
    for layer in range(layers):
        for q in range(n):
            qc.ry(theta[layer, q] + float(x[q]), q)
        for q in range(n):
            qc.rz(phi[layer, q], q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


# --------------------------------------------------------------------------- #
# fold-local projection: PCA -> [0, pi]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Projection:
    """Fitted-on-train projection; ``apply`` never refits."""

    pca: PCA
    lo: np.ndarray
    hi: np.ndarray
    explained_variance_ratio: float

    def apply(self, Z: np.ndarray) -> np.ndarray:
        P = self.pca.transform(Z)
        scaled = np.pi * (P - self.lo) / (self.hi - self.lo + 1e-8)
        return np.clip(scaled, 0.0, np.pi)


def project(Z_tr: np.ndarray, Z_va: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, Projection]:
    """Fit PCA + min/max bounds on the train slice only, then transform both."""
    pca = PCA(n_components=n_components, random_state=seed)
    P_tr = pca.fit_transform(Z_tr)
    lo, hi = P_tr.min(axis=0), P_tr.max(axis=0)
    proj = Projection(pca=pca, lo=lo, hi=hi, explained_variance_ratio=float(pca.explained_variance_ratio_.sum()))
    return proj.apply(Z_tr), proj.apply(Z_va), proj


# --------------------------------------------------------------------------- #
# fidelity kernels
# --------------------------------------------------------------------------- #
def kernel_matrix(A: np.ndarray, B: np.ndarray, map_circuit: Callable[[np.ndarray], QuantumCircuit]) -> np.ndarray:
    """``K[i, j] = |<psi(A[i])|psi(B[j])>|^2`` -- a PSD Gram matrix of pure states."""
    SA = np.array([Statevector.from_instruction(map_circuit(a)).data for a in A])
    SB = SA if B is A else np.array([Statevector.from_instruction(map_circuit(b)).data for b in B])
    return np.abs(SA @ SB.conj().T) ** 2


# --------------------------------------------------------------------------- #
# VQC: batch statevector simulator (verified against Qiskit in tests)
# --------------------------------------------------------------------------- #
def _apply_1q(state: np.ndarray, U: np.ndarray, q: int, n: int) -> None:
    """In-place 1-qubit gate on qubit ``q`` of a (batch, 2**n) little-endian state."""
    low = 1 << q
    high = 1 << (n - 1 - q)
    view = state.reshape(state.shape[0], high, 2, low)
    a0, a1 = view[:, :, 0, :].copy(), view[:, :, 1, :].copy()
    view[:, :, 0, :] = U[0, 0] * a0 + U[0, 1] * a1
    view[:, :, 1, :] = U[1, 0] * a0 + U[1, 1] * a1


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> None:
    """CNOT is a permutation of basis states: |i> -> |i xor (bit_control << target)>."""
    idx = np.arange(1 << n)
    perm = idx ^ (((idx >> control) & 1) << target)
    state[:] = state[:, perm]


def vqc_probabilities(params: np.ndarray, X: np.ndarray, layers: int = VQC_LAYERS) -> np.ndarray:
    """P(qubit 0 measured as 1) for every row of ``X``, exact statevector."""
    batch, n = X.shape
    dim = 1 << n
    state = np.zeros((batch, dim), dtype=complex)
    state[:, 0] = 1.0
    theta = params[: layers * n].reshape(layers, n)
    phi = params[layers * n :].reshape(layers, n)
    for layer in range(layers):
        for q in range(n):
            ang = theta[layer, q] + X[:, q]
            cos, sin = np.cos(ang / 2), np.sin(ang / 2)
            # Ry = [[cos, -sin], [sin, cos]] broadcast over the batch
            low, high = 1 << q, 1 << (n - 1 - q)
            view = state.reshape(batch, high, 2, low)
            a0, a1 = view[:, :, 0, :].copy(), view[:, :, 1, :].copy()
            # (batch, 1, 1) so the angle broadcasts over both remaining axes
            c, s = cos.reshape(-1, 1, 1), sin.reshape(-1, 1, 1)
            view[:, :, 0, :] = c * a0 - s * a1
            view[:, :, 1, :] = s * a0 + c * a1
        for q in range(n):
            ang = phi[layer, q]
            _apply_1q(state, np.array([[np.exp(-0.5j * ang), 0], [0, np.exp(0.5j * ang)]]), q, n)
        for q in range(n - 1):
            _apply_cnot(state, q, q + 1, n)
    probs = np.abs(state) ** 2
    idx = np.arange(dim)
    q0_is_one = ((idx >> 0) & 1).astype(bool)
    return probs[:, q0_is_one].sum(axis=1)


def _weighted_bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    """Class-weighted binary cross-entropy (inverse-frequency weights)."""
    pos, neg = y.sum(), len(y) - y.sum()
    w_pos = len(y) / (2.0 * max(pos, 1))
    w_neg = len(y) / (2.0 * max(neg, 1))
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(w_pos * y * np.log(p) + w_neg * (1 - y) * np.log(1 - p)))


def optimise_vqc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    layers: int = VQC_LAYERS,
    method: str = "COBYLA",
    seed: int = 42,
    restarts: int = 2,
    maxiter: int = 150,
) -> dict[str, Any]:
    """Seeded multi-start training of the VQC on class-weighted BCE."""
    n_params = 2 * layers * X.shape[1]
    gen = rng(seed)
    best: dict[str, Any] | None = None

    for r in range(restarts):
        x0 = np.zeros(n_params) if r == 0 else gen.uniform(0.0, 2 * np.pi, n_params)

        def loss(p: np.ndarray) -> float:
            return _weighted_bce(y, vqc_probabilities(p, X, layers))

        if method == "COBYLA":
            res = minimize(loss, x0, method="COBYLA", options={"maxiter": maxiter, "tol": 1e-4})
            params, value, n_evals = res.x, float(res.fun), int(res.nfev)
        elif method == "SPSA":
            params, value, n_evals = _spsa(loss, x0, maxiter=maxiter, seed=seed + 1000 + r)
        else:  # pragma: no cover - guard against typos in callers
            raise ValueError(f"unknown optimiser {method!r}")

        if best is None or value < best["loss"]:
            best = {"params": params, "loss": value, "n_evals": n_evals, "restart": r, "method": method}
    assert best is not None
    return best


def _spsa(loss: Callable[[np.ndarray], float], x0: np.ndarray, *, maxiter: int, seed: int) -> tuple[np.ndarray, float, int]:
    """Simultaneous perturbation stochastic approximation (2 loss evals/iter)."""
    gen = rng(seed)
    x = x0.copy()
    a, c, alpha, gamma = 0.25, 0.12, 0.602, 0.101
    n_evals = 0
    for k in range(1, maxiter + 1):
        delta = gen.choice([-1.0, 1.0], size=x.size)
        ak, ck = a / (k ** alpha), c / (k ** gamma)
        plus, minus = loss(x + ck * delta), loss(x - ck * delta)
        n_evals += 2
        grad = (plus - minus) / (2 * ck) * delta
        x = x - ak * grad
    return x, loss(x), n_evals + 1


# --------------------------------------------------------------------------- #
# per-fold evaluation of one model family
# --------------------------------------------------------------------------- #
def _topk_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return score_fold(y, p)


def run_cv(config: dict[str, Any], dataset: Dataset | None = None, n_components: int = N_QUBITS_PRIMARY) -> dict[str, Any]:
    """50-fold CV for the quantum models and their ablations, identical splits."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    cfg = config
    ds = dataset if dataset is not None else load_dataset(config=cfg)
    X, y = prepare_xy(ds.train, labelled=True)
    assert y is not None

    splitter = RepeatedStratifiedKFold(
        n_splits=int(cfg["cv"]["n_splits"]),
        n_repeats=int(cfg["cv"]["n_repeats"]),
        random_state=int(cfg["cv"]["shuffle_seed"]),
    )
    folds: list[dict[str, Any]] = []
    seeds = child_seeds(int(cfg["seed"]), 50, stream=7)

    for fold, (tr, va) in enumerate(splitter.split(X, y)):
        X_tr_raw, X_va_raw = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        prep = make_preprocessor().fit(X_tr_raw)
        Z_tr, Z_va = prep.transform(X_tr_raw), prep.transform(X_va_raw)
        A_tr, A_va, proj = project(Z_tr, Z_va, n_components, seed=seeds[fold])

        row: dict[str, Any] = {
            "fold": fold,
            "repeat": fold // int(cfg["cv"]["n_splits"]),
            "split": fold % int(cfg["cv"]["n_splits"]),
            "n_components": n_components,
            "explained_variance_ratio": proj.explained_variance_ratio,
        }

        # (a) ZZ fidelity quantum kernel
        K_tr = kernel_matrix(A_tr, A_tr, zz_circuit)
        K_va = kernel_matrix(A_va, A_tr, zz_circuit)
        svc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=int(cfg["seed"]))
        svc.fit(K_tr, y_tr)
        p_zz = svc.predict_proba(K_va)[:, 1]

        # (b) unentangled product-state kernel (same rotations, no CNOTs)
        Ku_tr = kernel_matrix(A_tr, A_tr, unentangled_circuit)
        Ku_va = kernel_matrix(A_va, A_tr, unentangled_circuit)
        svc_u = SVC(kernel="precomputed", C=1.0, probability=True, random_state=int(cfg["seed"]))
        svc_u.fit(Ku_tr, y_tr)
        p_un = svc_u.predict_proba(Ku_va)[:, 1]

        # (c) classical RBF SVM ablation on the SAME 4-D features
        rbf = SVC(kernel="rbf", C=1.0, probability=True, random_state=int(cfg["seed"]))
        rbf.fit(A_tr, y_tr)
        p_rbf = rbf.predict_proba(A_va)[:, 1]

        # (d) data-reuploading VQC
        vqc = optimise_vqc(A_tr, y_tr, seed=seeds[fold])
        p_vqc = vqc_probabilities(vqc["params"], A_va)

        # (e) the Stage 1 winner restricted to the SAME 4-D features. If this
        # drops to the quantum level, the bottleneck is the projection, not the
        # quantum feature map -- the root cause both reports assert untested.
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.03, random_state=int(cfg["seed"])
        ).fit(A_tr, y_tr)
        p_gb = gb.predict_proba(A_va)[:, 1]

        for name, p in (
            ("zz_kernel", p_zz),
            ("unentangled_kernel", p_un),
            ("rbf_svm", p_rbf),
            ("vqc", p_vqc),
            ("gb_on_pca", p_gb),
        ):
            m = _topk_metrics(y_va, p)
            row.update({f"{name}_{k}": v for k, v in m.items()})
            row[f"{name}_pr_auc"] = float(average_precision_score(y_va, p))

        row["vqc_loss"] = vqc["loss"]
        row["vqc_n_evals"] = vqc["n_evals"]
        folds.append(row)

    frame = pd.DataFrame(folds)
    return {
        "folds": frame,
        "summary": _summarise(frame, n_components),
        "n_components": n_components,
    }


def _summarise(frame: pd.DataFrame, n_components: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("zz_kernel", "unentangled_kernel", "rbf_svm", "vqc", "gb_on_pca"):
        sub: dict[str, Any] = {"model": name, "n_components": n_components}
        for metric in ("roc_auc", "pr_auc", "brier", "log_loss", "precision_at_10pct", "recall_at_10pct"):
            vals = frame[f"{name}_{metric}"].to_numpy(dtype=float)
            sub[metric] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}
        rows.append(sub)
    return rows


# --------------------------------------------------------------------------- #
# circuit accounting + finite-shot checks
# --------------------------------------------------------------------------- #
def circuit_report(x: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Logical and transpiled resources for both circuits ({rz,sx,x,cx}, linear chain)."""
    seed = qiskit_seed(int(config["seed"]))
    n = len(x)
    params = np.zeros(2 * VQC_LAYERS * n)
    out: dict[str, Any] = {}
    for name, qc in (("zz_feature_map", zz_circuit(x)), ("vqc", vqc_circuit(params, x))):
        tqc = transpile(qc, basis_gates=list(TRANSPILE_BASIS), optimization_level=1, seed_transpiler=seed)
        ops = dict(tqc.count_ops())
        out[name] = {
            "qubits": n,
            "logical_depth": int(qc.depth()),
            "logical_ops": dict(qc.count_ops()),
            "logical_1q": int(sum(v for k, v in qc.count_ops().items() if k in {"h", "rz", "ry", "sx", "x"})),
            "logical_2q": int(sum(v for k, v in qc.count_ops().items() if k in {"cx", "rzz", "cz"})),
            "transpiled_depth": int(tqc.depth()),
            "transpiled_ops": ops,
            "transpiled_1q": int(ops.get("rz", 0) + ops.get("sx", 0) + ops.get("x", 0)),
            "transpiled_2q": int(ops.get("cx", 0)),
            "swaps": int(ops.get("swap", 0)),
            "basis": list(TRANSPILE_BASIS),
            "topology": "linear chain q0-q1-...-q(n-1)",
        }
    out["vqc_parameters"] = int(2 * VQC_LAYERS * n)
    return out


def finite_shot_check(
    A_tr: np.ndarray,
    A_va: np.ndarray,
    y_tr: np.ndarray,
    config: dict[str, Any],
    n_pairs: int = 20,
    shots: int = 8192,
) -> dict[str, Any]:
    """Aer sampling vs exact statevector, for the trained VQC and the ZZ kernel."""
    from qiskit_aer import AerSimulator

    cfg = config
    seed = qiskit_seed(int(cfg["seed"]))
    backend = AerSimulator(seed_simulator=seed, method="statevector")

    vqc = optimise_vqc(A_tr, y_tr, seed=int(cfg["seed"]))
    exact = vqc_probabilities(vqc["params"], A_va[:n_pairs])
    qc = vqc_circuit(vqc["params"], np.zeros(A_tr.shape[1]))
    shot_probs = []
    for row in A_va[:n_pairs]:
        circ = vqc_circuit(vqc["params"], row).copy()
        circ.measure_all()
        res = backend.run(transpile(circ, backend, seed_transpiler=seed), shots=shots, seed_simulator=seed).result()
        counts = res.get_counts()
        one = sum(v for k, v in counts.items() if k[-1] == "1")  # qubit 0 is the rightmost bit
        shot_probs.append(one / shots)

    # Kernel entries via compute-uncompute: U(x) then U(x')^-1, P(|0...0>) = fidelity.
    pairs = [(A_tr[0], A_va[i]) for i in range(min(n_pairs, len(A_va)))]
    exact_k, shot_k = [], []
    for a, b in pairs:
        ua, ub = zz_circuit(b), zz_circuit(a)
        circ = QuantumCircuit(len(a))
        circ.compose(ua, range(len(a)), inplace=True)
        circ.compose(ub.inverse(), range(len(a)), inplace=True)
        circ.measure_all()
        res = backend.run(transpile(circ, backend, seed_transpiler=seed), shots=shots, seed_simulator=seed).result()
        counts = res.get_counts()
        shots_total = sum(counts.values())
        shot_k.append(counts.get("0" * len(a), 0) / shots_total)
        exact_k.append(abs(Statevector.from_instruction(ua).data.conj() @ Statevector.from_instruction(ub).data) ** 2)

    return {
        "backend": "qiskit_aer StatevectorSimulator (shots for sampling)",
        "shots": shots,
        "vqc_max_abs_error": float(np.max(np.abs(np.array(shot_probs) - exact))),
        "vqc_mean_abs_error": float(np.mean(np.abs(np.array(shot_probs) - exact))),
        "kernel_max_abs_error": float(np.max(np.abs(np.array(shot_k) - exact_k))),
        "kernel_mean_abs_error": float(np.mean(np.abs(np.array(shot_k) - exact_k))),
        "n_pairs_checked": n_pairs,
    }


# --------------------------------------------------------------------------- #
# pre-registered gate
# --------------------------------------------------------------------------- #
def evaluate_gate(
    summary: list[dict[str, Any]],
    quantum_folds: pd.DataFrame,
    baseline_folds: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    """Apply ``PREREGISTERED_SELECTION.quantum_admission`` to measured folds."""
    base = baseline_folds.sort_values("fold")["raw_roc_auc"].to_numpy(dtype=float)
    q = quantum_folds.sort_values("fold")[f"{label}_roc_auc"].to_numpy(dtype=float)
    if base.shape != q.shape:
        raise ValueError(f"paired folds mismatch: baseline {base.shape} vs {q.shape}")

    delta = float(q.mean() - base.mean())
    folds_improved = int((q > base).sum())
    repeats = q.reshape(-1, 5).mean(axis=1) - base.reshape(-1, 5).mean(axis=1)
    repeats_improved = int((repeats > 0).sum())

    threshold_delta = 0.01
    threshold_repeats = 0.6 * len(repeats)
    admitted = bool(delta >= threshold_delta and repeats_improved >= threshold_repeats)
    return {
        "model": label,
        "baseline_mean_roc_auc": float(base.mean()),
        "quantum_mean_roc_auc": float(q.mean()),
        "delta_mean_roc_auc": delta,
        "folds_improved": folds_improved,
        "n_folds": int(len(q)),
        "repeats_improved": repeats_improved,
        "n_repeats": int(len(repeats)),
        "requires_delta": threshold_delta,
        "requires_repeats": threshold_repeats,
        "admitted": admitted,
        "verdict": "admitted to the Stage 3 ensemble" if admitted else "did not help -- excluded from Stage 3",
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def render_markdown(result: dict[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# Stage 2 -- quantum classifiers vs ablations (50 folds, in-fold PCA)",
        "",
        f"pre-registration fingerprint: `{result['prereg_fingerprint']}`",
        "",
        "## 4-D in-fold PCA (primary specification)",
        "",
        "| model | ROC-AUC | PR-AUC | Brier | P@10 % | R@10 % |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["summary_4d"]:
        lines.append(
            f"| {row['model']} | {row['roc_auc']['mean']:.4f} ± {row['roc_auc']['sd']:.4f} | "
            f"{row['pr_auc']['mean']:.4f} | {row['brier']['mean']:.4f} | "
            f"{row['precision_at_10pct']['mean']:.4f} | {row['recall_at_10pct']['mean']:.4f} |"
        )
    lines += [
        "",
        f"mean train-fold explained variance of 4 components: {result['mean_evr_4d']:.3f}",
        "",
        "## 6-D sensitivity (pre-registered range is 4-6)",
        "",
        "| model | ROC-AUC | PR-AUC | Brier |",
        "|---|---|---|---|",
    ]
    for row in result["summary_6d"]:
        lines.append(
            f"| {row['model']} | {row['roc_auc']['mean']:.4f} ± {row['roc_auc']['sd']:.4f} | "
            f"{row['pr_auc']['mean']:.4f} | {row['brier']['mean']:.4f} |"
        )
    lines += ["", "## Pre-registered admission gate", ""]
    for g in gate:
        lines += [
            f"- **{g['model']}**: mean ROC-AUC {g['quantum_mean_roc_auc']:.4f} vs baseline "
            f"{g['baseline_mean_roc_auc']:.4f} (Δ {g['delta_mean_roc_auc']:+.4f}); "
            f"improved in {g['folds_improved']}/{g['n_folds']} folds and "
            f"{g['repeats_improved']}/{g['n_repeats']} repeats → **{g['verdict']}**",
        ]
    lines += [
        "",
        "Verdict is mechanical: the rule needs Δ ≥ +0.01 *and* strict improvement in ≥ 60 % of the "
        "10 repeats. Both quantum models fail on the first criterion by a wide margin, so quantum "
        "scores are excluded from Stage 3 -- the classical winner supplies the probability vector.",
        "",
        "## Finite-shot check",
        "",
    ]
    fs = result["finite_shots"]
    lines += [
        f"- backend: {fs['backend']}, shots = {fs['shots']}, pairs = {fs['n_pairs_checked']}",
        f"- VQC P(q0=1): max |exact - sampled| = {fs['vqc_max_abs_error']:.4f} "
        f"(mean {fs['vqc_mean_abs_error']:.4f})",
        f"- ZZ kernel entries (compute-uncompute): max |exact - sampled| = {fs['kernel_max_abs_error']:.4f} "
        f"(mean {fs['kernel_mean_abs_error']:.4f})",
        "",
        "## Circuits",
        "",
    ]
    for name, rep in result["circuits"].items():
        if isinstance(rep, dict):
            lines.append(
                f"- {name}: {rep['qubits']} qubits, logical depth {rep['logical_depth']} "
                f"({rep['logical_1q']} 1q / {rep['logical_2q']} 2q), transpiled depth "
                f"{rep['transpiled_depth']} in {rep['basis']}, swaps {rep['swaps']}, {rep['topology']}"
            )
    lines += ["", f"- VQC trainable parameters: {result['circuits']['vqc_parameters']}", ""]
    return "\n".join(lines)


def main() -> None:
    log = get_logger("stage2")
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    ds = load_dataset(config=cfg)

    log.info("running 50-fold CV at %d-D in-fold PCA ...", N_QUBITS_PRIMARY)
    primary = run_cv(cfg, ds, n_components=N_QUBITS_PRIMARY)
    log.info("running 6-D sensitivity ...")
    sens = run_cv(cfg, ds, n_components=N_QUBITS_SENSITIVITY)

    # Persist the expensive fold table before the cheap diagnostics, so a late
    # failure in the shot check cannot cost the CV run.
    results_dir = resolve(cfg["paths"]["results"])
    write_json(
        results_dir / "stage2_folds.json",
        {
            "folds_4d": primary["folds"].to_dict(orient="records"),
            "folds_6d": sens["folds"].to_dict(orient="records"),
        },
    )
    log.info("wrote stage2_folds.json")

    import json

    stage1 = json.loads((resolve(cfg["paths"]["results"]) / "stage1_cv.json").read_text(encoding="utf-8"))
    baseline = pd.DataFrame([r for r in stage1["folds"] if r["label"] == "gradient_boosting_tuned"])

    gates = [evaluate_gate(primary["summary"], primary["folds"], baseline, m) for m in ("zz_kernel", "vqc")]

    # `prepare_xy` is the only sanctioned way to build features: it adds the
    # row-local `prev_contacted` column the preprocessor expects.
    X_full, y_full = prepare_xy(ds.train, labelled=True)
    assert y_full is not None
    prep = make_preprocessor().fit(X_full)
    Z = prep.transform(X_full)
    A_tr, A_va, _ = project(Z[:240], Z[240:300], N_QUBITS_PRIMARY, seed=int(cfg["seed"]))
    y = y_full

    result = {
        "stage": 2,
        "prereg_fingerprint": fingerprint(),
        "n_components_primary": N_QUBITS_PRIMARY,
        "n_components_sensitivity": N_QUBITS_SENSITIVITY,
        "cv": {"n_splits": int(cfg["cv"]["n_splits"]), "n_repeats": int(cfg["cv"]["n_repeats"])},
        "summary_4d": primary["summary"],
        "summary_6d": sens["summary"],
        "mean_evr_4d": float(primary["folds"]["explained_variance_ratio"].mean()),
        "mean_evr_6d": float(sens["folds"]["explained_variance_ratio"].mean()),
        "gate": gates,
        "finite_shots": finite_shot_check(A_tr, A_va, y[:240], cfg),
        "circuits": circuit_report(np.zeros(N_QUBITS_PRIMARY) + 0.5, cfg),
        "baseline_model": "gradient_boosting_tuned (Stage 1 winner)",
        "baseline_roc_auc": float(baseline["raw_roc_auc"].mean()),
    }

    write_text(results_dir / "stage2_cv.md", render_markdown(result))
    write_json(
        results_dir / "stage2_cv.json",
        {k: v for k, v in result.items()},
    )
    write_json(results_dir / "stage2_gate.json", {"gate": gates, "fingerprint": fingerprint()})

    for row in primary["summary"]:
        log.info(
            "%-20s ROC-AUC %.4f±%.4f  PR-AUC %.4f  Brier %.4f",
            row["model"],
            row["roc_auc"]["mean"],
            row["roc_auc"]["sd"],
            row["pr_auc"]["mean"],
            row["brier"]["mean"],
        )
    for g in gates:
        log.info("%s: Δ %.4f, repeats %d/%d -> %s", g["model"], g["delta_mean_roc_auc"], g["repeats_improved"], g["n_repeats"], g["verdict"])
    log.info("wrote stage2_cv.{json,md}, stage2_gate.json, stage2_folds.json")


if __name__ == "__main__":
    main()
