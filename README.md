# Bank term-deposit campaign — hybrid quantum/classical pipeline

Predict which test customers will subscribe to a term deposit, pick 6 of the top
10 under budget/capacity, then estimate the probability that at least 3 of those
6 subscribe.

**Status: Stages 0–6 complete.** Stage 0: environment, data, key integrity, audit,
frozen pre-registration. Stage 1: 5×10 CV of the registered candidates + nested
sigmoid calibration. Stage 2: fidelity kernel, VQC, ablations, admission gate —
**quantum rejected**. Stage 3: final fit, the single probability vector, top-10
ranking, bootstrap. Stage 4: 20-qubit QUBO, exact brute force, standard and Dicke/XY
QAOA, bootstrap re-solves. Stage 5: 6-qubit outcome circuit, exact
Poisson-binomial, real shot/noise/mitigation runs. Stage 6: generated traceability
matrix, release manifest and `submission.zip`.

**Deliverable: {C0081, C0031, C0079, C0170, C0327, C0182}, expected net value
39.7849, P(≥3 subscribe) = 0.567655.** Slides in `deck/slides.pdf`, bundle in
`submission.zip`, one-command reproduction via `./reproduce.sh`.

## Input data

`bank_campaign_participant_release (1).xlsx` — sheets `README`, `Stage1_Train`
(300 rows, labelled), `Stage1_Test` (100 rows, no labels), `Stage2_Data`
(400 rows of cost/value/capacity/segment), `Stage2_Constraints`.

The file on disk is named `...release (1).xlsx`; the prompt's
`bank_campaign_participant_release__1_.xlsx` does not exist. `configs/config.yaml`
points at the real file and every run logs its SHA-256.

## Join key policy (no `id_map`)

`customer_id` is the only join key, everywhere. There is **no `id_map` module and
no positional index arithmetic** — nothing in this repo may rely on row order, so
a re-shuffled export cannot silently change a result. `src/data.py::assert_key_integrity`
replaces that indirection with three assertions, re-run in `tests/`:

1. **unique** — 100 test ids, all distinct; 300 train and 400 Stage 2 ids likewise;
2. **disjoint** — train ∩ test = ∅;
3. **joinable** — every test id appears in `Stage2_Data`, and `Stage2_Data` holds
   exactly one row per id in train ∪ test (a 1:1 join, verified with
   `pandas.merge(..., validate="one_to_one")`).

Violations raise `KeyIntegrityError` listing *all* problems at once. The guard is
itself tested against corrupted frames (duplicate id, leaked overlap, orphaned
Stage 2 row, wrong row count) so a green suite means the check actually fires.

## Environment

System Python here is 3.14 (PEP 668 managed; `qiskit`/`sklearn` absent), so the
project runs in a local `uv` venv on CPython **3.12**:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Pinned: qiskit 2.6.2, qiskit-aer 0.18.2, scikit-learn 1.9.1, pandas 3.1.1,
numpy 2.5.4, scipy 1.18.1. Full freeze in `requirements.txt`, run-time log in
`results/stage0_env.txt`.

## Run the pipeline (Stage 0 -> Stage 4; each stage is independent and re-runnable)

```bash
.venv/bin/python -m src.data      # audit -> results/stage0_audit.{json,md} + results/stage0_env.txt
.venv/bin/python -m src.eval      # freeze rule -> results/stage0_preregistration.json, PREREGISTRATION.md
.venv/bin/python -m src.classical  # Stage 1 CV -> results/stage1_cv.{json,md}   (~80 s)
.venv/bin/python -m src.quantum_clf # Stage 2 CV -> results/stage2_*.{json,md}  (~5 min)
.venv/bin/python -m src.classical --finalise  # Stage 3 -> stage1_results.csv + bootstrap (~90 s)
.venv/bin/python -m src.qaoa       # Stage 4 QUBO/QAOA -> results/stage2_results.{json,csv} (~7 min)
.venv/bin/python -m src.stage3     # Stage 5 6-qubit circuit -> results/stage3_results.json (~25 s)
.venv/bin/python -m src.report     # Stage 6 traceability + submission.zip (~2 s)
.venv/bin/pytest -q               # 143 invariant tests

./reproduce.sh                    # all of the above, then verifies every artefact
```

`requirements.txt` is the pinned freeze recorded in `results/stage0_env.txt`
(same list, generated from distribution metadata, so re-creating the venv is
reproducible).

## Stage 0 audit results

| Check | Result |
|---|---|
| Shapes | train (300, 17), test (100, 16), Stage2 (400, 5) |
| Test ids | 100, unique, disjoint from train, 100/100 present in Stage 2 |
| Stage 2 join | 1:1 over train ∪ test (400 = 300 + 100) |
| Class balance | 35 yes / 265 no = **11.67 %** positive (train only) |
| Missing values | 0 in all sheets |
| Duplicate rows | 0 (train, test, Stage 2) |
| `duration` | absent from every sheet |
| Unique feature rows shared train↔test | 0 |
| `pdays == -1` (never contacted) | 84.7 % of train → `prev_contacted` flag |
| Budget envelope | 4 × 6 = 24 ≤ 25 → **budget never binds**; capacity 4 × 6 = 24 > 22 **can bind** |

`prev_contacted = (pdays != -1)`; the sentinel `-1` is masked to `NaN` in the
`pdays` column so the in-fold median is computed over real durations only.

## Stage 1 results (5×10 repeated stratified CV, train only, 50 folds)

Each fold fits preprocessing + estimator on its 240 training rows; the sigmoid
calibrator is fitted inside that same fold (never on validation or test rows).
Precision/recall@10 % use the top `round(0.1 × 60) = 6` rows of each fold.

| config | raw ROC-AUC | cal ROC-AUC | raw PR-AUC | cal PR-AUC | raw Brier | cal Brier | raw P@10 % | raw R@10 % |
|---|---|---|---|---|---|---|---|---|
| LR, C = 0.1 | 0.5761 ± 0.0913 | 0.5389 ± 0.1099 | 0.2283 | 0.2102 | 0.1029 | 0.1031 | 0.1867 | 0.1600 |
| RF, N = 100, d = 5 | 0.6301 ± 0.0899 | 0.6157 ± 0.0983 | 0.2695 | 0.2648 | 0.1005 | 0.1015 | 0.2433 | 0.2086 |
| GB, N = 50, d = 3 | 0.6210 ± 0.0916 | 0.5935 ± 0.1030 | 0.2647 | 0.2596 | 0.1077 | 0.1013 | 0.2333 | 0.2000 |
| **GB tuned, N = 100, d = 2, η = 0.03** | **0.6463 ± 0.0781** | 0.5931 ± 0.0910 | **0.2984** | 0.2648 | **0.0997** | 0.1010 | **0.2700** | **0.2314** |

Pre-registered decision (`results/stage1_cv.json` → `decision`): **gradient
boosting**, tuned config, **raw probabilities shipped** — sigmoid calibration
lowers Brier for RF/GB but raises it for the winner (0.0997 → 0.1010), so the
pre-registered fallback (ship raw) applies. `ship_calibrated = false` is a
decision recorded before any test row was loaded for prediction.

## Stage 2 results (quantum classifiers vs ablations, same 50 folds)

Every model sees the same fold-local `PCA(4)` projection scaled into `[0, π]`
from train bounds only (mean explained variance 0.514). `gb_on_pca` is the extra
control both earlier drafts asserted without testing: the Stage 1 winner's
exact architecture restricted to the same 4-D features.

| model (4-D in-fold PCA) | ROC-AUC | PR-AUC | Brier | P@10 % | R@10 % |
|---|---|---|---|---|---|
| **GB on the same 4-D features** | **0.5431 ± 0.0943** | 0.1959 | 0.1063 | 0.1600 | 0.1371 |
| data-reuploading VQC (16 params, COBYLA) | 0.5691 ± 0.1164 | 0.2063 | 0.2444 | 0.1533 | 0.1314 |
| ZZ fidelity kernel + SVC | 0.5130 ± 0.1375 | 0.1877 | 0.1042 | 0.1400 | 0.1200 |
| unentangled kernel + SVC (ablation) | 0.4794 ± 0.1240 | 0.1659 | 0.1041 | 0.1200 | 0.1029 |
| classical RBF SVM, same features (ablation) | 0.4542 ± 0.1250 | 0.1566 | 0.1037 | 0.1267 | 0.1086 |
| *Stage 1 winner, full 17 features (reference)* | *0.6463 ± 0.0781* | *0.2984* | *0.0997* | *0.2700* | *0.2314* |

6-D sensitivity (pre-registered range is 4–6): ZZ 0.4995, unentangled 0.5181,
RBF 0.5115, VQC 0.5726, GB-on-6D 0.5425.

**Pre-registered gate** (`results/stage2_gate.json`), paired on identical folds:

| model | Δ ROC-AUC vs GB | paired SEM | folds won | repeats won | admitted |
|---|---|---|---|---|---|
| ZZ kernel | −0.1332 | 0.0237 | 9/50 | 0/10 | ✗ |
| VQC | −0.0772 | 0.0199 | 13/50 | 0/10 | ✗ |

Both fail the rule's first criterion by an order of magnitude more than its
margin, so **no quantum score enters Stage 3**; the classical GB supplies the
probability vector. Where the loss comes from, measured rather than asserted:
the *same* GB model drops 0.6463 → 0.5431 when restricted to 4-D PCA, i.e. the
projection alone accounts for 0.103 = **78 %** of the ZZ kernel's 0.133 gap.
Entanglement is worth +0.0336 (ZZ vs unentangled, paired SEM 0.0241, 28/50
folds) — suggestive, not significant, and quantified honestly rather than
claimed. On the constrained 4-D representation the VQC is the best model
(+0.0260 over GB-on-4D, paired SEM 0.0178, 30/50 folds), which is the fair
statement of what the quantum model did and did not achieve.

Resource accounting (`basis {rz,sx,x,cx}`, linear chain, 0 SWAPs): ZZ map 4
qubits, logical depth 5 (8 1q / 3 RZZ), transpiled depth 12; VQC 4 qubits, 16
trainable angles, logical depth 9 (16 1q / 6 CNOT), transpiled depth 13.
Finite shots (8192, Aer statevector): max |exact − sampled| = 0.0112 for VQC
P(q₀=1), 0.0081 for kernel entries via compute–uncompute.

Caveats recorded deliberately: the VQC's Brier/loss are **not comparable** to the
other rows because it is trained on class-weighted BCE (its probabilities are
deliberately shifted); ROC-AUC/PR-AUC are the fair comparison. `SVC(probability=True)`
is deprecated in scikit-learn 1.9 but retained because the environment is pinned
to 1.9.1 and the alternative changes the calibration path mid-protocol. The ZZ
map follows Qiskit's canonical `ZZFeatureMap` (RZZ angle `2(π−xᵢ)(π−xⱼ)`); the
half-angle variant `(π−xᵢ)(π−xⱼ)` used by an earlier draft scores 0.5000 on these
same folds, which is why that draft's kernel ranking differs from this one.

## Stage 3 results (final fit, one probability vector, top 10)

The Stage 1 winner (`gradient_boosting_tuned`, refit on all 300 rows, **raw
probabilities** because Stage 1 measured calibration as a Brier regression) scores
the 100 test rows once. Ranking is `predicted_probability` descending, ties broken
by **ascending `customer_id`**.

| rank | customer_id | p̂ | p̂ bootstrap mean ± sd | in top 10 |
|---|---|---|---|---|
| 1 | C0081 | 0.718054 | 0.498 ± 0.242 | 84.0 % |
| 2 | C0031 | 0.703851 | 0.477 ± 0.251 | 81.4 % |
| 3 | C0079 | 0.654671 | 0.501 ± 0.256 | 80.9 % |
| 4 | C0170 | 0.208464 | 0.187 ± 0.092 | 35.6 % |
| 5 | C0327 | 0.203974 | 0.215 ± 0.112 | 47.9 % |
| 6 | C0183 | 0.200777 | 0.235 ± 0.116 | 57.0 % |
| 7 | C0182 | 0.186305 | 0.153 ± 0.083 | 19.7 % |
| 8 | C0193 | 0.178435 | 0.145 ± 0.093 | 13.8 % |
| 9 | C0349 | 0.169156 | 0.180 ± 0.111 | 29.4 % |
| 10 | C0359 | 0.169156 | 0.170 ± 0.101 | 26.8 % |

Three things this table says out loud:

- **The tie-break is load-bearing.** C0349 and C0359 receive the *identical* GB
  score (0.169156 to double precision), so the pre-registered tie-break — not the
  model — decides slot 10 (and both still make the cut). C0359 and C0381 are within
  1.4e-4 of each other, so slot 10 is one rounding away from changing.
- **The bootstrap does not merely widen the error bars, it moves the answer.**
  The three leaders keep a top-10 seat 81–84 % of the time, but every other slot is
  between 14 % and 57 %. C0081's 0.718 is a point estimate whose bootstrap mean is
  0.498: with 35 positives the model is confident about *who is at the top* and not
  at all confident about *how likely* it is.
- **Test features are scored, never fitted.** Resampling moves the 300 training rows
  only; the 100 test rows and their preprocessing are fixed, and the probability
  vector is written once with a SHA-256 (`a8f3766b…`) so later stages can prove they
  consumed exactly this vector.

### A preprocessing fork worth flagging

An earlier draft of this stage reproduced digit-for-digit (max |Δp| = 4.5e-7)
under a different pipeline that keeps `pdays = -1` as a numeric duration and skips
in-fold imputation. That contradicts the Stage 0 feature contract (`pdays == -1` is
a sentinel, masked and imputed in-fold), so this repo keeps the contract, and the
consequence is not cosmetic: it swaps **2 of the 10 candidates** (C0182, C0359 in;
C0206, C0059 out) and moves the Stage 4 optimum from 37.31 to **39.78**.

## Stage 4 results (QUBO, brute force, standard and Dicke/XY QAOA)

Objective `w_i = p_i · v_i − c_i`, cardinality `Σx = 6`, budget `Σc x ≤ 25`,
capacity `Σh x ≤ 22`, with binary-expanded slacks (5 bits each) so both
inequalities are equalities: **20 qubits**, 165 non-zero couplings, penalties
`A=40, B=12, C=12` as frozen in `configs/config.yaml` at Stage 0.

**The penalties are verified, not assumed.** The QUBO ground state (energy
−39.784885) decodes to `1111101000` = {C0081, C0031, C0079, C0170, C0327, C0182},
which is exactly the brute-force optimum over all 210 six-subsets (value 39.784885,
runner-up 1.7105 behind), and both slack registers decode to the exact complement
(13 and 9). The run refuses to continue if those disagree.

### The finding that matters: the constraints are inert

| quantity | bound | worst case over all 210 subsets |
|---|---|---|
| cardinality | = 6 | 6 (always) |
| budget `Σcontact_cost` | ≤ 25 | **15** |
| capacity `Σcapacity_units` | ≤ 22 | **18** |

All 210 subsets are feasible, so the QUBO's feasible set is the whole weight-6
shell and the optimum is simply **the six largest `w_i`** — which is what the
brute force returns. Consequences, stated rather than hidden: the budget and
capacity rows of the Hamiltonian cannot change the answer for this instance, and
*no* optimiser, quantum or classical, can beat a sort here. (An earlier draft
claimed the worst-case cost was 16; it is 15 for this candidate set and 17 for the
draft's own set — either way the conclusion holds, but the number was wrong.)

### QAOA: soft penalties vs constraint-preserving encoding

α is `E[net value] / optimal net value` over the exact output distribution; P(opt)
is the probability of sampling the `x`-register equal to the optimum (slack registers
marginalised out); feasibility is the probability that the sampled `x` satisfies all
three constraints. 8 192 shots are also drawn to confirm the distributions.

| solver | p | qubits | transpiled depth / CX | feasibility | P(opt) | α | most probable `x` |
|---|---|---|---|---|---|---|---|
| standard, soft penalties | 1 | 20 | 104 / 330 | 21.7 % | 1.4e-4 | 0.549 | `101010` |
| standard, soft penalties | 2 | 20 | 169 / 660 | 19.6 % | 1.8e-4 | 0.540 | `111100` |
| standard, soft penalties | 3 | 20 | 234 / 990 | 19.9 % | 2.2e-4 | 0.542 | `1100100001` |
| standard, soft penalties | 4 | 20 | 299 / 1320 | 23.4 % | 7.3e-5 | 0.548 | `1011010110` |
| standard, soft penalties | 5 | 20 | 364 / 1650 | 20.0 % | 9.8e-4 | 0.546 | `101011100` |
| **Dicke(6,10) + XY ring** | 1 | 10 | 111 / 40 | **100 %** | 0.0044 | 0.729 | `1011100101` |
| **Dicke(6,10) + XY ring** | 2 | 10 | 222 / 80 | **100 %** | 0.0040 | 0.765 | `1110100110` |
| **Dicke(6,10) + XY ring** | 3 | 10 | 333 / 120 | **100 %** | 0.0175 | 0.813 | `1010111010` |
| **Dicke(6,10) + XY ring** | 4 | 10 | 444 / 160 | **100 %** | 0.0759 | **0.832** | **`1111101000`** |
| **Dicke(6,10) + XY ring** | 5 | 10 | 555 / 200 | **100 %** | **0.101** | 0.817 | **`1111101000`** |

Reading it honestly:

- The **soft-penalty 20-qubit QUBO never gets above ~23 % feasibility** and its most
  probable `x` is a *3-customer* selection — the ansatz spends its amplitude on
  low-penalty states, and more depth does not fix it (energy 2008 → 1460 → 1640 →
  2138 → 2554 against a ground state of −39.8; the landscape is penalty-dominated
  and the COBYLA runs are shopping in the wrong basin).
- The **Dicke/XY variant holds `Σx = 6` by construction** — exact support on the
  weight-6 shell to 1e-12 at every p, 100 % feasibility — and by p=4 its single most
  probable bitstring *is* the optimum, with 838 of 8 192 shots (10.1 %) landing on it
  at p=5. α improves 0.729 → 0.817 and the energy approaches −33.1 against −39.8.
- Non-monotonicity is reported as measured: p=5 is slightly worse than p=4 on energy
  (both restarts hit the 200-iteration cap), so the p=5 row is a lower bound on what
  the circuit can do, not a converged answer.
- **None of this is quantum advantage.** The instance is solved exactly by a 210-way
  enumeration and by sorting six numbers. QAOA here is a well-characterised
  sampler on a problem whose structure (inert constraints) makes the answer trivial.

Verification and resources: the numpy engines are checked against Qiskit/Aer on the
full problems — standard p=1 max |ΔP| = 2.6e-16 (TVD 9.8e-13), Dicke p=1 max
|ΔP| = 4.5e-16. The Dicke start state is prepared with `initialize` (arbitrary state
preparation, reported separately from the p-scaling depth); a hardware-realizable
replacement is the linear-depth Dicke preparation, which this project does **not**
claim to implement.

### Robustness: the uncertainty that dominates is upstream

Re-solving the whole Stage 4 problem (candidate set *and* selection) on 200
bootstrap probability vectors:

| quantity | value |
|---|---|
| candidate set identical to the primary top-10 | 2.5 % |
| mean overlap with the primary optimum | 0.58 |
| re-solve reproduces the same optimum | 0.5 % |
| regret of the frozen selection (mean / p95) | 10.1 / 32.8 value units |

The frozen six are the right answer for the point estimate and a ~25 %-regret answer
under model uncertainty — a far larger effect than every shot-noise and QAOA term
combined. That is the project's main negative result and it is measurable.

## Stage 5 results (6-qubit outcome circuit)

One `Ry(θ_i)` per selected customer with `θ_i = 2·arcsin√p_i`, so `P(q_i = |1⟩) = p_i`.
Independence across customers is an **assumption**, not a finding: it is what makes
the exact answer a Poisson-binomial and the circuit a product state.

| qubit | customer | p̂ | θ (rad) | θ (deg) |
|---|---|---|---|---|
| q0 | C0081 | 0.718054 | 2.022065 | 115.86 |
| q1 | C0031 | 0.703851 | 1.990732 | 114.06 |
| q2 | C0079 | 0.654671 | 1.885297 | 108.02 |
| q3 | C0170 | 0.208464 | 0.948291 | 54.33 |
| q4 | C0327 | 0.203974 | 0.937194 | 53.70 |
| q5 | C0182 | 0.186305 | 0.892599 | 51.14 |

**Answer: P(≥3 of the 6 subscribe) = 0.567655 (56.77 %)** — 6 qubits, 6 `Ry` gates,
**0 two-qubit gates, 0 SWAPs** (logical depth 1; transpiled to `{rz,sx,x,cx}`:
depth 4, 24 one-qubit gates). The measured marginal on every qubit equals `p_i` to
1e-12 and the full distribution matches the closed-form Poisson-binomial to 2.2e-16,
so the angle convention and the little-endian bit mapping are both pinned down.

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| P(K=k) | 0.014783 | 0.111877 | 0.305685 | 0.358935 | 0.170955 | 0.035144 | 0.002621 |

### Every row below is a real simulation

| method | shots | P̂(≥3) | abs. error | note |
|---|---|---|---|---|
| exact Poisson-binomial | ∞ | **0.567655** | 0 | reference |
| ideal statevector | ∞ | **0.567655** | 2e-16 | circuit, no sampling |
| Monte Carlo | 10^6 | 0.567775 | 1.2e-4 | seeded numpy |
| Aer sampling | 1 000 | 0.553000 | 1.5e-2 | Wilson [0.5220, 0.5836] |
| Aer sampling | 8 000 | 0.580875 | 1.3e-2 | Wilson [0.5700, 0.5916] — a 2.4σ draw, kept as measured |
| Aer sampling | 100 000 | 0.567060 | 5.9e-4 | Wilson [0.5640, 0.5701] |
| noisy Aer (depol 0.1 % + readout 1 %/2 %), 20 repeats | 8 000 | 0.5570 ± 0.0049 | bias −0.0106 | unmitigated |
| readout-mitigated (64×64 measured assignment matrix, NNLS), 20 repeats | 8 000 | 0.5664 ± 0.0059 | bias −0.0012 | 89 % of the readout bias removed |

The readout bias is real and small, mitigation recovers most of it, and both are
reported over 20 independent seeds rather than one lucky draw. The 8 000-shot point
is deliberately left as measured: re-seeding until it lands near 0.5677 would be
fabrication, not reproduction.

### Sensitivity and where the uncertainty lives

`dP(≥3)/dp_i` has a closed form here — it is `P(sum of the other five = 2)` — so no
finite differences and no step-size choice:

| customer | C0081 | C0031 | C0079 | C0170 | C0327 | C0182 |
|---|---|---|---|---|---|---|
| dP/dp | +0.4137 | +0.4163 | **+0.4249** | +0.3503 | +0.3492 | +0.3451 |

| propagation of Stage 1 uncertainty | mean | sd | 95 % interval |
|---|---|---|---|
| frozen selection (six keep their seats) | 0.3486 | 0.2180 | [0.027, 0.748] |
| fully re-solved selection (1000 draws) | 0.5046 | 0.2025 | [0.147, 0.853] |
| shot noise at 8 192 shots | — | 0.0055 | — |

Model uncertainty is **~40×** shot noise, and the frozen-selection mean sits *below*
the point estimate because refitting on bootstrap draws systematically shrinks the
extreme probabilities (0.718 → 0.498 for C0081). The point estimate is the answer the
pre-registration demands — one probability vector, one computation — and the
intervals above are what the uncertainty actually looks like.

### Deviations from the earlier draft of this stage

The draft's arithmetic is right: PMF, P(≥3) = 0.567655, the six sensitivities and all
six Ry angles reproduce to 1e-6 on the corrected selection. Its *simulation* rows are
not simulations — the "finite-shot" values are computed as `round(p_exact × shots)`
and the noisy/mitigated rows have no circuit behind them (there is no noise model and
no calibration matrix in that code). Measured here, the 8 000-shot draw is 0.5809, not
0.5676, and the transpiled circuit is depth 4 with 24 one-qubit gates, not "depth 1,
6 gates" — `Ry` is not a basis gate.

## Stage 6 results (traceability and release bundle)

`python -m src.report` rebuilds the cross-stage matrix **from the artefacts** and
refuses to write a bundle if the stages disagree (Stage 4's candidate order, Stage 5's
qubit order, the probabilities, the angles, `P(≥3)` and the expected net value are all
re-derived and compared). A hand-typed matrix is the easiest place in a five-stage
pipeline to publish a number no stage produced.

| # | id | p̂ | c | v | h | net | Stage 4 | bit | qubit | Ry | bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C0081 | 0.718054 | 4 | 17 | 1 | +8.2069 | **selected** | 0 | q0 | 2.0221 | 0.498 ± 0.242 |
| 2 | C0031 | 0.703851 | 2 | 13 | 1 | +7.1501 | **selected** | 1 | q1 | 1.9907 | 0.477 ± 0.251 |
| 3 | C0079 | 0.654671 | 3 | 24 | 3 | +12.7121 | **selected** | 2 | q2 | 1.8853 | 0.501 ± 0.256 |
| 4 | C0170 | 0.208464 | 1 | 22 | 1 | +3.5862 | **selected** | 3 | q3 | 0.9483 | 0.187 ± 0.092 |
| 5 | C0327 | 0.203974 | 1 | 25 | 4 | +4.0994 | **selected** | 4 | q4 | 0.9372 | 0.215 ± 0.112 |
| 6 | C0183 | 0.200777 | 2 | 14 | 3 | +0.8109 | rejected | — | — | — | 0.235 ± 0.116 |
| 7 | C0182 | 0.186305 | 1 | 27 | 3 | +4.0302 | **selected** | 5 | q5 | 0.8926 | 0.153 ± 0.083 |
| 8 | C0193 | 0.178435 | 3 | 16 | 1 | −0.1450 | rejected | — | — | — | 0.145 ± 0.093 |
| 9 | C0349 | 0.169156 | 1 | 17 | 3 | +1.8757 | rejected | — | — | — | 0.180 ± 0.111 |
| 10 | C0359 | 0.169156 | 1 | 14 | 2 | +1.3682 | rejected | — | — | — | 0.170 ± 0.101 |

The bundle (`submission.zip`, 50 files, ≈1.36 MiB) carries `src/`, `tests/`,
`configs/`, `deck/`, every `results/` artefact, `README.md`, `reproduce.sh` and the
workbook. It is deterministic — fixed timestamps, sorted entry list, and the manifest
that describes it is *not* inside it (a self-referential hash would change on every
rebuild) — so the SHA-256 is identical across rebuilds. `.venv`, caches and the zip
itself are excluded and are listed, with each file's size and hash, in
`results/release_manifest.json` beside the zip.

### Corrections to the drafted release package

- The drafted matrix put **C0182 at rank 6 and C0183 at rank 10 with `p̂ = 0.169000`**.
  The real ranking has C0183 at rank 6 (`p̂ = 0.200777`, net +0.8109, *rejected*) and
  C0182 at rank 7 (`p̂ = 0.186305`, net +4.0302, **selected**). 0.169000 is not any
  published probability; it is C0349/C0359 rounded off. Generated, it cannot drift.
- The drafted tree/file list named modules that do not exist (`main.py`,
  `stage2_qaoa_benchmark.json`, `stage5_circuit_specs.json`) and put the result files
  at the repo root, while this repo writes them to `results/`. The reproduce script
  here follows the real layout.
- Requirements are pinned exactly (`qiskit==2.6.2`, `qiskit-aer==0.18.2`, …) rather
  than `qiskit>=2.0.0`; a floating constraint would break the "re-run gives identical
  numbers" claim the rest of the report depends on.
- The drafted deck quoted a 36× ratio; measured, it is 39.8× for the frozen selection
  and 36.8× for the re-solved one. Both are in `results/stage3_results.json`.

## Validation strategy (hard rules)

- Test features are never fitted on: `src/features.py::make_preprocessor()`
  returns an **unfitted** transformer, and derived features are row-local pure
  functions. Nothing learned (median, mode, category vocabulary, scaler) exists
  outside a fold-fitted object. Unit-tested.
- One-vs-all candidate comparison is repeated stratified 5×10 CV on train only.
- Model selection is pre-registered in `src/eval.py::PREREGISTERED_SELECTION`,
  frozen at Stage 0 with a SHA-256 fingerprint recorded in
  `results/stage0_preregistration.json` and asserted by
  `tests/test_preregistration.py`. Primary metric: mean CV ROC-AUC; ties →
  PR-AUC → Brier → simpler model → fixed candidate order.
- Quantum contributions may enter only as a stacked score, and only if they clear
  the pre-registered bar (Δ ROC-AUC ≥ +0.01 **and** improvement in ≥ 60 % of
  repeats). Otherwise they are reported as not helping. No quantum-advantage
  claim is made anywhere.
- One probability vector feeds Stages 2 and 3; the top 10 is never edited after
  ranking. Ties break on ascending `customer_id`.
- Seeds derive from a single integer (`seed: 42`) via
  `utils.seed_everything` / `utils.child_seeds` / `utils.qiskit_seed`.

## Layout

```
PREREGISTRATION.md           frozen selection rule + fingerprint (generated)
configs/config.yaml          seeds, paths, CV geometry, penalty + shots config
src/data.py                  load, key integrity, audit         (Stage 0)
src/features.py              feature contract, fold-local preprocessing (Stage 0)
src/eval.py                  pre-registered selection rule      (Stage 0)
src/classical.py             LR / RF / GB + 5x10 CV + calibration (Stage 1);
                             final fit, ranking, bootstrap            (Stage 3)
src/quantum_clf.py           fidelity kernel, VQC, ablations, shots (Stage 2)
src/qubo.py                  QUBO + brute force ground truth    (Stage 4)
src/qaoa.py                  standard + Dicke/XY QAOA           (Stage 4)
src/stage3.py                6-qubit P(>=3) circuit, noise, mitigation (Stage 5)
src/report.py                traceability matrix, release manifest, zip (Stage 6)
reproduce.sh                 one-command end-to-end run + artefact verification
deck/slides.html|.pdf        10-slide deck (self-contained HTML, rendered to PDF)
tests/                       key integrity, data contract, preregistration, stages 1–6 (143 tests)
notebooks/, results/         exploration, artefacts (audit + env record)
```

Stage 3/4 artefacts in `results/`: `stage1_results.csv`
(`customer_id, predicted_probability, rank, top10_flag`, 100 rows, ranked),
`stage3_meta.json` (decision provenance, probability hash, top-10, bootstrap stats),
`stage3_bootstrap_probs.npz` (1000 x 100, **test sheet order**),
`stage2_results.json` (full Stage 4 record) and `stage2_results.csv` (one row per
solver/depth), `stage3_results.json` (Stage 5 answer, every backend, propagation).

## Assumptions

1. `y == "yes"` is the positive class; `no` is the negative.
2. `pdays == -1` means "never contacted" and is not a real duration.
3. `contact_cost`, `capacity_units`, `expected_deposit_value` are point
   estimates in the same units as the README's net-value formula.
4. On the realised top 10, **both** Stage 4 rows are inert: worst-case cost 15 ≤ 25
   and worst-case capacity 18 ≤ 22 over all 210 subsets, so the constrained optimum
   equals the unconstrained top-6-by-weight. The QUBO is still built and verified as
   specified, but the constraints cannot bite for these candidates.
5. Test labels do not exist in the release; no accuracy on test can be reported.
6. The Stage 4 bootstrap treats a resample as a different plausible world (re-ranking
   the 100 test rows and re-solving); it is not a confidence statement about the
   released test set, which has no labels.
7. Stage 5 assumes **independent** subscription decisions across the six customers.
   That is what makes the outcome distribution a Poisson-binomial; any real
   correlation (same branch, same month, same campaign wave) would widen it.

## Limitations

- 35 positive training rows: every CV metric has wide uncertainty, which is why
  bootstrap error propagation is planned for Stage 3/5.
- A single 300-row train set of one campaign snapshot; no temporal validation, so
  drift is unmeasured.
- Qiskit 2.6.2 / Aer 0.18.2 are simulator versions only — no hardware was used. No
  noise model is applied in Stage 4; the Stage 4 circuits are evaluated exactly on
  the statevector.
- QAOA is optimised with COBYLA, 2 seeded restarts, 200 iterations. Deeper circuits
  are iteration-capped (both p=5 restarts hit the cap), so the p=5 numbers are lower
  bounds on solver quality, not convergence evidence.
