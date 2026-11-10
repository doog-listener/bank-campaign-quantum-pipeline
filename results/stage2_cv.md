# Stage 2 -- quantum classifiers vs ablations (50 folds, in-fold PCA)

pre-registration fingerprint: `34ac751dbdd2488c6f08a3b05fc2e0b33a9c1346627732a857b06d32ed835cdc`

## 4-D in-fold PCA (primary specification)

| model | ROC-AUC | PR-AUC | Brier | P@10 % | R@10 % |
|---|---|---|---|---|---|
| zz_kernel | 0.5130 ± 0.1375 | 0.1877 | 0.1042 | 0.1400 | 0.1200 |
| unentangled_kernel | 0.4794 ± 0.1240 | 0.1659 | 0.1041 | 0.1200 | 0.1029 |
| rbf_svm | 0.4542 ± 0.1250 | 0.1566 | 0.1037 | 0.1267 | 0.1086 |
| vqc | 0.5691 ± 0.1164 | 0.2063 | 0.2444 | 0.1533 | 0.1314 |
| gb_on_pca | 0.5431 ± 0.0943 | 0.1959 | 0.1063 | 0.1600 | 0.1371 |

mean train-fold explained variance of 4 components: 0.514

## 6-D sensitivity (pre-registered range is 4-6)

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| zz_kernel | 0.4995 ± 0.1153 | 0.1638 | 0.1058 |
| unentangled_kernel | 0.5181 ± 0.1411 | 0.1835 | 0.1040 |
| rbf_svm | 0.5115 ± 0.1289 | 0.1781 | 0.1034 |
| vqc | 0.5726 ± 0.1151 | 0.2158 | 0.2449 |
| gb_on_pca | 0.5425 ± 0.1197 | 0.2048 | 0.1074 |

## Pre-registered admission gate

- **zz_kernel**: mean ROC-AUC 0.5130 vs baseline 0.6463 (Δ -0.1332); improved in 9/50 folds and 0/10 repeats → **did not help -- excluded from Stage 3**
- **vqc**: mean ROC-AUC 0.5691 vs baseline 0.6463 (Δ -0.0772); improved in 13/50 folds and 0/10 repeats → **did not help -- excluded from Stage 3**

Verdict is mechanical: the rule needs Δ ≥ +0.01 *and* strict improvement in ≥ 60 % of the 10 repeats. Both quantum models fail on the first criterion by a wide margin, so quantum scores are excluded from Stage 3 -- the classical winner supplies the probability vector.

## Finite-shot check

- backend: qiskit_aer StatevectorSimulator (shots for sampling), shots = 8192, pairs = 20
- VQC P(q0=1): max |exact - sampled| = 0.0112 (mean 0.0063)
- ZZ kernel entries (compute-uncompute): max |exact - sampled| = 0.0081 (mean 0.0034)

## Circuits

- zz_feature_map: 4 qubits, logical depth 5 (8 1q / 3 2q), transpiled depth 12 in ['rz', 'sx', 'x', 'cx'], swaps 0, linear chain q0-q1-...-q(n-1)
- vqc: 4 qubits, logical depth 9 (16 1q / 6 2q), transpiled depth 13 in ['rz', 'sx', 'x', 'cx'], swaps 0, linear chain q0-q1-...-q(n-1)

- VQC trainable parameters: 16
