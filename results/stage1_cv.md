# Stage 1 -- classical baseline (5x10 repeated stratified CV, train only)

pre-registration fingerprint: `34ac751dbdd2488c6f08a3b05fc2e0b33a9c1346627732a857b06d32ed835cdc`

| config | raw ROC-AUC | cal ROC-AUC | raw PR-AUC | cal PR-AUC | raw Brier | cal Brier | raw P@10% | cal P@10% | raw R@10% | cal R@10% |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression_l2 | 0.5761 ± 0.0913 | 0.5389 ± 0.1099 | 0.2283 ± 0.0956 | 0.2102 ± 0.0966 | 0.1029 ± 0.0044 | 0.1031 ± 0.0016 | 0.1867 ± 0.1414 | 0.1733 ± 0.1466 | 0.1600 ± 0.1212 | 0.1486 ± 0.1257 |
| random_forest | 0.6301 ± 0.0899 | 0.6157 ± 0.0983 | 0.2695 ± 0.0919 | 0.2648 ± 0.0975 | 0.1005 ± 0.0050 | 0.1015 ± 0.0024 | 0.2433 ± 0.1269 | 0.2200 ± 0.1280 | 0.2086 ± 0.1088 | 0.1886 ± 0.1097 |
| gradient_boosting | 0.6210 ± 0.0916 | 0.5935 ± 0.1030 | 0.2647 ± 0.0925 | 0.2596 ± 0.0952 | 0.1077 ± 0.0121 | 0.1013 ± 0.0029 | 0.2333 ± 0.1506 | 0.2367 ± 0.1351 | 0.2000 ± 0.1291 | 0.2029 ± 0.1158 |
| gradient_boosting_tuned | 0.6463 ± 0.0781 | 0.5931 ± 0.0910 | 0.2984 ± 0.0948 | 0.2648 ± 0.0973 | 0.0997 ± 0.0081 | 0.1010 ± 0.0031 | 0.2700 ± 0.1342 | 0.2400 ± 0.1474 | 0.2314 ± 0.1150 | 0.2057 ± 0.1263 |

## Pre-registered decision

- ranked models (ROC-AUC): logistic_regression_l2 0.5761, random_forest 0.6301, gradient_boosting 0.6463
- **selected model: `gradient_boosting`** (config `gradient_boosting_tuned`)
- params: `{'clf__n_estimators': 100, 'clf__max_depth': 2, 'clf__learning_rate': 0.03}`
- calibration: ship raw -- sigmoid hurts Brier on train CV -> ship raw probabilities (prereg fallback)

## Winner metrics (mean ± SD over 50 folds, uncalibrated)

- roc_auc: 0.6463 ± 0.0781
- pr_auc: 0.2984 ± 0.0948
- brier: 0.0997 ± 0.0081
- log_loss: 0.3489 ± 0.0253
- precision_at_10pct: 0.2700 ± 0.1342
- recall_at_10pct: 0.2314 ± 0.1150

Notes: 50 folds = 5 splits x 10 repeats; each fold's validation slice is 60 rows and the top-10% slice is 6 rows, so Precision@10% moves in steps of 1/6. Every transformer and the sigmoid calibrator are fitted on the fold's 240 training rows only.
