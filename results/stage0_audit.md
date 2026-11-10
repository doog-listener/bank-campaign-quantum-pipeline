# Stage 0 audit

- workbook: `bank_campaign_participant_release (1).xlsx`
- sha256: `d2f8e63705af2883c7618c463a6ac2f351119d59a8869d2677f8825148e9a709`
- rows: train [300, 17], test [100, 16], stage2 [400, 5]

## Key integrity (join on `customer_id` only, no id_map)

- test ids: 100 unique = True
- disjoint from train: True
- present in Stage2_Data: 100/100
- Stage2 join: 1:1 (train U test == Stage2 ids)

## Class balance

- 35 yes / 265 no = **11.67%** positive (train only)

## Data quality

- duplicate rows: train 0, test 0, stage2 0
- missing values: train 0, test 0, stage2 0
- range violations: {'train': [], 'test': [], 'stage2': []}

## Leakage checks

- `duration` absent everywhere: {'train': True, 'test': True, 'stage2': True}
- target column placement: {'train_has_y': True, 'test_has_y': False, 'stage2_has_y': False}
- identical feature rows shared train/test: 0
- constant columns: {'train': [], 'test': [], 'stage2': []}
- unseen test levels: {}

## Stage 2 feasibility envelope

- max cost 4 x K 6 = 24 vs budget 25 -> budget_binds=False
- max capacity 4 x K 6 = 24 vs capacity 22 -> capacity_binds=True

## Environment

- {'python': '3.12.13', 'numpy': '2.5.4', 'pandas': '3.1.1', 'scipy': '1.18.1', 'sklearn': '1.9.1', 'qiskit': '2.6.2', 'qiskit_aer': '0.18.2', 'openpyxl': '3.1.5'}
