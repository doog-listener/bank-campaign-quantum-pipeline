#!/usr/bin/env bash
# Full reproduction: Stage 0 -> Stage 6, then the assertion suite.
#
# Order matters only for the artefacts it consumes (each stage reads the previous
# stage's output), never for correctness: every stage is independently re-runnable
# and every stochastic step is seeded from configs/config.yaml.
#
# Usage:  ./reproduce.sh            # full pipeline (~10 min)
#         SKIP_TESTS=1 ./reproduce.sh
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: $PY not found. Create the environment first:" >&2
  echo "  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2
  exit 1
fi

echo "== interpreter =="
"$PY" -c "import sys, qiskit, sklearn, pandas, numpy, scipy; print(sys.version.split()[0], 'qiskit', qiskit.__version__)"

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  echo "== tests =="
  "$PY" -m pytest -q
fi

echo "== stage 0: audit + frozen pre-registration =="
"$PY" -m src.data
"$PY" -m src.eval

echo "== stage 1: classical CV (50 folds) =="
"$PY" -m src.classical

echo "== stage 2: quantum classifiers + admission gate =="
"$PY" -m src.quantum_clf

echo "== stage 3: final fit, one probability vector, bootstrap =="
"$PY" -m src.classical --finalise

echo "== stage 4: QUBO + QAOA =="
"$PY" -m src.qaoa

echo "== stage 5: 6-qubit outcome circuit =="
"$PY" -m src.stage3

echo "== stage 6: traceability + release bundle =="
"$PY" -m src.report

echo "== verify artefacts =="
for f in \
  results/stage0_audit.json \
  results/stage1_cv.json \
  results/stage2_cv.json \
  results/stage1_results.csv \
  results/stage3_bootstrap_probs.npz \
  results/stage2_results.json \
  results/stage2_results.csv \
  results/stage3_results.json \
  results/traceability.csv \
  results/traceability.md \
  results/release_manifest.json \
  submission.zip
do
  test -s "$f" || { echo "missing or empty: $f" >&2; exit 1; }
done

echo "== answer =="
"$PY" - <<'PY'
import json
from pathlib import Path

stage5 = json.loads(Path("results/stage3_results.json").read_text())
stage4 = json.loads(Path("results/stage2_results.json").read_text())
print("selected:", ", ".join(stage5["selected_customers"]))
print("expected net value:", round(stage5["expected_net_value"], 4))
print("P(>=3 subscribe):", round(stage5["answer"]["p_at_least_threshold_exact"], 6))
print("brute-force optimum matches QUBO:", stage4["qubo"]["matches_brute_force"])
PY

echo "done."
