#!/usr/bin/env bash
# Stage 1 end to end: the three seeds (skipping any that already finished), their best checkpoints, a check that
# every seed finished, then notebook 09. Needs scripts/aether_stage1/setup.sh first and AETHER_RUN_DIR exported.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${AETHER_RUN_DIR:?export AETHER_RUN_DIR (the folder for logs and kept checkpoints)}"
R="$AETHER_RUN_DIR"
mkdir -p "$R/best"
SEEDS=(12345 1 2)
todo=()
for s in "${SEEDS[@]}"; do
  if grep -q "=== seed $s end .* exit 0" "$R/runs.txt" 2>/dev/null && [ -f "$R/best/seed_$s.ckpt" ]; then
    echo "seed $s already done"
  else
    todo+=("$s")
  fi
done
echo "stage1 start $(date); seeds to run: ${todo[*]:-none}"
if [ ${#todo[@]} -gt 0 ]; then
  scripts/aether_stage1/keep_best.sh "${todo[@]}" > "$R/keep_best.log" 2>&1 &
  KB=$!
  scripts/aether_stage1/run_seeds.sh "${todo[@]}"
  wait $KB
  cat "$R/keep_best.log"
fi
for s in "${SEEDS[@]}"; do
  grep -q "=== seed $s end .* exit 0" "$R/runs.txt" || { echo "STOP: seed $s did not finish with exit 0"; exit 1; }
  [ -f "$R/best/seed_$s.ckpt" ] || { echo "STOP: missing best checkpoint for seed $s"; exit 1; }
done
cp scripts/aether_stage1/r1b_export.py "$R/r1b_export.py"
echo "== notebook 09 start $(date)"
set -a; source .env; set +a
export AETHER_RUN_DIR="$R"
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  notebooks/09_aether_reconstruction.ipynb
echo "== notebook 09 done $(date)"
