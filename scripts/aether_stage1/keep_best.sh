#!/usr/bin/env bash
# Copy each seed's best checkpoint aside as soon as its log reports it (the seeds share checkpoint file names).
# Arguments: the seeds to watch. Runs alongside run_seeds.sh.
set -uo pipefail
: "${AETHER_RUN_DIR:?export AETHER_RUN_DIR}"
R="$AETHER_RUN_DIR"
mkdir -p "$R/best"
for seed in "$@"; do
  until grep -q "Best ckpt path" "$R/seed_$seed.log" 2>/dev/null; do
    if grep -q "=== seed $seed end" "$R/runs.txt" 2>/dev/null; then break; fi
    sleep 5
  done
  p=$(grep "Best ckpt path" "$R/seed_$seed.log" 2>/dev/null | tail -1 \
      | sed -E 's/.*Best ckpt path: ([^ ]+\.ckpt).*/\1/' | sed 's/\x1b\[[0-9;]*m//g')
  if [ -n "$p" ] && [ -f "$p" ]; then
    cp "$p" "$R/best/seed_$seed.ckpt" && echo "seed $seed: kept $p"
  else
    echo "seed $seed: no checkpoint found ($p)"
  fi
done
