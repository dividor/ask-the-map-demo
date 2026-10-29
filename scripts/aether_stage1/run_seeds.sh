#!/usr/bin/env bash
# Run the AETHER SatBird alignment experiment with this project's reconstructed configuration files, one seed
# per argument, in the authors' own environment. Logs and run markers go to $AETHER_RUN_DIR.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${ASKMAP_DATA_ROOT:?set ASKMAP_DATA_ROOT in .env}"
: "${AETHER_RUN_DIR:?export AETHER_RUN_DIR (the folder for logs and kept checkpoints)}"

V="$ASKMAP_DATA_ROOT/vendor"
REC="$(pwd)/configs/aether_reconstruction"
mkdir -p "$AETHER_RUN_DIR/best"
for seed in "$@"; do
  echo "=== seed $seed start $(date)" >> "$AETHER_RUN_DIR/runs.txt"
  set +e
  (cd "$V/aether" && PYTHONPATH="$V/aether" "$V/aether/.venv/bin/python" src/train.py \
      experiment=satbird_alignment/aef_s01 hydra.mode=RUN "seed=$seed" "hydra.searchpath=[file://$REC]") \
      > "$AETHER_RUN_DIR/seed_$seed.log" 2>&1
  status=$?
  set -e
  echo "=== seed $seed end $(date) exit $status" >> "$AETHER_RUN_DIR/runs.txt"
done
