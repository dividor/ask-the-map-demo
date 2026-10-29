#!/usr/bin/env bash
# Fetch the AETHER authors' code and data and build their environment under $ASKMAP_DATA_ROOT/vendor/.
# Everything here is read before it is run; nothing is executed from the downloads.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${ASKMAP_DATA_ROOT:?set ASKMAP_DATA_ROOT in .env}"

AETHER_COMMIT=7b382c55645fb5a1d3f11ec8ba81a46c523fca48
ZENODO_URL="https://zenodo.org/records/23336313/files/data.zip?download=1"
V="$ASKMAP_DATA_ROOT/vendor"
mkdir -p "$V"

if [ ! -d "$V/aether/.git" ]; then
  git clone https://github.com/WUR-AI/aether.git "$V/aether"
fi
git -C "$V/aether" checkout --quiet "$AETHER_COMMIT"
echo "aether code at $(git -C "$V/aether" rev-parse --short HEAD)"

# Their environment: the pinned requirements this project ran with (their pyproject has no lock file)
if [ ! -x "$V/aether/.venv/bin/python" ]; then
  uv venv --python 3.12 "$V/aether/.venv"
  uv pip install --python "$V/aether/.venv/bin/python" -r scripts/aether_stage1/aether_requirements.txt
fi
echo "aether environment: $("$V/aether/.venv/bin/python" --version)"

# Their data (SatBird): about 1.5 GB
if [ ! -d "$V/aether_zenodo/data" ]; then
  mkdir -p "$V/aether_zenodo"
  curl -L --fail -o "$V/aether_zenodo/data.zip" "$ZENODO_URL"
  unzip -q "$V/aether_zenodo/data.zip" -d "$V/aether_zenodo"
fi
echo "aether data: $(du -sh "$V/aether_zenodo/data" | cut -f1)"
echo "set-up done"
