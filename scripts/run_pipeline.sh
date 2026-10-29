#!/usr/bin/env bash
# Run the notebooks in order, from the data download to the demo data and the checks, stopping at the first
# failure. Each notebook is executed in place, so its outputs are kept in the .ipynb file.
#
#   scripts/run_pipeline.sh            # everything except the AETHER paper reproduction (notebook 09)
#   scripts/run_pipeline.sh 11 12 13   # only the notebooks whose names start with these numbers
#
# Notebook 09 needs the AETHER authors' code and data and about 7 hours of CPU: see scripts/aether_stage1/.
# Notebooks 15 and 15b (rebuild checks) are for comparing a fresh build with an existing data folder and are not
# part of the main run.
set -euo pipefail
cd "$(dirname "$0")/.."

ORDER=(01_data 10_aether_osm 02_climate_summaries 03_land_maps 04_answer_key_and_land 04b_d0_diagnostic
       05_translation 06_embedding_search 07_verdicts 08_wetland_signal 11_aether_us 12_aether_compare
       20_general_land_maps 23_land_context 13_aether_demo_data 16_demo_check_d2 21_unseen_words
       22_demo_check_d3 24_demo_check_d4 25_three_methods 26_geometry_distances 27_aligned_retraining
       28_page_examples 17_static_site)

# The demo checks (16, 22, 24) ask the language model afresh and refuse to run over saved replies, so any saved
# replies for them are moved aside first.
move_aside() {
  local dir="outputs/llm/$1"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ]; then
    local kept="$dir.previous-$(date +%Y%m%d-%H%M%S)"
    mv "$dir" "$kept"
    echo "saved replies moved to $kept"
  fi
}

run_one() {
  local nb="$1"
  case "$nb" in
    16_demo_check_d2) move_aside d2 ;;
    22_demo_check_d3) move_aside d3 ;;
    24_demo_check_d4) move_aside d4 ;;
  esac
  echo "== $nb start $(date)"
  uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    "notebooks/$nb.ipynb"
  echo "== $nb done  $(date)"
}

if [ $# -eq 0 ]; then
  for nb in "${ORDER[@]}"; do run_one "$nb"; done
else
  for prefix in "$@"; do
    for nb in "${ORDER[@]}"; do
      if [[ "$nb" == "$prefix"* ]]; then run_one "$nb"; fi
    done
  done
fi
echo "pipeline finished $(date)"
