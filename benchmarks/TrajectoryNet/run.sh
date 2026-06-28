#!/bin/bash
# TrajectoryNet benchmark experiments
# Usage: ./run.sh                 (runs all experiments)
#        ./run.sh <experiment_id> (runs a specific experiment, e.g. ./run.sh 1)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK="$SCRIPT_DIR/notebooks/TrajectoryNet_for_Custom_Dataset.ipynb"

run_experiment() {
  local ID=$1 DESC=$2; shift 2
  echo "=== [$ID] $DESC ==="
  local OUTPUT="$SCRIPT_DIR/notebooks/output_${ID}.ipynb"
  papermill "$NOTEBOOK" "$OUTPUT" "$@"
}

# --- stem_cell_differentiation (train: 0,2,4 | unseen: 1,3) ---

# 1) stem_cell_differentiation, dim=2
[[ -z "$1" || "$1" == "1" ]] && run_experiment 1 "stem_cell_differentiation dim=2" \
  -p dataset stem_cell_differentiation -p d_red 2 \
  -y "days: [0, 2, 4]" -y "intermediate_days: [1, 3]"

# 2) stem_cell_differentiation, dim=4
[[ -z "$1" || "$1" == "2" ]] && run_experiment 2 "stem_cell_differentiation dim=4" \
  -p dataset stem_cell_differentiation -p d_red 4 \
  -y "days: [0, 2, 4]" -y "intermediate_days: [1, 3]"

# 3) stem_cell_differentiation, dim=8
[[ -z "$1" || "$1" == "3" ]] && run_experiment 3 "stem_cell_differentiation dim=8" \
  -p dataset stem_cell_differentiation -p d_red 8 \
  -y "days: [0, 2, 4]" -y "intermediate_days: [1, 3]"

# 4) stem_cell_differentiation, dim=16
[[ -z "$1" || "$1" == "4" ]] && run_experiment 4 "stem_cell_differentiation dim=16" \
  -p dataset stem_cell_differentiation -p d_red 16 \
  -y "days: [0, 2, 4]" -y "intermediate_days: [1, 3]"

# --- emt_72 (train: 0,4 | unseen: 1,2,3,8) ---

# 5) emt_72, dim=2
[[ -z "$1" || "$1" == "5" ]] && run_experiment 5 "emt_72 dim=2" \
  -p dataset emt_72 -p d_red 2 \
  -y "days: [0, 4]" -y "intermediate_days: [2]"

# --- LARRY_3000_benchmark (train: 2,6 | unseen: 4) ---

# 6) LARRY_3000_benchmark, dim=2
[[ -z "$1" || "$1" == "6" ]] && run_experiment 6 "LARRY_3000_benchmark dim=2" \
  -p dataset LARRY_3000_benchmark -p d_red 2 \
  -y "days: [2, 6]" -y "intermediate_days: [4]"

echo "=== All requested TrajectoryNet experiments complete ==="
