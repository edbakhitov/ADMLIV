#!/bin/bash

# ============================================================================
# Serial Monte Carlo Runner for mc_wad.py
# ============================================================================
# This script runs MC simulations in batches, with each replication saved
# separately. It can resume from previous runs automatically.
#
# Usage:
#   ./run_mc_wad_serial.sh --k 10 --n_obs 1000 --n_runs 100
#   ./run_mc_wad_serial.sh --k 10 --n_obs 1000 --n_runs 100 --n_workers 4
#   ./run_mc_wad_serial.sh --k 10 --n_obs 1000 --status
#   ./run_mc_wad_serial.sh --k 10 --n_obs 1000 --aggregate
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/mc_wad_serial.py"

# Default parameters
K=""
N_OBS=""
N_RUNS=100
N_WORKERS=1
BASE_SEED=1111
STATUS=false
AGGREGATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --k|-k)
            K="$2"
            shift 2
            ;;
        --n_obs|-n)
            N_OBS="$2"
            shift 2
            ;;
        --n_runs|-r)
            N_RUNS="$2"
            shift 2
            ;;
        --n_workers|-w)
            N_WORKERS="$2"
            shift 2
            ;;
        --base_seed|-s)
            BASE_SEED="$2"
            shift 2
            ;;
        --status)
            STATUS=true
            shift
            ;;
        --aggregate)
            AGGREGATE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --k K --n_obs N [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --k K           Number of covariates"
            echo "  --n_obs N       Sample size"
            echo ""
            echo "Options:"
            echo "  --n_runs N      Number of runs in this batch (default: 100)"
            echo "  --n_workers N   Number of parallel workers (default: 1)"
            echo "  --base_seed S   Base random seed (default: 1111)"
            echo "  --status        Show status of completed runs"
            echo "  --aggregate     Aggregate serial results into single file"
            echo ""
            echo "Examples:"
            echo "  # Run 100 simulations"
            echo "  $0 --k 10 --n_obs 1000 --n_runs 100"
            echo ""
            echo "  # Run with 4 parallel workers"
            echo "  $0 --k 10 --n_obs 1000 --n_runs 100 --n_workers 4"
            echo ""
            echo "  # Check status"
            echo "  $0 --k 10 --n_obs 1000 --status"
            echo ""
            echo "  # Aggregate all results"
            echo "  $0 --k 10 --n_obs 1000 --aggregate"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$K" ] || [ -z "$N_OBS" ]; then
    echo "Error: --k and --n_obs are required"
    echo "Use --help for usage information"
    exit 1
fi

# Build command
CMD="python3 $PYTHON_SCRIPT --k $K --n_obs $N_OBS"

if [ "$STATUS" = true ]; then
    CMD="$CMD --status"
elif [ "$AGGREGATE" = true ]; then
    CMD="$CMD --aggregate"
else
    CMD="$CMD --n_runs $N_RUNS --n_workers $N_WORKERS --base_seed $BASE_SEED"
fi

# Run
echo "Running: $CMD"
echo ""
exec $CMD
