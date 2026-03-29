#!/bin/bash

# ============================================================================
# Serial Monte Carlo Runner for mc_elasticity_logit.py
# ============================================================================
# This script runs MC simulations in batches, with each replication saved
# separately. It can resume from previous runs automatically.
#
# Usage:
#   ./run_mc_elasticity_serial.sh --n_markets 100 --n_products 4 --n_runs 100
#   ./run_mc_elasticity_serial.sh --n_markets 100 --n_products 4 --n_runs 100 --n_workers 4
#   ./run_mc_elasticity_serial.sh --n_markets 100 --n_products 4 --status
#   ./run_mc_elasticity_serial.sh --n_markets 100 --n_products 4 --aggregate
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/mc_elasticity_logit_serial.py"

# Default parameters
N_MARKETS=""
N_PRODUCTS=""
N_RUNS=100
N_WORKERS=1
BASE_SEED=1111
STATUS=false
AGGREGATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_markets|-T)
            N_MARKETS="$2"
            shift 2
            ;;
        --n_products|-J)
            N_PRODUCTS="$2"
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
            echo "Usage: $0 --n_markets T --n_products J [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --n_markets T   Number of markets"
            echo "  --n_products J  Number of products"
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
            echo "  $0 --n_markets 100 --n_products 4 --n_runs 100"
            echo ""
            echo "  # Run with 4 parallel workers"
            echo "  $0 --n_markets 100 --n_products 4 --n_runs 100 --n_workers 4"
            echo ""
            echo "  # Check status"
            echo "  $0 --n_markets 100 --n_products 4 --status"
            echo ""
            echo "  # Aggregate all results"
            echo "  $0 --n_markets 100 --n_products 4 --aggregate"
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
if [ -z "$N_MARKETS" ] || [ -z "$N_PRODUCTS" ]; then
    echo "Error: --n_markets and --n_products are required"
    echo "Use --help for usage information"
    exit 1
fi

# Build command
CMD="python3 $PYTHON_SCRIPT --n_markets $N_MARKETS --n_products $N_PRODUCTS"

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
