#!/bin/bash

# ============================================================================
# Monte Carlo Simulation Runner for mc_wad.py
# ============================================================================
# This script runs Monte Carlo simulations for the Weighted Average Derivative
# functional over different values of k (number of covariates) and n_obs
# (sample size).
#
# Usage:
#   ./run_mc_wad.sh              # Run sequentially
#   ./run_mc_wad.sh --parallel   # Run in parallel 
# ============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/mc_wad.py"
RESULTS_DIR="$SCRIPT_DIR/results"

# Simulation parameters
N_RUNS=1000
SEED=1111
IV_STRENGTH=0.8
R=0.5

# Parallel execution parameters
PARALLEL=false
MAX_JOBS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --max-jobs|-j)
            MAX_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel] [--max-jobs N]"
            exit 1
            ;;
    esac
done

# Parameter grids
K_VALUES=(5)
N_OBS_VALUES=(100 500 1000)

# ============================================================================
# Create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo "Monte Carlo Simulation Runner: Weighted Average Derivative"
echo "========================================================================"
echo "Script: $PYTHON_SCRIPT"
echo "Results directory: $RESULTS_DIR"
echo "Number of replications: $N_RUNS"
echo "Seed: $SEED"
echo "IV strength: $IV_STRENGTH"
echo "Endogeneity (r): $R"
echo ""
echo "Parameter grid:"
echo "  k values: ${K_VALUES[@]}"
echo "  n_obs values: ${N_OBS_VALUES[@]}"
echo ""
if [ "$PARALLEL" = true ]; then
    echo "Execution mode: PARALLEL (max $MAX_JOBS jobs)"
else
    echo "Execution mode: SEQUENTIAL"
fi
echo "========================================================================"
echo ""

# Counter for progress
total_jobs=$((${#K_VALUES[@]} * ${#N_OBS_VALUES[@]}))
current_job=0

# Function to run a single simulation
run_simulation() {
    local k=$1
    local n_obs=$2
    local job_num=$3
    local total=$4
    local output_file="$RESULTS_DIR/mc_results_toy_model_wad_k${k}_n${n_obs}.csv"

    echo "------------------------------------------------------------------------"
    echo "Job $job_num/$total: Running with k=$k, n_obs=$n_obs"
    echo "------------------------------------------------------------------------"
    echo "Output: $output_file"
    echo ""

    # Run simulation
    python "$PYTHON_SCRIPT" \
        --k $k \
        --n_obs $n_obs \
        --n_runs $N_RUNS \
        --iv_strength $IV_STRENGTH \
        --r $R \
        --seed $SEED \
        --output "$output_file"

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Job $job_num/$total completed successfully (k=$k, n_obs=$n_obs)"
        echo ""
    else
        echo ""
        echo "✗ Job $job_num/$total failed with exit code $? (k=$k, n_obs=$n_obs)"
        echo ""
    fi
}

# Export function and variables for parallel execution
export -f run_simulation
export PYTHON_SCRIPT N_RUNS IV_STRENGTH R SEED RESULTS_DIR

# Loop over parameter combinations
if [ "$PARALLEL" = true ]; then
    # Parallel execution using GNU parallel or xargs
    if command -v parallel &> /dev/null; then
        # Use GNU parallel if available
        echo "Using GNU parallel with $MAX_JOBS jobs"
        echo ""

        # Create array of job parameters
        job_params=()
        for k in "${K_VALUES[@]}"; do
            for n_obs in "${N_OBS_VALUES[@]}"; do
                current_job=$((current_job + 1))
                job_params+=("$k $n_obs $current_job $total_jobs")
            done
        done

        # Run jobs in parallel
        printf '%s\n' "${job_params[@]}" | parallel -j $MAX_JOBS --colsep ' ' run_simulation {1} {2} {3} {4}
    else
        # Fallback: manual parallel execution with background jobs
        echo "GNU parallel not found, using background jobs (max $MAX_JOBS)"
        echo ""

        for k in "${K_VALUES[@]}"; do
            for n_obs in "${N_OBS_VALUES[@]}"; do
                current_job=$((current_job + 1))

                # Wait if we've reached max jobs
                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    sleep 1
                done

                # Run in background
                run_simulation $k $n_obs $current_job $total_jobs &
            done
        done

        # Wait for all background jobs to complete
        wait
    fi
else
    # Sequential execution
    for k in "${K_VALUES[@]}"; do
        for n_obs in "${N_OBS_VALUES[@]}"; do
            current_job=$((current_job + 1))
            run_simulation $k $n_obs $current_job $total_jobs
        done
    done
fi

echo "========================================================================"
echo "All simulations completed!"
echo "========================================================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files created:"
ls -lh "$RESULTS_DIR"/mc_results_toy_model_wad_*.csv 2>/dev/null || echo "  No output files found"
echo "========================================================================"
