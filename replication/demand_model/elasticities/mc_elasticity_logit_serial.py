#!/usr/bin/env python3
# admliv/simulations/demand_model/elasticities/mc_elasticity_logit_serial.py

"""
Serial Monte Carlo Runner for Elasticity simulations.

Runs individual MC replications and saves each to a separate file.
Supports batch execution and can resume from previous runs.

Usage:
    # Run 100 replications for T=100, J=4 with KIV
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --n_runs 100

    # Run with double lasso and custom pgmm_c
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --mliv double_lasso --pgmm_c 0.001 --n_runs 100

    # Run with parallel workers
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --n_runs 100 --n_workers 4
"""

import sys
import os
import argparse
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from mc_elasticity_logit import MonteCarloElasticity, MonteCarloConfig


def _file_prefix(n_markets: int, n_products: int, mliv: str) -> str:
    """Return the filename prefix encoding (T, J, mliv)."""
    return f"mc_elasticity_T{n_markets}_J{n_products}_{mliv}"


def get_completed_seeds(output_dir: Path, n_markets: int, n_products: int, mliv: str) -> List[int]:
    """
    Get list of seeds that have already been completed.

    Parameters
    ----------
    output_dir : Path
        Directory containing serial outputs
    n_markets : int
        Number of markets
    n_products : int
        Number of products
    mliv : str
        MLIV estimator type

    Returns
    -------
    seeds : list
        List of completed seed values
    """
    prefix = _file_prefix(n_markets, n_products, mliv)
    pattern = output_dir / f"{prefix}_seed*.csv"
    completed_files = glob.glob(str(pattern))

    seeds = []
    for f in completed_files:
        # Extract seed from filename: {prefix}_seed{seed}.csv
        basename = os.path.basename(f)
        try:
            seed_part = basename.split('_seed')[1].replace('.csv', '')
            seeds.append(int(seed_part))
        except (IndexError, ValueError):
            continue

    return sorted(seeds)


def get_next_seeds(output_dir: Path, n_markets: int, n_products: int, mliv: str,
                   n_runs: int, base_seed: int = 1111) -> List[int]:
    """
    Get the next batch of seeds to run, avoiding already completed ones.

    Parameters
    ----------
    output_dir : Path
        Directory containing serial outputs
    n_markets : int
        Number of markets
    n_products : int
        Number of products
    mliv : str
        MLIV estimator type
    n_runs : int
        Number of runs requested for this batch
    base_seed : int
        Base seed to start from

    Returns
    -------
    seeds : list
        List of seed values to run
    """
    completed = set(get_completed_seeds(output_dir, n_markets, n_products, mliv))

    # Find next available seeds
    seeds = []
    candidate_seed = base_seed
    while len(seeds) < n_runs:
        if candidate_seed not in completed:
            seeds.append(candidate_seed)
        candidate_seed += 1

    return seeds


def run_single_seed(args: Tuple) -> Tuple[int, bool, str]:
    """
    Run a single simulation for a given seed.

    Parameters
    ----------
    args : tuple
        (seed, n_markets, n_products, mliv, pgmm_c, output_dir)

    Returns
    -------
    result : tuple
        (seed, success, message)
    """
    seed, n_markets, n_products, mliv, pgmm_c, output_dir = args
    prefix = _file_prefix(n_markets, n_products, mliv)
    output_file = output_dir / f"{prefix}_seed{seed}.csv"

    # Skip if already exists
    if output_file.exists():
        return (seed, True, "already exists")

    try:
        config = MonteCarloConfig(
            n_runs=1,
            n_markets=n_markets,
            n_products=n_products,
            mliv=mliv,
            pgmm_c=pgmm_c,
            seed=seed
        )

        mc = MonteCarloElasticity(config=config, verbose=False)
        mc.run()

        # Add seed and config columns to results
        mc.results_['seed'] = seed
        mc.results_['mliv'] = mliv
        mc.results_['pgmm_c'] = pgmm_c
        mc.results_.to_csv(output_file, index=False)

        return (seed, True, "completed")
    except Exception as e:
        return (seed, False, str(e))


def aggregate_results(output_dir: Path, n_markets: int, n_products: int, mliv: str,
                      results_dir: Path) -> pd.DataFrame:
    """
    Aggregate all serial results into a single file.

    Parameters
    ----------
    output_dir : Path
        Directory containing serial outputs
    n_markets : int
        Number of markets
    n_products : int
        Number of products
    mliv : str
        MLIV estimator type
    results_dir : Path
        Directory to save aggregated results

    Returns
    -------
    df : pd.DataFrame
        Aggregated results
    """
    prefix = _file_prefix(n_markets, n_products, mliv)
    pattern = output_dir / f"{prefix}_seed*.csv"
    files = sorted(glob.glob(str(pattern)))

    if not files:
        print(f"No results found for n_markets={n_markets}, n_products={n_products}, mliv={mliv}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    aggregated = pd.concat(dfs, ignore_index=True)

    # Save aggregated file
    output_file = results_dir / f"mc_results_elasticity_T{n_markets}_J{n_products}_{mliv}.csv"
    aggregated.to_csv(output_file, index=False)
    print(f"Aggregated {len(dfs)} results to {output_file}")

    return aggregated


def print_summary(results_dir: Path, n_markets: int, n_products: int, mliv: str,
                  iqr_multiplier: float = 100.0):
    """
    Print summary statistics with outlier removal.

    Parameters
    ----------
    results_dir : Path
        Directory containing aggregated results
    n_markets : int
        Number of markets
    n_products : int
        Number of products
    mliv : str
        MLIV estimator type
    iqr_multiplier : float, default=10.0
        IQR multiplier for outlier detection
    """
    results_file = results_dir / f"mc_results_elasticity_T{n_markets}_J{n_products}_{mliv}.csv"

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run with --aggregate first to create the aggregated results file.")
        return

    df = pd.read_csv(results_file)
    print(f"\n{'='*70}")
    print(f"Summary Statistics: n_markets={n_markets}, n_products={n_products}")
    print(f"{'='*70}")
    print(f"Total replications: {len(df)}")

    # Remove outliers using IQR method
    outlier_cols = ['admliv_bias', 'plugin_bias', 'admliv_mse', 'plugin_mse']
    mask = pd.Series(True, index=df.index)

    for col in outlier_cols:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) == 0:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & (col_mask | df[col].isna())

    df_clean = df[mask]
    n_outliers = len(df) - len(df_clean)

    if n_outliers > 0:
        print(f"Outliers removed: {n_outliers} ({100*n_outliers/len(df):.1f}%)")
    print(f"Valid replications: {len(df_clean)}")

    # Compute summary statistics
    methods = ['admliv', 'plugin']
    print(f"\n{'Method':<12} {'Bias':>10} {'SE':>10} {'SD':>10} {'RMSE':>10} {'Coverage':>10}")
    print("-" * 65)

    for method in methods:
        theta_col = f'{method}_theta'
        se_col = f'{method}_se'
        bias_col = f'{method}_bias'
        mse_col = f'{method}_mse'
        cp_col = f'{method}_cp'

        bias = df_clean[bias_col].mean()
        se = df_clean[se_col].median()
        sd = df_clean[theta_col].std()
        rmse = np.sqrt(df_clean[mse_col].mean())
        coverage = df_clean[cp_col].mean()

        print(f"{method:<12} {bias:>10.4f} {se:>10.4f} {sd:>10.4f} {rmse:>10.4f} {coverage:>10.4f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Serial Monte Carlo Runner for Elasticity simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 100 replications for T=100, J=4 with KIV (default)
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --n_runs 100

    # Run with double lasso and custom pgmm_c
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --mliv double_lasso --pgmm_c 0.001 --n_runs 100

    # Run with 4 parallel workers
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --n_runs 100 --n_workers 4

    # Aggregate all serial results into single file
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --mliv kiv --aggregate

    # Print summary statistics with outlier removal
    python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --mliv kiv --summary
        """
    )

    parser.add_argument('--n_markets', '-T', type=int, required=True,
                        help='Number of markets')
    parser.add_argument('--n_products', '-J', type=int, required=True,
                        help='Number of products')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Number of replications in this batch (default: 100)')
    parser.add_argument('--mliv', type=str, default='kiv',
                        choices=['kiv', 'double_lasso'],
                        help='MLIV estimator type (default: kiv)')
    parser.add_argument('--pgmm_c', type=float, default=0.0001,
                        help='PGMM penalty parameter (default: 0.0001)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--base_seed', type=int, default=1111,
                        help='Base random seed (default: 1111)')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate all serial results into single file')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics with outlier removal')
    parser.add_argument('--status', action='store_true',
                        help='Show status of completed runs')

    args = parser.parse_args()

    # Setup directories
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    serial_dir = results_dir / 'serial'

    results_dir.mkdir(exist_ok=True)
    serial_dir.mkdir(exist_ok=True)

    # Status mode
    if args.status:
        completed = get_completed_seeds(serial_dir, args.n_markets, args.n_products, args.mliv)
        print(f"\n{'='*60}")
        print(f"Status for n_markets={args.n_markets}, n_products={args.n_products}, mliv={args.mliv}")
        print(f"{'='*60}")
        print(f"Completed runs: {len(completed)}")
        if completed:
            print(f"Seed range: {min(completed)} - {max(completed)}")
        print(f"{'='*60}\n")
        return

    # Aggregate mode
    if args.aggregate:
        aggregate_results(serial_dir, args.n_markets, args.n_products, args.mliv, results_dir)
        return

    # Summary mode
    if args.summary:
        print_summary(results_dir, args.n_markets, args.n_products, args.mliv)
        return

    # Get seeds to run
    seeds_to_run = get_next_seeds(serial_dir, args.n_markets, args.n_products, args.mliv,
                                  args.n_runs, args.base_seed)
    completed_seeds = get_completed_seeds(serial_dir, args.n_markets, args.n_products, args.mliv)

    print(f"\n{'='*60}")
    print(f"Serial Monte Carlo Runner: Elasticity (Logit)")
    print(f"{'='*60}")
    print(f"Parameters: n_markets={args.n_markets}, n_products={args.n_products}")
    print(f"MLIV: {args.mliv}, pgmm_c: {args.pgmm_c}")
    print(f"Output directory: {serial_dir}")
    print(f"Already completed: {len(completed_seeds)} runs")
    print(f"Requested: {args.n_runs} runs")
    print(f"Seeds to run: {seeds_to_run[0]} - {seeds_to_run[-1]}")
    print(f"Workers: {args.n_workers}")
    print(f"{'='*60}\n")

    # Run simulations
    start_time = time.time()
    successful = 0
    failed = 0

    if args.n_workers == 1:
        # Sequential execution
        for i, seed in enumerate(seeds_to_run):
            seed, success, msg = run_single_seed((seed, args.n_markets, args.n_products,
                                                   args.mliv, args.pgmm_c, serial_dir))

            if success:
                successful += 1
            else:
                failed += 1
                print(f"  Seed {seed} failed: {msg}")

            # Progress update
            if (i + 1) % max(1, len(seeds_to_run) // 10) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(seeds_to_run) - i - 1)
                print(f"Progress: {i+1}/{len(seeds_to_run)} - "
                      f"Elapsed: {elapsed/60:.1f}m - ETA: {eta/60:.1f}m")
    else:
        # Parallel execution
        job_args = [(seed, args.n_markets, args.n_products, args.mliv, args.pgmm_c, serial_dir)
                    for seed in seeds_to_run]

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(run_single_seed, arg): arg[0] for arg in job_args}

            for i, future in enumerate(as_completed(futures)):
                seed, success, msg = future.result()

                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"  Seed {seed} failed: {msg}")

                # Progress update
                if (i + 1) % max(1, len(seeds_to_run) // 10) == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(seeds_to_run) - i - 1)
                    print(f"Progress: {i+1}/{len(seeds_to_run)} - "
                          f"Elapsed: {elapsed/60:.1f}m - ETA: {eta/60:.1f}m")

    total_time = time.time() - start_time
    total_completed = len(get_completed_seeds(serial_dir, args.n_markets, args.n_products, args.mliv))

    print(f"\n{'='*60}")
    print(f"Batch Complete")
    print(f"{'='*60}")
    print(f"This batch: {successful} successful, {failed} failed")
    print(f"Total completed: {total_completed}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"\nTo aggregate results into single file:")
    print(f"  python3 mc_elasticity_logit_serial.py --n_markets {args.n_markets} --n_products {args.n_products} --mliv {args.mliv} --aggregate")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
