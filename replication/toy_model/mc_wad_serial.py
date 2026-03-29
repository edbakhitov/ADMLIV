#!/usr/bin/env python3
# admliv/simulations/toy_model/mc_wad_serial.py

"""
Serial Monte Carlo Runner for WAD simulations.

Runs individual MC replications and saves each to a separate file.
Supports batch execution and can resume from previous runs.

Usage:
    # Run 100 replications for k=10, n=1000
    python mc_wad_serial.py --k 10 --n_obs 1000 --n_runs 100

    # Continue with another batch (auto-detects completed runs)
    python mc_wad_serial.py --k 10 --n_obs 1000 --n_runs 100

    # Run with parallel workers
    python mc_wad_serial.py --k 10 --n_obs 1000 --n_runs 100 --n_workers 4
"""

import sys
import os
import argparse
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from mc_wad import MonteCarloWAD, MonteCarloConfig


def get_completed_seeds(output_dir: Path, k: int, n_obs: int) -> List[int]:
    """
    Get list of seeds that have already been completed.

    Parameters
    ----------
    output_dir : Path
        Directory containing serial outputs
    k : int
        Number of covariates
    n_obs : int
        Sample size

    Returns
    -------
    seeds : list
        List of completed seed values
    """
    pattern = output_dir / f"mc_wad_k{k}_n{n_obs}_seed*.csv"
    completed_files = glob.glob(str(pattern))

    seeds = []
    for f in completed_files:
        # Extract seed from filename: mc_wad_k{k}_n{n_obs}_seed{seed}.csv
        basename = os.path.basename(f)
        try:
            seed_part = basename.split('_seed')[1].replace('.csv', '')
            seeds.append(int(seed_part))
        except (IndexError, ValueError):
            continue

    return sorted(seeds)


def get_next_seeds(output_dir: Path, k: int, n_obs: int, n_runs: int, base_seed: int = 1111) -> List[int]:
    """
    Get the next batch of seeds to run, avoiding already completed ones.

    Parameters
    ----------
    output_dir : Path
        Directory containing serial outputs
    k : int
        Number of covariates
    n_obs : int
        Sample size
    n_runs : int
        Number of runs requested for this batch
    base_seed : int
        Base seed to start from

    Returns
    -------
    seeds : list
        List of seed values to run
    """
    completed = set(get_completed_seeds(output_dir, k, n_obs))

    # Find next available seeds
    seeds = []
    candidate_seed = base_seed
    while len(seeds) < n_runs:
        if candidate_seed not in completed:
            seeds.append(candidate_seed)
        candidate_seed += 1

    return seeds


def run_single_seed(args: Tuple[int, int, int, Path]) -> Tuple[int, bool, str]:
    """
    Run a single simulation for a given seed.

    Parameters
    ----------
    args : tuple
        (seed, k, n_obs, output_dir)

    Returns
    -------
    result : tuple
        (seed, success, message)
    """
    seed, k, n_obs, output_dir = args
    output_file = output_dir / f"mc_wad_k{k}_n{n_obs}_seed{seed}.csv"

    # Skip if already exists
    if output_file.exists():
        return (seed, True, "already exists")

    try:
        config = MonteCarloConfig(
            n_runs=1,
            n_obs=n_obs,
            k=k,
            seed=seed
        )

        mc = MonteCarloWAD(config=config, verbose=False)
        mc.run()

        # Add seed column to results
        mc.results_['seed'] = seed
        mc.results_.to_csv(output_file, index=False)

        return (seed, True, "completed")
    except Exception as e:
        return (seed, False, str(e))


def aggregate_results(output_dir: Path, k: int, n_obs: int, results_dir: Path) -> pd.DataFrame:
    """
    Aggregate all serial results into a single file.

    Parameters
    ----------
    output_dir : Path
        Directory containing serial outputs
    k : int
        Number of covariates
    n_obs : int
        Sample size
    results_dir : Path
        Directory to save aggregated results

    Returns
    -------
    df : pd.DataFrame
        Aggregated results
    """
    pattern = output_dir / f"mc_wad_k{k}_n{n_obs}_seed*.csv"
    files = sorted(glob.glob(str(pattern)))

    if not files:
        print(f"No results found for k={k}, n_obs={n_obs}")
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
    output_file = results_dir / f"mc_results_toy_model_wad_k{k}_n{n_obs}.csv"
    aggregated.to_csv(output_file, index=False)
    print(f"Aggregated {len(dfs)} results to {output_file}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Serial Monte Carlo Runner for WAD simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 100 replications for k=10, n=1000
    python mc_wad_serial.py --k 10 --n_obs 1000 --n_runs 100

    # Continue with another batch (auto-detects completed runs)
    python mc_wad_serial.py --k 10 --n_obs 1000 --n_runs 100

    # Run with 4 parallel workers
    python mc_wad_serial.py --k 10 --n_obs 1000 --n_runs 100 --n_workers 4

    # Aggregate all serial results into single file
    python mc_wad_serial.py --k 10 --n_obs 1000 --aggregate
        """
    )

    parser.add_argument('--k', type=int, required=True,
                        help='Number of covariates and instruments')
    parser.add_argument('--n_obs', type=int, required=True,
                        help='Number of observations')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Number of replications in this batch (default: 100)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--base_seed', type=int, default=1111,
                        help='Base random seed (default: 1111)')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate all serial results into single file')
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
        completed = get_completed_seeds(serial_dir, args.k, args.n_obs)
        print(f"\n{'='*60}")
        print(f"Status for k={args.k}, n_obs={args.n_obs}")
        print(f"{'='*60}")
        print(f"Completed runs: {len(completed)}")
        if completed:
            print(f"Seed range: {min(completed)} - {max(completed)}")
        print(f"{'='*60}\n")
        return

    # Aggregate mode
    if args.aggregate:
        aggregate_results(serial_dir, args.k, args.n_obs, results_dir)
        return

    # Get seeds to run
    seeds_to_run = get_next_seeds(serial_dir, args.k, args.n_obs, args.n_runs, args.base_seed)
    completed_seeds = get_completed_seeds(serial_dir, args.k, args.n_obs)

    print(f"\n{'='*60}")
    print(f"Serial Monte Carlo Runner: WAD")
    print(f"{'='*60}")
    print(f"Parameters: k={args.k}, n_obs={args.n_obs}")
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
            seed, success, msg = run_single_seed((seed, args.k, args.n_obs, serial_dir))

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
        job_args = [(seed, args.k, args.n_obs, serial_dir) for seed in seeds_to_run]

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
    total_completed = len(get_completed_seeds(serial_dir, args.k, args.n_obs))

    print(f"\n{'='*60}")
    print(f"Batch Complete")
    print(f"{'='*60}")
    print(f"This batch: {successful} successful, {failed} failed")
    print(f"Total completed: {total_completed}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"\nTo aggregate results into single file:")
    print(f"  python3 mc_wad_serial.py --k {args.k} --n_obs {args.n_obs} --aggregate")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
