#!/usr/bin/env python3
"""
Summarize Monte Carlo results from WAD (Weighted Average Derivative) simulations.

Collects results from CSV files for different k and n_obs values and produces:
1. A formatted summary table with multi-index (k, n_obs)
2. A LaTeX table for paper inclusion

Metrics computed for each method (Plugin, DML, ADMLIV):
- Absolute Bias
- Median SE
- RMSE
- Coverage
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(results_dir: str, k_list: list, n_obs_list: list) -> dict:
    """
    Load results from CSV files for different k and n_obs combinations.

    Parameters
    ----------
    results_dir : str
        Directory containing result CSV files
    k_list : list
        List of k values (number of covariates)
    n_obs_list : list
        List of sample sizes to load

    Returns
    -------
    results : dict
        Dictionary mapping (k, n_obs) tuple to DataFrame
    """
    results = {}

    for k in k_list:
        for n_obs in n_obs_list:
            filename = f'mc_results_toy_model_wad_k{k}_n{n_obs}.csv'
            filepath = os.path.join(results_dir, filename)

            if os.path.exists(filepath):
                results[(k, n_obs)] = pd.read_csv(filepath)
                print(f"Loaded results for k={k}, n={n_obs}: {len(results[(k, n_obs)])} replications")
            else:
                print(f"Warning: File not found: {filepath}")

    return results


def compute_summary_statistics(results_dict: dict) -> pd.DataFrame:
    """
    Compute summary statistics for each method, k, and sample size.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping (k, n_obs) tuple to results DataFrame

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics with multi-index (k, n_obs) and columns for each metric/method
    """
    summary_data = []

    # Define method names and their column prefixes
    methods = [
        ('plugin', 'PI'),   # Plug-in
        ('dml', 'DML'),     # DML with Analytical RR
        ('admliv', 'ADMLIV') # ADMLIV
    ]

    for (k, n_obs), df in sorted(results_dict.items()):
        row = {'k': k, 'n_obs': n_obs}

        for method_col, method_name in methods:
            bias_col = f'{method_col}_bias'
            se_col = f'{method_col}_se'
            mse_col = f'{method_col}_mse'
            cp_col = f'{method_col}_cp'

            if bias_col in df.columns:
                # Absolute Bias: |mean(bias)|, NOT mean(|bias|)
                row[f'|Bias|_{method_name}'] = np.abs(df[bias_col].mean())

                # Median SE
                row[f'SE_{method_name}'] = df[se_col].median()

                # RMSE
                row[f'RMSE_{method_name}'] = np.sqrt(df[mse_col].mean())

                # Coverage
                row[f'Coverage_{method_name}'] = df[cp_col].mean()
            else:
                row[f'|Bias|_{method_name}'] = np.nan
                row[f'SE_{method_name}'] = np.nan
                row[f'RMSE_{method_name}'] = np.nan
                row[f'Coverage_{method_name}'] = np.nan

        summary_data.append(row)

    # Create DataFrame with multi-index
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index(['k', 'n_obs'])

    return summary_df


def format_table(summary_df: pd.DataFrame) -> str:
    """
    Format summary table as a nicely formatted string.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame with multi-index (k, n_obs)

    Returns
    -------
    table : str
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append("Monte Carlo Results Summary: WAD (Weighted Average Derivative)")
    lines.append("=" * 100)

    # Header
    header = f"{'k':<5} {'n':<8} {'Metric':<15} {'PI':>12} {'DML':>12} {'ADMLIV':>12}"
    lines.append(header)
    lines.append("-" * 100)

    # Data rows
    metrics = [
        ('|Bias|', 'Abs. Bias'),
        ('SE', 'Median SE'),
        ('RMSE', 'RMSE'),
        ('Coverage', 'Coverage')
    ]

    for (k, n_obs), row in summary_df.iterrows():
        # Print k and n_obs only on the first metric line for each (k, n_obs)
        for i, (metric_key, metric_label) in enumerate(metrics):
            if i == 0:
                k_str = f"{k:<5}"
                n_str = f"{n_obs:<8}"
            else:
                k_str = f"{'':<5}"
                n_str = f"{'':<8}"

            pi_val = row[f'{metric_key}_PI']
            dml_val = row[f'{metric_key}_DML']
            admliv_val = row[f'{metric_key}_ADMLIV']

            if not np.isnan(pi_val):
                line = f"{k_str} {n_str} {metric_label:<15} {pi_val:>12.4f} {dml_val:>12.4f} {admliv_val:>12.4f}"
            else:
                line = f"{k_str} {n_str} {metric_label:<15} {'---':>12} {'---':>12} {'---':>12}"

            lines.append(line)

        # Add separator between different (k, n_obs) combinations
        lines.append("-" * 100)

    lines.append("=" * 100)

    return "\n".join(lines)


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """
    Generate LaTeX table from summary statistics.

    Format: rows indexed by (k, n_obs), columns grouped by metric (Bias, Med SE, Coverage)
    with sub-columns for each method (PI, DML, ADMLIV).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame with multi-index (k, n_obs)

    Returns
    -------
    latex : str
        LaTeX table code
    """
    lines = []

    # Table preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Monte Carlo Results: Weighted Average Derivative Functional}")
    lines.append("\\label{tab:mc_wad}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{cc ccc ccc ccc}")
    lines.append("\\toprule")

    # Two-row header with metric groups
    lines.append(" & & \\multicolumn{3}{c}{Bias} & \\multicolumn{3}{c}{Med SE} & \\multicolumn{3}{c}{Coverage} \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}")
    lines.append("$k$ & $n$ & PI & DML & ADMLIV & PI & DML & ADMLIV & PI & DML & ADMLIV \\\\")
    lines.append("\\midrule")

    # Data rows - one row per (k, n_obs)
    for (k, n_obs), row in summary_df.iterrows():
        # Bias values
        bias_pi = row['|Bias|_PI']
        bias_dml = row['|Bias|_DML']
        bias_admliv = row['|Bias|_ADMLIV']

        # Median SE values
        se_pi = row['SE_PI']
        se_dml = row['SE_DML']
        se_admliv = row['SE_ADMLIV']

        # Coverage values
        cov_pi = row['Coverage_PI']
        cov_dml = row['Coverage_DML']
        cov_admliv = row['Coverage_ADMLIV']

        if not np.isnan(bias_pi):
            line = (
                f"{k} & {n_obs} & "
                f"{bias_pi:.3f} & {bias_dml:.3f} & {bias_admliv:.3f} & "
                f"{se_pi:.3f} & {se_dml:.3f} & {se_admliv:.3f} & "
                f"{cov_pi*100:.1f}\\% & {cov_dml*100:.1f}\\% & {cov_admliv*100:.1f}\\% \\\\"
            )
        else:
            line = f"{k} & {n_obs} & --- & --- & --- & --- & --- & --- & --- & --- & --- \\\\"

        lines.append(line)

    # Table closing
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item Notes: Results averaged over Monte Carlo replications. ")
    lines.append("PI = Plug-in estimator; DML = Debiased ML with analytical Riesz representer; ")
    lines.append("ADMLIV = Adaptive debiased ML with PGMM-estimated Riesz representer.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Configuration
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'

    # Define parameter grid
    k_list = [2, 5, 10]
    n_obs_list = [100, 500, 1000, 10000]

    print("\n" + "=" * 80)
    print("Monte Carlo Results Summary Generator - WAD")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"k values: {k_list}")
    print(f"Sample sizes: {n_obs_list}")
    print("=" * 80 + "\n")

    # Load results
    results_dict = load_results(results_dir, k_list, n_obs_list)

    if not results_dict:
        print("Error: No results files found!")
        sys.exit(1)

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_statistics(results_dict)

    # Format and print table
    print("\n")
    formatted_table = format_table(summary_df)
    print(formatted_table)

    # Generate LaTeX table
    latex_table = generate_latex_table(summary_df)

    # Save outputs
    output_dir = results_dir
    output_dir.mkdir(exist_ok=True)

    # Save formatted table
    table_file = output_dir / 'summary_wad_table.txt'
    with open(table_file, 'w') as f:
        f.write(formatted_table)
    print(f"\nFormatted table saved to: {table_file}")

    # Save LaTeX table
    latex_file = output_dir / 'summary_wad_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Save summary DataFrame as CSV
    csv_file = output_dir / 'summary_wad_statistics.csv'
    summary_df.to_csv(csv_file)
    print(f"Summary statistics saved to: {csv_file}")

    print("\n" + "=" * 80)
    print("Summary generation complete!")
    print("=" * 80 + "\n")

    # Print LaTeX table to console
    print("\nLaTeX Table:\n")
    print(latex_table)


if __name__ == '__main__':
    main()
