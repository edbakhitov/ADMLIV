#!/usr/bin/env python3
"""
Summarize Monte Carlo results from HD Linear IV simulations.

Collects results from CSV files for different sample sizes and produces:
1. A formatted summary table
2. A LaTeX table for paper inclusion
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(results_dir: str, n_obs_list: list) -> dict:
    """
    Load results from CSV files for different sample sizes.

    Parameters
    ----------
    results_dir : str
        Directory containing result CSV files
    n_obs_list : list
        List of sample sizes to load

    Returns
    -------
    results : dict
        Dictionary mapping sample size to DataFrame
    """
    results = {}

    for n_obs in n_obs_list:
        filename = f'mc_results_hd_linear_iv_n{n_obs}.csv'
        filepath = os.path.join(results_dir, filename)

        if os.path.exists(filepath):
            results[n_obs] = pd.read_csv(filepath)
            print(f"Loaded results for n={n_obs}: {len(results[n_obs])} replications")
        else:
            print(f"Warning: File not found: {filepath}")

    return results


def compute_summary_statistics(results_dict: dict) -> pd.DataFrame:
    """
    Compute summary statistics for each method and sample size.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping sample size to results DataFrame

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics with multi-level columns
    """
    summary_data = []

    # Define desired method order
    desired_order = ['DLasso', 'PGMM', 'PGMM-CV', 'A-PGMM', 'A-PGMM-CV']

    # Get all available methods from the first available results
    first_results = next(iter(results_dict.values()))
    available_methods = set(col.replace('_MSE', '').replace('_R2', '')
                           for col in first_results.columns
                           if '_MSE' in col or '_R2' in col)

    # Use only methods that exist in both desired order and data
    methods = [m for m in desired_order if m in available_methods]

    for method in methods:
        row = {'Method': method}

        for n_obs in sorted(results_dict.keys()):
            df = results_dict[n_obs]

            # Check if method exists in this dataset
            mse_col = f'{method}_MSE'
            r2_col = f'{method}_R2'

            if mse_col in df.columns:
                row[f'MSE_n{n_obs}'] = df[mse_col].mean()
                row[f'MSE_std_n{n_obs}'] = df[mse_col].std()
            else:
                row[f'MSE_n{n_obs}'] = np.nan
                row[f'MSE_std_n{n_obs}'] = np.nan

            if r2_col in df.columns:
                row[f'R2_n{n_obs}'] = df[r2_col].mean()
                row[f'R2_std_n{n_obs}'] = df[r2_col].std()
            else:
                row[f'R2_n{n_obs}'] = np.nan
                row[f'R2_std_n{n_obs}'] = np.nan

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def format_table(summary_df: pd.DataFrame, n_obs_list: list) -> str:
    """
    Format summary table as a nicely formatted string.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame
    n_obs_list : list
        List of sample sizes

    Returns
    -------
    table : str
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append("Monte Carlo Results Summary: High-Dimensional Linear IV Regression")
    lines.append("=" * 100)

    # Header
    header = f"{'Method':<15}"
    for n_obs in n_obs_list:
        header += f"{'n=' + str(n_obs):^20}"
    lines.append(header)

    subheader = f"{'':<15}"
    for n_obs in n_obs_list:
        subheader += f"{'MSE':>9} {'R²':>9}  "
    lines.append(subheader)
    lines.append("-" * 100)

    # Data rows
    for _, row in summary_df.iterrows():
        method = row['Method']
        line = f"{method:<15}"

        for n_obs in n_obs_list:
            mse = row[f'MSE_n{n_obs}']
            r2 = row[f'R2_n{n_obs}']

            if not np.isnan(mse):
                line += f"{mse:9.4f} {r2:9.4f}  "
            else:
                line += f"{'---':>9} {'---':>9}  "

        lines.append(line)

    lines.append("=" * 100)

    return "\n".join(lines)


def generate_latex_table(summary_df: pd.DataFrame, n_obs_list: list) -> str:
    """
    Generate LaTeX table from summary statistics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame
    n_obs_list : list
        List of sample sizes

    Returns
    -------
    latex : str
        LaTeX table code
    """
    lines = []

    # Table preamble
    n_cols = 1 + 2 * len(n_obs_list)  # Method + (MSE, R²) for each n
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Monte Carlo Results: High-Dimensional Linear IV Regression}")
    lines.append("\\label{tab:mc_hd_linear}")

    # Column specification
    col_spec = "l" + "rr" * len(n_obs_list)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Multi-column header
    header = "Method"
    for n_obs in n_obs_list:
        header += f" & \\multicolumn{{2}}{{c}}{{$n={n_obs}$}}"
    lines.append(header + " \\\\")

    # Sub-header
    subheader = ""
    for _ in n_obs_list:
        subheader += " & MSE & $R^2$"
    lines.append(subheader + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for _, row in summary_df.iterrows():
        method = row['Method'].replace('_', '\\_')  # Escape underscores
        line = method

        for n_obs in n_obs_list:
            mse = row[f'MSE_n{n_obs}']
            r2 = row[f'R2_n{n_obs}']

            if not np.isnan(mse):
                line += f" & {mse:.4f} & {r2:.4f}"
            else:
                line += " & --- & ---"

        lines.append(line + " \\\\")

    # Table closing
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item Notes: Results averaged over Monte Carlo replications. ")
    lines.append("MSE = Mean Squared Error; $R^2$ = Coefficient of determination.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Configuration
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    n_obs_list = [100, 1000, 10000]

    print("\n" + "=" * 80)
    print("Monte Carlo Results Summary Generator - HD Linear IV")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Sample sizes: {n_obs_list}")
    print("=" * 80 + "\n")

    # Load results
    results_dict = load_results(results_dir, n_obs_list)

    if not results_dict:
        print("Error: No results files found!")
        sys.exit(1)

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_statistics(results_dict)

    # Format and print table
    print("\n")
    formatted_table = format_table(summary_df, n_obs_list)
    print(formatted_table)

    # Generate LaTeX table
    latex_table = generate_latex_table(summary_df, n_obs_list)

    # Save outputs
    output_dir = results_dir
    output_dir.mkdir(exist_ok=True)

    # Save formatted table
    table_file = output_dir / 'summary_hd_linear_iv_table.txt'
    with open(table_file, 'w') as f:
        f.write(formatted_table)
    print(f"\nFormatted table saved to: {table_file}")

    # Save LaTeX table
    latex_file = output_dir / 'summary_hd_linear_iv_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Save summary DataFrame as CSV
    csv_file = output_dir / 'summary_hd_linear_iv_statistics.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"Summary statistics saved to: {csv_file}")

    print("\n" + "=" * 80)
    print("Summary generation complete!")
    print("=" * 80 + "\n")

    # Print LaTeX table to console
    print("\nLaTeX Table:\n")
    print(latex_table)


if __name__ == '__main__':
    main()
