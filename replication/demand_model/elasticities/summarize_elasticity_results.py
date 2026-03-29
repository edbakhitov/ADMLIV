# admliv/simulations/demand_model/elasticities/summarize_elasticity_resilts.py
"""
Summarize Monte Carlo results from Own-Price Elasticity simulations.

Collects results from CSV files for different J and T values and produces:
1. A formatted summary table with multi-index (J, T)
2. A LaTeX table for paper inclusion

Metrics computed for each method (Plugin, ADMLIV):
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

def load_results(results_dir: str, T_list: list, J_list: list) -> dict:
    """
    Load results from CSV files for different T and J combinations.

    Parameters
    ----------
    results_dir : str
        Directory containing result CSV files
    T_list : list
        List of T values (number of markets)
    J_list : list
        List of J values (number of products)

    Returns
    -------
    results : dict
        Dictionary mapping (J, T) tuple to DataFrame
    """
    results = {}

    for j in J_list:
        for t in T_list:
            filename = f'mc_results_elasticity_T{t}_J{j}.csv'
            filepath = os.path.join(results_dir, filename)

            if os.path.exists(filepath):
                results[(j, t)] = pd.read_csv(filepath)
                print(f"Loaded results for J={j}, T={t}: {len(results[(j, t)])} replications")
            else:
                print(f"Warning: File not found: {filepath}")

    return results


def compute_summary_statistics(results_dict: dict, iqr_multiplier: float = 10.0) -> pd.DataFrame:
    """
    Compute summary statistics for each method, J, and T.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping (J, T) tuple to results DataFrame
    iqr_multiplier : float, default=10.0
        IQR multiplier for outlier detection

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics with multi-index (J, T) and columns for each metric/method
    """
    summary_data = []

    # Define method names and their column prefixes
    methods = [
        ('plugin', 'PI'),   # Plug-in
        ('admliv', 'ADMLIV') # ADMLIV
    ]

    for (j, t), df in sorted(results_dict.items()):
        row = {'J': j, 'T': t}

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

        for method_col, method_name in methods:
            bias_col = f'{method_col}_bias'
            theta_col = f'{method_col}_theta'
            se_col = f'{method_col}_se'
            mse_col = f'{method_col}_mse'
            cp_col = f'{method_col}_cp'

            if bias_col in df_clean.columns:
                # Absolute Bias (absolute value of mean bias)
                row[f'|Bias|_{method_name}'] = np.abs(df_clean[bias_col].mean())

                # SD (standard deviation of theta estimates)
                row[f'SD_{method_name}'] = df_clean[theta_col].std()

                # Median SE
                row[f'SE_{method_name}'] = df_clean[se_col].median()

                # RMSE
                row[f'RMSE_{method_name}'] = np.sqrt(df_clean[mse_col].mean())

                # Coverage
                row[f'Coverage_{method_name}'] = df_clean[cp_col].mean()
            else:
                row[f'|Bias|_{method_name}'] = np.nan
                row[f'SD_{method_name}'] = np.nan
                row[f'SE_{method_name}'] = np.nan
                row[f'RMSE_{method_name}'] = np.nan
                row[f'Coverage_{method_name}'] = np.nan

        summary_data.append(row)

    # Create DataFrame with multi-index
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index(['J', 'T'])

    return summary_df


def format_table(summary_df: pd.DataFrame) -> str:
    """
    Format summary table as a nicely formatted string.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame with multi-index (J, T)

    Returns
    -------
    table : str
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append("Monte Carlo Results Summary: Own-Price Elasticity")
    lines.append("=" * 100)

    # Header
    header = f"{'J':<5} {'T':<8} {'Metric':<15} {'PI':>12} {'ADMLIV':>12}"
    lines.append(header)
    lines.append("-" * 100)

    # Data rows
    metrics = [
        ('|Bias|', 'Abs. Bias'),
        ('SD', 'SD'),
        ('SE', 'Median SE'),
        ('RMSE', 'RMSE'),
        ('Coverage', 'Coverage')
    ]

    for (j, t), row in summary_df.iterrows():
        # Print k and n_obs only on the first metric line for each (k, n_obs)
        for i, (metric_key, metric_label) in enumerate(metrics):
            if i == 0:
                k_str = f"{j:<5}"
                n_str = f"{t:<8}"
            else:
                k_str = f"{'':<5}"
                n_str = f"{'':<8}"

            pi_val = row[f'{metric_key}_PI']
            admliv_val = row[f'{metric_key}_ADMLIV']

            if not np.isnan(pi_val):
                line = f"{k_str} {n_str} {metric_label:<15} {pi_val:>12.4f} {admliv_val:>12.4f}"
            else:
                line = f"{k_str} {n_str} {metric_label:<15} {'---':>12} {'---':>12}"

            lines.append(line)

        # Add separator between different (J, T) combinations
        lines.append("-" * 100)

    lines.append("=" * 100)

    return "\n".join(lines)


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """
    Generate LaTeX table from summary statistics.

    Format: rows indexed by (J, T), columns grouped by metric (Bias, Med SE, Coverage)
    with sub-columns for each method (PI, ADMLIV).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame with multi-index (J, T)

    Returns
    -------
    latex : str
        LaTeX table code
    """
    lines = []

    # Table preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Monte Carlo Results: Own-Price Elasticity}")
    lines.append("\\label{tab:mc_wad}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{cc cc cc cc cc}")
    lines.append("\\toprule")

    # Two-row header with metric groups
    lines.append(" & & \\multicolumn{2}{c}{Bias} & \\multicolumn{2}{c}{SD} & \\multicolumn{2}{c}{Med SE} & \\multicolumn{2}{c}{Coverage} \\\\")
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}")
    lines.append("$J$ & $T$ & PI & ADMLIV & PI & ADMLIV & PI & ADMLIV & PI & ADMLIV \\\\")
    lines.append("\\midrule")

    # Data rows - one row per (k, n_obs)
    for (j, t), row in summary_df.iterrows():
        # Bias values
        bias_pi = row['|Bias|_PI']
        bias_admliv = row['|Bias|_ADMLIV']

        # SD values
        sd_pi = row['SD_PI']
        sd_admliv = row['SD_ADMLIV']

        # Median SE values
        se_pi = row['SE_PI']
        se_admliv = row['SE_ADMLIV']

        # Coverage values
        cov_pi = row['Coverage_PI']
        cov_admliv = row['Coverage_ADMLIV']

        if not np.isnan(bias_pi):
            line = (
                f"{j} & {t} & "
                f"{bias_pi:.3f} & {bias_admliv:.3f} & "
                f"{sd_pi:.3f} & {sd_admliv:.3f} & "
                f"{se_pi:.3f} & {se_admliv:.3f} & "
                f"{cov_pi*100:.1f}\\% & {cov_admliv*100:.1f}\\% \\\\"
            )
        else:
            line = f"{j} & {t} & --- & --- & --- & --- & --- & --- & --- & --- \\\\"

        lines.append(line)

    # Table closing
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item Notes: Results averaged over Monte Carlo replications. ")
    lines.append("PI = Plug-in estimator; ")
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
    J_list = [2, 5]
    T_list = [100, 200, 400, 800, 1000]

    print("\n" + "=" * 80)
    print("Monte Carlo Results Summary Generator - Own-Price Elasticity")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"J values: {J_list}")
    print(f"T values: {T_list}")
    print("=" * 80 + "\n")

    # Load results
    results_dict = load_results(results_dir, T_list, J_list)

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
    table_file = output_dir / 'summary_elasticity_table.txt'
    with open(table_file, 'w') as f:
        f.write(formatted_table)
    print(f"\nFormatted table saved to: {table_file}")

    # Save LaTeX table
    latex_file = output_dir / 'summary_elasticity_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Save summary DataFrame as CSV
    csv_file = output_dir / 'summary_elasticity_statistics.csv'
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
