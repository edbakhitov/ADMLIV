# admliv/simulations/linear_model/mc_hd_linear.py

"""
Monte Carlo Comparison: HD Linear Regression

This script replicates the Monte Carlo results from the paper
"Penalized GMM Framework for Inference on Functionals of Nonparametric
Instrumental Variable Estimators" by Bakhitov (2026).

Design:
    Y = X'β₀ + ε, ε ~ N(0, 1)
    X = (1, X₁, ..., X₁₀₀), where X_j ~ N(0, 1) i.i.d.
    β₀ = (1, 1, 1, 0, 0, ...) with dim(β₀) = 101
    n = 100 observations

Methods compared:
    1. SGD (sklearn LassoCV with stochastic gradient descent)
    2. LARS (sklearn LassoLarsCV with least angle regression)
    3. RMD Lasso (CNS method)
    4. RMD-CV (Cross-validated RMD Lasso)
    5. PGMM (Penalized GMM with Z=X, i.e., exogenous case)
    6. PGMM-CV (Cross-validated PGMM with Z=X)
    7. A-PGMM (Penalized GMM with adaptive weights)
    8. A-PGMM-CV (Cross-validated with adaptive weights)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LassoLarsCV

from admliv.utils.featurizers import SimpleFeaturizer
from admliv.moments.linear_model_moment import LinearModelMoment
from admliv.core.control import PGMMControl, PGMMCVControl
from admliv.core.pgmm import PGMM
from admliv.core.pgmm_cv import PGMMCV
from utils.rmd_lasso import RMDLasso
from utils.rmd_lasso_cv import RMDLassoCV
from utils.control import RMDControl, RMDCVControl


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_runs: int = 200
    n_obs: int = 100
    p_dim: int = 100
    seed: int = 1111
    n_nonzero: int = 3
    
    # Method-specific configs
    rmd_c: float = 0.5
    rmd_alpha: float = 0.1
    rmd_D_add: float = 0.2
    rmd_low_dim_divisor: int = 40
    rmd_cv_folds: int = 5
    rmd_c_vec: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.25])
    
    pgmm_c: float = 0.01
    pgmm_cv_folds: int = 5
    pgmm_c_vec: List[float] = field(default_factory=lambda: [0.005, 0.0075, 0.01, 0.0125, 0.015])
    
    sklearn_cv_folds: int = 5
    sklearn_alphas: int = 250


class MonteCarloHDLinear:
    """
    Monte Carlo simulation for high-dimensional linear regression.
    
    Compares RMD Lasso, PGMM, and sklearn methods for recovering
    sparse coefficients in linear regression.
    
    Parameters
    ----------
    config : MonteCarloConfig
        Configuration parameters for the simulation
    verbose : bool, default=True
        If True, prints progress during simulation
    
    Attributes
    ----------
    results_ : pd.DataFrame
        Results DataFrame with MSE for each method and run
    summary_ : pd.DataFrame
        Summary statistics (mean, std, median MSE) for each method
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None, verbose: bool = True):
        self.config = config if config is not None else MonteCarloConfig()
        self.verbose = verbose
        self.results_ = None
        self.summary_ = None
    
    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """Add intercept column to X."""
        return np.c_[np.ones(X.shape[0]), X]

    def _generate_data(self, seed: int) -> Dict[str, np.ndarray]:
        """
        Generate simulation data.
        
        Parameters
        ----------
        seed : int
            Random seed
            
        Returns
        -------
        data : dict
            Dictionary with 'Y', 'X', 'Z', 'beta_0' keys
        """
        np.random.seed(seed)
        
        p_total = self.config.p_dim + 1  # Including intercept
        beta_0 = np.zeros(p_total)
        beta_0[:self.config.n_nonzero] = 1.0
        
        X = np.random.normal(0, 1, size=(self.config.n_obs, self.config.p_dim))
        eps = np.random.normal(0, 1, size=self.config.n_obs)
        Y = self._add_intercept(X) @ beta_0 + eps
        
        return {
            'Y': Y,
            'X': X,
            'Z': X,  # Z = X for exogenous case
            'beta_0': beta_0
        }

    @staticmethod
    def _compute_r_squared(rho: np.ndarray, beta_0: np.ndarray) -> float:
        """
        Compute coefficient R-squared.
        
        R² = 1 - ||rho - beta_0||² / ||beta_0||²
        
        This measures how well the estimated coefficients match the true ones.
        """
        ss_res = np.sum((rho - beta_0) ** 2)
        ss_tot = np.sum(beta_0 ** 2)
        if ss_tot == 0:
            return np.nan
        return 1 - ss_res / ss_tot
    
    @staticmethod
    def _compute_pred_r_squared(Y: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Compute prediction R-squared.
        
        R² = 1 - SS_res / SS_tot = 1 - ||Y - Y_pred||² / ||Y - mean(Y)||²
        
        This is the standard R² used in regression.
        """
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        if ss_tot == 0:
            return np.nan
        return 1 - ss_res / ss_tot
    
    def _run_single_simulation(self, seed: int) -> Dict[str, float]:
        """
        Run a single Monte Carlo replication.
        
        Parameters
        ----------
        seed : int
            Random seed for this replication
            
        Returns
        -------
        results : dict
            MSE and R² for each method (keys: 'Method_MSE', 'Method_R2')
        """
        data = self._generate_data(seed)
        Y, X, beta_0 = data['Y'], data['X'], data['beta_0']
        W = {'Y': Y, 'X': X, 'Z': X}
        
        results = {}
        
        # SGD
        try:
            lasso = LassoCV(
                fit_intercept=True,
                alphas=np.logspace(-7, -1, self.config.sklearn_alphas),
                cv=self.config.sklearn_cv_folds,
                max_iter=2000,
                tol=1e-4
            )
            lasso.fit(X, Y)
            rho = np.append(lasso.intercept_, lasso.coef_)
            Y_pred = lasso.predict(X)
            results['SGD_MSE'] = np.sum((rho - beta_0) ** 2)
            results['SGD_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['SGD_MSE'] = np.nan
            results['SGD_R2'] = np.nan
        
        # LARS
        try:
            lasso = LassoLarsCV(
                fit_intercept=True,
                cv=self.config.sklearn_cv_folds,
                max_n_alphas=self.config.sklearn_alphas
            )
            lasso.fit(X, Y)
            rho = np.append(lasso.intercept_, lasso.coef_)
            Y_pred = lasso.predict(X)
            results['LARS_MSE'] = np.sum((rho - beta_0) ** 2)
            results['LARS_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['LARS_MSE'] = np.nan
            results['LARS_R2'] = np.nan
        
        # RMD
        try:
            featurizer = SimpleFeaturizer(include_bias=True)
            moment = LinearModelMoment()
            control = RMDControl(
                c=self.config.rmd_c,
                alpha=self.config.rmd_alpha,
                D_add=self.config.rmd_D_add,
                low_dim_divisor=self.config.rmd_low_dim_divisor
            )
            rmd = RMDLasso(x_featurizer=featurizer, control=control, verbose=False)
            rmd.fit(W, moment)
            rho = rmd.get_rho()
            Y_pred = rmd.predict(X)
            results['RMD_MSE'] = np.sum((rho - beta_0) ** 2)
            results['RMD_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['RMD_MSE'] = np.nan
            results['RMD_R2'] = np.nan
        
        # RMD-CV
        try:
            featurizer = SimpleFeaturizer(include_bias=True)
            moment = LinearModelMoment()
            control = RMDCVControl(
                alpha=self.config.rmd_alpha,
                D_add=self.config.rmd_D_add,
                n_folds=self.config.rmd_cv_folds,
                c_vec=self.config.rmd_c_vec,
                low_dim_divisor=self.config.rmd_low_dim_divisor
            )
            rmd_cv = RMDLassoCV(x_featurizer=featurizer, control=control, verbose=False)
            rmd_cv.fit(W, moment)
            rho = rmd_cv.get_rho()
            Y_pred = rmd_cv.predict(X)
            results['RMD-CV_MSE'] = np.sum((rho - beta_0) ** 2)
            results['RMD-CV_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['RMD-CV_MSE'] = np.nan
            results['RMD-CV_R2'] = np.nan
        
        # PGMM
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            moment = LinearModelMoment()
            control = PGMMControl(c=self.config.pgmm_c)
            pgmm = PGMM(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=False,
                control=control,
                verbose=False
            )
            pgmm.fit(W, moment)
            rho = pgmm.get_rho()
            Y_pred = pgmm.predict(X)
            results['PGMM_MSE'] = np.sum((rho - beta_0) ** 2)
            results['PGMM_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['PGMM_MSE'] = np.nan
            results['PGMM_R2'] = np.nan
        
        # PGMM-CV
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            moment = LinearModelMoment()
            control = PGMMCVControl(
                n_folds=self.config.pgmm_cv_folds,
                c_vec=np.array(self.config.pgmm_c_vec)
            )
            pgmm_cv = PGMMCV(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=False,
                control=control,
                verbose=False
            )
            pgmm_cv.fit(W, moment)
            rho = pgmm_cv.get_rho()
            Y_pred = pgmm_cv.predict(X)
            results['PGMM-CV_MSE'] = np.sum((rho - beta_0) ** 2)
            results['PGMM-CV_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['PGMM-CV_MSE'] = np.nan
            results['PGMM-CV_R2'] = np.nan

        # A-PGMM
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            moment = LinearModelMoment()
            control = PGMMControl(c=self.config.pgmm_c)
            a_pgmm = PGMM(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=True,
                control=control,
                verbose=False
            )
            a_pgmm.fit(W, moment)
            rho = a_pgmm.get_rho()
            Y_pred = a_pgmm.predict(X)
            results['A-PGMM_MSE'] = np.sum((rho - beta_0) ** 2)
            results['A-PGMM_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['A-PGMM_MSE'] = np.nan
            results['A-PGMM_R2'] = np.nan
        
        # A-PGMM-CV
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            moment = LinearModelMoment()
            control = PGMMCVControl(
                n_folds=self.config.pgmm_cv_folds,
                c_vec=np.array(self.config.pgmm_c_vec)
            )
            a_pgmm_cv = PGMMCV(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=True,
                control=control,
                verbose=False
            )
            a_pgmm_cv.fit(W, moment)
            rho = a_pgmm_cv.get_rho()
            Y_pred = a_pgmm_cv.predict(X)
            results['A-PGMM-CV_MSE'] = np.sum((rho - beta_0) ** 2)
            results['A-PGMM-CV_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception:
            results['A-PGMM-CV_MSE'] = np.nan
            results['A-PGMM-CV_R2'] = np.nan
        
        return results
    
    def run(self) -> pd.DataFrame:
        """
        Run the full Monte Carlo simulation.
        
        Returns
        -------
        results : pd.DataFrame
            Results DataFrame with MSE and R² for each method and run
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("Monte Carlo Simulation: HD Linear Regression")
            print("=" * 70)
            print(f"Number of replications: {self.config.n_runs}")
            print(f"Sample size: {self.config.n_obs}")
            print(f"Number of parameters: {self.config.p_dim + 1}")
            print(f"True non-zero coefficients: {self.config.n_nonzero}")
            print("=" * 70)
        
        results_list = []
        start_time = time.time()
        
        for i in range(self.config.n_runs):
            sim_seed = self.config.seed + i
            result = self._run_single_simulation(sim_seed)
            results_list.append(result)
            
            if self.verbose and (i + 1) % max(1, self.config.n_runs // 10) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (self.config.n_runs - i - 1)

                # Format elapsed time
                if elapsed < 60:
                    elapsed_str = f"{elapsed:.1f}s"
                elif elapsed < 3600:
                    elapsed_str = f"{elapsed/60:.1f}m"
                else:
                    elapsed_str = f"{elapsed/3600:.1f}h"

                # Format ETA
                if eta < 60:
                    eta_str = f"{eta:.1f}s"
                elif eta < 3600:
                    eta_str = f"{eta/60:.1f}m"
                else:
                    eta_str = f"{eta/3600:.1f}h"

                print(f"Progress: {i+1}/{self.config.n_runs} ({100*(i+1)/self.config.n_runs:.1f}%) - "
                      f"Elapsed: {elapsed_str} - ETA: {eta_str}")
        
        total_time = time.time() - start_time
        self.results_ = pd.DataFrame(results_list)
        
        # Compute summary for each method
        methods = ['SGD', 'LARS', 'RMD', 'RMD-CV', 'PGMM', 'PGMM-CV', 'A-PGMM', 'A-PGMM-CV']
        summary_data = []
        
        for method in methods:
            mse_col = f'{method}_MSE'
            r2_col = f'{method}_R2'
            if mse_col in self.results_.columns:
                summary_data.append({
                    'Method': method,
                    'Mean MSE': self.results_[mse_col].mean(),
                    'Std MSE': self.results_[mse_col].std(),
                    'Median MSE': self.results_[mse_col].median(),
                    'Mean R²': self.results_[r2_col].mean(),
                    'Std R²': self.results_[r2_col].std(),
                    'Median R²': self.results_[r2_col].median()
                })
        
        self.summary_ = pd.DataFrame(summary_data).set_index('Method')
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("Results Summary")
            print("=" * 70)
            print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
            print(f"\n{'Method':<12} {'Mean MSE':>10} {'Std MSE':>10} {'Mean R²':>10} {'Std R²':>10}")
            print("-" * 55)
            for method in self.summary_.index:
                row = self.summary_.loc[method]
                print(f"{method:<12} {row['Mean MSE']:>10.4f} {row['Std MSE']:>10.4f} "
                      f"{row['Mean R²']:>10.4f} {row['Std R²']:>10.4f}")
            print("=" * 70)
        
        return self.results_
    
    def save_results(self, filepath: str):
        """Save results to CSV file."""
        if self.results_ is None:
            raise ValueError("No results to save. Run simulation first.")
        self.results_.to_csv(filepath, index=False)
        if self.verbose:
            print(f"Results saved to '{filepath}'")
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics."""
        if self.summary_ is None:
            raise ValueError("No summary available. Run simulation first.")
        return self.summary_

def main():
    """Main entry point for Monte Carlo simulation."""
    import argparse
    from pathlib import Path

    
    parser = argparse.ArgumentParser(description='Monte Carlo comparison of HD linear regression methods')
    parser.add_argument('--n_runs', type=int, default=100, help='Number of Monte Carlo replications')
    parser.add_argument('--n_obs', type=int, default=1000, help='Number of observations')
    parser.add_argument('--p_dim', type=int, default=100, help='Number of covariates (excluding intercept)')
    parser.add_argument('--seed', type=int, default=1111, help='Base random seed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: mc_results_hd_linear_n{n_obs}.csv)')
    
    args = parser.parse_args()

    # Set default output filename with n_obs
    if args.output is None:
        script_dir = Path(__file__).parent
        results_dir = script_dir / 'results'
        args.output = str(results_dir / f'mc_results_hd_linear_n{args.n_obs}.csv')
    
    config = MonteCarloConfig(
        n_runs=args.n_runs,
        n_obs=args.n_obs,
        p_dim=args.p_dim,
        seed=args.seed
    )
    
    mc = MonteCarloHDLinear(config=config, verbose=True)
    mc.run()
    mc.save_results(args.output)


if __name__ == '__main__':
    main()