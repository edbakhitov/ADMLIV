# admliv/simulations/linear_model/mc_hd_linear_iv.py

"""
Monte Carlo simulation for high-dimensional linear IV regression.
    
Compares PGMM and Double Lasso for recovering sparse coefficients 
in linear IV regression.

Design (follows Belloni et al. 2012 exponential design):
    Y = X'β₀ + ε,  ε ~ N(0, 1)
    X = ΠZ + v,    first stage relationship
    
    β₀ = (1, 1, 1, 0, 0, ...) with dim(β₀) = 101 (including intercept)
    X = (1, X₁, ..., X₁₀₀)'
    Z = (Z₁, ..., Z₁₅₀) ~ N(0, Σ_Z), 150×1 vector
    
    Correlation structure: Corr(Z_h, Z_j) = 0.5^|h-j| (Toeplitz)
    First stage coefficients: Π = (1, 0.7, 0.7², ..., 0.7¹⁴⁹)
    
    Error structure:
        ε ~ N(0, 1)
        v|ε ~ N(rε·1, (1-r²)I), so Var(v) = I unconditionally
        r = 0.5 (endogeneity parameter)
    
    n = 100 observations

Methods compared:
    1. Double Lasso (Lasso in both stages)
    2. PGMM (Penalized GMM)
    3. A-PGMM (Penalized GMM with adaptive weights)
    4. PGMM-CV (Cross-validated PGMM)
    5. A-PGMM-CV (Cross-validated PGMM with adaptive weights)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from numpy.typing import NDArray
import time
import numpy as np
import pandas as pd

from admliv.utils.featurizers import SimpleFeaturizer
from admliv.core.control import PGMMControl, PGMMCVControl
from admliv.core.pgmm_linear_iv import PGMMLinearIV
from admliv.core.pgmm_linear_iv_cv import PGMMLinearIVCV
from admliv.estimators.sieve import DoubleLassoEstimator, LassoStageControl, DoubleLassoControl


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_runs: int = 100
    n_obs: int = 100
    dx: int = 100
    dz: int = 150
    r: float = 0.5
    seed: int = 1111
    n_nonzero: int = 3
    
    # Method-specific configs
    pgmm_c: float = 0.01
    pgmm_c_ada: float = 0.01
    pgmm_cv_folds: int = 5
    pgmm_c_vec: List[float] = field(default_factory=lambda: [0.005, 0.0075, 0.01, 0.0125, 0.015])
    pgmm_c_vec_ada: List[float] = field(default_factory=lambda: [0.005, 0.0075, 0.01, 0.0125, 0.015])
    
    
    dlasso_cv: int = 5
    dlasso_fs_alpha: float = 0.0001
    dlasso_alphas: NDArray[np.float64] = field(default_factory=lambda: np.logspace(-7, -1, 100))


class MonteCarloHDLinearIV:
    """
    Monte Carlo simulation for high-dimensional linear IV regression.
    
    Compares PGMM and Double Lasso for recovering
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
        Results DataFrame with MSE and R^2 for each method and run
    summary_ : pd.DataFrame
        Summary statistics (mean, std, median MSE and R^2) for each method
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

        n = self.config.n_obs
        dx = self.config.dx
        dz = self.config.dz
        r = self.config.r
        
        # True coefficients: β₀ = (1, 1, 1, 0, ..., 0)
        beta_0 = np.zeros(dx + 1)  # +1 for intercept
        beta_0[:self.config.n_nonzero] = 1.0
        
        # Generate Z ~ N(0, Σ_Z) with Toeplitz correlation
        Sigma_z = np.zeros((dz, dz))
        for h in range(dz):
            for j in range(dz):
                Sigma_z[h, j] = 0.5 ** np.abs(h - j)
        
        Z = np.random.multivariate_normal(
            mean=np.zeros(dz),
            cov=Sigma_z,
            size=n
        )
        
        # Generate structural errors
        eps = np.random.normal(0, 1, size=n)
        
        # Generate first-stage errors v|ε ~ N(rε·1, (1-r²)I)
        # This ensures Cov(v) = I unconditionally
        v = np.vstack([
            np.random.multivariate_normal(
                mean=r * np.ones(dx) * e,
                cov=np.eye(dx) * (1 - r ** 2),
                size=1
            )
            for e in eps
        ]).squeeze()
        
        # First stage coefficients: Π_j = 0.7^(j-1)
        pi = np.array([0.7 ** j for j in range(dz)])
        Pi = np.tile(pi, (dx, 1))  # Each X_j has same relationship with Z
        
        # Generate X = ΠZ + v
        X = Z @ Pi.T + v
        
        # Generate Y = X'β₀ + ε (with intercept)
        X_with_intercept = np.c_[np.ones(n), X]
        Y = X_with_intercept @ beta_0 + eps
        
        return {
            'Y': Y,
            'X': X,
            'Z': Z,
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
        Y, X, Z, beta_0 = data['Y'], data['X'], data['Z'], data['beta_0']
        W = {'Y': Y, 'X': X, 'Z': Z}
        
        results = {}

        # Double Lasso (CV in 2nd stage only)
        try:
            x_feat = SimpleFeaturizer(include_bias=False)
            z_feat = SimpleFeaturizer(include_bias=False)
            fs = LassoStageControl(use_cv=False, alpha=self.config.dlasso_fs_alpha)
            ss = LassoStageControl(use_cv=True, cv=self.config.dlasso_cv, alphas=self.config.dlasso_alphas)
            control = DoubleLassoControl(first_stage=fs, second_stage=ss)
            dlasso = DoubleLassoEstimator(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                control=control
            )
            dlasso.fit(W)
            rho = dlasso.coef_
            Y_pred = dlasso.predict(X)
            results['DLasso_MSE'] = np.sum((rho - beta_0) ** 2)
            results['DLasso_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception as e:
            if self.verbose:
                print(f"  DLasso failed: {type(e).__name__}: {e}")
            results['DLasso_MSE'] = np.nan
            results['DLasso_R2'] = np.nan

        # PGMM
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            control = PGMMControl(c=self.config.pgmm_c)
            pgmm = PGMMLinearIV(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=False,
                control=control,
                verbose=False
            )
            pgmm.fit(W)
            rho = pgmm.get_rho()
            Y_pred = pgmm.predict(X)
            results['PGMM_MSE'] = np.sum((rho - beta_0) ** 2)
            results['PGMM_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception as e:
            if self.verbose:
                print(f"  PGMM failed: {type(e).__name__}: {e}")
            results['PGMM_MSE'] = np.nan
            results['PGMM_R2'] = np.nan

        # PGMM-CV
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            control = PGMMCVControl(
                n_folds=self.config.pgmm_cv_folds,
                c_vec=np.array(self.config.pgmm_c_vec)
            )
            pgmm_cv = PGMMLinearIVCV(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=False,
                control=control,
                verbose=False
            )
            pgmm_cv.fit(W)
            rho = pgmm_cv.get_rho()
            Y_pred = pgmm_cv.predict(X)
            results['PGMM-CV_MSE'] = np.sum((rho - beta_0) ** 2)
            results['PGMM-CV_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception as e:
            if self.verbose:
                print(f"  PGMM-CV failed: {type(e).__name__}: {e}")
            results['PGMM-CV_MSE'] = np.nan
            results['PGMM-CV_R2'] = np.nan

        # A-PGMM
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            control = PGMMControl(c=self.config.pgmm_c_ada)
            a_pgmm = PGMMLinearIV(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=True,
                control=control,
                verbose=False
            )
            a_pgmm.fit(W)
            rho = a_pgmm.get_rho()
            Y_pred = a_pgmm.predict(X)
            results['A-PGMM_MSE'] = np.sum((rho - beta_0) ** 2)
            results['A-PGMM_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception as e:
            if self.verbose:
                print(f"  A-PGMM failed: {type(e).__name__}: {e}")
            results['A-PGMM_MSE'] = np.nan
            results['A-PGMM_R2'] = np.nan
        
        # A-PGMM-CV
        try:
            x_feat = SimpleFeaturizer(include_bias=True)
            z_feat = SimpleFeaturizer(include_bias=True)
            control = PGMMCVControl(
                n_folds=self.config.pgmm_cv_folds,
                c_vec=np.array(self.config.pgmm_c_vec_ada)
            )
            a_pgmm_cv = PGMMLinearIVCV(
                x_featurizer=x_feat,
                z_featurizer=z_feat,
                adaptive=True,
                control=control,
                verbose=False
            )
            a_pgmm_cv.fit(W)
            rho = a_pgmm_cv.get_rho()
            Y_pred = a_pgmm_cv.predict(X)
            results['A-PGMM-CV_MSE'] = np.sum((rho - beta_0) ** 2)
            results['A-PGMM-CV_R2'] = self._compute_pred_r_squared(Y, Y_pred)
        except Exception as e:
            if self.verbose:
                print(f"  A-PGMM-CV failed: {type(e).__name__}: {e}")
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
            print("Monte Carlo Simulation: HD Linear IV Regression")
            print("=" * 70)
            print(f"Number of replications: {self.config.n_runs}")
            print(f"Sample size: {self.config.n_obs}")
            print(f"Number of parameters: {self.config.dx + 1}")
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
        methods = ['DLasso', 'PGMM', 'PGMM-CV', 'A-PGMM', 'A-PGMM-CV']
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

    parser = argparse.ArgumentParser(description='Monte Carlo comparison of HD linear IV regression methods')
    parser.add_argument('--n_runs', type=int, default=100, help='Number of Monte Carlo replications')
    parser.add_argument('--n_obs', type=int, default=1000, help='Number of observations')
    parser.add_argument('--dx', type=int, default=100, help='Number of endogenous covariates (excluding intercept)')
    parser.add_argument('--dz', type=int, default=150, help='Number of instruments (excluding intercept)')
    parser.add_argument('--r', type=float, default=0.5, help='Endogeneity strength')
    parser.add_argument('--seed', type=int, default=1111, help='Base random seed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: mc_results_hd_linear_iv_n{n_obs}.csv)')

    args = parser.parse_args()

    # Set default output filename with n_obs (relative to script directory)
    if args.output is None:
        script_dir = Path(__file__).parent
        results_dir = script_dir / 'results'
        args.output = str(results_dir / f'mc_results_hd_linear_iv_n{args.n_obs}.csv')
    
    # A-PGMM penalty depends on sample size (see hyperparameters.txt)
    if args.n_obs <= 100:
        pgmm_c_ada = 0.01
        pgmm_c_vec_ada = [0.005, 0.0075, 0.01, 0.0125, 0.015]
    else:
        pgmm_c_ada = 0.001
        pgmm_c_vec_ada = [0.0005, 0.00075, 0.001, 0.00125, 0.0015]

    config = MonteCarloConfig(
        n_runs=args.n_runs,
        n_obs=args.n_obs,
        dx=args.dx,
        dz=args.dz,
        r=args.r,
        seed=args.seed,
        pgmm_c_ada=pgmm_c_ada,
        pgmm_c_vec_ada=pgmm_c_vec_ada,
    )
    
    mc = MonteCarloHDLinearIV(config=config, verbose=True)
    mc.run()
    mc.save_results(args.output)


if __name__ == '__main__':
    main()
    