# admliv/simulations/toy_model/mc_wa.py

"""
Monte Carlo Comparison: Toy model with Weighted Average Derivative Functional
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataclasses import dataclass, field
from typing import Dict, Optional
from numpy.typing import NDArray
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from admliv.utils.featurizers import CoordinatePolyTransform
from admliv.moments import WeightedAverageDerivative
from admliv.core.pgmm import PGMMControl
from admliv.main import ADMLIV, ADMLIVControl
from admliv.estimators.sieve import DoubleLassoEstimator, LassoStageControl, DoubleLassoControl

# Import DML with Analytical RR
from utils.dml_analytical_ad import DMLAnalyticalAD, DMLAnalyticalADControl


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_runs: int = 1000
    n_obs: int = 100
    k: int = 5
    iv_strength: float = 0.8
    r: float = 0.5
    seed: int = 1111
    
    # Method-specific configs
    admliv_n_folds: int = 5
    admliv_use_adaptive_pgmm: bool = True
    admliv_use_cv_for_pgmm: bool = False
    admliv_feat_degree: int = 3

    pgmm_c: float = 0.0001

    dlasso_feat_degree: int = 3
    dlasso_cv: int = 3
    dlasso_fs_alpha: float = 0.0001
    dlasso_tol: float = 0.001
    dlasso_alphas: NDArray[np.float64] = field(default_factory=lambda: np.logspace(-7, -1, 100))

    # DML Analytical RR configs
    dml_n_folds: int = 5
    dml_feat_degree: int = 3

class MonteCarloWAD:
    """
    Monte Carlo simulation for nonlinear IV model.

    Estimate weighted average derivative of a simple nonlinear function.
    Compares three methods:
    - ADMLIV: Debiased estimator using PGMM for Riesz representer
    - DML: Debiased estimator using analytical RR (Chen et al. 2023)
    - Plug-in: Naive plug-in estimator (biased)

    Uses Double Lasso to estimate the underlying structural function.

    Parameters
    ----------
    config : MonteCarloConfig
        Configuration parameters for the simulation
    verbose : bool, default=True
        If True, prints progress during simulation

    Attributes
    ----------
    results_ : pd.DataFrame
        Results DataFrame with Bias, SD, RMSE and Coverage for all methods
    summary_ : pd.DataFrame
        Summary statistics (Bias, SD, RMSE and Coverage) for each method
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None, verbose: bool = True):
        self.config = config if config is not None else MonteCarloConfig()
        self.verbose = verbose
        self.results_ = None
        self.summary_ = None

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
            Dictionary with 'Y', 'X', 'Z', 'gamma_0' keys
        """
        np.random.seed(seed)

        # validate inputs: require at least two covariates 
        if self.config.k < 2:
            raise ValueError("The DGP requires at least two covariates")

        # draw (X, Z, U) tuples
        Sigma = np.eye(3)
        Sigma[0, 1] = Sigma[1, 0] = self.config.iv_strength
        Sigma[0, 2] = Sigma[2, 0] = self.config.r
        np.all(np.linalg.eigvals(Sigma) > 0)
        xzu = np.random.multivariate_normal(mean=np.zeros(3), cov=Sigma, size=self.config.n_obs * self.config.k)
        X = xzu[:, 0].reshape(self.config.n_obs, self.config.k)
        Z = xzu[:, 1].reshape(self.config.n_obs, self.config.k)
        U = xzu[:, 2].reshape(self.config.n_obs, self.config.k) / np.sqrt(self.config.k)
        # structural function
        gamma_0 = X[:, 0] + np.array([np.exp(-X[i, 1:] @ X[i, 1:].T / 2) for i in range(self.config.n_obs)])
 
        # dependent variable
        V = U.sum(axis=1)
        Y = gamma_0 + V
        
        return {'X': X, 'Y': Y, 'Z': Z, 'gamma_0': gamma_0, 'theta_0': 1.0}
    
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
            Theta_hat, SE, Bias, MSE and Coverage for ADMLIV and plug-in 
            (keys: Method_theta', 'Method_se', 'Method_bias', 'Method_mse', 'Method_cp')
        """
        data = self._generate_data(seed)
        Y, X, Z, theta_0 = data['Y'], data['X'], data['Z'], data['theta_0']
        W = {'Y': Y, 'X': X, 'Z': Z}
        
        results = {}

        try:
            # specify MLIV factory
            def mliv_factory():
                x_feat = PolynomialFeatures(degree=self.config.dlasso_feat_degree, include_bias=False)
                z_feat = PolynomialFeatures(degree=self.config.dlasso_feat_degree, include_bias=False)
                fs = LassoStageControl(
                    use_cv=False, 
                    alpha=self.config.dlasso_fs_alpha, 
                    tol=self.config.dlasso_tol
                    )
                ss = LassoStageControl(
                    use_cv=True, 
                    cv=self.config.dlasso_cv, 
                    alphas=self.config.dlasso_alphas, 
                    tol=self.config.dlasso_tol
                    )
                control = DoubleLassoControl(first_stage=fs, second_stage=ss)
            
                return DoubleLassoEstimator(
                    x_featurizer=x_feat,
                    z_featurizer=z_feat,
                    control=control
                )
            
            # specify ADMLIV
            control = ADMLIVControl(
                n_folds=self.config.admliv_n_folds, 
                use_adaptive_pgmm=self.config.admliv_use_adaptive_pgmm,
                use_cv_for_pgmm=self.config.admliv_use_cv_for_pgmm,
                pgmm_control=PGMMControl(c=self.config.pgmm_c),
                verbose=False
                )
            admliv = ADMLIV(
                mliv_estimator=mliv_factory,
                x_featurizer=PolynomialFeatures(degree=self.config.admliv_feat_degree),
                z_featurizer=PolynomialFeatures(degree=self.config.admliv_feat_degree),
                control=control
            )
            
            # specify moment function
            moment = WeightedAverageDerivative()
            def weight_func(X):
                return np.ones(X.shape[0])
            
            # run estimator
            admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

            # Theta_hat
            results['admliv_theta'] = admliv.result_.theta_debiased
            results['plugin_theta'] = admliv.result_.theta_plugin

            # Theta_hat SE
            results['admliv_se'] = admliv.result_.se_debiased
            results['plugin_se'] = admliv.result_.se_plugin

            # Bias
            results['admliv_bias'] = admliv.result_.theta_debiased - theta_0
            results['plugin_bias'] = admliv.result_.theta_plugin - theta_0

            # MSE
            results['admliv_mse'] = (admliv.result_.theta_debiased - theta_0) ** 2
            results['plugin_mse'] = (admliv.result_.theta_plugin - theta_0) ** 2

            # Coverage probabilities
            results['admliv_cp'] = admliv.result_.ci_lower <= theta_0 <= admliv.result_.ci_upper
            results['plugin_cp'] = admliv.result_.ci_lower_plugin <= theta_0 <= admliv.result_.ci_upper_plugin
        except Exception as e:
            if self.verbose:
                print(f"  ADMLIV failed: {type(e).__name__}: {e}")

            # Theta_hat
            results['admliv_theta'] = np.nan
            results['plugin_theta'] = np.nan

            # Theta_hat SE
            results['admliv_se'] = np.nan
            results['plugin_se'] = np.nan

            # Bias
            results['admliv_bias'] = np.nan
            results['plugin_bias'] = np.nan

            # MSE
            results['admliv_mse'] = np.nan
            results['plugin_mse'] = np.nan

            # Coverage probabilities
            results['admliv_cp'] = np.nan
            results['plugin_cp'] = np.nan

        # DML with Analytical RR
        try:
            control_dml = DMLAnalyticalADControl(
                n_folds=self.config.dml_n_folds,
                verbose=False
            )
            dml = DMLAnalyticalAD(
                mliv_estimator=mliv_factory(),
                # x_featurizer=PolynomialFeatures(degree=self.config.dml_feat_degree),
                # z_featurizer=PolynomialFeatures(degree=self.config.dml_feat_degree),
                x_featurizer=CoordinatePolyTransform(degree=self.config.dml_feat_degree, pairwise_interactions=True),
                z_featurizer=CoordinatePolyTransform(degree=self.config.dml_feat_degree, pairwise_interactions=True),
                control=control_dml
            )
            dml.fit(W, moment, weight_func=weight_func, deriv_index=0)

            results['dml_theta'] = dml.result_.theta_debiased
            results['dml_se'] = dml.result_.se_debiased
            results['dml_bias'] = dml.result_.theta_debiased - theta_0
            results['dml_mse'] = (dml.result_.theta_debiased - theta_0) ** 2
            results['dml_cp'] = dml.result_.ci_lower <= theta_0 <= dml.result_.ci_upper
        except Exception as e:
            if self.verbose:
                print(f"  DML failed: {type(e).__name__}: {e}")
            results['dml_theta'] = np.nan
            results['dml_se'] = np.nan
            results['dml_bias'] = np.nan
            results['dml_mse'] = np.nan
            results['dml_cp'] = np.nan

        return results
    
    def run(self) -> pd.DataFrame:
        """
        Run the full Monte Carlo simulation.
        
        Returns
        -------
        results : pd.DataFrame
            Results DataFrame with Theta_hat, SE, Bias, MSE and Coverage for each method and run
        """

        if self.verbose:
            print("\n" + "=" * 70)
            print("Monte Carlo Simulation: Toy Model")
            print("=" * 70)
            print(f"Number of replications: {self.config.n_runs}")
            print(f"Sample size: {self.config.n_obs}")
            print(f"Number of covariates and instruments: {self.config.k}")
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
        methods = ['admliv', 'dml', 'plugin']
        summary_data = []

        for method in methods:
            theta_col = f'{method}_theta'
            se_col = f'{method}_se'
            bias_col = f'{method}_bias'
            mse_col = f'{method}_mse'
            cp_col = f'{method}_cp'
            summary_data.append({
                'Method': method,
                'BIAS': self.results_[bias_col].mean(),
                'SE': self.results_[se_col].median(),
                'SD': self.results_[theta_col].std(),
                'RMSE': np.sqrt(self.results_[mse_col].mean()),
                'Coverage': self.results_[cp_col].mean()
            })

        self.summary_ = pd.DataFrame(summary_data).set_index('Method')

        if self.verbose:
            print("\n" + "=" * 70)
            print("Results Summary")
            print("=" * 70)
            print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
            print(f"\n{'Method':<12} {'Bias':>10} {'SE':>10} {'SD':>10} {'RMSE':>10} {'Coverage':>10}")
            print("-" * 55)
            for method in self.summary_.index:
                row = self.summary_.loc[method]
                print(f"{method:<12} {row['BIAS']:>10.4f} {row['SD']:>10.4f} {row['SE']:>10.4f} "
                      f"{row['RMSE']:>10.4f} {row['Coverage']:>10.4f}")
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
    
    parser = argparse.ArgumentParser(description='Monte Carlo | Toy Model | Weighted Average Derivative Functional')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of Monte Carlo replications')
    parser.add_argument('--n_obs', type=int, default=100, help='Number of observations')
    parser.add_argument('--k', type=int, default=5, help='Number of covariates and instruments')
    parser.add_argument('--iv_strength', type=float, default=0.8, help='IV strength')
    parser.add_argument('--r', type=float, default=0.5, help='Endogeneity strength')
    parser.add_argument('--seed', type=int, default=1111, help='Base random seed')
    parser.add_argument('--pgmm_c', type=float, default=None, help='PGMM penalty constant (default: 0.01 for k=2, 0.001 for k=5, 0.0001 for k=10)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: results/mc_results_toy_model_k{k}_n{n_obs}.csv)')

    args = parser.parse_args()

    # Set default output filename with n_obs
    if args.output is None:
        script_dir = Path(__file__).parent
        results_dir = script_dir / 'results'
        args.output = str(results_dir / f'mc_results_toy_model_wad_k{args.k}_n{args.n_obs}.csv')

    # Default pgmm_c depends on k (see hyperparameters.txt)
    if args.pgmm_c is not None:
        pgmm_c = args.pgmm_c
    elif args.k <= 2:
        pgmm_c = 0.01
    elif args.k <= 5:
        pgmm_c = 0.001
    else:
        pgmm_c = 0.0001

    config = MonteCarloConfig(
        n_runs=args.n_runs,
        n_obs=args.n_obs,
        k=args.k,
        iv_strength=args.iv_strength,
        r=args.r,
        seed=args.seed,
        pgmm_c=pgmm_c,
    )
    
    mc = MonteCarloWAD(config=config, verbose=True)
    mc.run()
    mc.save_results(args.output)


if __name__ == '__main__':
    main()
