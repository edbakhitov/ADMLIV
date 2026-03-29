# admliv/simulations/demand_model/elasticities/mc_elasticity_logit.py

"""
Monte Carlo Comparison: Average Own-Price Elasticity Functional for Logit
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import Dict, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
import time
from scipy.stats import truncnorm

from admliv.utils.featurizers import CoordinatePolyTransform
from admliv.estimators.kiv import KIVEstimator
from admliv.main.admliv import MLIVEstimator
from admliv.estimators.sieve import (
    DoubleLassoEstimator, DoubleLassoControl, LassoStageControl
)
from utils.omega_transformer import OmegaTransformer
from utils.em_basis_featurizer import EMBasisFeaturizer
from utils.raw_data import RawData
from utils.admliv_elasticity import ADMLIVElasticity, ADMLIVElasticityControl
from utils.pgmm_elasticity import PGMMElasticityControl
from utils.own_price_elasticity import OwnPriceElasticity, get_omega_structure


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    dgp: str = 'uniform'
    n_runs: int = 200
    n_markets: int = 100
    n_products: int = 4
    K_1: int = 1
    K_2: int = 3
    K_w: int = 1
    seed: int = 1111
    product_id: int = 0  # wlog, focus on the first product in the market
    
    # Method-specific configs
    admliv_n_folds: int = 5
    admliv_use_adaptive_pgmm: bool = True
    admliv_feat_degree: int = 2
    admliv_high_cond_number: float = 100.0

    pgmm_c: float = 1e-7
    pgmm_featurizer: str = 'poly'
    compute_ill_cond_diagnostic: bool = False

    mliv: str = 'kiv'
    # KIV params
    kiv_bandwidth_method: str = 'median'
    kiv_bandwidth_scale: float = 25.0
    # DLasso params
    dl_fs_alpha: float = 1e-04
    dl_ss_alpha: float = 1e-08
    dl_feat_degree: int = 2
    dl_max_iter: int = 10000


class MonteCarloElasticity:
    """
    Monte Carlo simulation for semiparametric demand model.

    Estimate average own-price elasticity for one product.
    Compares two methods:
    - ADMLIV: Debiased estimator using PGMM for Riesz representer
    - Plug-in: Naive plug-in estimator 

    Uses KIV to estimate the inverse demand function.

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
        self.theta_0_pop_ = None

    def _compute_population_theta0(
        self, n_markets_large: int = 100_000, seed: int = 9999
    ) -> float:
        """
        Approximate population θ₀ = E[β_p · p_{jt} · (1 - s_{jt})] using a large sample.

        The population parameter is fixed across MC runs. We approximate it by
        drawing a very large number of markets and computing the sample average
        of true elasticities. This is computed once before the MC loop.

        Parameters
        ----------
        n_markets_large : int, default=100_000
            Number of markets for the approximation
        seed : int, default=9999
            Random seed for reproducibility

        Returns
        -------
        theta_0_pop : float
            Approximation of the population average own-price elasticity
        """
        saved_n_markets = self.config.n_markets
        self.config.n_markets = n_markets_large
        data = self._generate_data(seed)
        self.config.n_markets = saved_n_markets
        return data['elasticities'][self.config.product_id]

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
            Dictionary with 'price', 'x1', 'x2', 'shares', 'w', 'elasticities', 'market_ids' keys
        """
        np.random.seed(seed)

        # validate inputs: require at least one x_1  
        if self.config.K_1 < 1:
            raise ValueError("The DGP requires at least one linear characteristic")
        
        # generate market indices
        T, J = self.config.n_markets, self.config.n_products
        n = T * J
        market_ids = np.repeat(np.arange(T), J)
        
        beta_0 = np.array([-2.0, 1.0, -0.5, 0.5, 1.0])

        if self.config.dgp == 'tnormal':
            # Truncated Normal
            x_1 = truncnorm.rvs(-2, 2, loc=0, scale=1, size=(n, self.config.K_1))
            x_2 = truncnorm.rvs(-2, 2, loc=0, scale=1, size=(n, self.config.K_2))
            x = np.c_[x_1, x_2]
            w = truncnorm.rvs(-2, 2, loc=0, scale=1, size=(n, self.config.K_w))
            xi = truncnorm.rvs(-2, 2, loc=0, scale=1, size=(n, 1))
            e = truncnorm.rvs(-2, 2, loc=0, scale=1, size=(n, 1))
            price = 0.5 * np.abs(2 + 1.1 * (x.sum(1)[:, np.newaxis] + w) + 0.5 * xi + e)
        elif self.config.dgp == 'uniform':
            # Uniform
            x_1 = np.random.uniform(0, 1, size=(n, self.config.K_1))
            x_2 = np.random.uniform(0, 1, size=(n, self.config.K_2))
            x = np.c_[x_1, x_2]
            w = np.random.uniform(0, 1, size=(n, self.config.K_w))
            xi = np.random.normal(1, 0.15, size=(n, 1))
            e = np.random.uniform(0, 0.1, size=(n, 1))
            price = 0.5 * np.abs(1 + x.sum(1).reshape(n, 1) + xi + w + e)
        else:
            raise ValueError("Unrecognized dgp value. dgp should be either 'tnormal' or 'uniform'.")
        
        delta = (np.c_[price, x] @ beta_0)[:, np.newaxis] + xi
        
        # Compute logit shares (vectorized: reshape to T x J)
        delta_mat = delta.reshape(T, J)
        delta_max = delta_mat.max(axis=1, keepdims=True)
        exp_delta_mat = np.exp(delta_mat - delta_max)
        denom = np.exp(-delta_max) + exp_delta_mat.sum(axis=1, keepdims=True)
        shares_mat = exp_delta_mat / denom
        shares = shares_mat.flatten()

        # True average own-price elasticity for each product (vectorized)
        price_mat = price.reshape(T, J)
        elasticities = {}
        for j in range(J):
            # Logit elasticity: ε_{jj} = β_p * p * (1 - s)
            eps_j = beta_0[0] * price_mat[:, j] * (1 - shares_mat[:, j])
            elasticities[j] = float(np.mean(eps_j))

        return {
            'price': price,
            'x1': x_1,
            'x2': x_2,
            'shares': shares,
            'w': w,
            'elasticities': elasticities,
            'market_ids': market_ids
        }
    

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
        data_dict = self._generate_data(seed)
        raw_data = RawData(
            price=data_dict['price'],
            x1=data_dict['x1'],
            x2=data_dict['x2'],
            shares=data_dict['shares'],
            market_ids=data_dict['market_ids'],
            w=data_dict['w']
        )

        omega_transformer = OmegaTransformer(
            price_in_diffs=True,
            include_prices=True,
            include_shares=True,
            share_representation='all'
        )

        omega_iv_transformer = OmegaTransformer(
            include_prices=False,
            include_shares=False
        )

        if self.config.pgmm_featurizer == 'poly':
            omega_featurizer = CoordinatePolyTransform(
                degree=self.config.admliv_feat_degree,
                pairwise_interactions=True,
                include_bias=True
            )
            omega_iv_featurizer = CoordinatePolyTransform(
                degree=self.config.admliv_feat_degree,
                pairwise_interactions=False,
                include_bias=True
            )
        elif self.config.pgmm_featurizer == 'em':
            J = self.config.n_products
            # Fit transformers first so get_omega_structure can access n_characteristics_
            omega_transformer.fit(
                raw_data.x2, raw_data.market_ids,
                price=raw_data.price, shares=raw_data.shares
            )
            x_w = np.c_[raw_data.x1[:, np.newaxis] if raw_data.x1.ndim == 1 else raw_data.x1,
                         raw_data.x2,
                         raw_data.w if raw_data.w.ndim == 2 else raw_data.w[:, np.newaxis]]
            omega_iv_transformer.fit(x_w, raw_data.market_ids)
            omega_struct = get_omega_structure(omega_transformer, J)
            omega_iv_struct = get_omega_structure(omega_iv_transformer, J)

            moment_featurizer = CoordinatePolyTransform(
                degree=self.config.admliv_feat_degree,
                pairwise_interactions=True,
                include_bias=True
            )
            moment_iv_featurizer = CoordinatePolyTransform(
                degree=self.config.admliv_feat_degree,
                pairwise_interactions=False,
                include_bias=True
            )

            omega_featurizer = EMBasisFeaturizer(
                max_moment_order=3, min_moment_order=2, include_bias=False,
                moment_featurizer=moment_featurizer,
                omega_structure=omega_struct,
            )
            omega_iv_featurizer = EMBasisFeaturizer(
                max_moment_order=2, min_moment_order=1, include_bias=False,
                moment_featurizer=moment_iv_featurizer,
                omega_structure=omega_iv_struct,
            )
        else:
            raise ValueError("pgmm_featurizer not found! It should be either 'poly' or 'em'.")


        results = {}

        # True elasticity: population (fixed) and run-specific (varies)
        theta_0_pop = self.theta_0_pop_
        theta_0_run = data_dict['elasticities'][self.config.product_id]
        results['theta_0_pop'] = theta_0_pop
        results['theta_0_run'] = theta_0_run

        try:
            # specify MLIV factory
            if self.config.mliv == 'kiv':
                def mliv_factory():
                    return KIVEstimator(
                        bandwidth_method=self.config.kiv_bandwidth_method,
                        bandwidth_scale=self.config.kiv_bandwidth_scale
                    )
            elif self.config.mliv == 'double_lasso':
                def mliv_factory():
                    control = DoubleLassoControl.with_fixed_alpha(
                        fs_alpha=self.config.dl_fs_alpha,
                        ss_alpha=self.config.dl_ss_alpha,
                        max_iter=self.config.dl_max_iter,
                    )
                    x_featurizer = CoordinatePolyTransform(
                        degree=self.config.dl_feat_degree,
                        pairwise_interactions=True,
                        include_bias=False  
                    )
                    z_featurizer = CoordinatePolyTransform(
                        degree=self.config.dl_feat_degree + 1,  # use more instruments
                        pairwise_interactions=True,
                        include_bias=False
                    )
                    return DoubleLassoEstimator(
                        x_featurizer=x_featurizer,
                        z_featurizer=z_featurizer,
                        control=control
                    ) 
            else:
                raise ValueError("MLIV type not found. Should be either 'kiv' or 'double_lasso'.")
            
            # specify ADMLIV
            control = ADMLIVElasticityControl(
                n_folds=self.config.admliv_n_folds, 
                use_adaptive_pgmm=self.config.admliv_use_adaptive_pgmm,
                pgmm_control=PGMMElasticityControl(c=self.config.pgmm_c),
                random_state=42,
                verbose=False
            )

            admliv = ADMLIVElasticity(
                mliv_estimator=mliv_factory(),
                omega_transformer=omega_transformer,
                omega_iv_transformer=omega_iv_transformer,
                omega_featurizer=omega_featurizer,
                omega_iv_featurizer=omega_iv_featurizer,
                control=control
            )

            # run estimator
            admliv.fit(raw_data, product_ids=[self.config.product_id])
            admliv_result = admliv.results_[self.config.product_id]

            # Theta_hat
            results['admliv_theta'] = admliv_result.theta_debiased
            results['plugin_theta'] = admliv_result.theta_plugin

            # Theta_hat SE
            results['admliv_se'] = admliv_result.se_debiased
            results['plugin_se'] = admliv_result.se_plugin

            # Bias (relative to population θ₀)
            results['admliv_bias'] = admliv_result.theta_debiased - theta_0_pop
            results['plugin_bias'] = admliv_result.theta_plugin - theta_0_pop

            # MSE (relative to population θ₀)
            results['admliv_mse'] = (admliv_result.theta_debiased - theta_0_pop) ** 2
            results['plugin_mse'] = (admliv_result.theta_plugin - theta_0_pop) ** 2

            # Coverage (relative to population θ₀)
            results['admliv_cp'] = admliv_result.ci_lower <= theta_0_pop <= admliv_result.ci_upper
            results['plugin_cp'] = admliv_result.ci_lower_plugin <= theta_0_pop <= admliv_result.ci_upper_plugin

            # Number of ill-conditioned markets (optional, expensive diagnostic)
            if self.config.compute_ill_cond_diagnostic:
                results['n_ill_cond_markets'] = self._n_ill_conditioned_markets(
                    raw_data=raw_data,
                    mliv_estimator=mliv_factory(),
                    omega_transformer=omega_transformer,
                    omega_iv_transformer=omega_iv_transformer
                )
            else:
                results['n_ill_cond_markets'] = np.nan
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

            # Number of ill-conditioned markets
            results['n_ill_cond_markets'] = np.nan

        return results
    
    def _n_ill_conditioned_markets(
        self,
        raw_data: RawData,
        mliv_estimator: Union[MLIVEstimator, Callable[[], MLIVEstimator]],
        omega_transformer: OmegaTransformer,
        omega_iv_transformer: OmegaTransformer
    ) -> int:
        """
        Calculate the number of markets where A matrix is ill-conditioned.

        Parameters
        ----------
        raw_data : RawData
            Simulated data in RawData format
        mliv_estimator : Callable
            Factory function that returns a fresh MLIV estimator instance.
        omega_transformer : OmegaTransformer
            Transformer for constructing omega from raw data
        omega_iv_transformer : OmegaTransformer
            Transformer for constructing IV omega from raw data

        Returns
        -------
        n_ill_conditioned : int
            Number of ill-conditioned markets
        """

        # Construct training data
        if raw_data.has_instruments:
            x_w_full = np.c_[raw_data.x1, raw_data.x2, raw_data.w]
        else:
            x_w_full = np.c_[raw_data.x1, raw_data.x2]
        omega_full = omega_transformer.transform(
            raw_data.x2, raw_data.market_ids,
            price=raw_data.price, shares=raw_data.shares
        )
        omega_iv_full = omega_iv_transformer.transform(x_w_full, raw_data.market_ids)

        n = raw_data.n_obs
        T = raw_data.n_markets
        s0 = np.zeros(n)
        for t in range(T):
            mask = raw_data.market_ids == t
            s0[mask] = 1 - raw_data.shares[mask].sum()
        y = (np.log(raw_data.shares) - np.log(s0) - raw_data.x1).flatten()

        # Fit a single MLIV on full data to check its behavior
        mliv_estimator.fit({'X': omega_full, 'Z': omega_iv_full, 'Y': y})

        # Check A matrix conditioning for each market
        elasticity = OwnPriceElasticity(omega_transformer, product_id=self.config.product_id)
        cond_numbers = []

        for t in range(raw_data.n_markets):
            try:
                _, components = elasticity.compute_market_elasticity(
                    mliv_estimator, raw_data, t, self.config.product_id, return_components=True
                )
                A = components['A']
                cond = np.linalg.cond(A)
                cond_numbers.append(cond)
            except Exception as e:
                cond_numbers.append(np.nan)

        cond_arr = np.array(cond_numbers)
        high_cond_markets = np.where(cond_arr > self.config.admliv_high_cond_number)[0]

        return len(high_cond_markets)


    def _remove_outliers(
        self,
        df: pd.DataFrame,
        iqr_multiplier: float = 10.0
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers from results using IQR method.

        Outliers are detected based on bias and MSE columns for both methods.
        A row is considered an outlier if ANY of its bias/MSE values fall
        outside [Q1 - k*IQR, Q3 + k*IQR] where k = iqr_multiplier.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        iqr_multiplier : float, default=10.0
            Multiplier for IQR bounds. Higher values are more permissive.
            Default of 10.0 catches only extreme outliers (numerical failures).

        Returns
        -------
        df_clean : pd.DataFrame
            Cleaned DataFrame with outliers removed
        n_outliers : int
            Number of outliers removed
        """
        # Columns to check for outliers
        outlier_cols = ['admliv_bias', 'plugin_bias', 'admliv_mse', 'plugin_mse']

        # Initialize mask (True = keep)
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

            # Mark as outlier if outside bounds
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            mask = mask & (col_mask | df[col].isna())

        df_clean = df[mask].copy()
        n_outliers = len(df) - len(df_clean)

        return df_clean, n_outliers

    def run(self) -> pd.DataFrame:
        """
        Run the full Monte Carlo simulation.
        
        Returns
        -------
        results : pd.DataFrame
            Results DataFrame with Theta_hat, SE, Bias, MSE and Coverage for each method and run
        """

        # Compute population θ₀ via large pre-simulation draw
        if self.verbose:
            print("Computing population θ₀ via large sample (T=100,000)...")
        self.theta_0_pop_ = self._compute_population_theta0()

        if self.verbose:
            print("\n" + "=" * 70)
            print("Monte Carlo Simulation: Demand Model | Own-Price Elasticity")
            print("=" * 70)
            print(f"Population θ₀: {self.theta_0_pop_:.6f}")
            print(f"Number of replications: {self.config.n_runs}")
            print(f"Number of markets: {self.config.n_markets}")
            print(f"Number of products: {self.config.n_products}")
            print(f"PGMM penalty: {self.config.pgmm_c}")
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

        # Detect and remove outliers using IQR method on bias and MSE
        results_clean, n_outliers = self._remove_outliers(self.results_)
        self.results_clean_ = results_clean
        self.n_outliers_ = n_outliers

        # Compute summary for each method (on cleaned data)
        methods = ['admliv', 'plugin']
        summary_data = []

        for method in methods:
            theta_col = f'{method}_theta'
            se_col = f'{method}_se'
            bias_col = f'{method}_bias'
            mse_col = f'{method}_mse'
            cp_col = f'{method}_cp'
            ill_cond_col = 'n_ill_cond_markets'
            summary_data.append({
                'Method': method,
                'BIAS': results_clean[bias_col].mean(),
                'SE': results_clean[se_col].median(),
                'SD': results_clean[theta_col].std(),
                'RMSE': np.sqrt(results_clean[mse_col].mean()),
                'Coverage': results_clean[cp_col].mean(),
                '# ill-cond markets': results_clean[ill_cond_col].mean()
            })

        self.summary_ = pd.DataFrame(summary_data).set_index('Method')

        if self.verbose:
            print("\n" + "=" * 70)
            print("Results Summary")
            print("=" * 70)
            print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
            if n_outliers > 0:
                print(f"Outliers removed: {n_outliers} ({100*n_outliers/len(self.results_):.1f}%)")
            print(f"Valid replications: {len(results_clean)}")
            print(f"Average number of ill-conditioned markets: {results_clean[ill_cond_col].mean():.2f}")
            print(f"\n{'Method':<12} {'Bias':>10} {'SE':>10} {'SD':>10} {'RMSE':>10} {'Coverage':>10}") 
            print("-" * 70)
            for method in self.summary_.index:
                row = self.summary_.loc[method]
                print(f"{method:<12} {row['BIAS']:>10.4f} {row['SE']:>10.4f} {row['SD']:>10.4f} "
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
    
    parser = argparse.ArgumentParser(description='Monte Carlo | Demand Model |Average Own-Price Elasticity Functional')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of Monte Carlo replications')
    parser.add_argument('--n_markets', type=int, default=100, help='Number of markets')
    parser.add_argument('--n_products', type=int, default=4, help='Number of products')
    parser.add_argument('--seed', type=int, default=1111, help='Base random seed')
    parser.add_argument('--mliv', type=str, default='double_lasso', help='MLIV estimator')
    parser.add_argument('--pgmm_c', type=float, default=0.0001, help='PGMM penalty')
    parser.add_argument('--pgmm_feat', type=str, default='poly', help='PGMM featurizer')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: results/mc_results_elasticity_T{n_markets}_J{n_products}.csv)')
    
    args = parser.parse_args()

    # Set default output filename with n_obs
    if args.output is None:
        script_dir = Path(__file__).parent
        results_dir = script_dir / 'results'
        args.output = str(results_dir / f'mc_results_elasticity_T{args.n_markets}_J{args.n_products}.csv')
    
    config = MonteCarloConfig(
        n_runs=args.n_runs,
        n_markets=args.n_markets,
        n_products=args.n_products,
        mliv=args.mliv,
        pgmm_c=args.pgmm_c,
        pgmm_featurizer=args.pgmm_feat,
        seed=args.seed
    )
    
    mc = MonteCarloElasticity(config=config, verbose=True)
    mc.run()
    mc.save_results(args.output)


if __name__ == '__main__':
    main()