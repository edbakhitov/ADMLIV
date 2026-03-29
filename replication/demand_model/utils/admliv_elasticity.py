# simulations/demand_model/utils/admliv_elasticity.py

"""
Panel Data ADMLIV for Own-Price Elasticity Estimation.

This module implements ADMLIV for the own-price elasticity functional:
    θ_j = E[ε_{jj,t}(γ)] = E[(p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj}]

Key difference from ADMLIVPanel (demand derivatives):
The elasticity functional is NONLINEAR in γ, which requires an additional
layer of cross-fitting when computing the moment matrix M for PGMM.

Double Cross-Fitting:
----------------------------------
For nonlinear functionals, the Gateaux derivative M = D_γ θ[d_k] depends on γ.
Using the same γ to compute M and fit PGMM introduces bias.

Solution: Double cross-fitting using the same K-fold structure.
For fold k (test fold), M is computed using all observations outside fold k:
- For each fold ℓ ≠ k:
  - Train γ_{k,ℓ} on data NOT in fold k AND NOT in fold ℓ
  - Compute Gateaux derivative D_{k,ℓ} on data in fold ℓ using γ_{k,ℓ}
- M[t] = D_{k,ℓ}[t] where market t belongs to fold ℓ

This uses all observations outside fold k to construct M, with proper
cross-fitting so that each M[t] is computed using a γ trained without market t.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, List
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
import scipy.stats as ss

from .raw_data import RawData
from .omega_transformer import OmegaTransformer
from .pgmm_elasticity import PGMMElasticity, PGMMElasticityControl
from .own_price_elasticity import OwnPriceElasticity, FeaturizerWithDerivative
from admliv.main.admliv import MLIVEstimator



# ---------------------------------------------------------------------------
# Inlined from admliv_panel.py (MarketKFold + ADMLIVElasticityResult)
# ---------------------------------------------------------------------------

@dataclass
class ADMLIVElasticityResult:
    """
    Results from ADMLIVElasticity estimation for a single product.

    Attributes
    ----------
    product_id : int or str
        Product identifier
    theta_debiased : float
        Debiased estimate of the own-price elasticity
    theta_plugin : float
        Plug-in estimate (without debiasing)
    se_debiased : float
        Standard error of debiased estimate
    se_plugin : float
        Standard error of plug-in estimate
    variance_debiased : float
        Variance estimate for debiased estimator
    variance_plugin : float
        Variance estimate for plug-in estimator
    ci_lower : float
        Lower bound of confidence interval (debiased)
    ci_upper : float
        Upper bound of confidence interval (debiased)
    ci_lower_plugin : float
        Lower bound of confidence interval (plug-in)
    ci_upper_plugin : float
        Upper bound of confidence interval (plug-in)
    confidence_level : float
        Confidence level used for confidence interval
    n_markets : int
        Number of markets where product exists
    n_folds : int
        Number of cross-fitting folds
    """
    product_id: Union[int, str]
    theta_debiased: float
    theta_plugin: float
    se_debiased: float
    se_plugin: float
    variance_debiased: float
    variance_plugin: float
    ci_lower: float
    ci_upper: float
    ci_lower_plugin: float
    ci_upper_plugin: float
    confidence_level: float
    n_markets: int
    n_folds: int

    def __repr__(self) -> str:
        ci_pct = int(self.confidence_level * 100)
        return (
            f"ADMLIVElasticityResult(product={self.product_id},\n"
            f"  theta_debiased={self.theta_debiased:.6f} (SE={self.se_debiased:.6f})\n"
            f"    {ci_pct}% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  theta_plugin={self.theta_plugin:.6f} (SE={self.se_plugin:.6f})\n"
            f"    {ci_pct}% CI: [{self.ci_lower_plugin:.6f}, {self.ci_upper_plugin:.6f}]\n"
            f"  n_markets={self.n_markets}, n_folds={self.n_folds}\n"
            f")"
        )


class MarketKFold:
    """
    K-Fold cross-validator that splits by market, not observation.

    Ensures all observations within a market are assigned to the same fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle markets before splitting
    random_state : int, default=None
        Random seed for shuffling
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, market_ids: NDArray):
        """
        Generate train/test indices based on market-level splits.

        Parameters
        ----------
        market_ids : NDArray, shape (n,)
            Market identifier for each observation

        Yields
        ------
        train_idx : NDArray
            Training observation indices
        test_idx : NDArray
            Test observation indices
        """
        unique_markets = np.unique(market_ids)
        n_markets = len(unique_markets)

        if self.n_splits > n_markets:
            raise ValueError(
                f"Cannot have {self.n_splits} folds with only {n_markets} markets"
            )

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for train_market_idx, test_market_idx in kfold.split(unique_markets):
            train_markets = unique_markets[train_market_idx]
            test_markets = unique_markets[test_market_idx]

            train_mask = np.isin(market_ids, train_markets)
            test_mask = np.isin(market_ids, test_markets)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits

@dataclass
class ADMLIVElasticityControl:
    """
    Control parameters for ADMLIVElasticity estimation.

    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-fitting folds (splits markets). Used for both main
        cross-fitting and double cross-fitting for M computation.
        Must be >= 3 for proper double cross-fitting.
    random_state : int, default=42
        Random seed for reproducibility
    pgmm_control : Optional[PGMMElasticityControl], default=None
        Control parameters for PGMMElasticity estimation
    use_adaptive_pgmm : bool, default=True
        If True, PGMM uses adaptive weights based on preliminary estimation
    confidence_level : float, default=0.95
        Confidence level for confidence intervals
    verbose : bool, default=True
        If True, prints progress information
    regularize : bool, default=True
        If True, apply Tikhonov regularization to ill-conditioned A matrix
    cond_threshold : float, default=100.0
        Condition number threshold for A matrix.
    exclude_ill_conditioned_markets : bool, default=True
        If True, exclude markets with ill-conditioned A matrix from Gateaux
        derivative (M) computation. This avoids bias in Riesz estimation.
    gateaux_cond_threshold : float, default=100.0
        Condition number threshold for excluding markets from M computation.
        Markets with cond(A) > threshold are excluded when
        exclude_ill_conditioned_markets=True.
    """
    n_folds: int = 5
    random_state: int = 42
    pgmm_control: Optional[PGMMElasticityControl] = None
    use_adaptive_pgmm: bool = True
    pgmm_single_stage: bool = False
    confidence_level: float = 0.95
    verbose: bool = True
    regularize: bool = True
    cond_threshold: float = 100.0
    regularize_gateaux: bool = False  
    exclude_ill_conditioned_markets: bool = True
    gateaux_cond_threshold: float = 100.0

    def __post_init__(self):
        """Validate control parameters."""
        if self.n_folds < 3:
            raise ValueError("n_folds must be at least 3 for double cross-fitting")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


class ADMLIVElasticity(BaseEstimator):
    """
    Panel Data ADMLIV for estimating own-price elasticity with debiasing.

    Estimates θ_j = E[ε_{jj,t}(γ)] for each product j using double cross-fitting
    to handle the nonlinear dependence of the functional on γ.

    The estimator computes:
        θ̂_j = (1/T_j) Σ_ℓ Σ_{t∈I_ℓ} {ε_{jj,t}(γ̂_{-ℓ}) + α̂_jℓ(Z_jt)[Y_jt - γ̂_{-ℓ}(ω_jt)]}

    where:
        - T_j is the number of markets where product j exists
        - γ̂_{-ℓ} is the MLIV estimator fitted on markets not in fold ℓ
        - α̂_jℓ is the product-specific Riesz representer from PGMMElasticity
        - ε_{jj,t}(γ) = (p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj} is the elasticity

    Double Cross-Fitting:
    For fold k, the Gateaux derivative M is computed using all data outside fold k:
    - For each fold ℓ ≠ k: train γ_{k,ℓ} on folds ≠ k and ≠ ℓ
    - Compute D_{k,ℓ} on fold ℓ using γ_{k,ℓ}
    - M[t] = D_{k,ℓ}[t] where market t belongs to fold ℓ
    This ensures M is computed with a γ independent of each market's data.

    Parameters
    ----------
    mliv_estimator : Callable
        Factory function that returns a fresh MLIV estimator instance.
        The estimator must have fit(), predict(), and predict_derivative() methods.
    omega_transformer : OmegaTransformer
        Transformer for constructing omega from raw data
    omega_iv_transformer : OmegaTransformer
        Transformer for constructing IV omega from raw data
    omega_featurizer : FeaturizerWithDerivative
        Featurizer for omega space with transform_derivative method.
        Used both for PGMM basis expansion and Gateaux derivative computation.
    omega_iv_featurizer : TransformerMixin
        Featurizer for IV space (b(Z) basis expansion) for PGMM
    control : Optional[ADMLIVElasticityControl], default=None
        Control parameters for estimation

    Attributes
    ----------
    results_ : Dict[product_id, ADMLIVElasticityResult]
        Estimation results per product (available after fit)
    gamma_estimators_ : List
        Fitted γ estimators from each outer fold
    is_fitted_ : bool
        Whether the estimator has been fitted

    Examples
    --------
    >>> # Setup transformers
    >>> omega_transformer = OmegaTransformer(price_in_diffs=True, include_shares=True)
    >>> omega_iv_transformer = OmegaTransformer(include_prices=False, include_shares=False)
    >>>
    >>> # Create ADMLIV estimator for elasticity
    >>> admliv = ADMLIVElasticity(
    ...     mliv_estimator=lambda: KIVEstimator(),
    ...     omega_transformer=omega_transformer,
    ...     omega_iv_transformer=omega_iv_transformer,
    ...     omega_featurizer=CoordinatePolyTransform(degree=2),
    ...     omega_iv_featurizer=CoordinatePolyTransform(degree=2)
    ... )
    >>>
    >>> # Prepare data
    >>> raw_data = RawData(price, x1, x2, shares, market_ids, w=cost_shifters)
    >>> results = admliv.fit(raw_data, product_ids=[0, 1, 2])
    """

    def __init__(
        self,
        mliv_estimator: Union[MLIVEstimator, Callable[[], MLIVEstimator]],
        omega_transformer: OmegaTransformer,
        omega_iv_transformer: OmegaTransformer,
        omega_featurizer: FeaturizerWithDerivative,
        omega_iv_featurizer: TransformerMixin,
        control: Optional[ADMLIVElasticityControl] = None,
    ):
        self.mliv_estimator = mliv_estimator
        self.omega_transformer = omega_transformer
        self.omega_iv_transformer = omega_iv_transformer
        self.omega_featurizer = omega_featurizer
        self.omega_iv_featurizer = omega_iv_featurizer
        self.control = control if control is not None else ADMLIVElasticityControl()
        self.is_fitted_ = False

    def _get_mliv_instance(self) -> MLIVEstimator:
        """Get a fresh instance of the MLIV estimator."""
        if callable(self.mliv_estimator) and not hasattr(self.mliv_estimator, 'fit'):
            return self.mliv_estimator()
        elif hasattr(self.mliv_estimator, 'fit'):
            try:
                return clone(self.mliv_estimator)
            except TypeError:
                return self.mliv_estimator
        else:
            raise ValueError(
                "mliv_estimator must be either a factory function or an estimator "
                "instance with fit(), predict(), and predict_derivative() methods"
            )

    def _get_pgmm_instance(self, featurizer=None, iv_featurizer=None):
        """Get a fresh instance of the PGMMElasticity estimator.

        Both featurizers should be pre-fitted on product j's data.
        """
        pgmm_control = self.control.pgmm_control
        if pgmm_control is None:
            pgmm_control = PGMMElasticityControl()
        omega_feat = featurizer if featurizer is not None else clone(self.omega_featurizer)
        omega_iv_feat = iv_featurizer if iv_featurizer is not None else clone(self.omega_iv_featurizer)
        return PGMMElasticity(
            omega_featurizer=omega_feat,
            omega_iv_featurizer=omega_iv_feat,
            adaptive=self.control.use_adaptive_pgmm,
            single_stage=self.control.pgmm_single_stage,
            control=pgmm_control,
            verbose=self.control.verbose
        )

    def _construct_y(self, raw_data: RawData) -> NDArray:
        """
        Construct the outcome variable y for gamma estimation.

        Default: y = log(s_j/s_0) - x^(1)

        Parameters
        ----------
        raw_data : RawData
            Raw panel data

        Returns
        -------
        y : NDArray, shape (n,)
            Outcome variable
        """
        n = raw_data.n_obs

        s0 = np.zeros(n)
        for t in raw_data.unique_markets:
            mask = raw_data.market_ids == t
            s0[mask] = 1 - raw_data.shares[mask].sum()
        y = (np.log(raw_data.shares) - np.log(s0) - raw_data.x1).flatten()
        return y

    def _make_elasticity(self, product_id: Union[int, str]) -> OwnPriceElasticity:
        """
        Create an elasticity functional instance.
        """

        return OwnPriceElasticity(
            self.omega_transformer,
            product_id,
            regularize=self.control.regularize,  
            regularize_gateaux=self.control.regularize_gateaux,  
            cond_threshold=self.control.cond_threshold  
        )

    def _precompute_all_inner_gammas(
        self,
        omega_full: NDArray,
        omega_iv_full: NDArray,
        y: NDArray,
        fold_assignments: NDArray,
        n_folds: int
    ) -> Dict[frozenset, MLIVEstimator]:
        """
        Precompute all unique inner gamma estimators for double cross-fitting.

        For each unique unordered pair {k, ell}, fits one gamma on observations
        whose fold is NOT in {k, ell}. Since gamma_{k,ell} and gamma_{ell,k}
        share the same training data, this yields C(K,2) = K*(K-1)/2 fits
        instead of K*(K-1) per product.

        Parameters
        ----------
        omega_full : NDArray, shape (n, d_omega)
        omega_iv_full : NDArray, shape (n, d_iv)
        y : NDArray, shape (n,)
        fold_assignments : NDArray, shape (n,)
        n_folds : int

        Returns
        -------
        gamma_cache : Dict[frozenset, MLIVEstimator]
            Maps frozenset({k, ell}) -> fitted gamma estimator.
        """
        gamma_cache = {}

        for k in range(n_folds):
            for ell in range(k + 1, n_folds):
                pair = frozenset({k, ell})
                train_mask = (fold_assignments != k) & (fold_assignments != ell)

                if not train_mask.any():
                    continue

                gamma_kl = self._get_mliv_instance()
                gamma_kl.fit({
                    'X': omega_full[train_mask],
                    'Z': omega_iv_full[train_mask],
                    'Y': y[train_mask]
                })
                gamma_cache[pair] = gamma_kl

        return gamma_cache

    def _compute_double_crossfit_M(
        self,
        raw_data: RawData,
        omega: NDArray,
        omega_iv: NDArray,
        y: NDArray,
        current_fold: int,
        fold_assignments: NDArray,
        product_id: Union[int, str],
        featurizer: FeaturizerWithDerivative,
        gamma_cache: Optional[Dict[frozenset, MLIVEstimator]] = None
    ) -> NDArray:
        """
        Compute moment matrix M using double cross-fitting.

        For fold k (test fold), we compute M using all data outside fold k:
        - For each fold ℓ ≠ k:
          - Train γ_{k,ℓ} on all data NOT in fold k AND NOT in fold ℓ
          - Compute Gateaux derivative D_{k,ℓ} on data in fold ℓ using γ_{k,ℓ}
        - M[t] = D_{k,ℓ}[t] where market t belongs to fold ℓ

        This uses all observations outside fold k to construct M, with proper
        cross-fitting to ensure γ is independent of the data used for each M[t].

        Parameters
        ----------
        raw_data : RawData
            Full raw panel data
        omega : NDArray
            Omega for full data
        omega_iv : NDArray
            IV omega for full data
        y : NDArray
            Outcome for full data
        current_fold : int
            Index of the current test fold (k)
        fold_assignments : NDArray
            Array mapping each observation to its fold index
        product_id : int or str
            Product identifier
        featurizer : FeaturizerWithDerivative
            Fitted featurizer for Gateaux derivatives
        gamma_cache : Dict[frozenset, MLIVEstimator], optional
            Pre-computed inner gammas keyed by frozenset({k, ell}).
            If provided, looks up gamma instead of fitting inline.

        Returns
        -------
        M : NDArray, shape (T_train, n_basis)
            Moment matrix computed with cross-fitted γ
        markets : NDArray
            Market identifiers corresponding to rows of M
        """
        n_folds = self.control.n_folds

        # Get all markets with product that are not in current test fold
        train_mask = fold_assignments != current_fold
        train_markets = np.unique(raw_data.market_ids[train_mask])
        markets_with_product = raw_data.get_markets_with_product(product_id)
        train_markets_j = [t for t in train_markets if t in markets_with_product]

        if len(train_markets_j) == 0:
            raise ValueError(f"No training markets with product {product_id}")

        # Check if we have enough folds for double cross-fitting
        # Need at least 3 folds (1 test, 2+ for training γ)
        if n_folds < 3:
            # Fall back to using single γ on all training data
            gamma_full = self._get_mliv_instance()
            gamma_full.fit({
                'X': omega[train_mask],
                'Z': omega_iv[train_mask],
                'Y': y[train_mask]
            })

            train_idx = np.where(train_mask)[0]
            raw_data_train = raw_data.subset(train_idx)

            elasticity = self._make_elasticity(product_id)
            M, markets_used = elasticity.compute_basis_gateaux(
                gamma_full, featurizer, raw_data_train, product_id,
                return_markets=True,
                exclude_ill_conditioned=self.control.exclude_ill_conditioned_markets,
                cond_threshold=self.control.gateaux_cond_threshold,
                omega=omega[train_idx]
            )
            return M, markets_used

        # Storage for M values by market
        M_by_market = {}
        n_basis = None

        # Iterate over all folds ℓ ≠ k
        for fold_ell in range(n_folds):
            if fold_ell == current_fold:
                continue

            # Look up or fit γ_{k,ℓ}
            pair_key = frozenset({current_fold, fold_ell})

            if gamma_cache is not None:
                if pair_key not in gamma_cache:
                    continue
                gamma_kl = gamma_cache[pair_key]
            else:
                # Fallback: fit inline (backward compatibility)
                gamma_train_mask = (fold_assignments != current_fold) & (fold_assignments != fold_ell)
                if not gamma_train_mask.any():
                    continue
                gamma_kl = self._get_mliv_instance()
                gamma_kl.fit({
                    'X': omega[gamma_train_mask],
                    'Z': omega_iv[gamma_train_mask],
                    'Y': y[gamma_train_mask]
                })

            # Get markets in fold ℓ that have product j
            fold_ell_mask = fold_assignments == fold_ell
            fold_ell_markets = np.unique(raw_data.market_ids[fold_ell_mask])
            fold_ell_markets_j = [t for t in fold_ell_markets if t in markets_with_product]

            if len(fold_ell_markets_j) == 0:
                continue

            # Create subset for fold ℓ
            fold_ell_idx = np.where(fold_ell_mask)[0]
            raw_data_ell = raw_data.subset(fold_ell_idx)

            # Compute Gateaux derivatives for all markets in fold ℓ with product
            # Exclude ill-conditioned markets to avoid biased Riesz estimation
            elasticity = self._make_elasticity(product_id)
            M_ell, markets_ell = elasticity.compute_basis_gateaux(
                gamma_kl, featurizer, raw_data_ell, product_id,
                return_markets=True,
                exclude_ill_conditioned=self.control.exclude_ill_conditioned_markets,
                cond_threshold=self.control.gateaux_cond_threshold,
                omega=omega[fold_ell_idx]
            )

            # Store M values by market
            for i, t in enumerate(markets_ell):
                M_by_market[t] = M_ell[i]

            if n_basis is None and len(markets_ell) > 0:
                n_basis = M_ell.shape[1]

        # Assemble M matrix in sorted market order
        markets_ordered = sorted(M_by_market.keys())
        M = np.vstack([M_by_market[t] for t in markets_ordered])

        return M, np.array(markets_ordered)

    def fit(
        self,
        raw_data: RawData,
        product_ids: Optional[List] = None
    ) -> Dict[Union[int, str], ADMLIVElasticityResult]:
        """
        Fit ADMLIVElasticity for specified products using double cross-fitting.

        Parameters
        ----------
        raw_data : RawData
            Raw panel data containing:
            - price, shares, market_ids: basic panel structure
            - x1: special regressor
            - x2: characteristics for omega construction
            - w (optional): external instruments for omega_iv construction
        product_ids : List, optional
            Products to estimate. If None, estimates for all products.

        Returns
        -------
        results : Dict[product_id, ADMLIVElasticityResult]
            Estimation results for each product
        """
        if product_ids is None:
            product_ids = list(raw_data.unique_products)

        n = raw_data.n_obs
        T = raw_data.n_markets

        # Construct y (can be overridden in subclasses)
        y = self._construct_y(raw_data)

        if self.control.verbose:
            print("=" * 70)
            print("ADMLIVElasticity: Panel Data ADMLIV for Own-Price Elasticity")
            print("=" * 70)
            print(f"Total observations: {n}")
            print(f"Number of markets: {T}")
            print(f"Products to estimate: {len(product_ids)}")
            print(f"Cross-fitting folds: {self.control.n_folds}")
            print(f"Double cross-fitting: enabled (using same fold structure)")
            print("=" * 70)

        # Build IV input for omega_iv_transformer:
        # [x1, x2, w] — all exogenous variables that should be differenced.
        # w contains instruments like cost shifters that go through the transformer.
        # w_external is appended after transformation.
        if raw_data.w is not None:
            x_iv = np.c_[raw_data.x1, raw_data.x2, raw_data.w]
        else:
            x_iv = np.c_[raw_data.x1, raw_data.x2]

        # Fit transformers on full panel
        if self.control.verbose:
            print("\nFitting omega transformers on full panel...")

        self.omega_transformer.fit(
            raw_data.x2, raw_data.market_ids,
            price=raw_data.price, shares=raw_data.shares
        )
        self.omega_iv_transformer.fit(x_iv, raw_data.market_ids)

        # Compute omega and omega_iv for full data (used for all folds)
        omega_full = self.omega_transformer.transform(
            raw_data.x2, raw_data.market_ids,
            price=raw_data.price, shares=raw_data.shares
        )
        omega_iv_full = self.omega_iv_transformer.transform(x_iv, raw_data.market_ids)

        # Append external instruments after transformation
        if raw_data.w_external is not None:
            omega_iv_full = np.c_[omega_iv_full, raw_data.w_external]

        # Setup market-level cross-validation (outer folds)
        outer_kfold = MarketKFold(
            n_splits=self.control.n_folds,
            shuffle=True,
            random_state=self.control.random_state
        )

        # Compute fold assignments for all observations (needed for double cross-fitting)
        fold_assignments = np.zeros(n, dtype=int)
        fold_splits = list(outer_kfold.split(raw_data.market_ids))
        for fold_idx, (_, test_idx) in enumerate(fold_splits):
            fold_assignments[test_idx] = fold_idx

        # Precompute all inner gammas for double cross-fitting
        n_unique_pairs = self.control.n_folds * (self.control.n_folds - 1) // 2
        if self.control.verbose:
            print(f"\nPrecomputing {n_unique_pairs} inner gamma estimators "
                  f"for double cross-fitting...")

        gamma_cache = self._precompute_all_inner_gammas(
            omega_full, omega_iv_full, y,
            fold_assignments, self.control.n_folds
        )

        if self.control.verbose:
            print(f"  Cached {len(gamma_cache)} gamma estimators.\n")

        # Storage for cross-fitted values per product
        psi_by_product = {j: {} for j in product_ids}
        eps_by_product = {j: {} for j in product_ids}

        self.gamma_estimators_ = []

        # Outer cross-fitting loop
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            if self.control.verbose:
                print(f"\n{'='*50}")
                print(f"Outer Fold {fold_idx + 1}/{self.control.n_folds}")
                print(f"{'='*50}")

            train_markets = np.unique(raw_data.market_ids[train_idx])
            test_markets = np.unique(raw_data.market_ids[test_idx])

            if self.control.verbose:
                print(f"  Train markets: {len(train_markets)}")
                print(f"  Test markets: {len(test_markets)}")

            # Create train/test RawData
            raw_data_train = raw_data.subset(train_idx)
            raw_data_test = raw_data.subset(test_idx)

            # Slice precomputed omega arrays 
            omega_train = omega_full[train_idx]
            omega_iv_train = omega_iv_full[train_idx]
            y_train = y[train_idx]

            omega_test = omega_full[test_idx]
            omega_iv_test = omega_iv_full[test_idx]
            y_test = y[test_idx]

            # Stage 1: Fit γ on full training data (for test evaluation)
            if self.control.verbose:
                print("  Stage 1: Fitting MLIV estimator on training data...")

            gamma_outer = self._get_mliv_instance()
            gamma_outer.fit({'X': omega_train, 'Z': omega_iv_train, 'Y': y_train})
            self.gamma_estimators_.append(gamma_outer)

            # Stage 2: For each product, fit PGMM with double cross-fitting
            for j in product_ids:
                train_markets_j = raw_data_train.get_markets_with_product(j)
                test_markets_j = raw_data_test.get_markets_with_product(j)

                if len(train_markets_j) == 0 or len(test_markets_j) == 0:
                    continue

                if self.control.verbose:
                    print(f"  Product {j}: fitting PGMM ({len(train_markets_j)} train markets)...")

                # Fit featurizers on product j's training data
                omega_j_train = np.vstack([
                    omega_train[raw_data_train.market_ids == t][
                        raw_data_train.get_local_index(t, j)
                    ]
                    for t in train_markets_j
                ])
                omega_iv_j_train = np.vstack([
                    omega_iv_train[raw_data_train.market_ids == t][
                        raw_data_train.get_local_index(t, j)
                    ]
                    for t in train_markets_j
                ])

                featurizer_j = clone(self.omega_featurizer)
                featurizer_j.fit(omega_j_train)
                iv_featurizer_j = clone(self.omega_iv_featurizer)
                iv_featurizer_j.fit(omega_iv_j_train)

                # Compute M using double cross-fitting
                # Uses all folds ≠ current fold, with proper cross-fitting
                M_crossfit, markets_M = self._compute_double_crossfit_M(
                    raw_data, omega_full, omega_iv_full, y,
                    fold_idx, fold_assignments,
                    j, featurizer_j,
                    gamma_cache=gamma_cache
                )

                if self.control.verbose:
                    print(f"    Double cross-fit M shape: {M_crossfit.shape}")

                # Fit PGMM using cross-fitted M (featurizers are pre-fitted)
                pgmm = self._get_pgmm_instance(featurizer=featurizer_j, iv_featurizer=iv_featurizer_j)
                pgmm.fit(
                    raw_data_train, omega_train, omega_iv_train,
                    M_crossfit, j, markets=markets_M
                )

                # Stage 3: Evaluate on test fold
                elasticity = self._make_elasticity(j)

                # Compute elasticity ε_{jj,t}(γ) for test markets
                eps_test, markets_test = elasticity.compute(
                    gamma_outer, raw_data_test, j, return_markets=True,
                    omega=omega_test
                )

                # Extract product j's data for all test markets (vectorized)
                omega_j_test = np.vstack([
                    omega_test[raw_data_test.market_ids == t][raw_data_test.get_local_index(t, j)]
                    for t in markets_test
                ])
                omega_iv_j_test = np.vstack([
                    omega_iv_test[raw_data_test.market_ids == t][raw_data_test.get_local_index(t, j)]
                    for t in markets_test
                ])
                y_j_test = np.array([
                    y_test[raw_data_test.market_ids == t][raw_data_test.get_local_index(t, j)]
                    for t in markets_test
                ])

                # Compute α_j(Z_jt) and γ(ω_jt) in batch
                alpha_test = pgmm.predict(omega_iv_j_test)
                gamma_test = gamma_outer.predict(omega_j_test).flatten()

                # Compute residuals and influence adjustment
                residuals_test = y_j_test - gamma_test
                influence_adj_test = alpha_test * residuals_test

                # Orthogonal moment: ε + α(Y - γ)
                psi_test = eps_test + influence_adj_test

                # Store by market
                for t_idx, t in enumerate(markets_test):
                    psi_by_product[j][t] = psi_test[t_idx]
                    eps_by_product[j][t] = eps_test[t_idx]

        # Compute final estimates for each product
        self.results_ = {}

        if self.control.verbose:
            print(f"\n{'='*70}")
            print("Final Estimates")
            print(f"{'='*70}")

        for j in product_ids:
            if len(psi_by_product[j]) == 0:
                if self.control.verbose:
                    print(f"Product {j}: No valid estimates (insufficient data)")
                continue

            # Extract values
            markets_j = sorted(psi_by_product[j].keys())
            psi_j = np.array([psi_by_product[j][t] for t in markets_j])
            eps_j = np.array([eps_by_product[j][t] for t in markets_j])
            T_j = len(markets_j)

            # Point estimates
            theta_debiased = np.mean(psi_j)
            theta_plugin = np.mean(eps_j)

            # Variance estimates
            influence_debiased = psi_j - theta_debiased
            variance_debiased = np.mean(influence_debiased ** 2)

            influence_plugin = eps_j - theta_plugin
            variance_plugin = np.mean(influence_plugin ** 2)

            # Standard errors
            se_debiased = np.sqrt(variance_debiased / T_j)
            se_plugin = np.sqrt(variance_plugin / T_j)

            # Confidence intervals
            alpha = 1 - self.control.confidence_level
            z_alpha = ss.norm.ppf(1 - alpha / 2)
            ci_lower = theta_debiased - z_alpha * se_debiased
            ci_upper = theta_debiased + z_alpha * se_debiased
            ci_lower_plugin = theta_plugin - z_alpha * se_plugin
            ci_upper_plugin = theta_plugin + z_alpha * se_plugin

            self.results_[j] = ADMLIVElasticityResult(
                product_id=j,
                theta_debiased=theta_debiased,
                theta_plugin=theta_plugin,
                se_debiased=se_debiased,
                se_plugin=se_plugin,
                variance_debiased=variance_debiased,
                variance_plugin=variance_plugin,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_lower_plugin=ci_lower_plugin,
                ci_upper_plugin=ci_upper_plugin,
                confidence_level=self.control.confidence_level,
                n_markets=T_j,
                n_folds=self.control.n_folds
            )

            if self.control.verbose:
                print(f"\nProduct {j}:")
                print(f"  θ_debiased (avg elasticity) = {theta_debiased:.6f} (SE = {se_debiased:.6f})")
                print(f"  θ_plugin   (avg elasticity) = {theta_plugin:.6f} (SE = {se_plugin:.6f})")
                print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"  Markets: {T_j}")

        self.is_fitted_ = True

        if self.control.verbose:
            print(f"\n{'='*70}")

        return self.results_

    def get_results(self) -> Dict[Union[int, str], ADMLIVElasticityResult]:
        """Get estimation results."""
        if not self.is_fitted_:
            raise ValueError("ADMLIVElasticity must be fitted before accessing results")
        return self.results_
