# admliv/main/admliv.py

"""
Automatic Debiased Machine Learning for IV (ADMLIV) Estimator.

Implements the two-stage cross-fitting procedure for debiased estimation of
functionals of MLIV estimators, as described in:

- Bakhitov (2026): "Penalized GMM Framework for Inference on Functionals of
  Nonparametric Instrumental Variable Estimators"
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Union, Any, Protocol, List
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
import scipy.stats as ss

from ..moments.base import BaseMoment
from ..core.pgmm import PGMM
from ..core.pgmm_cv import PGMMCV
from ..core.control import PGMMControl, PGMMCVControl


# Type definitions for MLIV estimator protocol
# Note: ADMLIV expects estimators with Dict interface: fit(W: Dict)
# All standard MLIV estimators (DoubleLassoEstimator, NpivSieveEstimator,
# KIVEstimator, DeepGMMEstimator) implement this interface
class MLIVEstimator(Protocol):
    """
    Protocol defining the Dict-based interface expected by ADMLIV.

    This is the interface that ADMLIV uses internally during cross-fitting.
    All standard BaseMLIVEstimator instances implement this interface with
    fit(W: Dict) where W contains keys 'X', 'Z', 'Y'.

    See Also
    --------
    admliv.estimators.BaseMLIVEstimator : Base class for MLIV estimators
    """

    def fit(self, W: Dict[str, NDArray[np.float64]], **kwargs) -> 'MLIVEstimator':
        """Fit the MLIV estimator using Dict interface."""
        ...

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict gamma(X)."""
        ...


@dataclass
class ADMLIVControl:
    """
    Control parameters for ADMLIV estimation.

    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-fitting folds (L in the paper)
    random_state : int, default=42
        Random seed for reproducibility
    pgmm_control : Optional[Any], default=None
        Control parameters for PGMM estimation. If None, uses defaults.
    use_cv_for_pgmm : bool, default=False
        If True, uses cross-validated PGMM for Riesz representer estimation
    use_adaptive_pgmm : bool, default=True
        If True, PGMM uses adaptive weights based on preliminary estimation.
        This implements adaptive Lasso weights: w_j = 1 / |rho_j|
    confidence_level : float, default=0.95
        Confidence level for confidence intervals (e.g., 0.95 for 95% CI)
    verbose : bool, default=True
        If True, prints progress information
    """
    n_folds: int = 5
    random_state: int = 42
    pgmm_control: Optional[Any] = None
    use_cv_for_pgmm: bool = False
    use_adaptive_pgmm: bool = True
    confidence_level: float = 0.95
    verbose: bool = True
    
    def __post_init__(self):
        """Validate control parameters."""
        if self.n_folds <= 1:
            raise ValueError("n_folds must be greater than 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass
class ADMLIVResult:
    """
    Results from ADMLIV estimation.

    Attributes
    ----------
    theta_debiased : float
        Debiased estimate of the parameter of interest
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
    influence_functions : NDArray[np.float64]
        Influence function values for each observation
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
    n_samples : int
        Number of observations
    n_folds : int
        Number of cross-fitting folds
    fold_estimates : List[Dict[str, Any]]
        Per-fold estimates and diagnostics
    """
    theta_debiased: float
    theta_plugin: float
    se_debiased: float
    se_plugin: float
    variance_debiased: float
    variance_plugin: float
    influence_functions: NDArray[np.float64]
    ci_lower: float
    ci_upper: float
    ci_lower_plugin: float
    ci_upper_plugin: float
    confidence_level: float
    n_samples: int
    n_folds: int
    fold_estimates: List[Dict[str, Any]] = field(default_factory=list)
    
    def __repr__(self) -> str:
        ci_pct = int(self.confidence_level * 100)
        return (
            f"ADMLIVResult(\n"
            f"  theta_debiased={self.theta_debiased:.6f} (SE={self.se_debiased:.6f})\n"
            f"    {ci_pct}% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  theta_plugin={self.theta_plugin:.6f} (SE={self.se_plugin:.6f})\n"
            f"    {ci_pct}% CI: [{self.ci_lower_plugin:.6f}, {self.ci_upper_plugin:.6f}]\n"
            f"  n_samples={self.n_samples}, n_folds={self.n_folds}\n"
            f")"
        )
    
    def summary(self) -> str:
        """Return a formatted summary of results."""
        z_stat_debiased = self.theta_debiased / self.se_debiased if self.se_debiased > 0 else np.inf
        p_value_debiased = 2 * (1 - ss.norm.cdf(np.abs(z_stat_debiased)))

        z_stat_plugin = self.theta_plugin / self.se_plugin if self.se_plugin > 0 else np.inf
        p_value_plugin = 2 * (1 - ss.norm.cdf(np.abs(z_stat_plugin)))

        ci_pct = int(self.confidence_level * 100)

        lines = [
            "=" * 70,
            "ADMLIV Estimation Results",
            "=" * 70,
            f"{'Parameter':<25} {'Estimate':<12} {'Std. Err.':<12} {'z-stat':<10} {'p-value':<10}",
            "-" * 70,
            f"{'Debiased θ':<25} {self.theta_debiased:<12.6f} {self.se_debiased:<12.6f} "
            f"{z_stat_debiased:<10.4f} {p_value_debiased:<10.4f}",
            f"{'Plug-in θ':<25} {self.theta_plugin:<12.6f} {self.se_plugin:<12.6f} "
            f"{z_stat_plugin:<10.4f} {p_value_plugin:<10.4f}",
            "-" * 70,
            f"{ci_pct}% Confidence Intervals:",
            f"  Debiased: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
            f"  Plug-in:  [{self.ci_lower_plugin:.6f}, {self.ci_upper_plugin:.6f}]",
            "-" * 70,
            f"Number of observations: {self.n_samples}",
            f"Number of folds: {self.n_folds}",
            "=" * 70,
        ]
        return "\n".join(lines)


class ADMLIV(BaseEstimator):
    """
    Automatic Debiased Machine Learning for IV (ADMLIV) Estimator.
    
    Implements the two-stage cross-fitting procedure for debiased estimation
    of functionals of MLIV estimators.
    
    The estimator computes:
        θ̂ = (1/n) Σ_ℓ Σ_{i∈I_ℓ} {m(W_i, γ̂_ℓ) + α̂_ℓ(Z_i)[Y_i - γ̂_ℓ(X_i)]}
    
    where:
        - γ̂_ℓ is the MLIV estimator fitted on data not in fold ℓ
        - α̂_ℓ is the Riesz representer estimated via PGMM on data not in fold ℓ
        - m(W, γ) is the moment function defining the parameter of interest
    
    Parameters
    ----------
    mliv_estimator : Union[MLIVEstimator, Callable]
        MLIV estimator instance or factory function with Dict interface.

        All standard MLIV estimators (DoubleLassoEstimator, NpivSieveEstimator,
        KIVEstimator, DeepGMMEstimator) use the Dict interface: fit(W: Dict).

        If instance: must have fit(W: Dict) and predict(X) methods
        If factory: must return a new estimator instance when called

        Example:
            from admliv.estimators import DoubleLassoEstimator

            # Use estimator directly
            estimator = DoubleLassoEstimator()

            # Or use a factory
            def mliv_factory():
                return DoubleLassoEstimator()
    x_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion d(X) for PGMM
    z_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion b(Z) for PGMM
    control : Optional[ADMLIVControl], default=None
        Control parameters for estimation
    
    Attributes
    ----------
    result_ : ADMLIVResult
        Estimation results (available after fit)
    is_fitted_ : bool
        Whether the estimator has been fitted
    fold_estimators_ : List[Dict]
        Estimators fitted on each fold
        
    Examples
    --------
    >>> from admliv import ADMLIV, ADMLIVControl
    >>> from admliv.estimators import DoubleLassoEstimator
    >>> from admliv.moments import WeightedAverage
    >>> from admliv.utils.featurizers import CoordinatePolyTransform
    >>>
    >>> # Setup
    >>> x_feat = CoordinatePolyTransform(degree=2)
    >>> z_feat = CoordinatePolyTransform(degree=2)
    >>> moment = WeightedAverage()
    >>>
    >>> # Create MLIV estimator factory
    >>> def mliv_factory():
    ...     return DoubleLassoEstimator(
    ...         x_featurizer=CoordinatePolyTransform(degree=2),
    ...         z_featurizer=CoordinatePolyTransform(degree=2)
    ...     )
    >>>
    >>> # Fit ADMLIV
    >>> control = ADMLIVControl(n_folds=5, verbose=True)
    >>> admliv = ADMLIV(
    ...     mliv_estimator=mliv_factory,
    ...     x_featurizer=x_feat,
    ...     z_featurizer=z_feat,
    ...     control=control
    ... )
    >>> result = admliv.fit(W, moment, weight_func=lambda x: np.ones(x.shape[0]))
    >>> print(result.summary())
    """
    
    def __init__(
        self,
        mliv_estimator: Union[MLIVEstimator, Callable[[], MLIVEstimator]],
        x_featurizer: TransformerMixin,
        z_featurizer: TransformerMixin,
        control: Optional[ADMLIVControl] = None
    ):
        self.mliv_estimator = mliv_estimator
        self.x_featurizer = x_featurizer
        self.z_featurizer = z_featurizer
        self.control = control if control is not None else ADMLIVControl()
        self.is_fitted_ = False

    def _get_mliv_instance(self) -> MLIVEstimator:
        """Get a fresh instance of the MLIV estimator."""
        if callable(self.mliv_estimator) and not hasattr(self.mliv_estimator, 'fit'):
            # It's a factory function
            return self.mliv_estimator()
        elif hasattr(self.mliv_estimator, 'fit'):
            # It's an estimator instance - clone it
            try:
                return clone(self.mliv_estimator)
            except TypeError:
                # If clone fails, assume the estimator handles its own state
                return self.mliv_estimator
        else:
            raise ValueError(
                "mliv_estimator must be either a factory function or an estimator "
                "instance with fit() and predict() methods"
            )
    
    def _get_pgmm_instance(self):
        """Get a fresh instance of the PGMM estimator."""
        if self.control.use_cv_for_pgmm:
            pgmm_control = self.control.pgmm_control
            if pgmm_control is None:
                pgmm_control = PGMMCVControl()
            elif isinstance(pgmm_control, PGMMControl) and not isinstance(pgmm_control, PGMMCVControl):
                # Convert PGMMControl to PGMMCVControl, preserving settings
                pgmm_control = PGMMCVControl(
                    maxiter=pgmm_control.maxiter,
                    optTol=pgmm_control.optTol,
                    zeroThreshold=pgmm_control.zeroThreshold,
                    intercept_penalty=pgmm_control.intercept_penalty,
                    c=pgmm_control.c,
                    adaptive_threshold=pgmm_control.adaptive_threshold,
                    adaptive_max_weight=pgmm_control.adaptive_max_weight
                )
            return PGMMCV(
                x_featurizer=clone(self.x_featurizer),
                z_featurizer=clone(self.z_featurizer),
                adaptive=self.control.use_adaptive_pgmm,
                control=pgmm_control,
                verbose=False  # Quiet during cross-fitting
            )
        else:
            pgmm_control = self.control.pgmm_control
            if pgmm_control is None:
                pgmm_control = PGMMControl()
            elif isinstance(pgmm_control, PGMMCVControl):
                # Convert PGMMCVControl to PGMMControl, preserving settings
                pgmm_control = PGMMControl(
                    maxiter=pgmm_control.maxiter,
                    optTol=pgmm_control.optTol,
                    zeroThreshold=pgmm_control.zeroThreshold,
                    intercept_penalty=pgmm_control.intercept_penalty,
                    c=pgmm_control.c,
                    adaptive_threshold=pgmm_control.adaptive_threshold,
                    adaptive_max_weight=pgmm_control.adaptive_max_weight
                )
            return PGMM(
                x_featurizer=clone(self.x_featurizer),
                z_featurizer=clone(self.z_featurizer),
                adaptive=self.control.use_adaptive_pgmm,
                control=pgmm_control,
                verbose=False
            )
    
    @staticmethod
    def _is_nonlinear_moment(moment: 'BaseMoment') -> bool:
        """
        Detect whether a moment is nonlinear by checking if it overrides
        compute_all_basis_gamma from BaseMoment.

        Uses duck typing: if the method on the concrete class is different
        from the one on BaseMoment (which raises NotImplementedError),
        the moment is nonlinear.
        """
        method = getattr(type(moment), 'compute_all_basis_gamma', None)
        base_method = getattr(BaseMoment, 'compute_all_basis_gamma', None)
        return method is not base_method

    def _precompute_inner_gammas(
        self,
        W: Dict[str, NDArray],
        fold_assignments: NDArray,
        n_folds: int
    ) -> Dict:
        """
        Precompute all C(K,2) inner gamma estimators for double cross-fitting.

        For each unordered pair {k, ell}, fits gamma on observations whose fold
        is NOT in {k, ell}. Since gamma_{k,ell} = gamma_{ell,k}, this yields
        C(K,2) = K*(K-1)/2 fits instead of K*(K-1).

        Returns
        -------
        gamma_cache : Dict[frozenset, MLIVEstimator]
            Maps frozenset({k, ell}) -> fitted gamma estimator
        """
        Y = np.asarray(W['Y']).flatten()
        X = np.asarray(W['X'])
        Z = np.asarray(W['Z'])

        gamma_cache = {}

        for k in range(n_folds):
            for ell in range(k + 1, n_folds):
                pair = frozenset({k, ell})
                train_mask = (fold_assignments != k) & (fold_assignments != ell)

                if not train_mask.any():
                    if self.control.verbose:
                        print(f"  WARNING: No training data for "
                              f"pair ({k}, {ell})")
                    continue

                gamma_kl = self._get_mliv_instance()
                gamma_kl.fit({
                    'X': X[train_mask],
                    'Z': Z[train_mask],
                    'Y': Y[train_mask]
                })
                gamma_cache[pair] = gamma_kl

                if self.control.verbose:
                    n_train = train_mask.sum()
                    print(f"  Fitted gamma for pair ({k}, {ell}): "
                          f"{n_train} training obs")

        return gamma_cache

    def _compute_double_crossfit_M(
        self,
        W: Dict[str, NDArray],
        current_fold: int,
        fold_assignments: NDArray,
        moment: 'BaseMoment',
        gamma_cache: Dict,
        x_featurizer_train,
        **moment_kwargs
    ) -> NDArray:
        """
        Compute moment matrix M using double cross-fitting.

        For outer fold k (current_fold), assembles M using all data
        outside fold k. For each inner fold ell != k:
          - Look up gamma_{k,ell} from cache
          - Compute M_ell on fold ell using that gamma
        Stack all M_ell and return aligned with training data order.

        Parameters
        ----------
        W : Dict
            Full data dictionary
        current_fold : int
            Outer test fold index
        fold_assignments : NDArray
            Fold assignment per observation
        moment : BaseMoment
            Nonlinear moment
        gamma_cache : Dict
            Pre-computed inner gammas
        x_featurizer_train : TransformerMixin
            Featurizer fitted on training data (folds != current_fold)
        **moment_kwargs
            Extra args for moment.compute_all_basis_gamma()

        Returns
        -------
        M_train : NDArray, shape (n_train, p)
        """
        Y = np.asarray(W['Y']).flatten()
        X = np.asarray(W['X'])
        Z = np.asarray(W['Z'])
        n = len(Y)
        n_folds = self.control.n_folds

        train_mask = fold_assignments != current_fold

        # Collect M by fold
        M_by_fold = {}

        for fold_ell in range(n_folds):
            if fold_ell == current_fold:
                continue

            pair_key = frozenset({current_fold, fold_ell})
            if pair_key not in gamma_cache:
                if self.control.verbose:
                    print(f"    WARNING: No gamma for pair "
                          f"({current_fold}, {fold_ell}), skipping")
                continue

            gamma_kl = gamma_cache[pair_key]

            # Data for fold ell
            fold_mask = fold_assignments == fold_ell
            W_ell = {
                'Y': Y[fold_mask],
                'X': X[fold_mask],
                'Z': Z[fold_mask]
            }

            def gamma_func(X_eval, _est=gamma_kl):
                return _est.predict(X_eval)

            M_ell = moment.compute_all_basis_gamma(
                gamma_func, x_featurizer_train, W_ell, **moment_kwargs
            )

            M_by_fold[fold_ell] = M_ell

        # Determine feature dimension p
        p = None
        for m_arr in M_by_fold.values():
            if m_arr.shape[0] > 0:
                p = m_arr.shape[1]
                break

        if p is None:
            raise ValueError(
                f"Could not compute M for any inner fold "
                f"(current_fold={current_fold})"
            )

        # Assemble M_full, fill by fold, then extract training rows
        M_full = np.zeros((n, p))
        for fold_ell, M_ell in M_by_fold.items():
            fold_mask = fold_assignments == fold_ell
            M_full[fold_mask] = M_ell

        M_train = M_full[train_mask]
        return M_train

    def fit(
        self,
        W: Dict[str, NDArray[np.float64]],
        moment: 'BaseMoment',
        **moment_kwargs
    ) -> ADMLIVResult:
        """
        Fit the ADMLIV estimator using cross-fitting.

        Automatically detects whether the moment is nonlinear (by checking
        for compute_all_basis_gamma method) and uses double cross-fitting
        if needed. For linear moments, uses standard single cross-fitting.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': shape (n, 1) or (n,) - outcome variable
            - 'X': shape (n, d_x) - endogenous variables
            - 'Z': shape (n, d_z) - instrumental variables
        moment : BaseMoment
            Moment function instance defining the parameter of interest.
            If it implements compute_all_basis_gamma(), double cross-fitting
            is used automatically.
        **moment_kwargs : dict
            Additional arguments passed to moment.compute() and
            moment.compute_all_basis()

        Returns
        -------
        result : ADMLIVResult
            Estimation results containing debiased estimate, standard errors,
            confidence intervals, and diagnostics
        """
        Y = np.asarray(W['Y']).flatten()
        X = np.asarray(W['X'])
        Z = np.asarray(W['Z'])
        n = X.shape[0]

        # Detect nonlinear
        nonlinear = self._is_nonlinear_moment(moment)

        if self.control.verbose:
            print("=" * 70)
            print("ADMLIV: Automatic Debiased Machine Learning for IV")
            print("=" * 70)
            print(f"Number of observations: {n}")
            print(f"Number of cross-fitting folds: {self.control.n_folds}")
            mode = ("NONLINEAR (double cross-fitting)"
                    if nonlinear else "LINEAR (single cross-fitting)")
            print(f"Functional type: {mode}")
            print("=" * 70)

        if nonlinear and self.control.n_folds < 3:
            raise ValueError(
                "Double cross-fitting requires n_folds >= 3. "
                f"Got n_folds={self.control.n_folds}."
            )

        if nonlinear and self.control.use_cv_for_pgmm:
            raise ValueError(
                "Cross-validated PGMM (use_cv_for_pgmm=True) is not supported "
                "for nonlinear functionals. Double cross-fitting already requires "
                "three layers of data splitting; adding CV would slice data too "
                "thin. Set use_cv_for_pgmm=False and specify pgmm_c directly."
            )

        # Setup cross-validation splits
        kfold = KFold(
            n_splits=self.control.n_folds,
            shuffle=True,
            random_state=self.control.random_state
        )

        # Compute fold assignments
        fold_assignments = np.full(n, -1, dtype=int)
        fold_splits = list(kfold.split(X))
        for fold_idx, (_, test_idx) in enumerate(fold_splits):
            fold_assignments[test_idx] = fold_idx

        # Precompute inner gammas for nonlinear double cross-fitting
        gamma_cache = None
        if nonlinear:
            n_pairs = self.control.n_folds * (self.control.n_folds - 1) // 2
            if self.control.verbose:
                print(f"\nPrecomputing {n_pairs} inner gamma estimators "
                      f"for double cross-fitting...")

            gamma_cache = self._precompute_inner_gammas(
                W, fold_assignments, self.control.n_folds
            )

            if self.control.verbose:
                print(f"  Cached {len(gamma_cache)} gamma estimators.\n")

        # Storage
        psi_tilde = np.zeros(n)
        m_tilde = np.zeros(n)

        fold_estimates = []
        self.fold_estimators_ = []

        # Cross-fitting loop
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            if self.control.verbose:
                print(f"\n--- Fold {fold_idx + 1}/{self.control.n_folds} ---")

            W_train = {
                'Y': Y[train_idx],
                'X': X[train_idx],
                'Z': Z[train_idx]
            }
            W_test = {
                'Y': Y[test_idx],
                'X': X[test_idx],
                'Z': Z[test_idx]
            }

            # Stage 1a: Fit gamma on training data
            if self.control.verbose:
                print("  Stage 1a: Fitting MLIV estimator...")

            gamma_estimator = self._get_mliv_instance()
            gamma_estimator.fit(W_train)

            def gamma_func(X_eval, _est=gamma_estimator):
                return _est.predict(X_eval)

            # Stage 1b: Fit PGMM
            if self.control.verbose:
                print("  Stage 1b: Fitting PGMM for Riesz representer...")

            pgmm_estimator = self._get_pgmm_instance()

            if nonlinear:
                # Fit featurizer on training data for M computation
                x_featurizer_train = clone(self.x_featurizer)
                x_featurizer_train.fit(X[train_idx])

                # Double cross-fit M
                M_train = self._compute_double_crossfit_M(
                    W, fold_idx, fold_assignments, moment,
                    gamma_cache, x_featurizer_train, **moment_kwargs
                )

                if self.control.verbose:
                    print(f"    Double cross-fit M shape: {M_train.shape}")

                # Fit PGMM with precomputed M
                pgmm_estimator.fit_with_M(W_train, M_train)
            else:
                # Linear: PGMM computes M internally
                pgmm_estimator.fit(W_train, moment, **moment_kwargs)

            # Stage 2: Evaluate on test fold
            if self.control.verbose:
                print("  Stage 2: Evaluating on test fold...")

            m_test = moment.compute(gamma_func, W_test, **moment_kwargs)
            alpha_test = pgmm_estimator.predict(W_test['Z'])
            gamma_test = gamma_func(W_test['X']).flatten()
            residuals_test = W_test['Y'] - gamma_test
            influence_adj_test = alpha_test * residuals_test
            psi_test = m_test + influence_adj_test

            m_tilde[test_idx] = m_test
            psi_tilde[test_idx] = psi_test

            fold_info = {
                'fold_idx': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'm_mean': np.mean(m_test),
                'alpha_mean': np.mean(alpha_test),
                'residual_mean': np.mean(residuals_test),
                'influence_adj_mean': np.mean(influence_adj_test),
                'psi_mean': np.mean(psi_test)
            }
            fold_estimates.append(fold_info)
            self.fold_estimators_.append({
                'gamma': gamma_estimator,
                'pgmm': pgmm_estimator
            })

            if self.control.verbose:
                print(f"  Fold {fold_idx + 1} complete:")
                print(f"    Train/Test: "
                      f"{fold_info['train_size']}/{fold_info['test_size']}")
                print(f"    Mean m(W, gamma): {fold_info['m_mean']:.6f}")
                print(f"    Mean alpha(Z): {fold_info['alpha_mean']:.6f}")
                print(f"    Mean residual: "
                      f"{fold_info['residual_mean']:.6f}")
                print(f"    Mean influence adj: "
                      f"{fold_info['influence_adj_mean']:.6f}")

        # Aggregate (identical for linear and nonlinear)
        theta_debiased = np.mean(psi_tilde)
        theta_plugin = np.mean(m_tilde)

        influence_debiased = psi_tilde - theta_debiased
        influence_plugin = m_tilde - theta_plugin

        variance_debiased = np.mean(influence_debiased ** 2)
        variance_plugin = np.mean(influence_plugin ** 2)

        se_debiased = np.sqrt(variance_debiased / n)
        se_plugin = np.sqrt(variance_plugin / n)

        alpha_ci = 1 - self.control.confidence_level
        z_alpha = ss.norm.ppf(1 - alpha_ci / 2)
        ci_lower = theta_debiased - z_alpha * se_debiased
        ci_upper = theta_debiased + z_alpha * se_debiased
        ci_lower_plugin = theta_plugin - z_alpha * se_plugin
        ci_upper_plugin = theta_plugin + z_alpha * se_plugin

        self.result_ = ADMLIVResult(
            theta_debiased=theta_debiased,
            theta_plugin=theta_plugin,
            se_debiased=se_debiased,
            se_plugin=se_plugin,
            variance_debiased=variance_debiased,
            variance_plugin=variance_plugin,
            influence_functions=influence_debiased,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_lower_plugin=ci_lower_plugin,
            ci_upper_plugin=ci_upper_plugin,
            confidence_level=self.control.confidence_level,
            n_samples=n,
            n_folds=self.control.n_folds,
            fold_estimates=fold_estimates
        )

        self.is_fitted_ = True

        if self.control.verbose:
            print("\n" + self.result_.summary())

        return self.result_

    def get_result(self) -> ADMLIVResult:
        """
        Get the estimation result.
        
        Returns
        -------
        result : ADMLIVResult
            Estimation results
        
        Raises
        ------
        ValueError
            If estimator has not been fitted
        """
        if not self.is_fitted_:
            raise ValueError("ADMLIV must be fitted before accessing results")
        return self.result_


# Convenience function for simpler API
def fit_admliv(
    W: Dict[str, NDArray[np.float64]],
    mliv_estimator: Union[MLIVEstimator, Callable[[], MLIVEstimator]],
    moment: 'BaseMoment',
    x_featurizer: TransformerMixin,
    z_featurizer: TransformerMixin,
    n_folds: int = 5,
    use_cv_for_pgmm: bool = True,
    verbose: bool = True,
    **moment_kwargs
) -> ADMLIVResult:
    """
    Convenience function for fitting ADMLIV estimator.
    
    Parameters
    ----------
    W : Dict[str, NDArray]
        Data dictionary with keys 'Y', 'X', 'Z'
    mliv_estimator : Union[MLIVEstimator, Callable]
        MLIV estimator or factory function with Dict interface.
        All standard estimators (DoubleLassoEstimator, etc.) already use this interface
    moment : BaseMoment
        Moment function
    x_featurizer : TransformerMixin
        Featurizer for d(X) basis
    z_featurizer : TransformerMixin
        Featurizer for b(Z) basis
    n_folds : int, default=5
        Number of cross-fitting folds
    use_cv_for_pgmm : bool, default=True
        Whether to use CV for PGMM penalty selection
    verbose : bool, default=True
        Whether to print progress
    **moment_kwargs : dict
        Additional moment arguments
    
    Returns
    -------
    result : ADMLIVResult
        Estimation results
    
    Examples
    --------
    >>> result = fit_admliv(
    ...     W={'Y': Y, 'X': X, 'Z': Z},
    ...     mliv_estimator=my_mliv_factory,
    ...     moment=WeightedAverage(),
    ...     x_featurizer=CoordinatePolyTransform(degree=3),
    ...     z_featurizer=CoordinatePolyTransform(degree=3),
    ...     weight_func=lambda x: np.ones(x.shape[0])
    ... )
    """
    control = ADMLIVControl(
        n_folds=n_folds,
        use_cv_for_pgmm=use_cv_for_pgmm,
        verbose=verbose
    )
    
    admliv = ADMLIV(
        mliv_estimator=mliv_estimator,
        x_featurizer=x_featurizer,
        z_featurizer=z_featurizer,
        control=control
    )
    
    return admliv.fit(W, moment, **moment_kwargs)