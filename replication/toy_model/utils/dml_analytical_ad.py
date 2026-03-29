# simulations/toy_model/utils/dml_analytical_ad.py

"""
DML estimator for Average Derivative using analytical Riesz representer.

This estimator combines:
1. Cross-fitting for nuisance parameter estimation (like ADMLIV)
2. Analytical RR coefficient computation instead of PGMM (from Chen et al. 2023)
3. Any moment functional (via moment.compute())
"""

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
import scipy.stats as ss

from .rr_ad_analytical import AnalyticalRieszRepresenter


class MLIVEstimator(Protocol):
    """Protocol for MLIV estimators with Dict interface."""

    def fit(self, W: Dict[str, NDArray[np.float64]], **kwargs) -> 'MLIVEstimator':
        """Fit the MLIV estimator."""
        ...

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict gamma(X)."""
        ...


@dataclass
class DMLAnalyticalADControl:
    """
    Control parameters for DML with Analytical AD.

    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-fitting folds
    random_state : int, default=42
        Random seed for reproducibility
    confidence_level : float, default=0.95
        Confidence level for confidence intervals
    verbose : bool, default=True
        If True, prints progress information
    """
    n_folds: int = 5
    random_state: int = 42
    confidence_level: float = 0.95
    verbose: bool = True

    def __post_init__(self):
        """Validate control parameters."""
        if self.n_folds <= 1:
            raise ValueError("n_folds must be greater than 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass
class DMLAnalyticalADResult:
    """
    Results from DML with Analytical AD estimation.

    Attributes
    ----------
    theta_debiased : float
        Debiased estimate of the functional
    theta_plugin : float
        Plug-in estimate (biased)
    se_debiased : float
        Standard error of debiased estimate
    se_plugin : float
        Standard error of plug-in estimate
    ci_lower : float
        Lower bound of debiased CI
    ci_upper : float
        Upper bound of debiased CI
    ci_lower_plugin : float
        Lower bound of plug-in CI
    ci_upper_plugin : float
        Upper bound of plug-in CI
    psi_tilde_mean : float
        Mean of orthogonal moments (should be close to 0)
    psi_tilde_std : float
        Standard deviation of orthogonal moments
    """
    theta_debiased: float
    theta_plugin: float
    se_debiased: float
    se_plugin: float
    ci_lower: float
    ci_upper: float
    ci_lower_plugin: float
    ci_upper_plugin: float
    psi_tilde_mean: float
    psi_tilde_std: float


class DMLAnalyticalAD(BaseEstimator):
    """
    Double Machine Learning estimator with analytical Riesz representer.

    This estimator implements cross-fitted estimation using:
    - Any MLIV estimator for the structural function gamma(X)
    - Analytical Riesz representer computation (Chen et al. 2023) instead of PGMM
    - Any moment functional (via BaseMoment interface)

    The estimator computes:
        θ̂ = (1/n) Σ_i [m(W_i, γ̂) + α̂(Z_i) · ψ(W_i, γ̂)]

    where:
    - m(W_i, γ̂) is the plug-in moment (e.g., ∂γ̂(X_i)/∂X_j for average derivative)
    - ψ(W_i, γ̂) is the residual moment (e.g., Y_i - γ̂(X_i))
    - α̂(Z_i) = Wz(Z_i)' ρ̂ where ρ̂ is from analytical RR formula

    Parameters
    ----------
    mliv_estimator : MLIVEstimator
        Base MLIV estimator (must have fit(W) and predict(X) methods)
    x_featurizer : TransformerMixin
        Featurizer for endogenous variables
    z_featurizer : TransformerMixin
        Featurizer for instruments (used in RR computation)
    control : Optional[DMLAnalyticalADControl], default=None
        Control parameters. If None, uses defaults.

    Attributes
    ----------
    result_ : DMLAnalyticalADResult
        Estimation results (available after calling fit)
    fold_estimators_ : list
        List of fitted estimators from each fold

    References
    ----------
    Chen, X., Christensen, T.M., and Tamer, E. (2023). "Efficient estimation
    of average derivatives in NPIV models: Simulation comparisons of neural
    network estimators"

    Examples
    --------
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> from admliv.estimators import DoubleLassoEstimator
    >>> from admliv.moments import WeightedAverageDerivative
    >>>
    >>> # Setup
    >>> mliv = DoubleLassoEstimator(...)
    >>> x_feat = PolynomialFeatures(degree=3)
    >>> z_feat = PolynomialFeatures(degree=3)
    >>> moment = WeightedAverageDerivative(deriv_index=0)
    >>>
    >>> # Estimate
    >>> dml = DMLAnalyticalAD(mliv, x_feat, z_feat)
    >>> result = dml.fit(W, moment)
    >>> print(f"Estimate: {result.theta_debiased:.4f} ({result.se_debiased:.4f})")
    """

    def __init__(
        self,
        mliv_estimator: MLIVEstimator,
        x_featurizer: TransformerMixin,
        z_featurizer: TransformerMixin,
        control: Optional[DMLAnalyticalADControl] = None
    ):
        self.mliv_estimator = mliv_estimator
        self.x_featurizer = x_featurizer
        self.z_featurizer = z_featurizer
        self.control = control if control is not None else DMLAnalyticalADControl()

        # Results (set after fitting)
        self.result_ = None
        self.fold_estimators_ = []

    def _get_mliv_instance(self) -> MLIVEstimator:
        """Get a fresh instance of the MLIV estimator."""
        return clone(self.mliv_estimator)

    def _get_rr_instance(self, deriv_index: int) -> AnalyticalRieszRepresenter:
        """
        Get a fresh instance of the analytical RR estimator.

        Parameters
        ----------
        deriv_index : int
            Index of X variable to differentiate with respect to

        Returns
        -------
        rr_estimator : AnalyticalRieszRepresenter
            Fresh RR estimator instance
        """
        return AnalyticalRieszRepresenter(
            x_featurizer=clone(self.x_featurizer),
            z_featurizer=clone(self.z_featurizer),
            deriv_index=deriv_index
        )

    def fit(
        self,
        W: Dict[str, NDArray[np.float64]],
        moment: Any,  # BaseMoment instance
        **moment_kwargs
    ) -> DMLAnalyticalADResult:
        """
        Fit the DML estimator using cross-fitting with analytical RR.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': shape (n, 1) or (n,) - outcome variable
            - 'X': shape (n, d_x) - endogenous variables
            - 'Z': shape (n, d_z) - instrumental variables
        moment : BaseMoment
            Moment function instance defining the parameter of interest
        **moment_kwargs : dict
            Additional arguments passed to moment.compute()

        Returns
        -------
        result : DMLAnalyticalADResult
            Estimation results with debiased estimate, standard errors, and CIs
        """
        # Extract and validate data
        Y = np.asarray(W['Y']).flatten()
        X = np.asarray(W['X'])
        Z = np.asarray(W['Z'])
        n = X.shape[0]

        if self.control.verbose:
            print("=" * 70)
            print("DML with Analytical RR (Chen et al. 2023)")
            print("=" * 70)
            print(f"Number of observations: {n}")
            print(f"Number of cross-fitting folds: {self.control.n_folds}")
            print("=" * 70)

        # Setup cross-validation splits
        kfold = KFold(
            n_splits=self.control.n_folds,
            shuffle=True,
            random_state=self.control.random_state
        )

        # Storage for cross-fitted values
        psi_tilde = np.zeros(n)  # Orthogonal moment
        m_tilde = np.zeros(n)    # Plug-in moment

        # Storage for fold diagnostics
        self.fold_estimators_ = []

        # Cross-fitting loop
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            if self.control.verbose:
                print(f"\n--- Fold {fold_idx + 1}/{self.control.n_folds} ---")

            # Split data
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

            # Stage 1a: Fit gamma (structural function) on training data
            if self.control.verbose:
                print("  Stage 1a: Fitting MLIV estimator...")

            gamma_estimator = self._get_mliv_instance()
            gamma_estimator.fit(W_train)

            # Create gamma function for moment evaluation
            def gamma_func(X_eval):
                return gamma_estimator.predict(X_eval)

            # Stage 1b: Fit analytical RR on training data
            if self.control.verbose:
                print("  Stage 1b: Fitting analytical Riesz representer...")

            # Get deriv_index from moment if it has one
            deriv_index = getattr(moment, 'deriv_index', 0)

            # Fit RR estimator on training data
            rr_estimator = self._get_rr_instance(deriv_index)
            rr_estimator.fit(W_train)

            # Stage 2: Evaluate on test fold
            if self.control.verbose:
                print("  Stage 2: Evaluating on test fold...")

            # Compute m(W_i, gamma_hat) for test observations (plug-in moment)
            m_test = moment.compute(gamma_func, W_test, **moment_kwargs)

            # Predict alpha(Z_i) on test observations
            alpha_test = rr_estimator.predict(W_test['Z'])

            # Compute residuals: Y_i - gamma(X_i)
            gamma_test = gamma_func(W_test['X']).flatten()
            residuals_test = W_test['Y'] - gamma_test

            # Orthogonal moment: m(W, gamma) + alpha(Z) * (Y - gamma(X))
            psi_tilde[test_idx] = m_test + alpha_test * residuals_test

            # Compute plug-in moment
            m_tilde[test_idx] = m_test

            # Store fold estimators
            self.fold_estimators_.append({
                'gamma_estimator': gamma_estimator,
                'rr_estimator': rr_estimator
            })

        # Compute estimates
        theta_debiased = np.mean(psi_tilde)
        theta_plugin = np.mean(m_tilde)

        # Compute standard errors
        se_debiased = np.std(psi_tilde, ddof=1) / np.sqrt(n)
        se_plugin = np.std(m_tilde, ddof=1) / np.sqrt(n)

        # Compute confidence intervals
        z_crit = ss.norm.ppf(1 - (1 - self.control.confidence_level) / 2)
        ci_lower = theta_debiased - z_crit * se_debiased
        ci_upper = theta_debiased + z_crit * se_debiased
        ci_lower_plugin = theta_plugin - z_crit * se_plugin
        ci_upper_plugin = theta_plugin + z_crit * se_plugin

        # Store results
        self.result_ = DMLAnalyticalADResult(
            theta_debiased=theta_debiased,
            theta_plugin=theta_plugin,
            se_debiased=se_debiased,
            se_plugin=se_plugin,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_lower_plugin=ci_lower_plugin,
            ci_upper_plugin=ci_upper_plugin,
            psi_tilde_mean=np.mean(psi_tilde),
            psi_tilde_std=np.std(psi_tilde, ddof=1)
        )

        if self.control.verbose:
            print("\n" + "=" * 70)
            print("Estimation Results")
            print("=" * 70)
            print(f"Debiased estimate: {theta_debiased:.6f} (SE: {se_debiased:.6f})")
            print(f"Plug-in estimate:  {theta_plugin:.6f} (SE: {se_plugin:.6f})")
            print(f"{self.control.confidence_level*100:.0f}% CI (debiased): [{ci_lower:.6f}, {ci_upper:.6f}]")
            print(f"{self.control.confidence_level*100:.0f}% CI (plug-in):  [{ci_lower_plugin:.6f}, {ci_upper_plugin:.6f}]")
            print("=" * 70)

        return self.result_
