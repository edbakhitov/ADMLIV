# admliv/estimators/sieve.py
"""
Sieve-based estimators for MLIV.

This module provides:
- NpivSieveEstimator: 2SLS with basis expansion 
- DoubleLassoEstimator: Two-stage Lasso for high-dimensional IV

All estimators follow the BaseMLIVEstimator interface and can be used
directly with moment functions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV

from .base import BaseMLIVEstimator


class NpivSieveEstimator(BaseMLIVEstimator):
    """
    Nonparametric IV estimator using 2SLS with sieve basis expansion.
    
    This estimator handles the endogenous case where X is correlated with
    the error term but Z is a valid instrument.
    
    Model: Y = d(X)'β + ε where E[ε|Z] = 0 but E[ε|X] ≠ 0
    First stage: Project d(X) onto b(Z)
    Second stage: Regress Y on projected d(X)
    
    Parameters
    ----------
    x_featurizer : TransformerMixin, optional
        Transformer for X basis expansion d(X). If None, uses identity.
    z_featurizer : TransformerMixin, optional
        Transformer for Z basis expansion b(Z). If None, uses identity.
    ridge_penalty : float, default=1e-10
        Ridge penalty for numerical stability.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features_x,)
        Estimated coefficients β̂
    n_features_X_ : int
        Number of X basis functions
    n_features_Z_ : int
        Number of Z basis functions
    """
    
    def __init__(
        self,
        x_featurizer: Optional[TransformerMixin] = None,
        z_featurizer: Optional[TransformerMixin] = None,
        ridge_penalty: float = 1e-10
    ):
        super().__init__(x_featurizer=x_featurizer, z_featurizer=z_featurizer)
        self.ridge_penalty = ridge_penalty
    
    def fit(self, W: Dict[str, NDArray[np.float64]]) -> 'NpivSieveEstimator':
        """
        Fit the NPIV estimator using 2SLS.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': outcome, shape (n,) or (n, 1)
            - 'X': endogenous regressors, shape (n, d_x)
            - 'Z': instrumental variables, shape (n, d_z)

        Returns
        -------
        self : NpivSieveEstimator
            Fitted estimator
        """
        # Extract data from dictionary
        X = W['X']
        Z = W['Z']
        Y = W['Y']

        # Transform to basis
        X_basis = self._transform_X(X, fit=True)
        Z_basis = self._transform_Z(Z, fit=True)
        Y = self._check_y(Y)
        
        self.n_features_X_ = X_basis.shape[1]
        self.n_features_Z_ = Z_basis.shape[1]
        
        # Check rank condition
        if self.n_features_Z_ < self.n_features_X_:
            raise ValueError(
                f"Rank condition violated: n_features_Z ({self.n_features_Z_}) "
                f"< n_features_X ({self.n_features_X_})"
            )
        
        # First stage: projection matrix P_Z = Z(Z'Z)^{-1}Z'
        ZZ = Z_basis.T @ Z_basis
        for attempt in range(10):
            try:
                Pz = Z_basis @ np.linalg.pinv(ZZ) @ Z_basis.T
                break
            except np.linalg.LinAlgError:
                ZZ += self.ridge_penalty * np.eye(self.n_features_Z_)
        else:
            raise np.linalg.LinAlgError(
                f"First stage failed after 10 ridge-stabilization attempts "
                f"(ridge_penalty={self.ridge_penalty})"
            )

        # Second stage: β̂ = (X'P_Z X)^{-1} X'P_Z Y
        XX_hat = X_basis.T @ Pz @ X_basis
        for attempt in range(10):
            try:
                self.coef_ = np.linalg.solve(XX_hat, X_basis.T @ Pz @ Y)
                break
            except np.linalg.LinAlgError:
                XX_hat += self.ridge_penalty * np.eye(self.n_features_X_)
        else:
            raise np.linalg.LinAlgError(
                f"Second stage failed after 10 ridge-stabilization attempts "
                f"(ridge_penalty={self.ridge_penalty})"
            )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict gamma(X) = d(X)'β̂.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Input data

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Estimator is not fitted. Call fit() first.")

        X_basis = self._transform_X(X, fit=False)
        return X_basis @ self.coef_

    def predict_derivative(
        self,
        X: NDArray[np.float64],
        wrt: Optional[int] = None,
        eps: float = 1e-6
    ) -> NDArray[np.float64]:
        """
        Compute the derivative of gamma(X) with respect to input features.

        Uses analytical derivatives if the featurizer has a `transform_derivative`
        method, otherwise falls back to numerical differentiation.

        For gamma(X) = d(X)'β, the derivative is:
            ∂γ/∂X_j = Σ_k β_k * (∂d_k(X)/∂X_j)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Input data
        wrt : int or None, default=None
            Index of feature to differentiate with respect to.
            If None, returns full Jacobian (derivatives w.r.t. all features).
        eps : float, default=1e-6
            Step size for numerical differentiation (if needed)

        Returns
        -------
        gradients : ndarray
            If wrt is int: shape (n_samples,) - derivative ∂γ/∂X_j
            If wrt is None: shape (n_samples, n_features) - full Jacobian
        """
        if not self.is_fitted_:
            raise ValueError("Estimator is not fitted. Call fit() first.")

        X = self._check_X(X)
        n_samples, n_features = X.shape

        # Check if featurizer has analytical derivative
        has_analytical = (self.x_featurizer is not None and
                         hasattr(self.x_featurizer, 'transform_derivative'))

        if wrt is not None:
            # Single derivative
            if has_analytical:
                d_basis = self.x_featurizer.transform_derivative(X, wrt=wrt)
                return d_basis @ self.coef_
            else:
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, wrt] += eps
                X_minus[:, wrt] -= eps
                return (self.predict(X_plus) - self.predict(X_minus)) / (2 * eps)
        else:
            # Full Jacobian
            jacobian = np.zeros((n_samples, n_features))
            for j in range(n_features):
                if has_analytical:
                    d_basis = self.x_featurizer.transform_derivative(X, wrt=j)
                    jacobian[:, j] = d_basis @ self.coef_
                else:
                    X_plus = X.copy()
                    X_minus = X.copy()
                    X_plus[:, j] += eps
                    X_minus[:, j] -= eps
                    jacobian[:, j] = (self.predict(X_plus) - self.predict(X_minus)) / (2 * eps)
            return jacobian


@dataclass
class LassoStageControl:
    """
    Control parameters for a single Lasso stage.

    Parameters
    ----------
    use_cv : bool, default=False
        Whether to use cross-validation for alpha selection
    alpha : float, default=0.001
        Fixed alpha value (used when use_cv=False)
    alphas : NDArray, optional
        Alpha grid for CV (used when use_cv=True)
        If None, uses default np.logspace(-7, -1, 100)
    cv : int, default=3
        Number of CV folds (used when use_cv=True)
    max_iter : int, default=5000
        Maximum iterations for Lasso
    tol : float, default=0.001
        Tolerance for Lasso convergence
    """
    use_cv: bool = False
    alpha: float = 0.001
    alphas: Optional[NDArray[np.float64]] = None
    cv: int = 3
    max_iter: int = 5000
    tol: float = 0.001

    def __post_init__(self):
        """Set default alphas if not provided and CV is enabled."""
        if self.use_cv and self.alphas is None:
            self.alphas = np.logspace(-7, -1, 100)


@dataclass
class DoubleLassoControl:
    """
    Control parameters for Double Lasso estimator.

    This provides a clean, hierarchical way to configure the two stages
    of Double Lasso independently or together.

    Parameters
    ----------
    first_stage : LassoStageControl, optional
        Configuration for first stage. If None, uses default.
    second_stage : LassoStageControl, optional
        Configuration for second stage. If None, uses CV by default.

    Examples
    --------
    # Example 1: Use CV on both stages with same settings
    >>> control = DoubleLassoControl.with_cv(cv=5, alphas=np.logspace(-5, 0, 50))

    # Example 2: CV on second stage only (default)
    >>> control = DoubleLassoControl()

    # Example 3: Fixed alpha on both stages
    >>> control = DoubleLassoControl.with_fixed_alpha(fs_alpha=0.01, ss_alpha=0.001)

    # Example 4: Different settings per stage
    >>> fs = LassoStageControl(use_cv=False, alpha=0.01)
    >>> ss = LassoStageControl(use_cv=True, cv=5)
    >>> control = DoubleLassoControl(first_stage=fs, second_stage=ss)
    """
    first_stage: LassoStageControl = field(default_factory=LassoStageControl)
    second_stage: LassoStageControl = field(
        default_factory=lambda: LassoStageControl(use_cv=True)
    )

    @classmethod
    def with_cv(
        cls,
        cv: int = 3,
        alphas: Optional[NDArray[np.float64]] = None,
        max_iter: int = 10000
    ) -> 'DoubleLassoControl':
        """
        Create control with CV enabled on both stages.

        Parameters
        ----------
        cv : int, default=3
            Number of CV folds for both stages
        alphas : NDArray, optional
            Alpha grid for both stages. If None, uses default.
        max_iter : int, default=5000
            Max iterations for both stages

        Returns
        -------
        DoubleLassoControl
            Configured control object
        """
        fs = LassoStageControl(use_cv=True, cv=cv, alphas=alphas, max_iter=max_iter)
        ss = LassoStageControl(use_cv=True, cv=cv, alphas=alphas, max_iter=max_iter)
        return cls(first_stage=fs, second_stage=ss)

    @classmethod
    def with_fixed_alpha(
        cls,
        fs_alpha: float = 0.001,
        ss_alpha: float = 0.001,
        max_iter: int = 5000
    ) -> 'DoubleLassoControl':
        """
        Create control with fixed alphas (no CV).

        Parameters
        ----------
        fs_alpha : float, default=0.001
            Fixed alpha for first stage
        ss_alpha : float, default=0.001
            Fixed alpha for second stage
        max_iter : int, default=5000
            Max iterations for both stages

        Returns
        -------
        DoubleLassoControl
            Configured control object
        """
        fs = LassoStageControl(use_cv=False, alpha=fs_alpha, max_iter=max_iter)
        ss = LassoStageControl(use_cv=False, alpha=ss_alpha, max_iter=max_iter)
        return cls(first_stage=fs, second_stage=ss)


class DoubleLassoEstimator(BaseMLIVEstimator):
    """
    Double Lasso estimator for high-dimensional IV regression (improved architecture).

    This is an improved version of DoubleLassoEstimator with cleaner configuration
    using the DoubleLassoControl class.

    First stage: For each column of d(X), use Lasso to predict from b(Z)
    Second stage: Use Lasso to regress Y on predicted d̂(X)

    Parameters
    ----------
    x_featurizer : TransformerMixin, optional
        Transformer for X basis expansion d(X). If None, uses identity.
    z_featurizer : TransformerMixin, optional
        Transformer for Z basis expansion b(Z). If None, uses identity.
    control : DoubleLassoControl, optional
        Control parameters for both stages. If None, uses default
        (no CV on first stage, CV on second stage).

    Attributes
    ----------
    coef_ : ndarray of shape (n_features_x + 1,)
        Estimated coefficients [intercept, β̂]
    n_nonzero_coef_ : int
        Number of non-zero coefficients (excluding intercept)
    fs_best_alphas_ : list of float
        Best alpha for each first stage regression (if first_stage.use_cv=True)
    ss_best_alpha_ : float
        Best alpha for second stage (if second_stage.use_cv=True)

    Examples
    --------
    # Example 1: Default (CV on second stage only)
    >>> estimator = DoubleLassoEstimator()
    >>> estimator.fit(W)

    # Example 2: CV on both stages
    >>> control = DoubleLassoControl.with_cv(cv=5)
    >>> estimator = DoubleLassoEstimator(control=control)
    >>> estimator.fit(W)

    # Example 3: Fixed alphas on both stages
    >>> control = DoubleLassoControl.with_fixed_alpha(fs_alpha=0.01, ss_alpha=0.001)
    >>> estimator = DoubleLassoEstimator(control=control)
    >>> estimator.fit(W)

    # Example 4: Custom per-stage configuration
    >>> fs = LassoStageControl(use_cv=True, cv=10, alphas=np.logspace(-5, 0, 20))
    >>> ss = LassoStageControl(use_cv=False, alpha=0.0001)
    >>> control = DoubleLassoControl(first_stage=fs, second_stage=ss)
    >>> estimator = DoubleLassoEstimator(control=control)
    >>> estimator.fit(W)
    """

    def __init__(
        self,
        x_featurizer: Optional[TransformerMixin] = None,
        z_featurizer: Optional[TransformerMixin] = None,
        control: Optional[DoubleLassoControl] = None
    ):
        super().__init__(x_featurizer=x_featurizer, z_featurizer=z_featurizer)
        self.control = control if control is not None else DoubleLassoControl()

    def _create_lasso_estimator(
        self,
        stage_control: LassoStageControl,
        fit_intercept: bool = True
    ):
        """
        Create a Lasso or LassoCV estimator based on stage configuration.

        Parameters
        ----------
        stage_control : LassoStageControl
            Configuration for this stage
        fit_intercept : bool, default=True
            Whether to fit intercept

        Returns
        -------
        Lasso or LassoCV
            Configured estimator
        """
        if stage_control.use_cv:
            return LassoCV(
                cv=stage_control.cv,
                alphas=stage_control.alphas,
                fit_intercept=fit_intercept,
                tol=stage_control.tol,
                max_iter=stage_control.max_iter
            )
        else:
            return Lasso(
                alpha=stage_control.alpha,
                fit_intercept=fit_intercept,
                tol=stage_control.tol,
                max_iter=stage_control.max_iter
            )

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, W: Dict[str, NDArray[np.float64]]) -> 'DoubleLassoEstimator':
        """
        Fit the Double Lasso estimator.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': outcome, shape (n,) or (n, 1)
            - 'X': endogenous regressors, shape (n, d_x) (NO intercept - added internally)
            - 'Z': instrumental variables, shape (n, d_z) (NO intercept - added internally)

        Returns
        -------
        self : DoubleLassoEstimator
            Fitted estimator
        """
        # Extract data from dictionary
        X = W['X']
        Z = W['Z']
        Y = W['Y']

        # Transform to basis (no intercept expected)
        X_basis = self._transform_X(X, fit=True)
        Z_basis = self._transform_Z(Z, fit=True)
        Y = self._check_y(Y)

        self.n_features_X_ = X_basis.shape[1]
        self.n_features_Z_ = Z_basis.shape[1]

        # ========== First Stage ==========
        # Predict each column of X_basis from Z_basis
        lasso_fs = self._create_lasso_estimator(
            self.control.first_stage,
            fit_intercept=True
        )

        if self.control.first_stage.use_cv:
            self.fs_best_alphas_ = []

        X_hat = np.zeros_like(X_basis)
        for k in range(self.n_features_X_):
            lasso_fs.fit(Z_basis, X_basis[:, k])
            X_hat[:, k] = lasso_fs.predict(Z_basis)
            if self.control.first_stage.use_cv:
                self.fs_best_alphas_.append(lasso_fs.alpha_)

        # ========== Second Stage ==========
        # Regress Y on X_hat
        lasso_ss = self._create_lasso_estimator(
            self.control.second_stage,
            fit_intercept=True
        )

        lasso_ss.fit(X_hat, Y)

        # Store coefficients [intercept, slopes]
        self.coef_ = np.concatenate([[lasso_ss.intercept_], lasso_ss.coef_])
        self.n_nonzero_coef_ = np.count_nonzero(lasso_ss.coef_)

        if self.control.second_stage.use_cv:
            self.ss_best_alpha_ = lasso_ss.alpha_

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict gamma(X) = intercept + d(X)'β̂.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Input data

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Estimator is not fitted. Call fit() first.")

        X_basis = self._transform_X(X, fit=False)
        # coef_[0] is intercept, coef_[1:] are slopes
        return self.coef_[0] + X_basis @ self.coef_[1:]

    def predict_derivative(
        self,
        X: NDArray[np.float64],
        wrt: Optional[int] = None,
        eps: float = 1e-6
    ) -> NDArray[np.float64]:
        """
        Compute the derivative of gamma(X) with respect to input features.

        Uses analytical derivatives if the featurizer has a `transform_derivative`
        method, otherwise falls back to numerical differentiation.

        For gamma(X) = intercept + d(X)'β, the derivative is:
            ∂γ/∂X_j = Σ_k β_k * (∂d_k(X)/∂X_j)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Input data
        wrt : int or None, default=None
            Index of feature to differentiate with respect to.
            If None, returns full Jacobian (derivatives w.r.t. all features).
        eps : float, default=1e-6
            Step size for numerical differentiation (if needed)

        Returns
        -------
        gradients : ndarray
            If wrt is int: shape (n_samples,) - derivative ∂γ/∂X_j
            If wrt is None: shape (n_samples, n_features) - full Jacobian
        """
        if not self.is_fitted_:
            raise ValueError("Estimator is not fitted. Call fit() first.")

        X = self._check_X(X)
        n_samples, n_features = X.shape

        # Get slope coefficients (exclude intercept)
        beta = self.coef_[1:]

        # Check if featurizer has analytical derivative
        has_analytical = (self.x_featurizer is not None and
                         hasattr(self.x_featurizer, 'transform_derivative'))

        if wrt is not None:
            # Single derivative
            if has_analytical:
                d_basis = self.x_featurizer.transform_derivative(X, wrt=wrt)
                return d_basis @ beta
            else:
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, wrt] += eps
                X_minus[:, wrt] -= eps
                return (self.predict(X_plus) - self.predict(X_minus)) / (2 * eps)
        else:
            # Full Jacobian
            jacobian = np.zeros((n_samples, n_features))
            for j in range(n_features):
                if has_analytical:
                    d_basis = self.x_featurizer.transform_derivative(X, wrt=j)
                    jacobian[:, j] = d_basis @ beta
                else:
                    X_plus = X.copy()
                    X_minus = X.copy()
                    X_plus[:, j] += eps
                    X_minus[:, j] -= eps
                    jacobian[:, j] = (self.predict(X_plus) - self.predict(X_minus)) / (2 * eps)
            return jacobian

    @property
    def intercept_(self) -> float:
        """Return estimated intercept."""
        if hasattr(self, 'coef_'):
            return self.coef_[0]
        raise AttributeError("Estimator is not fitted.")

    @property
    def slopes_(self) -> NDArray[np.float64]:
        """Return estimated slope coefficients (excluding intercept)."""
        if hasattr(self, 'coef_'):
            return self.coef_[1:]
        raise AttributeError("Estimator is not fitted.")
