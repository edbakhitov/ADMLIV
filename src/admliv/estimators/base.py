# admliv/estimators/base.py
"""
Base class for MLIV estimators.

All estimators in this module follow a consistent interface:
- fit(W) for training, where W is a data dictionary W with keys 'X', 'Z', 'Y'
- predict(X) for predictions
- The predict method can be passed directly to moment functions as gamma
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True

    def _is_jax_tracer(x):
        """Check if x is a JAX tracer (used during autodiff)."""
        return isinstance(x, jax.core.Tracer)
except ImportError:
    JAX_AVAILABLE = False

    def _is_jax_tracer(x):
        return False


class BaseMLIVEstimator(BaseEstimator, ABC):
    """
    Abstract base class for Machine Learning IV estimators.

    All MLIV estimators must implement:
    - fit(W): Estimate gamma(X) using instruments Z 
    - predict(X): Return predictions gamma(X)

    The predict method returns values that can be used directly with
    moment functions: m(W, gamma) where gamma = estimator.predict

    Parameters
    ----------
    x_featurizer : TransformerMixin, optional
        Transformer for X basis expansion. If None, uses identity.
    z_featurizer : TransformerMixin, optional
        Transformer for Z basis expansion (for IV methods). If None, uses identity.

    Notes
    -----
    This base class uses the Dict interface fit(W) to be compatible with
    ADMLIV's cross-fitting procedure. Estimators with different signatures
    (e.g., DeepGMM) should be adapted using appropriate adapter classes.
    """

    def __init__(
        self,
        x_featurizer: Optional[TransformerMixin] = None,
        z_featurizer: Optional[TransformerMixin] = None
    ):
        self.x_featurizer = x_featurizer
        self.z_featurizer = z_featurizer
        self.is_fitted_ = False
    
    def _check_X(self, X: NDArray) -> NDArray:
        """Ensure X is 2D array. JAX-compatible."""
        # Check if it's a JAX tracer (during autodiff) or JAX array
        if JAX_AVAILABLE and (_is_jax_tracer(X) or isinstance(X, jnp.ndarray)):
            # Use JAX operations for JAX arrays/tracers - avoid np.asarray()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X
        else:
            # Use NumPy for regular arrays
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X

    def _check_y(self, y: NDArray) -> NDArray:
        """Ensure y is 1D array. JAX-compatible."""
        # Check if it's a JAX tracer (during autodiff) or JAX array
        if JAX_AVAILABLE and (_is_jax_tracer(y) or isinstance(y, jnp.ndarray)):
            # Use JAX operations for JAX arrays/tracers - avoid np.asarray()
            if y.ndim == 2:
                y = y.flatten()
            return y
        else:
            # Use NumPy for regular arrays
            y = np.asarray(y)
            if y.ndim == 2:
                y = y.flatten()
            return y
    
    def _transform_X(self, X: NDArray, fit: bool = False) -> NDArray:
        """Transform X using featurizer. JAX-compatible if featurizer supports it."""
        X = self._check_X(X)
        if self.x_featurizer is None:
            return X
        if fit:
            return self.x_featurizer.fit_transform(X)
        return self.x_featurizer.transform(X)
    
    def _transform_Z(self, Z: NDArray, fit: bool = False) -> NDArray:
        """Transform Z using featurizer. JAX-compatible if featurizer supports it."""
        Z = self._check_X(Z)  # Same shape check
        if self.z_featurizer is None:
            return Z
        if fit:
            return self.z_featurizer.fit_transform(Z)
        return self.z_featurizer.transform(Z)
    
    @abstractmethod
    def fit(self, W: Dict[str, NDArray[np.float64]]) -> 'BaseMLIVEstimator':
        """
        Fit the estimator using data dictionary.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': outcome, shape (n,) or (n, 1)
            - 'X': endogenous regressors, shape (n, d_x)
            - 'Z': instrumental variables, shape (n, d_z)

        Returns
        -------
        self : BaseMLIVEstimator
            Fitted estimator
        """
        pass
    
    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict gamma(X).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Input data
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values gamma(X)
        """
        pass
    
    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Allow estimator to be called as a function.
        
        This enables passing the estimator directly to moment functions
        that expect gamma as a callable.
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        y_pred : ndarray
            Predicted values gamma(X)
        """
        return self.predict(X)
    
    @property
    def coef(self) -> NDArray[np.float64]:
        """Return estimated coefficients if available."""
        if hasattr(self, 'coef_'):
            return self.coef_
        raise AttributeError("Estimator does not have coefficients or is not fitted.")
