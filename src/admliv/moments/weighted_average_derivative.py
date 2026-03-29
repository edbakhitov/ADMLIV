# admliv/moments/weighted_average_derivative.py

from typing import Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .base import BaseMoment


class WeightedAverageDerivative(BaseMoment):
    """
    Weighted average derivative moment function.
    
    Computes: m(W, gamma) = w(X) * ∂gamma(X)/∂X_i
    Parameter: theta = E[w(X) * ∂gamma(X)/∂X_i]
    
    Required kwargs:
    - 'weight_func': Callable that takes X and returns weights
    - 'deriv_index': int, index i indicating which X_i to differentiate w.r.t.
    """
    
    def __init__(self, use_jax: bool = True):
        """
        Parameters
        ----------
        use_jax : bool, default=True
            Whether to use JAX for automatic differentiation.
            If False, uses numerical differentiation.
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        if use_jax and not JAX_AVAILABLE:
            import warnings
            warnings.warn("JAX not available. Falling back to numerical differentiation.")
    
    def compute(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute weighted average derivative moment for a single gamma function.
        
        Parameters
        ----------
        gamma : Callable
            Function that maps X to gamma(X)
        W : Dict[str, NDArray]
            Data dictionary with 'X' key
        **kwargs : Any
            Must contain:
            - 'weight_func': Callable for weights w(X)
            - 'deriv_index': int, index of X to differentiate w.r.t.
        
        Returns
        -------
        moment : NDArray[np.float64], shape (n,)
            w(X) * ∂gamma(X)/∂X_i for each observation
        """
        if 'weight_func' not in kwargs:
            raise ValueError("WeightedAverageDerivative requires 'weight_func' in kwargs")
        if 'deriv_index' not in kwargs:
            raise ValueError("WeightedAverageDerivative requires 'deriv_index' in kwargs")
        
        weight_func = kwargs['weight_func']
        deriv_index = kwargs['deriv_index']
        X = W['X']
        n, d_x = X.shape
        
        if not (0 <= deriv_index < d_x):
            raise ValueError(f"deriv_index must be in [0, {d_x-1}], got {deriv_index}")
        
        weights = weight_func(X)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)

        # Priority 1: Try JAX automatic differentiation (if enabled)
        # This is most accurate for differentiating through gamma functions
        if self.use_jax:
            try:
                gamma_deriv = self._compute_derivative_jax(gamma, X, deriv_index)
            except Exception:
                # Fall back to numerical differentiation if JAX fails
                # (e.g., gamma is not JAX-differentiable, uses NumPy-only operations)
                gamma_deriv = self._compute_derivative_numerical(gamma, X, deriv_index)
        else:
            # Priority 2: Numerical differentiation (fallback)
            gamma_deriv = self._compute_derivative_numerical(gamma, X, deriv_index)

        moment = (weights * gamma_deriv).flatten()
        return moment
    
    def compute_all_basis(
        self,
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute weighted derivative moment for all basis functions.
        
        For PGMM: M[i, j] = w(X_i) * ∂d_j(X_i)/∂X_deriv_index
        
        Tries methods in priority order:
        1. Analytical derivative (if featurizer has transform_derivative)
        2. JAX automatic differentiation (if JAX available)
        3. Numerical differentiation (fallback)
        
        Parameters
        ----------
        x_featurizer : TransformerMixin
            Fitted featurizer for basis expansion d(X)
        W : Dict[str, NDArray]
            Data dictionary with 'X' key
        **kwargs : Any
            Must contain:
            - 'weight_func': Callable for weights
            - 'deriv_index': int, index to differentiate w.r.t.
        
        Returns
        -------
        M : NDArray[np.float64], shape (n, k)
            Derivative moment matrix
        """
        if 'weight_func' not in kwargs:
            raise ValueError("WeightedAverageDerivative requires 'weight_func' in kwargs")
        if 'deriv_index' not in kwargs:
            raise ValueError("WeightedAverageDerivative requires 'deriv_index' in kwargs")
        
        X = W['X']
        weight_func = kwargs['weight_func']
        deriv_index = kwargs['deriv_index']
        
        weights = weight_func(X)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        
        # Priority 1: Analytical derivatives
        if hasattr(x_featurizer, 'transform_derivative'):
            dWx = x_featurizer.transform_derivative(X, wrt=deriv_index)
            M = weights * dWx
            return M
        
        # Priority 2: JAX automatic differentiation
        if self.use_jax and JAX_AVAILABLE:
            dWx = self._compute_featurizer_derivative_jax(x_featurizer, X, deriv_index)
            M = weights * dWx
            return M
        
        # Priority 3: Numerical differentiation (fallback)
        dWx = self._compute_featurizer_derivative_numerical(x_featurizer, X, deriv_index)
        M = weights * dWx
        return M
    
    def _compute_featurizer_derivative_jax(
        self,
        x_featurizer: TransformerMixin,
        X: NDArray[np.float64],
        deriv_index: int
    ) -> NDArray[np.float64]:
        """
        Compute derivatives of all basis functions using JAX.
        
        Note: This uses finite differences computed via JAX for compatibility
        with sklearn transformers which don't natively support JAX.
        
        Returns
        -------
        dWx : NDArray, shape (n, k)
            Derivative of each basis function w.r.t. X[:, deriv_index]
        """
        # TODO: For sklearn transformers, we can't use JAX's automatic differentiation
        # because they use numpy internally. We have to construct a featurizer copy 
        # using jax.numpy. Instead, use numerical differentiation
        # which is still reasonably accurate.
        
        # This is essentially the same as _compute_featurizer_derivative_numerical
        # We keep this method separate in case we want to optimize for JAX-native
        # featurizers in the future
        
        eps = 1e-5
        
        X_plus = X.copy()
        X_plus[:, deriv_index] += eps
        
        X_minus = X.copy()
        X_minus[:, deriv_index] -= eps
        
        Wx_plus = x_featurizer.transform(X_plus)
        Wx_minus = x_featurizer.transform(X_minus)
        
        dWx = (Wx_plus - Wx_minus) / (2 * eps)
        return dWx
    
    def _compute_featurizer_derivative_numerical(
        self,
        x_featurizer: TransformerMixin,
        X: NDArray[np.float64],
        deriv_index: int,
        eps: float = 1e-5
    ) -> NDArray[np.float64]:
        """
        Compute derivatives of all basis functions using finite differences.
        
        Returns
        -------
        dWx : NDArray, shape (n, k)
            Derivative of each basis function w.r.t. X[:, deriv_index]
        """
        X_plus = X.copy()
        X_plus[:, deriv_index] += eps
        
        X_minus = X.copy()
        X_minus[:, deriv_index] -= eps
        
        Wx_plus = x_featurizer.transform(X_plus)
        Wx_minus = x_featurizer.transform(X_minus)
        
        dWx = (Wx_plus - Wx_minus) / (2 * eps)
        return dWx
    
    def _compute_derivative_jax(
        self,
        gamma: Callable,
        X: NDArray[np.float64],
        deriv_index: int
    ) -> NDArray[np.float64]:
        """Compute derivative using JAX automatic differentiation."""
        def gamma_single(x):
            return gamma(x.reshape(1, -1))[0]
        
        grad_func = jax.grad(lambda x: jnp.sum(gamma_single(x)))
        
        derivatives = []
        for i in range(X.shape[0]):
            grad_val = grad_func(jnp.array(X[i]))
            derivatives.append(float(grad_val[deriv_index]))
        
        return np.array(derivatives).reshape(-1, 1)
    
    def _compute_derivative_numerical(
        self,
        gamma: Callable,
        X: NDArray[np.float64],
        deriv_index: int,
        eps: float = 1e-5
    ) -> NDArray[np.float64]:
        """Compute derivative using numerical differentiation."""
        X_plus = X.copy()
        X_plus[:, deriv_index] += eps

        X_minus = X.copy()
        X_minus[:, deriv_index] -= eps

        gamma_plus = gamma(X_plus)
        gamma_minus = gamma(X_minus)

        derivative = (gamma_plus - gamma_minus) / (2 * eps)
        return derivative.reshape(-1, 1)