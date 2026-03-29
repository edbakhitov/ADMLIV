# admliv/moments/weighted_average.py

from typing import Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

from .base import BaseMoment


class WeightedAverage(BaseMoment):
    """
    Weighted average moment function.
    
    Computes: m(W, gamma) = w(X) * gamma(X)
    Parameter: theta = E[w(X) * gamma(X)]
    
    The weight function w(X) must be provided via kwargs['weight_func'].
    """
    
    def compute(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute weighted average moment for a single gamma function.
        
        Parameters
        ----------
        gamma : Callable
            Function that maps X to gamma(X)
        W : Dict[str, NDArray]
            Data dictionary with 'X' key
        **kwargs : Any
            Must contain 'weight_func': Callable that takes X and returns weights
        
        Returns
        -------
        moment : NDArray[np.float64], shape (n,)
            w(X) * gamma(X) for each observation
        """
        if 'weight_func' not in kwargs:
            raise ValueError("WeightedAverage requires 'weight_func' in kwargs")
        
        weight_func = kwargs['weight_func']
        X = W['X']
        
        weights = weight_func(X)
        gamma_vals = gamma(X)
        
        # Ensure proper shapes
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        if gamma_vals.ndim == 1:
            gamma_vals = gamma_vals.reshape(-1, 1)
        
        # Element-wise multiplication
        gamma_weighted = weights * gamma_vals
        
        return gamma_weighted.flatten()
    
    def compute_all_basis(
        self,
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute weighted average moment for all basis functions (vectorized).
        
        This is an optimized implementation for PGMM that computes moments
        for all basis functions at once: M[i, j] = w(X_i) * d_j(X_i)
        
        Parameters
        ----------
        x_featurizer : TransformerMixin
            Fitted featurizer for basis expansion d(X)
        W : Dict[str, NDArray]
            Data dictionary with 'X' key
        **kwargs : Any
            Must contain 'weight_func': Callable for weights w(X)
        
        Returns
        -------
        M : NDArray[np.float64], shape (n, k)
            Moment matrix where k = number of basis functions
        """
        if 'weight_func' not in kwargs:
            raise ValueError("WeightedAverage requires 'weight_func' in kwargs")
        
        weight_func = kwargs['weight_func']
        X = W['X']
        
        # Transform once (efficient!)
        Wx = x_featurizer.transform(X)
        weights = weight_func(X)
        
        # Ensure proper shape for broadcasting
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        
        # Vectorized: (n, 1) * (n, k) = (n, k)
        M = weights * Wx
        
        return M