# admliv/moments/linear_model_moment.py

from typing import Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

from .base import BaseMoment


class LinearModelMoment(BaseMoment):
    """
    Linear model moment function.
    
    Can be used to estimate linear model coefficients.

    For linear exogenous model: Y = X'β + ε with E[X'ε] = 0
    
    Computes: m(W, gamma) = Y * gamma(X)
    Parameter: theta = E[Y * gamma(X)]

    Logic (for IV):
    1. Under linearity, gamma(X) = X'β
    2. Moment condition: E[X'ε] = E[X'(Y - X'β)] = 0
    3. Since E[X'ε] = 0, E[X'Y] = E[X'b(X)] = E[m(W, b)], where b is a dictionary of basis functions
    
    This moment function allows PGMM to estimate coefficients in linear sparse models.
    Note: Unlike WeightedAverage, the "weight" here is Y (the outcome), not a 
    function of X.
    """
    
    def compute(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute linear model moment for a single gamma function.
        
        Parameters
        ----------
        gamma : Callable
            Function that maps X to gamma(X)
        W : Dict[str, NDArray]
            Data dictionary with 'Y' and 'X' keys
        **kwargs : Any
            Unused for this moment function
        
        Returns
        -------
        moment : NDArray[np.float64], shape (n,)
            Y * gamma(X) for each observation
        """
        Y = W['Y']
        X = W['X']
        
        gamma_vals = gamma(X)
        
        # Ensure proper shapes
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.flatten()
        if gamma_vals.ndim == 1:
            gamma_vals = gamma_vals.reshape(-1, 1)
        elif gamma_vals.ndim == 2 and gamma_vals.shape[1] > 1:
            # If gamma returns multiple values, take mean
            gamma_vals = np.mean(gamma_vals, axis=1, keepdims=True)
        
        moment = Y * gamma_vals.flatten()
        return moment
    
    def compute_all_basis(
        self,
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute linear model moment for all basis functions (vectorized).
        
        For PGMM: M[i, j] = Y_i * d_j(X_i)
        
        This is the key moment for estimating Riesz representers in linear IV.
        
        Parameters
        ----------
        x_featurizer : TransformerMixin
            Fitted featurizer for basis expansion d(X)
            Set to CoordinatePolynomials with degree 1 to use X as basis
        W : Dict[str, NDArray]
            Data dictionary with 'Y' and 'X' keys
        **kwargs : Any
            Unused for this moment function
        
        Returns
        -------
        M : NDArray[np.float64], shape (n, k)
            Moment matrix where M[i,j] = Y_i * d_j(X_i)
        """
        Y = W['Y']
        X = W['X']
        
        # Transform X to basis
        Wx = x_featurizer.transform(X)  # (n, k)
        
        # Ensure Y is (n, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Vectorized: (n, 1) * (n, k) = (n, k)
        M = Y * Wx
        
        return M