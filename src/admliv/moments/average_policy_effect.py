# admliv/moments/average_policy_effect.py

from typing import Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

from .base import BaseMoment


class AveragePolicyEffect(BaseMoment):
    """
    Average policy effect moment function.
    
    Computes: m(W, gamma) = gamma(h(X)) - gamma(X)
    Parameter: theta = E[gamma(h(X)) - gamma(X)]
    
    where h is a policy function representing covariate shift.
    
    Required kwargs:
    - 'policy_func': Callable that takes X and returns h(X)
    """
    
    def compute(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute average policy effect moment for a single gamma function.
        
        Parameters
        ----------
        gamma : Callable
            Function that maps X to gamma(X)
        W : Dict[str, NDArray]
            Data dictionary with 'X' key
        **kwargs : Any
            Must contain 'policy_func': Callable that takes X and returns h(X)
        
        Returns
        -------
        moment : NDArray[np.float64], shape (n,)
            gamma(h(X)) - gamma(X) for each observation
        """
        if 'policy_func' not in kwargs:
            raise ValueError("AveragePolicyEffect requires 'policy_func' in kwargs")
        
        policy_func = kwargs['policy_func']
        X = W['X']
        
        X_policy = policy_func(X)
        gamma_policy = gamma(X_policy)
        gamma_original = gamma(X)
        
        moment = (gamma_policy - gamma_original).flatten()
        return moment
    
    def compute_all_basis(
        self,
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute policy effect moment for all basis functions (vectorized).
        
        For PGMM: M[i, j] = d_j(h(X_i)) - d_j(X_i)
        
        Parameters
        ----------
        x_featurizer : TransformerMixin
            Fitted featurizer for basis expansion d(X)
        W : Dict[str, NDArray]
            Data dictionary with 'X' key
        **kwargs : Any
            Must contain 'policy_func': Callable for policy h(X)
        
        Returns
        -------
        M : NDArray[np.float64], shape (n, k)
            Policy effect moment matrix
        """
        if 'policy_func' not in kwargs:
            raise ValueError("AveragePolicyEffect requires 'policy_func' in kwargs")
        
        policy_func = kwargs['policy_func']
        X = W['X']
        
        # Apply policy function
        X_policy = policy_func(X)
        
        # Transform both original and policy-shifted X
        Wx_policy = x_featurizer.transform(X_policy)
        Wx_original = x_featurizer.transform(X)
        
        # Vectorized difference: (n, k)
        M = Wx_policy - Wx_original
        
        return M