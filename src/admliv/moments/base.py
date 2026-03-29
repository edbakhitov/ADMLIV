# admliv/moments/base.py

from abc import ABC, abstractmethod
from typing import Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin


class BaseMoment(ABC):
    """
    Base class for moment functions in ADML.
    
    The parameter of interest theta is defined as the solution to:
        theta = E[m(W, gamma)]
    
    where gamma is estimated by the MLIV algorithm and E denotes expectation.
    
    This class supports two modes:
    1. Single function mode: for ADML estimation of theta
    2. Basis expansion mode: for PGMM estimation of Riesz representer
    """
    
    @abstractmethod
    def compute(
        self, 
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]], 
        W: Dict[str, NDArray[np.float64]], 
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Evaluate moment function m(W, gamma) for a single gamma function.
        
        Used in ADML for estimating theta = E[m(W, gamma)].
        
        Parameters
        ----------
        gamma : Callable
            Estimated function from MLIV. Takes X and returns gamma(X).
            Input shape: (n, d_x), Output shape: (n, 1) 
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': shape (n, 1) - outcome variable
            - 'X': shape (n, d_x) - endogenous variables
            - 'Z': shape (n, d_z) - instrumental variables
        **kwargs : Any
            Additional moment-specific arguments (e.g., weight_func, policy_func)
        
        Returns
        -------
        moment : NDArray[np.float64], shape (n,)
            Moment function values for each observation.
            theta = E[moment] is the parameter of interest.
        """
        pass
    
    def compute_all_basis(
        self,
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute moments for all basis functions from featurizer.
        
        Used in PGMM for estimating Riesz representer coefficients.
        Returns moment matrix M where M[i, j] = m(W_i, d_j(X_i)) and d_j is the
        j-th basis function from the featurizer.
        
        Default implementation transforms once then loops over basis functions.
        Subclasses should override for vectorized computation when possible.
        
        Parameters
        ----------
        x_featurizer : TransformerMixin
            Fitted sklearn-style transformer for basis expansion d(X)
        W : Dict[str, NDArray]
            Data dictionary with keys 'Y', 'X', 'Z'
        **kwargs : Any
            Additional moment-specific arguments
        
        Returns
        -------
        M : NDArray[np.float64], shape (n, k)
            Moment matrix where k = number of basis functions
        """
        pass
    
    def compute_all_basis_gamma(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute Gateaux derivative matrix M that depends on gamma.

        For nonlinear functionals, the Gateaux derivative D_gamma theta[d_k]
        depends on the MLIV estimate gamma. This method computes:

            M[i, k] = D_gamma m(W_i; gamma)[d_k]

        the directional derivative of the moment at observation i in the
        direction of basis function d_k, evaluated at the given gamma.

        This method is ONLY needed for nonlinear functionals. ADMLIV detects
        nonlinear functionals by checking whether this method is overridden
        (i.e., does not raise NotImplementedError).

        When this method is available, ADMLIV uses double cross-fitting:
        for each inner fold, a separate gamma is used to compute M, ensuring
        that gamma is independent of the data used for each M[i].

        Parameters
        ----------
        gamma : Callable
            MLIV estimate used to evaluate the Gateaux derivative.
            Must support gamma(X) -> predictions.
        x_featurizer : TransformerMixin
            Fitted sklearn-style transformer for basis expansion d(X)
        W : Dict[str, NDArray]
            Data dictionary with keys 'Y', 'X', 'Z'
        **kwargs : Any
            Additional moment-specific arguments

        Returns
        -------
        M : NDArray[np.float64], shape (n, k)
            Gateaux derivative matrix where k = number of basis functions

        Raises
        ------
        NotImplementedError
            If the moment is linear and does not require gamma-dependent M.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_all_basis_gamma. "
            "This is required for nonlinear functionals. For linear functionals, "
            "implement compute_all_basis instead."
        )

    @property
    def is_linear(self) -> bool:
        """Whether the functional is linear in gamma.

        Nonlinear moments should override this to return False.
        ADMLIV uses duck-typing (checking for compute_all_basis_gamma) as
        the primary detection mechanism, but this property is available for
        introspection and documentation.
        """
        return True

    @property
    def dim(self) -> int:
        """Dimension of the parameter theta (always 1 for scalar)."""
        return 1