# admliv/moments/squared_gamma_average.py

"""
Nonlinear moment: theta = E[gamma(X)^2].

A simple nonlinear functional useful for testing the double
cross-fitting machinery in ADMLIV.

The moment function is:
    m(W, gamma) = gamma(X)^2

The Gateaux derivative is:
    D_gamma m(W; gamma)[zeta] = 2 * gamma(X) * zeta(X)

So for basis function d_k:
    M[i, k] = 2 * gamma(X_i) * d_k(X_i)
"""

from typing import Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

from .base import BaseMoment


class SquaredGammaAverage(BaseMoment):
    """
    Nonlinear moment: theta = E[gamma(X)^2].

    This is the simplest nonlinear functional of gamma.

    Moment: m(W, gamma) = gamma(X)^2
    Gateaux: D_gamma theta[d_k] = E[2 * gamma(X) * d_k(X)]
    Per-observation: M[i, k] = 2 * gamma(X_i) * d_k(X_i)
    """

    def compute(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """Compute m(W, gamma) = gamma(X)^2."""
        X = W['X']
        gamma_vals = gamma(X).flatten()
        return gamma_vals ** 2

    def compute_all_basis(
        self,
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """Not applicable for nonlinear functionals."""
        raise NotImplementedError(
            "SquaredGammaAverage is nonlinear. Use compute_all_basis_gamma()."
        )

    def compute_all_basis_gamma(
        self,
        gamma: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x_featurizer: TransformerMixin,
        W: Dict[str, NDArray[np.float64]],
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """
        Compute M[i, k] = 2 * gamma(X_i) * d_k(X_i).

        Parameters
        ----------
        gamma : Callable
            MLIV estimate
        x_featurizer : TransformerMixin
            Fitted featurizer for basis expansion d(X)
        W : Dict[str, NDArray]
            Data dictionary
        **kwargs : Any
            Unused

        Returns
        -------
        M : NDArray, shape (n, p)
            Gateaux derivative matrix
        """
        X = W['X']
        gamma_vals = gamma(X).flatten()  # (n,)
        Wx = x_featurizer.transform(X)   # (n, p)

        # M[i, k] = 2 * gamma(X_i) * d_k(X_i)
        M = 2 * gamma_vals[:, np.newaxis] * Wx
        return M

    @property
    def is_linear(self) -> bool:
        return False
