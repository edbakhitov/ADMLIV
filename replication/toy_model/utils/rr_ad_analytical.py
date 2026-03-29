# admliv/utils/rr_ad_analytical.py

import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

def _compute_featurizer_derivative_numerical(
        x_featurizer: TransformerMixin,
        X: NDArray[np.float64],
        deriv_index: int,
        eps: float = 1e-5
    ) -> NDArray[np.float64]:
    """
    Compute derivatives of all basis functions using finite differences.

    Uses central difference approximation: f'(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)

    Parameters
    ----------
    x_featurizer : TransformerMixin
        Fitted featurizer (e.g., PolynomialFeatures)
    X : NDArray, shape (n, d_x)
        Input data
    deriv_index : int
        Index of variable to differentiate with respect to
    eps : float, default=1e-5
        Step size for finite differences

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

def _rr_analytical_ad(
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        x_featurizer: TransformerMixin,
        z_featurizer: TransformerMixin,
        deriv_index: int = 0
) -> NDArray[np.float64]:
    """
    Compute Riesz representer for average derivative functional.

    Implements the analytical identity score estimator from Chen et al. (2023)
    for efficiently estimating average partial derivatives in NPIV models.

    The algorithm computes α̂ such that:
        E[α̂(Z) · ψ(Z, θ)] ≈ E[∂h(X,θ)/∂X_j]

    Parameters
    ----------
    X : NDArray, shape (n, d_x)
        Endogenous covariates
    Z : NDArray, shape (n, d_z)
        Instrumental variables
    x_featurizer : TransformerMixin
        Featurizer for endogenous variables (e.g., PolynomialFeatures)
    z_featurizer : TransformerMixin
        Featurizer for instruments (e.g., PolynomialFeatures)
    deriv_index : int, default=0
        Index of X variable to differentiate with respect to

    Returns
    -------
    rho_hat : NDArray, shape (q, )
        Estimated Riesz representer coefficients

    References
    ----------
    Chen, X., Christensen, T.M., and Tamer, E. (2023). "Efficient estimation
    of average derivatives in NPIV models: Simulation comparisons of neural
    network estimators"

    Notes
    -----
    Uses analytical derivatives if x_featurizer has 'transform_derivative'
    method, otherwise falls back to numerical differentiation.
    """
    # Input validation
    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError("X and Z must be 2D arrays")
    if X.shape[0] != Z.shape[0]:
        raise ValueError(f"X and Z must have same number of observations: X has {X.shape[0]}, Z has {Z.shape[0]}")
    if deriv_index < 0 or deriv_index >= X.shape[1]:
        raise ValueError(f"deriv_index must be in [0, {X.shape[1]}), got {deriv_index}")

    # Fit featurizers and transform
    x_featurizer.fit(X)
    z_featurizer.fit(Z)

    Wx = x_featurizer.transform(X)  # (n, p)
    Wz = z_featurizer.transform(Z)  # (n, q)
    n = X.shape[0]

    # Compute projection matrix with small ridge regularization for stability
    q = Wz.shape[1]
    ridge_reg = 1e-8 * np.eye(q)
    Wz_inv = np.linalg.inv(Wz.T @ Wz + ridge_reg)

    # Compute derivatives of basis functions
    if hasattr(x_featurizer, 'transform_derivative'):
        # Priority 1: Analytical derivatives
        dWx = x_featurizer.transform_derivative(X, wrt=deriv_index)  # (n, p)
    else:
        # Priority 2: Numerical differentiation (fallback)
        dWx = _compute_featurizer_derivative_numerical(x_featurizer, X, deriv_index)  # (n, p)

    # Solve for beta_hat and get w_hat
    B0 = dWx.T @ np.ones(n)[:, np.newaxis]  # (p, 1)
    B1 = Wx.T @ (Wz @ Wz_inv @ Wz.T) @ Wx / n + B0 @ B0.T / n**2  # (p, p)

    # Use pinv with ridge regularization for numerical stability
    p = B1.shape[0]
    ridge_reg_b = 1e-8 * np.eye(p)
    beta_hat = - np.linalg.pinv(B1 + ridge_reg_b) @ (B0 / n)  # (p, 1)
    w_hat = (Wx @ beta_hat).flatten()  # (n, )

    # Compute Riesz representer
    denominator = 1 + (dWx @ beta_hat).mean()
    if np.abs(denominator) < 1e-10:
        raise ValueError(f"Near-zero denominator in RR computation: {denominator:.2e}. "
                        f"This may indicate numerical instability or model misspecification.")

    nu_hat = - w_hat / denominator
    rho_hat = Wz_inv @ Wz.T @ nu_hat

    return rho_hat


class AnalyticalRieszRepresenter:
    """
    Analytical Riesz Representer estimator for average derivative functional.

    This class provides a scikit-learn style interface (fit/predict) for computing
    the analytical Riesz representer from Chen et al. (2023).

    The Riesz representer α(Z) is represented as α(Z) = Wz(Z)' ρ̂, where ρ̂ are
    coefficients computed using the analytical formula.

    Parameters
    ----------
    x_featurizer : TransformerMixin
        Featurizer for endogenous variables (e.g., PolynomialFeatures)
    z_featurizer : TransformerMixin
        Featurizer for instruments (e.g., PolynomialFeatures)
    deriv_index : int, default=0
        Index of X variable to differentiate with respect to

    Attributes
    ----------
    rho_hat_ : NDArray, shape (q,)
        Estimated RR coefficients (available after calling fit)
    z_featurizer_fitted_ : TransformerMixin
        Fitted z_featurizer (available after calling fit)

    References
    ----------
    Chen, X., Christensen, T.M., and Tamer, E. (2023). "Efficient estimation
    of average derivatives in NPIV models: Simulation comparisons of neural
    network estimators"

    Examples
    --------
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> x_feat = PolynomialFeatures(degree=3)
    >>> z_feat = PolynomialFeatures(degree=3)
    >>> rr = AnalyticalRieszRepresenter(x_feat, z_feat, deriv_index=0)
    >>> rr.fit(W_train)
    >>> alpha_test = rr.predict(Z_test)
    """

    def __init__(
        self,
        x_featurizer: TransformerMixin,
        z_featurizer: TransformerMixin,
        deriv_index: int = 0
    ):
        self.x_featurizer = x_featurizer
        self.z_featurizer = z_featurizer
        self.deriv_index = deriv_index

        # Fitted attributes (set after calling fit)
        self.rho_hat_ = None
        self.z_featurizer_fitted_ = None

    def fit(self, W: dict) -> 'AnalyticalRieszRepresenter':
        """
        Compute analytical RR coefficients on training data.

        Parameters
        ----------
        W : dict
            Data dictionary with keys:
            - 'X': shape (n, d_x) - endogenous variables
            - 'Z': shape (n, d_z) - instrumental variables

        Returns
        -------
        self : AnalyticalRieszRepresenter
            Fitted estimator
        """
        X = np.asarray(W['X'])
        Z = np.asarray(W['Z'])

        # Compute RR coefficients
        self.rho_hat_ = _rr_analytical_ad(
            X=X,
            Z=Z,
            x_featurizer=self.x_featurizer,
            z_featurizer=self.z_featurizer,
            deriv_index=self.deriv_index
        )

        # Fit z_featurizer for later prediction
        self.z_featurizer_fitted_ = self.z_featurizer
        self.z_featurizer_fitted_.fit(Z)

        return self

    def predict(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict Riesz representer values α(Z) = Wz(Z)' ρ̂.

        Parameters
        ----------
        Z : NDArray, shape (n, d_z)
            Instrumental variables

        Returns
        -------
        alpha : NDArray, shape (n,)
            Riesz representer values at each observation
        """
        if self.rho_hat_ is None or self.z_featurizer_fitted_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Transform Z and compute α(Z) = Wz(Z)' ρ̂
        Wz = self.z_featurizer_fitted_.transform(Z)
        alpha = Wz @ self.rho_hat_
        return alpha.ravel()

    def get_coefficients(self) -> NDArray[np.float64]:
        """
        Get the RR coefficients ρ̂.

        Returns
        -------
        rho_hat : NDArray, shape (q,)
            RR coefficients
        """
        if self.rho_hat_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.rho_hat_


