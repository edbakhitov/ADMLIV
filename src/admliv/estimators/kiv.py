# admliv/estimators/kiv.py
"""
Kernel Instrumental Variables (KIV) estimator.

Based on the KIV method of Singh et al. (2019):
"Kernel Instrumental Variable Regression"

This is a nonparametric IV estimator using kernel ridge regression
with automatic hyperparameter tuning via sample splitting.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from .base import BaseMLIVEstimator


class KIVEstimator(BaseMLIVEstimator):
    """
    Kernel Instrumental Variables (KIV) estimator.

    A nonparametric IV estimator that uses kernel ridge regression
    with automatic hyperparameter tuning via sample splitting.

    The method:
    1. Splits data into two halves
    2. Tunes first-stage regularization λ to minimize projection error
    3. Tunes second-stage regularization ξ to minimize prediction error
    4. Stores optimal hyperparameters for prediction

    This is an optimized version of the original implementation:
    1. Vectorized RBF kernel computation using scipy.spatial.distance.cdist
    2. Cached X1 for faster prediction (no re-splitting)
    3. Cholesky factorization for efficient matrix solves
    4. Precomputed alpha coefficients for O(n) prediction
    5. Brent's method for faster 1D hyperparameter optimization

    Parameters
    ----------
    bandwidth_scale : float, default=1.0
        Scale factor for bandwidth (bandwidth = scale * std(X))
    split_frac : float, default=0.5
        Fraction of data for first split
    lam_init : float, default=0.05
        Initial value for first-stage hyperparameter search
    xi_init : float, default=0.05
        Initial value for second-stage hyperparameter search

    Attributes
    ----------
    lam_star_ : float
        Optimal first-stage regularization
    xi_star_ : float
        Optimal second-stage regularization
    bandwidth_x_ : ndarray
        Bandwidths for X kernel
    bandwidth_z_ : ndarray
        Bandwidths for Z kernel
    alpha_ : ndarray
        Precomputed dual coefficients for fast prediction

    Examples
    --------
    >>> est = KIVEstimator2()
    >>> W = {'X': X, 'Z': Z, 'Y': Y}
    >>> est.fit(W)
    >>> gamma_hat = est.predict(X_test)

    References
    ----------
    Singh, R., Sahani, M., & Gretton, A. (2019).
    Kernel Instrumental Variable Regression.
    """

    def __init__(
        self,
        bandwidth_scale: float = 1.0,
        bandwidth_method: str = 'median',
        split_frac: float = 0.5,
        lam_bounds: Tuple[float, float] = (1e-6, 10.0),
        xi_bounds: Tuple[float, float] = (1e-6, 10.0),
        bandwidth_subsample: int = 1000,
        verbose: bool = False
    ):
        super().__init__(x_featurizer=None, z_featurizer=None)
        self.bandwidth_scale = bandwidth_scale
        self.bandwidth_method = bandwidth_method
        self.split_frac = split_frac
        self.lam_bounds = lam_bounds
        self.xi_bounds = xi_bounds
        self.bandwidth_subsample = bandwidth_subsample
        self.verbose = verbose

    @staticmethod
    def _median_interpoint_distance(
        X: NDArray,
        max_points: int = 1000
    ) -> NDArray:
        """
        Compute median interpoint distance for each dimension.

        When n > max_points, a random subsample is used for efficiency.
        This reduces cost from O(n^2 * d) to O(m^2 * d) where m = max_points.

        Parameters
        ----------
        X : array of shape (n, d)
            Data points
        max_points : int, default=1000
            Maximum number of points to use. If n > max_points, a random
            subsample is drawn. Standard practice for kernel bandwidth
            selection.

        Returns
        -------
        bandwidth : array of shape (d,)
            Median interpoint distance for each dimension
        """
        X = np.atleast_2d(X)
        n, d = X.shape

        # Subsample for efficiency
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            X = X[idx]
            n = max_points

        bandwidth = np.zeros(d)

        for j in range(d):
            col = X[:, j]
            # Compute pairwise absolute differences
            A = np.repeat(col.reshape(n, 1), n, axis=1)
            dist = np.abs(A - A.T)
            # Use upper triangle to avoid zeros on diagonal and double counting
            upper_tri = dist[np.triu_indices(n, k=1)]
            bandwidth[j] = np.median(upper_tri) if len(upper_tri) > 0 else 1.0

        return bandwidth

    @staticmethod
    def _rbf_kernel_matrix(
        X: NDArray,
        Y: NDArray,
        bandwidth: NDArray
    ) -> NDArray:
        """
        Compute RBF kernel matrix K(X, Y) using vectorized operations.

        K(x, y) = exp(-||x/h - y/h||^2 / 2)

        Uses scipy.spatial.distance.cdist for efficient computation.

        Parameters
        ----------
        X : array of shape (n_x, d)
            First set of points
        Y : array of shape (n_y, d)
            Second set of points
        bandwidth : array of shape (d,)
            Bandwidth for each dimension

        Returns
        -------
        K : array of shape (n_x, n_y)
            Kernel matrix
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Scale by bandwidth
        X_scaled = X / bandwidth
        Y_scaled = Y / bandwidth

        # Compute squared Euclidean distances efficiently
        sq_dist = cdist(X_scaled, Y_scaled, metric='sqeuclidean')

        return np.exp(-sq_dist / 2)

    def _split_data(
        self,
        X: NDArray,
        Z: NDArray,
        Y: NDArray
    ) -> Dict[str, NDArray]:
        """Split data into two parts."""
        n = len(Y)
        n1 = int(np.round(self.split_frac * n))

        return {
            'X1': X[:n1], 'X2': X[n1:],
            'Z1': Z[:n1], 'Z2': Z[n1:],
            'Y1': Y[:n1], 'Y2': Y[n1:]
        }

    def _compute_kernels(
        self,
        split: Dict[str, NDArray]
    ) -> Dict[str, NDArray]:
        """Precompute all kernel matrices needed for KIV."""
        # Compute bandwidths based on method
        if self.bandwidth_method == 'median':
            self.bandwidth_x_ = self._median_interpoint_distance(split['X1'], self.bandwidth_subsample) * self.bandwidth_scale
            self.bandwidth_z_ = self._median_interpoint_distance(split['Z1'], self.bandwidth_subsample) * self.bandwidth_scale
        else:  # 'std' (default)
            self.bandwidth_x_ = np.std(split['X1'], axis=0) * self.bandwidth_scale
            self.bandwidth_z_ = np.std(split['Z1'], axis=0) * self.bandwidth_scale

        # Avoid zero bandwidth
        self.bandwidth_x_ = np.maximum(self.bandwidth_x_, 1e-10)
        self.bandwidth_z_ = np.maximum(self.bandwidth_z_, 1e-10)

        # Compute kernel matrices
        K_XX = self._rbf_kernel_matrix(split['X1'], split['X1'], self.bandwidth_x_)
        K_xx = self._rbf_kernel_matrix(split['X2'], split['X2'], self.bandwidth_x_)
        K_xX = self._rbf_kernel_matrix(split['X2'], split['X1'], self.bandwidth_x_)
        K_ZZ = self._rbf_kernel_matrix(split['Z1'], split['Z1'], self.bandwidth_z_)
        K_Zz = self._rbf_kernel_matrix(split['Z1'], split['Z2'], self.bandwidth_z_)

        # Make symmetric and add small ridge for numerical stability
        eps = 1e-10
        K_XX = (K_XX + K_XX.T) / 2 + eps * np.eye(K_XX.shape[0])
        K_xx = (K_xx + K_xx.T) / 2 + eps * np.eye(K_xx.shape[0])
        K_ZZ = (K_ZZ + K_ZZ.T) / 2 + eps * np.eye(K_ZZ.shape[0])

        return {
            'K_XX': K_XX, 'K_xx': K_xx, 'K_xX': K_xX,
            'K_ZZ': K_ZZ, 'K_Zz': K_Zz,
            'Y1': split['Y1'], 'Y2': split['Y2'],
            'n1': len(split['Y1']), 'n2': len(split['Y2'])
        }

    def _first_stage_solve(
        self,
        kernels: Dict,
        lam: float
    ) -> Tuple[NDArray, Optional[Tuple]]:
        """
        Solve first stage with Cholesky factorization.

        Returns gamma and the Cholesky factor for potential reuse.
        Falls back to least squares if Cholesky fails.
        """
        n = kernels['n1']

        # Form regularized matrix and factorize
        A = kernels['K_ZZ'] + lam * n * np.eye(n)

        try:
            L = cho_factor(A)
            gamma = cho_solve(L, kernels['K_Zz'])
            return gamma, L
        except np.linalg.LinAlgError:
            # Cholesky failed - fall back to lstsq
            gamma = np.linalg.lstsq(A, kernels['K_Zz'], rcond=None)[0]
            return gamma, None

    def _compute_alpha(
        self,
        kernels: Dict,
        lam: float,
        xi: float
    ) -> NDArray:
        """
        Compute alpha coefficients for prediction.

        Falls back to least squares if Cholesky fails.
        """
        n = kernels['n1']
        m = kernels['n2']

        # First stage
        gamma, _ = self._first_stage_solve(kernels, lam)
        W = kernels['K_XX'] @ gamma

        # Second stage
        WWT = W @ W.T
        WWT = (WWT + WWT.T) / 2  # Ensure symmetry
        A2 = WWT + m * xi * kernels['K_XX']

        try:
            L2 = cho_factor(A2)
            alpha = cho_solve(L2, W @ kernels['Y2'])
        except np.linalg.LinAlgError:
            # Cholesky failed - fall back to lstsq
            alpha = np.linalg.lstsq(A2, W @ kernels['Y2'], rcond=None)[0]

        return alpha

    def _first_stage_loss(
        self,
        lam: float,
        kernels: Dict
    ) -> float:
        """First stage loss for hyperparameter tuning."""
        m = kernels['n2']

        gamma, _ = self._first_stage_solve(kernels, lam)

        # Loss = trace(K_xx - 2*K_xX @ gamma + gamma.T @ K_XX @ gamma) / m
        # Compute efficiently
        term1 = np.trace(kernels['K_xx'])
        term2 = 2 * np.sum(kernels['K_xX'] * gamma.T)  # trace(A @ B) = sum(A * B.T)
        term3 = np.sum((kernels['K_XX'] @ gamma) * gamma)

        loss = (term1 - term2 + term3) / m

        return loss

    def _second_stage_loss(
        self,
        xi: float,
        kernels: Dict,
        lam: float
    ) -> float:
        """Second stage loss for hyperparameter tuning."""
        alpha = self._compute_alpha(kernels, lam, xi)
        Y1_pred = (alpha.T @ kernels['K_XX']).flatten()
        loss = np.sum((kernels['Y1'] - Y1_pred) ** 2) / kernels['n1']
        return loss

    def fit(self, W: Dict[str, NDArray[np.float64]]) -> 'KIVEstimator':
        """
        Fit the KIV estimator.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': outcome, shape (n,) or (n, 1)
            - 'X': endogenous regressors, shape (n, d_x)
            - 'Z': instrumental variables, shape (n, d_z)

        Returns
        -------
        self : KIVEstimator2
            Fitted estimator
        """
        # Extract data from dictionary
        X = W['X']
        Z = W['Z']
        Y = W['Y']

        X = self._check_X(X)
        Z = self._check_X(Z)
        Y = self._check_y(Y)

        # Split data and compute kernels
        split = self._split_data(X, Z, Y)
        kernels = self._compute_kernels(split)
        self.kernels_ = kernels

        # Store X1 for prediction 
        self.X1_ = split['X1'].copy()

        # Tune first stage using Brent's method 
        result_lam = optimize.minimize_scalar(
            lambda lam: self._first_stage_loss(lam, kernels),
            bounds=self.lam_bounds,
            method='bounded'
        )
        self.lam_star_ = result_lam.x

        # Tune second stage using Brent's method
        result_xi = optimize.minimize_scalar(
            lambda xi: self._second_stage_loss(xi, kernels, self.lam_star_),
            bounds=self.xi_bounds,
            method='bounded'
        )
        self.xi_star_ = result_xi.x

        # Precompute alpha for fast prediction 
        self.alpha_ = self._compute_alpha(kernels, self.lam_star_, self.xi_star_)

        if self.verbose:
            print(f"KIV2: λ* = {self.lam_star_:.6f}, ξ* = {self.xi_star_:.6f}")

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict gamma(X).

        Uses precomputed alpha for O(n_test * n_train) prediction.

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

        X = self._check_X(X)

        # Compute test kernel using cached X1 
        K_test = self._rbf_kernel_matrix(self.X1_, X, self.bandwidth_x_)

        # Use precomputed alpha
        return (self.alpha_.T @ K_test).flatten()

    def predict_derivative(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the gradient of gamma(X) with respect to X.

        For each test point x, returns ∂γ̂(x)/∂x.

        The derivative of the RBF kernel K(x_i, x) w.r.t. x_j is:
            ∂K/∂x_j = K(x_i, x) * (x_i,j - x_j) / σ_j²

        So the gradient of γ̂(x) = Σ_i α_i K(X_1,i, x) is:
            ∂γ̂/∂x_j = (1/σ_j²) Σ_i (X_1,i,j - x_j) α_i K(X_1,i, x)

        In matrix form:
            ∇γ̂(x) = Γ⁻¹ (X_1 - x)ᵀ (K ⊙ α)
        where Γ = diag(σ₁², ..., σ_d²).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Test points at which to evaluate the gradient

        Returns
        -------
        gradients : ndarray of shape (n_samples, n_features_x)
            Gradient ∂γ̂/∂x for each test point
        """
        if not self.is_fitted_:
            raise ValueError("Estimator is not fitted. Call fit() first.")

        X = self._check_X(X)
        n_test = X.shape[0]
        d_x = X.shape[1]

        # Compute kernel matrix K(X1, X_test): shape (n1, n_test)
        K_test = self._rbf_kernel_matrix(self.X1_, X, self.bandwidth_x_)

        # K ⊙ α for all test points: (n1, n_test)
        # Broadcast alpha (n1,) to multiply each column of K_test
        K_alpha = K_test * self.alpha_.reshape(-1, 1)

        # Inverse bandwidth squared: 1/σ² for each dimension
        inv_bw_sq = 1.0 / (self.bandwidth_x_ ** 2)

        # Compute gradients efficiently
        # For each dimension j: ∂γ̂/∂x_j = (1/σ_j²) Σ_i (X1_i,j - x_j) K_alpha_i
        gradients = np.zeros((n_test, d_x))

        for j in range(d_x):
            # diff[i, k] = X1[i, j] - X[k, j], shape (n1, n_test)
            diff_j = self.X1_[:, j:j+1] - X[:, j:j+1].T
            # Sum over i: Σ_i diff_j[i, k] * K_alpha[i, k]
            gradients[:, j] = (diff_j * K_alpha).sum(axis=0) * inv_bw_sq[j]

        return gradients
