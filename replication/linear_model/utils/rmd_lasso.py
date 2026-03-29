# simulations/linear_model/utils/rmd_lasso.py

"""
Regularized Minimum Distance (RMD) Lasso estimator for Riesz representer.

This implements the CNS (Chernozhukov, Newey, Singh) method for estimating
Riesz representers in the exogenous case (no instruments needed).

Reference:
    Chernozhukov, V., Newey, W. K., & Singh, R. (2022).
    Automatic debiased machine learning of causal and structural effects.
    Econometrica.
"""

from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray
import scipy.stats as ss
from sklearn.base import BaseEstimator, TransformerMixin
from admliv.moments.base import BaseMoment
from .control import RMDControl


class RMDLasso(BaseEstimator):
    """
    Regularized Minimum Distance (RMD) Lasso estimator for Riesz representer.
    
    Implements the CNS method for the exogenous case. The objective is:
    
        min_rho (1/2) rho'G rho - rho'M + r_L ||D*rho||_1
    
    where:
        - G = B'B/n is the Gram matrix of basis functions
        - M = mean(m(W, b)) is the sample average moment
        - D is an iteratively updated normalization matrix
        - r_L = c * Phi^{-1}(1 - alpha/(2p)) / sqrt(n) is the penalty
    
    Parameters
    ----------
    x_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion b(X)
    control : Optional[RMDControl], default=None
        Control parameters for optimization
    verbose : bool, default=True
        If True, prints optimization progress
    
    Attributes
    ----------
    rho_ : NDArray
        Estimated Riesz representer coefficients
    n_samples_ : int
        Number of samples used in fitting
    n_features_ : int
        Number of basis functions
    is_fitted_ : bool
        Whether the estimator has been fitted
    """
    
    def __init__(
        self,
        x_featurizer: TransformerMixin,
        control: Optional[RMDControl] = None,
        verbose: bool = True
    ):
        self.x_featurizer = x_featurizer
        self.control = control if control is not None else RMDControl()
        self.verbose = verbose
        self.is_fitted_ = False
    
    def _compute_normalization_D(
        self,
        B: NDArray[np.float64],
        M: NDArray[np.float64],
        rho: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute normalization matrix D for RMD.
        
        D_j = sqrt(mean([(b_i @ rho) * b_{i,j} - M_{i,j}]^2))
        
        Parameters
        ----------
        B : array (n, p)
            Basis matrix b(X)
        M : array (n, p)
            Individual moment matrix m(W_i, b_j)
        rho : array (p,)
            Current coefficient estimate
            
        Returns
        -------
        D : array (p,)
            Normalization vector
        """
        alpha = B @ rho  # (n,) predicted values
        df = alpha[:, np.newaxis] * B - M  # (n, p)
        D = np.sqrt(np.mean(df ** 2, axis=0))
        return D
    
    def _coordinate_descent(
        self,
        M_hat: NDArray[np.float64],
        G_hat: NDArray[np.float64],
        D: NDArray[np.float64],
        lambda_: float,
        rho_start: Optional[NDArray[np.float64]] = None
    ) -> Dict:
        """
        Core RMD Lasso coordinate descent solver.
        
        Minimizes: (1/2) rho'G rho - rho'M + lambda * ||D * L * rho||_1
        
        Parameters
        ----------
        M_hat : array (p,)
            Sample average of moments
        G_hat : array (p, p)
            Sample Gram matrix B'B/n
        D : array (p,)
            Normalization vector
        lambda_ : float
            Penalty parameter r_L
        rho_start : array (p,), optional
            Starting values for coefficients
            
        Returns
        -------
        dict with keys:
            'coefs': estimated coefficients
            'num_it': number of iterations
        """
        p = G_hat.shape[1]
        
        # Penalty loadings: L = [intercept_penalty, 1, 1, ..., 1]
        L = np.concatenate([[self.control.intercept_penalty], np.ones(p - 1)])
        lambda_vec = lambda_ * L * D
        
        # Initialize
        if rho_start is None:
            rho = np.zeros(p)
        else:
            rho = rho_start.copy()
        
        # Coordinate descent
        for mm in range(1, self.control.maxiter + 1):
            rho_old = rho.copy()
            
            for j in range(p):
                z_j = G_hat[j, j]
                pi_j = M_hat[j] - G_hat[j, :] @ rho + G_hat[j, j] * rho[j]
                
                if np.isnan(pi_j):
                    rho[j] = 0
                    continue
                
                # Coordinate update
                if pi_j < -lambda_vec[j]:
                    rho[j] = (pi_j + lambda_vec[j]) / z_j
                elif np.abs(pi_j) <= lambda_vec[j]:
                    rho[j] = 0
                else:
                    rho[j] = (pi_j - lambda_vec[j]) / z_j
            
            if np.nansum(np.abs(rho - rho_old)) < self.control.optTol:
                break
        
        # Soft-thresholding
        rho = np.where(np.abs(rho) < self.control.zeroThreshold, 0, rho)
        
        return {'coefs': rho, 'num_it': mm}
    
    def _compute_lambda(self, n: int, p: int, c: Optional[float] = None) -> float:
        """
        Compute penalty parameter.
        
        lambda = c * Phi^{-1}(1 - alpha/(2p)) / sqrt(n)
        
        Parameters
        ----------
        n : int
            Sample size
        p : int
            Number of parameters
        c : float, optional
            Penalty scaling. If None, uses self.control.c
            
        Returns
        -------
        lambda_ : float
            Penalty parameter
        """
        if c is None:
            c = self.control.c
        return c * ss.norm.ppf(1 - self.control.alpha / (2 * p)) / np.sqrt(n)
    
    def _low_dim_init(
        self,
        G_hat: NDArray[np.float64],
        M_hat: NDArray[np.float64],
        p_x: int,
        p: int
    ) -> NDArray[np.float64]:
        """
        Low-dimensional initialization for RMD.
        
        Solves G0 @ rho0 = M0 on first p0 features.
        
        Parameters
        ----------
        G_hat : array (p, p)
            Full Gram matrix
        M_hat : array (p,)
            Full moment vector
        p_x : int
            Number of original features (without intercept)
        p : int
            Total number of basis functions
            
        Returns
        -------
        rho_init : array (p,)
            Initial coefficient estimate
        """
        p0 = max(1, p_x // self.control.low_dim_divisor)
        p0_total = p0 + 1  # including intercept
        
        if p0_total < p:
            G_hat0 = G_hat[:p0_total, :p0_total]
            M_hat0 = M_hat[:p0_total]
            try:
                rho_hat0 = np.linalg.solve(G_hat0, M_hat0)
            except np.linalg.LinAlgError:
                rho_hat0 = np.linalg.lstsq(G_hat0, M_hat0, rcond=None)[0]
            return np.concatenate([rho_hat0, np.zeros(p - p0_total)])
        else:
            return np.zeros(p)
    
    def _fit_with_c(
        self,
        B: NDArray[np.float64],
        M_full: NDArray[np.float64],
        M_hat: NDArray[np.float64],
        G_hat: NDArray[np.float64],
        rho_init: NDArray[np.float64],
        c: float
    ) -> NDArray[np.float64]:
        """
        Fit RMD Lasso with a specific penalty parameter c.
        
        Parameters
        ----------
        B : array (n, p)
            Basis matrix
        M_full : array (n, p)
            Full moment matrix
        M_hat : array (p,)
            Sample average moment
        G_hat : array (p, p)
            Gram matrix
        rho_init : array (p,)
            Initial coefficient estimate
        c : float
            Penalty scaling constant
            
        Returns
        -------
        rho_hat : array (p,)
            Estimated coefficients
        """
        n, p = B.shape
        lambda_ = self._compute_lambda(n, p, c)
        rho_hat = rho_init.copy()
        
        # Outer iteration for normalization updates
        for _ in range(1, self.control.max_outer_iter + 1):
            rho_hat_old = rho_hat.copy()
            
            if self.control.normalize:
                D_hat = self._compute_normalization_D(B, M_full, rho_hat)
                D_hat = np.maximum(D_hat, self.control.D_LB)
                D_hat = D_hat + self.control.D_add
            else:
                D_hat = np.ones(p)
            
            result = self._coordinate_descent(M_hat, G_hat, D_hat, lambda_, rho_hat)
            rho_hat = result['coefs']
            
            diff_rho = np.linalg.norm(rho_hat - rho_hat_old)
            if diff_rho < self.control.outer_tol:
                break
        
        return rho_hat
    
    def fit(
        self,
        W: Dict[str, NDArray[np.float64]],
        moment: BaseMoment,
        **moment_kwargs
    ):
        """
        Fit the RMD Lasso estimator.
        
        Parameters
        ----------
        W : dict
            Data dictionary with keys 'Y', 'X', etc.
        moment : BaseMoment
            Moment class implementing compute_all_basis
        **moment_kwargs
            Additional arguments passed to moment computation
            
        Returns
        -------
        self
        """
        Y = W['Y']
        X = W['X']
        n = len(Y)
        p_x = X.shape[1] if X.ndim > 1 else 1
        
        # Fit featurizer and transform
        self.x_featurizer.fit(X)
        B = self.x_featurizer.transform(X)
        p = B.shape[1]
        
        self.n_samples_ = n
        self.n_features_ = p
        
        # Compute moments
        M_full = moment.compute_all_basis(self.x_featurizer, W, **moment_kwargs)
        M_hat = M_full.mean(axis=0)
        G_hat = B.T @ B / n
        
        # Store for potential later use
        self._B = B
        self._M_full = M_full
        self._M_hat = M_hat
        self._G_hat = G_hat
        
        if self.verbose:
            print("=" * 60)
            print("RMD Lasso Estimation (CNS method)")
            print("=" * 60)
            print(f"n = {n}, p = {p}")
        
        # Low-dimensional initialization
        rho_init = self._low_dim_init(G_hat, M_hat, p_x, p)
        
        if self.verbose:
            print(f"Penalty lambda = {self._compute_lambda(n, p):.6f}")
        
        # Fit with default c
        self.rho_ = self._fit_with_c(B, M_full, M_hat, G_hat, rho_init, self.control.c)
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"Non-zero coefficients: {np.count_nonzero(self.rho_)}")
            print("=" * 60)
        
        return self
    
    def get_rho(self) -> NDArray[np.float64]:
        """Return the estimated coefficients."""
        if not self.is_fitted_:
            raise ValueError("Estimator not fitted. Call fit() first.")
        return self.rho_
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict alpha(X) = b(X)' @ rho.
        
        Parameters
        ----------
        X : array (n, p_x)
            Feature matrix
            
        Returns
        -------
        alpha : array (n,)
            Predicted Riesz representer values
        """
        if not self.is_fitted_:
            raise ValueError("Estimator not fitted. Call fit() first.")
        B = self.x_featurizer.transform(X)
        return B @ self.rho_