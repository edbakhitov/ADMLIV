# admliv/core/pgmm_linear_iv.py

"""
Penalized GMM for High-Dimensional Linear IV Regression.

Implements the PGMM estimator for the standard IV moment condition
E[Z'(Y - X'ρ)] = 0 as described in Section C.1.2 of Bakhitov (2026).

This differs from the main PGMM class which estimates Riesz representer
coefficients using orthogonal moments. Here we directly estimate the
structural coefficients ρ.

Reference:
    Bakhitov, E. (2026). "Penalized GMM Framework for Inference on Functionals
    of Nonparametric Instrumental Variable Estimators", Section C.1.2.

    Caner, M. & Kock, A. (2018). "High Dimensional Linear GMM"
"""

from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from .control import PGMMControl
from .pgmm import _coordinate_descent_optimized
from admliv.utils.featurizers import SimpleFeaturizer


class PGMMLinearIV(BaseEstimator):
    """
    Penalized GMM for High-Dimensional Linear IV Regression.
    
    Estimates structural coefficients ρ in the linear IV model:
        Y = X'ρ + ε
        E[Z'ε] = 0
        
    using the standard IV GMM moment condition:
        E[Z'(Y - X'ρ)] = 0
        
    The GMM problem is:
        min_ρ (1/2)(M - Gρ)'Ω(M - Gρ) + λ Σ_j w_j |ρ_j|

    where:
        G = (1/n) b(Z)' b(X)  ∈ R^{q × p}
        M = (1/n) b(Z)' Y     ∈ R^{q}
        
    Key Differences from PGMM (Riesz Representer):
    -------------------------------------------------
    1. Estimates structural coeffs not Riesz representer coeffs
    2. G = Z'X (instruments × regressors) not X'Z
    3. M = Z'Y (instruments × outcome) not E[m(W, d_j)]
    4. Weighting matrix dimension is q × q (number of instruments)
       not p × p (number of regressors)
    
    Parameters
    ----------
    x_featurizer : TransformerMixin, default=SimpleFeaturizer
        Sklearn-style transformer for basis expansion b(X) of regressors
    z_featurizer : TransformerMixin, default=SimpleFeaturizer
        Sklearn-style transformer for basis expansion b(Z) of instruments
    lambda_ : Optional[float], default=None
        Penalty parameter. If None, computed as c * sqrt(log(p) / n)
    adaptive : bool, default=True
        If True, uses adaptive weights based on preliminary estimation
    Omega : Optional[NDArray], default=None
        Weight matrix for GMM criterion (q × q). If None, uses identity
        for preliminary and optimal diagonal for adaptive step.
    control : Optional[PGMMControl], default=None
        Control parameters for optimization
    verbose : bool, default=True
        If True, prints optimization progress
        
    Attributes
    ----------
    rho_ : NDArray, shape (p,)
        Estimated structural coefficients
    n_samples_ : int
        Number of training samples
    n_features_x_ : int
        Number of features in b(X)
    n_features_z_ : int
        Number of features in b(Z)
    is_fitted_ : bool
        Whether the estimator has been fitted
    """

    def __init__(
        self,
        x_featurizer: TransformerMixin = SimpleFeaturizer,
        z_featurizer: TransformerMixin = SimpleFeaturizer,
        lambda_: Optional[float] = None,
        adaptive: bool = True,
        Omega: Optional[NDArray[np.float64]] = None,
        control: Optional[PGMMControl] = None,
        verbose: bool = True
    ):
        self.x_featurizer = x_featurizer
        self.z_featurizer = z_featurizer
        self.lambda_ = lambda_
        self.adaptive = adaptive
        self.Omega = Omega
        self.control = control if control is not None else PGMMControl()
        self.verbose = verbose
        self.is_fitted_ = False
    
    def fit(
        self,
        W: Dict[str, NDArray[np.float64]]
    ) -> 'PGMMLinearIV':
        """
        Fit the PGMM Linear IV estimator.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': outcome, shape (n,) or (n, 1)
            - 'X': endogenous regressors, shape (n, d_x)
            - 'Z': instruments, shape (n, d_z)

        Returns
        -------
        self : PGMMLinearIV
            Fitted estimator
        """
        Y = W['Y']
        X = W['X']
        Z = W['Z']
        n = X.shape[0]

        # Fit featurizers
        self.x_featurizer.fit(X)
        self.z_featurizer.fit(Z)

        # Transform data
        Wx = self.x_featurizer.transform(X)
        Wz = self.z_featurizer.transform(Z)

        self.n_samples_ = n
        self.n_features_x_ = Wx.shape[1]  # p (number of regressors)
        self.n_features_z_ = Wz.shape[1]  # q (number of instruments)

        # Check identification: need at least as many instruments as regressors
        if self.n_features_z_ < self.n_features_x_:
            raise ValueError(
                f"Under-identified: dim(b(Z))={self.n_features_z_} < "
                f"dim(b(X))={self.n_features_x_}. Need at least as many "
                "instruments as regressors."
            )

        # Compute penalty parameter
        if self.lambda_ is None:
            # Use q (number of instruments) for penalty calculation
            self.lambda_ = self.control.c * np.sqrt(
                np.log(self.n_features_z_) / n
            )

        # Always run two-stage procedure
        self.rho_ = self._fit_two_stage(Y, Wx, Wz)

        self.is_fitted_ = True
        return self
    
    def _fit_two_stage(
        self,
        Y: NDArray[np.float64],
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Two-step adaptive PGMM estimation.

        Step 1: Preliminary PGMM with Omega = I
        Step 2: Adaptive PGMM with optimal Omega and adaptive weights

        Parameters
        ----------
        Y : NDArray, shape (n,)
            Outcome
        Wx : NDArray, shape (n, p)
            Transformed regressors
        Wz : NDArray, shape (n, q)
            Transformed instruments

        Returns
        -------
        rho : NDArray, shape (p,)
            Estimated coefficients
        """
        if self.verbose:
            print("=" * 60)
            print("PGMM Two-Stage Estimation")
            print("=" * 60)
            print(f"Adaptive weights: {self.adaptive}")
            print(f"Lambda: {self.lambda_:.6f}")
            print("-" * 60)
            print("Stage 1: Preliminary PGMM with Omega = I")

        # ===== Stage 1: Preliminary estimate with identity weight matrix =====
        Omega_I = np.eye(self.n_features_z_)
        rho_tilde = self._fit_pgmm(
            Y, Wx, Wz,
            Omega=Omega_I,
            weights=None,
            rho_start=None
        )

        # Store preliminary estimate
        self.rho_preliminary_ = rho_tilde.copy()

        if self.verbose:
            print(f"  Preliminary estimate: {np.count_nonzero(rho_tilde)} non-zero coefficients")

        # Step 2: Compute optimal weight matrix
        # psi_i = (Y_i - X_i'ρ) * Z_i
        residual = Y - Wx @ rho_tilde
        psi_tilde = residual[:, np.newaxis] * Wz  # (n, q)

        if self.Omega is None:
            self.Omega_opt_ = self._compute_optimal_weight_matrix(psi_tilde)
        else:
            self.Omega_opt_ = self.Omega

        if self.verbose:
            print("-" * 60)
            print("Stage 2: PGMM with optimal Omega" +
                  (" and adaptive weights" if self.adaptive else ""))
        
        # ===== Stage 2: Re-estimate with optimal Omega =====
        if self.adaptive:
            # Compute adaptive weights from preliminary estimate
            weights = self._compute_adaptive_weights(rho_tilde)
        else:
            weights = None

        rho_hat = self._fit_pgmm(
            Y, Wx, Wz,
            Omega=self.Omega_opt_,
            weights=weights,
            rho_start=rho_tilde  # Warm start from stage 1
        )

        if self.verbose:
            print(f"  Final estimate: {np.count_nonzero(rho_hat)} non-zero coefficients")
            print("=" * 60)

        return rho_hat

    def _fit_pgmm(
        self,
        Y: NDArray[np.float64],
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        Omega: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]],
        rho_start: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Coordinate descent for PGMM-Lasso.

        Minimizes:
            (1/2)(M - Gρ)'Ω(M - Gρ) + λ Σ_j w_j |ρ_j|

        where:
            G = (1/n) Z'X
            M = (1/n) Z'Y

        The coordinate descent update for ρ_j is:
            ρ_j = S(A_j, λ_j) / B_j

        where:
            B_j = G[:,j]' Ω G[:,j]
            A_j = G[:,j]' Ω (M - Gρ + G[:,j]ρ_j)
            S(a, λ) = sign(a) * max(|a| - λ, 0)  (soft-thresholding)

        Parameters
        ----------
        Y : NDArray, shape (n,)
            Outcome
        Wx : NDArray, shape (n, p)
            Transformed regressors
        Wz : NDArray, shape (n, q)
            Transformed instruments
        Omega : NDArray, shape (q, q)
            GMM weighting matrix
        weights : Optional[NDArray], shape (p,)
            Adaptive weights for Lasso penalty
        rho_start : Optional[NDArray], shape (p,)
            Starting values for coefficients

        Returns
        -------
        rho : NDArray, shape (p,)
            Estimated coefficients
        """
        # Ensure Y is 1D
        Y = np.asarray(Y)
        if Y.ndim == 2:
            Y = Y.flatten()

        n = Wx.shape[0]
        p = Wx.shape[1]  # number of regressors
        q = Wz.shape[1]  # number of moments

        # Setup penalty vector
        # Intercept (first element) gets smaller penalty
        L = np.concatenate([
            np.array([self.control.intercept_penalty]),
            np.ones(p - 1)
        ])
        
        if weights is not None:
            lambda_vec = self.lambda_ * L * weights
        else:
            lambda_vec = self.lambda_ * L
        
        # Initialize rho
        if rho_start is None:
            rho = np.zeros(p)
        else:
            rho = rho_start.copy()

        # Precompute auxiliary matrices
        G = Wz.T @ Wx / n  # (q, p)
        M = Wz.T @ Y / n   # (q,)

        # ===== Precompute diagonal elements =====
        # B_j = G[:,j]' Ω G[:,j] doesn't change across iterations
        GOmega = G.T @ Omega / q  # (p, q)
        B_diag = np.sum(GOmega * G.T, axis=1)  # (p,)

        # ===== Initialize active set =====
        # Start with all active
        active_set = np.ones(p, dtype=np.bool_)

        # ===== Use optimized coordinate descent =====
        # Pass Omega / q to match the B_diag calculation
        rho, num_iter, n_active, diff_rho = _coordinate_descent_optimized(
            rho=rho,
            G=G,
            Omega=Omega / q,
            M=M,
            lambda_vec=lambda_vec,
            B_diag=B_diag,
            active_set=active_set,
            max_iter=self.control.maxiter,
            tol=self.control.optTol,
            check_frequency=self.control.check_frequency,
            buffer_factor=self.control.buffer_factor
        )

        # Apply zero threshold
        rho = np.where(np.abs(rho) < self.control.zeroThreshold, 0, rho)

        if self.verbose:
            print(f"  Coordinate descent converged in {num_iter} iterations")
            print(f"  Final active set size: {n_active}")
            print(f"  Final convergence criterion: {diff_rho:.2e}")

        return rho
    
    @staticmethod
    def _compute_optimal_weight_matrix(psi_tilde: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute optimal diagonal weight matrix from preliminary residuals.

        Following Caner and Kock (2018), uses diagonal weight matrix with
        diagonal elements equal to inverse of moment variances.

        Parameters
        ----------
        psi_tilde : NDArray, shape (n, q)
            Moment from preliminary estimation

        Returns
        -------
        Omega_opt : NDArray, shape (q, q)
            Optimal diagonal weight matrix (before scaling by q)
        """
        # Compute diagonal elements: 1 / var(psi_j)
        psi_var = np.var(psi_tilde, axis=0, ddof=1)

        # Avoid division by zero: add a small Ridge penalty
        psi_var_safe = np.where(psi_var > 1e-10, psi_var, 1e-10)
        Omega_diag = 1.0 / psi_var_safe

        return np.diag(Omega_diag)

    def _compute_adaptive_weights(
        self,
        rho_prelim: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute adaptive weights: w_j = 1 / |rho_j| with numerical safeguards.

        Parameters
        ----------
        rho_prelim : NDArray, shape (p,)
            Preliminary coefficient estimates

        Returns
        -------
        weights : NDArray, shape (p,)
            Adaptive weights for penalty
        """
        p = len(rho_prelim)
        weights = np.zeros(p)

        for j in range(p):
            if np.abs(rho_prelim[j]) <= self.control.adaptive_threshold:
                weights[j] = self.control.adaptive_max_weight
            else:
                weights[j] = 1.0 / np.abs(rho_prelim[j])

        return weights

    def get_rho(self) -> NDArray[np.float64]:
        """
        Get the estimated structural coefficients.

        Returns
        -------
        rho : NDArray, shape (dim(b),)
            Estimated coefficients
        """
        if not self.is_fitted_:
            raise ValueError("PGMM must be fitted before accessing rho")
        return self.rho_
    
    def get_omega(self) -> NDArray[np.float64]:
        """
        Get the optimal weight matrix from stage 1.
        
        Returns
        -------
        Omega : NDArray, shape (k, k)
            Optimal diagonal weight matrix
        """
        if not self.is_fitted_:
            raise ValueError("PGMM must be fitted before accessing Omega")
        return self.Omega_opt_
    
    def compute_criterion(
        self,
        W: Dict[str, NDArray[np.float64]],
        Omega: Optional[NDArray[np.float64]] = None
    ) -> float:
        """
        Compute GMM criterion: psi' Omega psi / (2q).

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary
        Omega : Optional[NDArray], default=None
            Weight matrix. If None, uses identity

        Returns
        -------
        criterion : float
            GMM criterion value
        """
        if not self.is_fitted_:
            raise ValueError("PGMM must be fitted before computing criterion")

        Y = W['Y']
        X = W['X']
        Z = W['Z']
        n = X.shape[0]

        Wx = self.x_featurizer.transform(X)
        Wz = self.z_featurizer.transform(Z)
        q = self.n_features_z_

        G = Wz.T @ Wx / n
        M = Wz.T @ Y / n

        if Omega is None:
            Omega = np.eye(q)

        residual = M - G @ self.rho_
        criterion = 0.5 * residual.T @ Omega @ residual / q

        return float(criterion)
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute prediction Y_hat = b(X)' rho.
        
        Parameters
        ----------
        Z : NDArray, shape (n, d_z)
            Instrumental variables
        
        Returns
        -------
        alpha : NDArray, shape (n,)
            Riesz representer values
        """
        if not self.is_fitted_:
            raise ValueError("PGMM must be fitted before prediction")
        
        Wx = self.x_featurizer.transform(X)
        alpha = Wx @ self.rho_
        return alpha