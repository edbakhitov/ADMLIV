# admliv/core/pgmm.py

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from numba import jit
from numba.core.errors import NumbaPerformanceWarning
import warnings

from .control import PGMMControl
from ..moments.base import BaseMoment

# Suppress Numba performance warnings
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


@jit(nopython=True)
def _coordinate_descent_optimized(
    rho,
    G,
    Omega,
    M,
    lambda_vec,
    B_diag,
    active_set,
    max_iter,
    tol,
    check_frequency,
    buffer_factor
):
    """
    Optimized coordinate descent with active set strategy.

    Optimizations:
    1. Precomputed diagonal elements (B_j cached)
    2. Active set strategy - only update non-zero coefficients
    3. Numba JIT compilation

    Parameters
    ----------
    rho : ndarray, shape (p,)
        Coefficient vector (modified in-place)
    G : ndarray, shape (d, p)
        Gram matrix X'Z / n
    Omega : ndarray, shape (d, d)
        Weight matrix
    M : ndarray, shape (d,)
        Moment vector
    lambda_vec : ndarray, shape (p,)
        Penalty parameters for each coordinate
    B_diag : ndarray, shape (p,)
        Precomputed diagonal elements B_j = G[:,j]' Omega G[:,j]
    active_set : ndarray, shape (p,)
        Boolean array indicating which coordinates are active
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    check_frequency : int
        How often to check inactive coordinates
    buffer_factor : float
        Buffer for KKT violation check

    Returns
    -------
    rho : ndarray
        Updated coefficients
    num_iter : int
        Number of iterations
    n_active : int
        Final number of active coordinates
    diff_rho : float
        Final convergence criterion value
    """
    p = len(rho)

    diff_rho = float('inf')

    for num_iter in range(max_iter):
        rho_old = rho.copy()

        # Compute residual once per iteration
        residual = M - G @ rho

        # ===== Phase 1: Update active coordinates =====
        for j in range(p):
            if not active_set[j]:
                continue

            B_j = B_diag[j]

            if B_j < 1e-12:
                continue

            # A_j = G[:,j]' Ω residual + B_j * rho[j]
            A_j = np.dot(G[:, j], Omega @ residual) + B_j * rho[j]

            # Soft-thresholding
            if A_j < -lambda_vec[j]:
                rho_new = (A_j + lambda_vec[j]) / B_j
            elif abs(A_j) <= lambda_vec[j]:
                rho_new = 0.0
            else:
                rho_new = (A_j - lambda_vec[j]) / B_j

            # Update residual incrementally
            if rho_new != rho[j]:
                delta = rho_new - rho[j]
                residual -= G[:, j] * delta
                rho[j] = rho_new

            # Mark as inactive if became zero
            if rho[j] == 0:
                active_set[j] = False

        # ===== Phase 2: Check inactive coordinates periodically =====
        if (num_iter + 1) % check_frequency == 0:
            # Recompute residual for accuracy
            residual = M - G @ rho

            for j in range(p):
                if active_set[j]:
                    continue

                # Check KKT violation
                grad_j = np.dot(G[:, j], Omega @ residual)

                if abs(grad_j) > lambda_vec[j] * buffer_factor:
                    # Add back to active set
                    active_set[j] = True

        # Check convergence
        diff_rho = np.sum(np.abs(rho - rho_old))

        if diff_rho <= tol:
            n_active = np.sum(active_set)
            return rho, num_iter + 1, n_active, diff_rho

    n_active = np.sum(active_set)
    return rho, max_iter, n_active, diff_rho


class PGMM(BaseEstimator):
    """
    Penalized Generalized Method of Moments (PGMM) estimator for Riesz representer.
    
    Estimates the Riesz representer alpha(Z) = b(Z)'rho using adaptive PGMM
    with coordinate descent. This approach uses L1 penalization for sparse estimation
    of the Riesz representer coefficients.
    
    The orthogonal moment condition is:
        psi(W, gamma, alpha) = m(W, gamma) + alpha(Z) * (Y - gamma(X))
    
    where the Gateaux derivative of psi w.r.t. gamma is zero, yielding the
    sample moment condition for PGMM:
        1/n * sum_i [m(W_i, d_j) - d_j(X_i) * b(Z_i)' rho] = 0
    
    for j = 1, ..., dim(d(X)), where d(X) is a basis expansion of X.
    
    Parameters
    ----------
    x_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion d(X)
    z_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion b(Z)
    lambda_ : float, optional
        Penalty parameter. If None, computed as c * sqrt(log(p) / n)
    adaptive : bool, default=True
        If True, uses adaptive weights based on preliminary estimation
    Omega : Optional[NDArray], default=None
        Weight matrix for 2nd stage GMM criterion. If None, uses the 
        optimal diagonal matrix from Caner and Kock (2018)
    control : Optional[PGMMControl], default=None
        Control parameters for optimization
    verbose : bool, default=True
        If True, prints optimization progress

    Attributes
    ----------
    rho_ : NDArray
        Estimated Riesz representer coefficients (shape: dim(b(Z)))
    lambda_ : float
        Penalty parameter used in estimation
    Omega_opt_ : NDArray
        Optimal weight matrix computed from stage 1
    rho_preliminary_ : NDArray
        Preliminary estimate from stage 1
    n_samples_ : int
        Number of samples in training data
    n_features_x_ : int
        Number of features in d(X) basis expansion
    n_features_z_ : int
        Number of features in b(Z) basis expansion
    is_fitted_ : bool
        Whether the estimator has been fitted
    """
    
    def __init__(
        self,
        x_featurizer: TransformerMixin,
        z_featurizer: TransformerMixin,
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
        W: Dict[str, NDArray[np.float64]],
        moment: BaseMoment,
        **moment_kwargs
    ):
        """
        Fit the PGMM estimator.
        
        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys 'Y', 'X', 'Z'
        moment : BaseMoment
            Moment function instance
        **moment_kwargs : dict
            Additional arguments passed to moment.compute_all_basis()
        
        Returns
        -------
        self : PGMM
            Fitted estimator
        """
        Y = W['Y']
        X = W['X']
        Z = W['Z']
        n = X.shape[0]
        
        # Fit and transform basis expansions
        self.x_featurizer.fit(X)
        self.z_featurizer.fit(Z)
        
        # Transform data
        Wx = self.x_featurizer.transform(X)
        Wz = self.z_featurizer.transform(Z)
        
        self.n_samples_ = n
        self.n_features_x_ = Wx.shape[1]
        self.n_features_z_ = Wz.shape[1]
        
        # Check identification condition
        if self.n_features_x_ < self.n_features_z_:
            raise ValueError(
                f"Under-identified: dim(d(X))={self.n_features_x_} < "
                f"dim(b(Z))={self.n_features_z_}. Need dim(d(X)) >= dim(b(Z))."
            )
        
        # Compute penalty parameter
        if self.lambda_ is None:
            self.lambda_ = self.control.c * np.sqrt(np.log(self.n_features_x_) / n)
        
        # Always run two-stage procedure
        self.rho_ = self._fit_two_stage(Y, X, Z, Wx, Wz, moment, moment_kwargs)
        
        self.is_fitted_ = True
        return self
    
    def _fit_two_stage(
        self,
        Y: NDArray[np.float64],
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        moment: BaseMoment,
        moment_kwargs: dict
    ) -> NDArray[np.float64]:
        """
        Two-step adaptive PGMM estimation.
        
        Step 1: Preliminary PGMM with Omega = I
        Step 2: Adaptive PGMM with optimal Omega and adaptive weights
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
        Omega_I = np.eye(self.n_features_x_) 
        rho_tilde = self._fit_pgmm(
            Y, X, Z, Wx, Wz, moment, moment_kwargs,
            Omega=Omega_I,
            weights=None,
            rho_start=None
        )
        
        # Store preliminary estimate
        self.rho_preliminary_ = rho_tilde.copy()
        
        if self.verbose:
            print(f"  Preliminary estimate: {np.count_nonzero(rho_tilde)} non-zero coefficients")
        
        # ===== Compute optimal weight matrix from stage 1 =====
        if self.Omega is None:
            psi_tilde = self._compute_orthogonal_moment(
                Y, X, Wx, Wz, moment, moment_kwargs, rho_tilde
            )
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
            Y, X, Z, Wx, Wz, moment, moment_kwargs,
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
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        moment: BaseMoment,
        moment_kwargs: dict,
        Omega: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]],
        rho_start: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Coordinate descent for PGMM estimation.

        Minimizes: psi' Omega psi / 2 + lambda * sum(w_j * |rho_j|)
        where w_j are adaptive weights if provided, or 1 otherwise.
        """
        n = X.shape[0]

        # Precompute Gram matrix
        G = Wx.T @ Wz / n

        # Compute moment matrix M from the moment object
        W_temp = {'Y': Y, 'X': X, 'Z': Z}
        M = moment.compute_all_basis(self.x_featurizer, W_temp, **moment_kwargs)
        M_mean = M.mean(axis=0)

        return self._fit_pgmm_core(G, Omega, M_mean, weights, rho_start)

    def _fit_pgmm_with_M(
        self,
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        M: NDArray[np.float64],
        Omega: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]],
        rho_start: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Coordinate descent for PGMM with precomputed moment matrix M.

        Same as _fit_pgmm but takes M directly instead of computing from moment.
        Used for nonlinear functionals where M depends on gamma.
        """
        n = Wx.shape[0]
        G = Wx.T @ Wz / n
        M_mean = M.mean(axis=0)
        return self._fit_pgmm_core(G, Omega, M_mean, weights, rho_start)

    def _fit_pgmm_core(
        self,
        G: NDArray[np.float64],
        Omega: NDArray[np.float64],
        M_mean: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]],
        rho_start: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Core coordinate descent, shared by _fit_pgmm and _fit_pgmm_with_M.

        Parameters
        ----------
        G : NDArray, shape (q, p)
            Gram matrix Wx.T @ Wz / n
        Omega : NDArray, shape (q, q)
            Weight matrix
        M_mean : NDArray, shape (q,)
            Mean moment vector
        weights : Optional[NDArray], shape (p,)
            Adaptive weights (None for uniform)
        rho_start : Optional[NDArray], shape (p,)
            Warm start (None for zeros)
        """
        p = self.n_features_z_
        q = self.n_features_x_

        # Setup penalty vector
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

        # Precompute diagonal elements
        GOmega = G.T @ Omega / q  # (p, d)
        B_diag = np.sum(GOmega * G.T, axis=1)  # (p,)

        # Initialize active set (all coordinates start active)
        active_set = np.ones(p, dtype=np.bool_)

        # Optimized coordinate descent
        rho, num_iter, n_active, diff_rho = _coordinate_descent_optimized(
            rho=rho,
            G=G,
            Omega=Omega / q,
            M=M_mean,
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
    
    def _compute_orthogonal_moment(
        self,
        Y: NDArray[np.float64],
        X: NDArray[np.float64],
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        moment: BaseMoment,
        moment_kwargs: dict,
        rho: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute orthogonal moment: psi(W, gamma, alpha) = m(W, d_j) - d_j(X) * alpha(Z)
        
        This is used for computing the optimal weight matrix.
        
        Returns
        -------
        psi : NDArray, shape (n, k)
            Orthogonal moment values
        """
        # Compute moment matrix M using new interface
        W_temp = {'Y': Y, 'X': X, 'Z': None}
        M = moment.compute_all_basis(self.x_featurizer, W_temp, **moment_kwargs)
        
        # Compute alpha(Z) = b(Z)' rho
        alpha = Wz @ rho
        
        # Orthogonal moment: m(W, d_j) - d_j(X) * alpha(Z)
        # Broadcasting: (n, k) - (n, 1) * (n, k)
        psi = M - alpha[:, np.newaxis] * Wx
        
        return psi
    
    def _compute_orthogonal_moment_with_M(
        self,
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        M: NDArray[np.float64],
        rho: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute orthogonal moment with precomputed M.

        psi[i, k] = M[i, k] - d_k(X_i) * alpha(Z_i)

        Used for computing the optimal weight matrix when M is provided
        externally (nonlinear functionals).

        Parameters
        ----------
        Wx : NDArray, shape (n, q)
            Basis expansion d(X)
        Wz : NDArray, shape (n, p)
            Basis expansion b(Z)
        M : NDArray, shape (n, q)
            Precomputed moment matrix
        rho : NDArray, shape (p,)
            Current Riesz representer coefficients

        Returns
        -------
        psi : NDArray, shape (n, q)
            Orthogonal moment values
        """
        alpha = Wz @ rho
        psi = M - alpha[:, np.newaxis] * Wx
        return psi

    def _compute_adaptive_weights(
        self,
        rho_prelim: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute adaptive weights: w_j = 1 / |rho_j| with numerical safeguards.
        """
        p = len(rho_prelim)
        weights = np.zeros(p)
        
        for j in range(p):
            if np.abs(rho_prelim[j]) <= self.control.adaptive_threshold:
                weights[j] = self.control.adaptive_max_weight
            else:
                weights[j] = 1.0 / np.abs(rho_prelim[j])
        
        return weights

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

    def fit_with_M(
        self,
        W: Dict[str, NDArray[np.float64]],
        M: NDArray[np.float64],
    ) -> 'PGMM':
        """
        Fit PGMM with a precomputed moment matrix M.

        For nonlinear functionals, M = D_gamma theta[d_k] depends on gamma
        and must be computed externally with proper double cross-fitting.

        The only difference from fit() is that M is provided directly
        instead of being computed internally via moment.compute_all_basis().

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys 'Y', 'X', 'Z'.
        M : NDArray, shape (n, q)
            Precomputed moment matrix where q = dim(d(X)).
            M[i, k] = D_gamma m(W_i; gamma)[d_k].

        Returns
        -------
        self : PGMM
            Fitted estimator
        """
        X = W['X']
        Z = W['Z']
        n = X.shape[0]

        # Fit and transform basis expansions
        self.x_featurizer.fit(X)
        self.z_featurizer.fit(Z)

        Wx = self.x_featurizer.transform(X)
        Wz = self.z_featurizer.transform(Z)

        self.n_samples_ = n
        self.n_features_x_ = Wx.shape[1]
        self.n_features_z_ = Wz.shape[1]

        # Validate M dimensions
        if M.shape[0] != n:
            raise ValueError(
                f"M has {M.shape[0]} rows but data has {n} observations"
            )
        if M.shape[1] != self.n_features_x_:
            raise ValueError(
                f"M has {M.shape[1]} columns but d(X) has "
                f"{self.n_features_x_} features"
            )

        # Check identification condition
        if self.n_features_x_ < self.n_features_z_:
            raise ValueError(
                f"Under-identified: dim(d(X))={self.n_features_x_} < "
                f"dim(b(Z))={self.n_features_z_}. "
                f"Need dim(d(X)) >= dim(b(Z))."
            )

        # Compute penalty parameter — nonlinear rate n^{-1/4}
        # For nonlinear functionals, double cross-fitting introduces
        # additional estimation error from inner gamma estimates,
        # requiring the slower rate lambda = c * sqrt(log(p)) * n^{-1/4}
        # instead of the linear rate c * sqrt(log(p) / n) = c * sqrt(log(p)) * n^{-1/2}
        if self.lambda_ is None:
            self.lambda_ = (self.control.c
                            * np.sqrt(np.log(self.n_features_x_))
                            / n ** (1/4))

        # Run two-stage procedure with precomputed M
        self.rho_ = self._fit_two_stage_with_M(Wx, Wz, M)

        self.is_fitted_ = True
        return self

    def _fit_two_stage_with_M(
        self,
        Wx: NDArray[np.float64],
        Wz: NDArray[np.float64],
        M: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Two-step adaptive PGMM with precomputed M.

        Mirrors _fit_two_stage but uses _fit_pgmm_with_M and
        _compute_orthogonal_moment_with_M instead of moment-based versions.
        """
        if self.verbose:
            print("=" * 60)
            print("PGMM Two-Stage Estimation (precomputed M)")
            print("=" * 60)
            print(f"Adaptive weights: {self.adaptive}")
            print(f"Lambda: {self.lambda_:.6f}")
            print("-" * 60)
            print("Stage 1: Preliminary PGMM with Omega = I")

        # Stage 1: Preliminary estimate with identity weight matrix
        Omega_I = np.eye(self.n_features_x_)
        rho_tilde = self._fit_pgmm_with_M(
            Wx, Wz, M,
            Omega=Omega_I,
            weights=None,
            rho_start=None
        )

        self.rho_preliminary_ = rho_tilde.copy()

        if self.verbose:
            print(f"  Preliminary estimate: "
                  f"{np.count_nonzero(rho_tilde)} non-zero coefficients")

        # Compute optimal weight matrix from stage 1
        if self.Omega is None:
            psi_tilde = self._compute_orthogonal_moment_with_M(
                Wx, Wz, M, rho_tilde
            )
            self.Omega_opt_ = self._compute_optimal_weight_matrix(psi_tilde)
        else:
            self.Omega_opt_ = self.Omega

        if self.verbose:
            print("-" * 60)
            print("Stage 2: PGMM with optimal Omega" +
                  (" and adaptive weights" if self.adaptive else ""))

        # Stage 2: Re-estimate with optimal Omega
        if self.adaptive:
            weights = self._compute_adaptive_weights(rho_tilde)
        else:
            weights = None

        rho_hat = self._fit_pgmm_with_M(
            Wx, Wz, M,
            Omega=self.Omega_opt_,
            weights=weights,
            rho_start=rho_tilde
        )

        if self.verbose:
            print(f"  Final estimate: "
                  f"{np.count_nonzero(rho_hat)} non-zero coefficients")
            print("=" * 60)

        return rho_hat

    def predict(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute Riesz representer alpha(Z) = b(Z)' rho.
        
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
        
        Wz = self.z_featurizer.transform(Z)
        alpha = Wz @ self.rho_
        return alpha
    
    def get_rho(self) -> NDArray[np.float64]:
        """
        Get the estimated Riesz representer coefficients.
        
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
        moment: BaseMoment,
        Omega: Optional[NDArray[np.float64]] = None,
        **moment_kwargs
    ) -> float:
        """
        Compute GMM criterion: psi' Omega psi / 2q.
        
        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary
        moment : BaseMoment
            Moment function
        Omega : Optional[NDArray], default=None
            Weight matrix. If None, uses identity
        **moment_kwargs : dict
            Additional moment arguments
        
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
        
        Wx = self.x_featurizer.transform(X)
        Wz = self.z_featurizer.transform(Z)
        q = self.n_features_x_
        
        psi = self._compute_orthogonal_moment(
            Y, X, Wx, Wz, moment, moment_kwargs, self.rho_
        )
        psi_mean = psi.mean(axis=0)
        
        if Omega is None:
            Omega = np.eye(q) 
        
        criterion = 0.5 * psi_mean.T @ Omega @ psi_mean / q
        return float(criterion)