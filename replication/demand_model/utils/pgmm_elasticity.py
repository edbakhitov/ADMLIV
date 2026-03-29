# simulations/demand_model/utils/pgmm_elasticity.py

"""
Panel Data PGMM for Own-Price Elasticity Estimation.

This module implements PGMM for estimating the Riesz representer needed to
debias the own-price elasticity functional:

    ε_{jj}(γ) = (p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj}

Key differences from PGMMPanel (demand derivative):
1. Moment M (Gateaux derivatives) is passed in explicitly, not computed internally
2. This allows for double cross-fitting in ADMLIVElasticity
3. The functional is nonlinear in γ, requiring Gateaux derivatives

The Gateaux derivative for basis function d_k is:
    D_γ ε_{jj}[d_k] = (p_j/s_j) [(A⁻¹ Zᵖ_k)_{jj} + (A⁻¹ Zˢ_k A⁻¹ Γᵖ)_{jj}]
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass

from .raw_data import RawData

# Import coordinate descent from main PGMM module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from admliv.core.pgmm import _coordinate_descent_optimized


@dataclass
class PGMMElasticityControl:
    """
    Control parameters for PGMM elasticity estimation.

    Parameters
    ----------
    maxiter : int, default=5000
        Maximum number of coordinate descent iterations
    optTol : float, default=1e-5
        Convergence tolerance for coordinate descent
    zeroThreshold : float, default=1e-6
        Threshold below which coefficients are set to zero
    intercept_penalty : float, default=0.1
        Penalty multiplier for intercept term (should be small)
    c : float, default=0.01
        Penalty parameter multiplier for lambda = c * sqrt(log(q)) * n^(-1/4)
    warm_start_fraction : float, default=0.4
        Fraction of sample size to use for lower-dimensional warm start.
        Sets p_low = int(warm_start_fraction * T), capped at the full IV
        dimension p.  Only used when warm_start=True in PGMMElasticity.
    adaptive_threshold : float, default=1e-10
        Threshold for adaptive weights calculation (numerical stability)
    adaptive_max_weight : float, default=1e10
        Maximum value for adaptive weights (numerical stability)
    check_frequency : int, default=5
        How often to check inactive coordinates in active set strategy
    buffer_factor : float, default=1.1
        Buffer for KKT violation check in active set (larger = more conservative)
    """
    maxiter: int = 5000
    optTol: float = 1e-5
    zeroThreshold: float = 1e-6
    intercept_penalty: float = 0.1
    c: float = 0.01
    warm_start_fraction: float = 0.4
    adaptive_threshold: float = 1e-10
    adaptive_max_weight: float = 1e10
    check_frequency: int = 5
    buffer_factor: float = 1.1

    def __post_init__(self):
        """Validate control parameters."""
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.optTol <= 0:
            raise ValueError("optTol must be positive")
        if self.zeroThreshold < 0:
            raise ValueError("zeroThreshold must be non-negative")
        if self.intercept_penalty < 0:
            raise ValueError("intercept_penalty must be non-negative")
        if self.c <= 0:
            raise ValueError("c must be positive")
        if not (0 < self.warm_start_fraction < 1):
            raise ValueError("warm_start_fraction must be in (0, 1)")
        if self.check_frequency <= 0:
            raise ValueError("check_frequency must be positive")
        if self.buffer_factor <= 0:
            raise ValueError("buffer_factor must be positive")


class PGMMElasticity(BaseEstimator):
    """
    Panel Data PGMM for estimating Riesz representers for own-price elasticity.

    For own-price elasticity estimation, the parameter of interest is:
        θ_j = E[ε_{jj,t}(γ)] = E[(p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj}]

    This class estimates the Riesz representer α_j(Z) needed for debiasing,
    using a precomputed Gateaux derivative matrix M.

    Parameters
    ----------
    omega_featurizer : TransformerMixin
        Pre-fitted featurizer for omega space (d(ω) basis expansion).
        Used to construct the Gram matrix G = d(ω)' b(Z).
        Must be fitted on product j's omega data before calling fit().
    omega_iv_featurizer : TransformerMixin
        Pre-fitted featurizer for IV space (b(Z) basis expansion).
        Must be fitted on product j's omega_iv data before calling fit().
    lambda_ : float, optional
        Penalty parameter. If None, computed as c * sqrt(log(q)) * n^(-1/4)
    adaptive : bool, default=True
        If True, uses adaptive weights based on preliminary estimation
    warm_start : bool, default=False
        If True, Stage 1 is initialized from a lower-dimensional unpenalized
        GMM solved in closed form. The reduced dimension
        p_low = int(control.warm_start_fraction * T) is capped at p.
    control : PGMMElasticityControl, optional
        Control parameters for optimization
    verbose : bool, default=True
        If True, prints optimization progress

    Attributes
    ----------
    rho_ : NDArray
        Estimated Riesz representer coefficients
    product_id_ : int or str
        Product identifier for which α was estimated
    n_markets_ : int
        Number of markets used in estimation
    is_fitted_ : bool
        Whether the estimator has been fitted

    Examples
    --------
    >>> # Setup
    >>> from admliv.utils.featurizers import CoordinatePolyTransform
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>>
    >>> # Compute M (Gateaux derivatives) externally
    >>> elasticity = OwnPriceElasticity(omega_transformer, product_id=0)
    >>> M = elasticity.compute_basis_gateaux(kiv, omega_featurizer, raw_data, product_id=0)
    >>>
    >>> # Create PGMM for elasticity
    >>> pgmm = PGMMElasticity(
    ...     omega_featurizer=CoordinatePolyTransform(degree=2),
    ...     omega_iv_featurizer=PolynomialFeatures(degree=3)
    ... )
    >>>
    >>> # Fit for product 0 with precomputed M
    >>> pgmm.fit(raw_data, omega, omega_iv, M, product_id=0)
    """

    def __init__(
        self,
        omega_featurizer: TransformerMixin,
        omega_iv_featurizer: TransformerMixin,
        lambda_: Optional[float] = None,
        adaptive: bool = True,
        warm_start: bool = False,
        single_stage: bool = False,
        control: Optional[PGMMElasticityControl] = None,
        verbose: bool = True
    ):
        self.omega_featurizer = omega_featurizer
        self.omega_iv_featurizer = omega_iv_featurizer
        self.lambda_ = lambda_
        self.adaptive = adaptive
        self.warm_start = warm_start
        self.single_stage = single_stage
        self.control = control if control is not None else PGMMElasticityControl()
        self.verbose = verbose
        self.is_fitted_ = False

    def fit(
        self,
        raw_data: RawData,
        omega: NDArray,
        omega_iv: NDArray,
        M: NDArray,
        product_id: Union[int, str],
        markets: Optional[NDArray] = None
    ) -> 'PGMMElasticity':
        """
        Fit PGMM for elasticity Riesz representer.

        Parameters
        ----------
        raw_data : RawData
            Raw panel data (price, x, shares, market_ids)
        omega : NDArray, shape (n, d_omega)
            Pre-computed omega for full panel
        omega_iv : NDArray, shape (n, d_iv)
            Pre-computed IV omega for full panel
        M : NDArray, shape (T_j, q)
            Pre-computed Gateaux derivative matrix.
            M[t, k] = D_γ ε_{jj,t}[d_k] (derivative w.r.t. basis function k)
            Should be computed externally via OwnPriceElasticity.compute_basis_gateaux()
        product_id : int or str
            Product identifier to estimate Riesz representer for
        markets : NDArray, optional
            Market IDs corresponding to rows of M. If None, uses all markets
            with the product in raw_data.

        Returns
        -------
        self : PGMMElasticity
        """
        self.product_id_ = product_id

        # Determine which markets to use
        if markets is not None:
            markets_with_product = markets
        else:
            markets_with_product = raw_data.get_markets_with_product(product_id)

        self.n_markets_ = len(markets_with_product)

        if self.n_markets_ == 0:
            raise ValueError(f"Product {product_id} not found in any market")

        if self.verbose:
            print(f"Fitting PGMM Elasticity for product {product_id}")
            print(f"  Markets with product: {self.n_markets_}")

        # Extract product j's omega and IV omega from each market
        Wx_list = []  # omega for product j
        Wz_list = []  # IV omega for product j

        for t in markets_with_product:
            mask = raw_data.market_ids == t
            j_local = raw_data.get_local_index(t, product_id)

            # Extract product j's row from precomputed arrays
            Wx_list.append(omega[mask][j_local])
            Wz_list.append(omega_iv[mask][j_local])

        # Stack into arrays: (T, d_omega) and (T, d_omega_iv)
        omega_j = np.vstack(Wx_list)
        omega_iv_j = np.vstack(Wz_list)

        # Transform to basis expansions (featurizers should be pre-fitted)
        Wx = self.omega_featurizer.transform(omega_j)  # (T, q)
        Wz = self.omega_iv_featurizer.transform(omega_iv_j)  # (T, p)

        self.n_features_omega_ = Wx.shape[1]
        self.n_features_omega_iv_ = Wz.shape[1]

        if self.verbose:
            print(f"  Omega basis dim: {self.n_features_omega_}")
            print(f"  IV basis dim: {self.n_features_omega_iv_}")

        # Check identification
        if self.n_features_omega_ < self.n_features_omega_iv_:
            raise ValueError(
                f"Under-identified: dim(d(omega))={self.n_features_omega_} < "
                f"dim(b(omega_iv))={self.n_features_omega_iv_}"
            )

        # Validate M dimensions
        if M.shape[0] != self.n_markets_:
            raise ValueError(
                f"M has {M.shape[0]} rows but expected {self.n_markets_} markets"
            )
        if M.shape[1] != self.n_features_omega_:
            raise ValueError(
                f"M has {M.shape[1]} columns but omega basis has {self.n_features_omega_} features"
            )

        # Compute penalty parameter
        if self.lambda_ is None:
            self.lambda_computed_ = self.control.c * np.sqrt(np.log(self.n_features_omega_)) / self.n_markets_**(1/4)
        else:
            self.lambda_computed_ = self.lambda_

        if self.verbose:
            print(f"  Moment matrix M (Gateaux): {M.shape}")
            print(f"  M range: [{M.min():.4f}, {M.max():.4f}]")

        # Store for diagnostics
        self.M_ = M

        # Run two-stage PGMM
        self.rho_ = self._fit_two_stage(Wx, Wz, M)

        self.is_fitted_ = True
        return self

    def _fit_two_stage(
        self,
        Wx: NDArray,
        Wz: NDArray,
        M: NDArray
    ) -> NDArray:
        """
        Two-stage adaptive PGMM estimation.

        Stage 1: Preliminary PGMM with Omega = I
        Stage 2: Adaptive PGMM with optimal Omega
        """
        if self.verbose:
            print("-" * 50)
            print("Stage 1: Preliminary PGMM with Omega = I")

        # Stage 1: optionally warm-start from low-dimensional closed-form solution
        if self.warm_start:
            M_mean = M.mean(axis=0)
            rho_init = self._compute_warm_start(Wx, Wz, M_mean)
        else:
            rho_init = None

        Omega_I = np.eye(self.n_features_omega_)
        rho_tilde = self._fit_pgmm(Wx, Wz, M, Omega_I, weights=None, rho_start=rho_init)

        self.rho_preliminary_ = rho_tilde.copy()

        if self.verbose:
            print(f"  Preliminary estimate: {np.count_nonzero(rho_tilde)} non-zero coefficients")

        if self.single_stage:
            if self.verbose:
                print("  Single-stage mode: skipping Stage 2")
                print("-" * 50)
            return rho_tilde

        # Compute optimal weight matrix
        psi_tilde = self._compute_orthogonal_moment(Wx, Wz, M, rho_tilde)
        self.Omega_opt_ = self._compute_optimal_weight_matrix(psi_tilde)

        if self.verbose:
            print("-" * 50)
            print("Stage 2: PGMM with optimal Omega" +
                  (" and adaptive weights" if self.adaptive else ""))

        # Stage 2
        if self.adaptive:
            weights = self._compute_adaptive_weights(rho_tilde)
        else:
            weights = None

        rho_hat = self._fit_pgmm(Wx, Wz, M, self.Omega_opt_, weights, rho_start=rho_tilde)

        if self.verbose:
            print(f"  Final non-zero coefficients: {np.count_nonzero(rho_hat)}")
            print("-" * 50)

        return rho_hat

    def _compute_warm_start(
        self,
        Wx: NDArray,
        Wz: NDArray,
        M_mean: NDArray
    ) -> NDArray:
        """
        Compute warm-start initial estimate from a lower-dimensional
        unpenalized GMM.

        Selects the first p_low columns of b(Z) (where
        p_low = int(warm_start_fraction * T), capped at p) and solves
        the closed-form unpenalized problem:

            rho_low = (G_low' G_low)^{-1} G_low' M_mean

        The solution is then zero-padded to the full p-dimensional space.

        Parameters
        ----------
        Wx : NDArray, shape (T, q)
            Basis expansion of omega for product j
        Wz : NDArray, shape (T, p)
            Basis expansion of IV omega for product j
        M_mean : NDArray, shape (q,)
            Mean moment vector

        Returns
        -------
        rho_init : NDArray, shape (p,)
            Initial estimate, zero-padded from the low-dimensional solution
        """
        T, p = Wz.shape

        p_low = min(int(self.control.warm_start_fraction * T), p)
        if p_low < 1:
            p_low = 1

        if self.verbose:
            print(f"  Warm start: solving {p_low}-dim unpenalized GMM "
                  f"(fraction={self.control.warm_start_fraction}, T={T}, p={p})")

        # Reduced dictionary: first p_low basis functions
        Wz_low = Wz[:, :p_low]  # (T, p_low)

        # Reduced Gram matrix: G_low = Wx' Wz_low / T
        G_low = Wx.T @ Wz_low / T  # (q, p_low)

        # Closed-form solution: rho_low = (G_low' G_low)^{-1} G_low' M_mean
        GtG = G_low.T @ G_low + 1e-5 * np.eye(p_low)  # (p_low, p_low)
        GtM = G_low.T @ M_mean  # (p_low,)
        rho_low = np.linalg.solve(GtG, GtM)  # (p_low,)

        # Zero-pad to full dimension
        rho_init = np.zeros(p)
        rho_init[:p_low] = rho_low

        if self.verbose:
            print(f"  Warm start rho_low: norm={np.linalg.norm(rho_low):.4f}, "
                  f"nonzero={np.count_nonzero(rho_low)}/{p_low}")

        return rho_init

    def _fit_pgmm(
        self,
        Wx: NDArray,
        Wz: NDArray,
        M: NDArray,
        Omega: NDArray,
        weights: Optional[NDArray],
        rho_start: Optional[NDArray]
    ) -> NDArray:
        """
        Coordinate descent for PGMM.

        Parameters
        ----------
        Wx : NDArray, shape (T, q)
            Basis expansion of omega for product j
        Wz : NDArray, shape (T, p)
            Basis expansion of IV omega for product j
        M : NDArray, shape (T, q)
            Gateaux derivative matrix from compute_basis_gateaux
        Omega : NDArray, shape (q, q)
            Weight matrix
        weights : NDArray, shape (p,), optional
            Adaptive weights
        rho_start : NDArray, shape (p,), optional
            Initial coefficients (warm start)
        """
        T = Wx.shape[0]
        p = self.n_features_omega_iv_
        q = self.n_features_omega_

        # Setup penalty vector
        L = np.concatenate([
            np.array([self.control.intercept_penalty]),
            np.ones(p - 1)
        ])

        if weights is not None:
            lambda_vec = self.lambda_computed_ * L * weights
        else:
            lambda_vec = self.lambda_computed_ * L

        # Initialize rho
        if rho_start is None:
            rho = np.zeros(p)
        else:
            rho = rho_start.copy()

        # Gram matrix: G = Wx' Wz / T
        G = Wx.T @ Wz / T  # (q, p)

        # Mean moment
        M_mean = M.mean(axis=0)  # (q,)

        # Precompute diagonal elements
        GOmega = G.T @ Omega / q  # (p, q)
        B_diag = np.sum(GOmega * G.T, axis=1)  # (p,)

        # Diagnostics
        if self.verbose:
            # Initial gradient at rho=0: A_j = G[:,j]' (Omega/q) M_mean
            grad_init = GOmega @ M_mean  # (p,)
            n_active_b = (B_diag > 1e-12).sum()
            # Among active coordinates, how many have gradient > lambda?
            active_mask = B_diag > 1e-12
            n_grad_active = ((np.abs(grad_init) > lambda_vec) & active_mask).sum()
            print(f"  PGMM diagnostics:")
            print(f"    lambda = {self.lambda_computed_:.6e}")
            print(f"    lambda_vec range = [{lambda_vec.min():.6e}, {lambda_vec.max():.6e}]")
            print(f"    ||G||_max = {np.abs(G).max():.6e}, ||G||_mean = {np.abs(G).mean():.6e}")
            print(f"    ||M_mean||_inf = {np.abs(M_mean).max():.6e}, ||M_mean||_2 = {np.linalg.norm(M_mean):.6e}")
            print(f"    max|init_gradient| = {np.abs(grad_init).max():.6e}")
            print(f"    # coords with B_diag > 1e-12: {n_active_b}/{p}")
            print(f"    # active coords with |grad| > lambda: {n_grad_active}/{n_active_b}")
            print(f"    B_diag range = [{B_diag.min():.6e}, {B_diag.max():.6e}]")
            if n_active_b > 0:
                print(f"    B_diag (active only) range = [{B_diag[active_mask].min():.6e}, {B_diag[active_mask].max():.6e}]")

        # Initialize active set
        active_set = np.ones(p, dtype=np.bool_)

        # Coordinate descent
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
        Wx: NDArray,
        Wz: NDArray,
        M: NDArray,
        rho: NDArray
    ) -> NDArray:
        """
        Compute orthogonal moment: psi = M - alpha * Wx

        where alpha = Wz @ rho
        """
        alpha = Wz @ rho  # (T,)
        psi = M - alpha[:, np.newaxis] * Wx  # (T, q)
        return psi

    def _compute_adaptive_weights(
        self,
        rho_prelim: NDArray
    ) -> NDArray:
        """Compute adaptive weights: w_j = 1 / |rho_j|"""
        abs_rho = np.abs(rho_prelim)
        weights = np.where(
            abs_rho <= self.control.adaptive_threshold,
            self.control.adaptive_max_weight,
            1.0 / np.maximum(abs_rho, self.control.adaptive_threshold)
        )
        return weights

    @staticmethod
    def _compute_optimal_weight_matrix(psi_tilde: NDArray) -> NDArray:
        """Compute optimal diagonal weight matrix."""
        psi_var = np.var(psi_tilde, axis=0, ddof=1)
        psi_var_safe = np.where(psi_var > 1e-10, psi_var, 1e-10)
        Omega_diag = 1.0 / psi_var_safe
        return np.diag(Omega_diag)

    def predict(self, omega_iv: NDArray) -> NDArray:
        """
        Compute Riesz representer α(Z) = b(Z)' ρ.

        Parameters
        ----------
        omega_iv : NDArray, shape (n, d_iv)
            IV omega (not featurized)

        Returns
        -------
        alpha : NDArray, shape (n,)
            Riesz representer values
        """
        if not self.is_fitted_:
            raise ValueError("PGMMElasticity must be fitted before prediction")

        Wz = self.omega_iv_featurizer.transform(omega_iv)
        return Wz @ self.rho_

    def get_rho(self) -> NDArray:
        """Get estimated Riesz representer coefficients."""
        if not self.is_fitted_:
            raise ValueError("PGMMElasticity must be fitted before accessing rho")
        return self.rho_

    def get_omega(self) -> NDArray:
        """Get optimal weight matrix from stage 1."""
        if not self.is_fitted_:
            raise ValueError("PGMMElasticity must be fitted before accessing Omega")
        return self.Omega_opt_

