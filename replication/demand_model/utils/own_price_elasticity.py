# simulations/demand_model/utils/own_price_elasticity.py

"""
Own-Price Elasticity Functional for ADMLIV.

This module implements the moment function for estimating the own-price elasticity
functional in nonparametric demand estimation:

    ε_{jj}(γ) = (p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj}

where:
    - L = D_s^{-1} + (1/s_0) 11' is the logit-like share Jacobian
    - Γˢ_{jk} = ∂γ(ω_j)/∂s_k is the share derivative matrix
    - Γᵖ_{jk} = ∂γ(ω_j)/∂p_k is the price derivative matrix

Key Design:
-----------
This class uses ANALYTICAL derivatives from the gamma estimator (e.g., KIV's
predict_derivative method) combined with chain rule through the omega transformation.
This avoids numerical differentiation issues when shares are close to zero.

Gateaux Derivative (for debiasing):
----------------------------------
D_γ ε_{jj}[ζ] = (p_j/s_j) [(A⁻¹ Zᵖ)_{jj} + (A⁻¹ Zˢ A⁻¹ Γᵖ)_{jj}]

where A = L - Γˢ, and Zᵖ, Zˢ are the perturbation derivative matrices.

"""

from typing import Optional, Union, Dict, Tuple, Protocol, Callable, Any
from dataclasses import dataclass
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from sklearn.base import TransformerMixin

from .raw_data import RawData
from .omega_transformer import OmegaTransformer
from admliv.moments.base import BaseMoment


class GammaEstimatorWithDerivative(Protocol):
    """Protocol for gamma estimators that support analytical derivatives."""

    def predict(self, X: NDArray) -> NDArray:
        """Predict gamma(omega)."""
        ...

    def predict_derivative(self, X: NDArray) -> NDArray:
        """Return gradient ∂γ/∂ω of shape (n, d_omega)."""
        ...


class FeaturizerWithDerivative(Protocol):
    """Protocol for featurizers that support analytical derivatives."""

    def transform(self, X: NDArray) -> NDArray:
        """Transform X to basis features of shape (n, n_basis)."""
        ...

    def transform_derivative(self, X: NDArray, wrt: int) -> NDArray:
        """
        Return derivative of each basis function w.r.t. X[:, wrt].

        Returns array of shape (n, n_basis) where column k is ∂d_k/∂X[:, wrt].
        """
        ...


@dataclass
class OmegaStructure:
    """
    Describes the structure of the omega vector for a specific market.

    Used to map omega indices to economic variables (prices, shares, characteristics).

    Attributes
    ----------
    J : int
        Number of inside goods in the market
    K : int
        Number of characteristics (excluding price)
    share_start : int
        Starting index of shares in omega
    share_end : int
        Ending index of shares (exclusive)
    s0_index : int or None
        Index of outside good share (if included)
    own_price_index : int or None
        Index of own price (if price_in_diffs=False)
    own_char_start : int
        Starting index of own characteristics (including price if price_in_diffs=True)
    own_char_end : int
        Ending index of own characteristics
    diff_start : int
        Starting index of characteristic differences
    diff_end : int
        Ending index of characteristic differences
    price_in_diffs : bool
        Whether price is included in characteristic differences
    share_representation : str
        How shares are represented ('all', 'inner', 'others')
    """
    J: int
    J_max: int
    K: int
    share_start: int
    share_end: int
    s0_index: Optional[int]
    own_price_index: Optional[int]
    own_char_start: int
    own_char_end: int
    diff_start: int
    diff_end: int
    price_in_diffs: bool
    share_representation: str

    @property
    def n_chars_with_price(self) -> int:
        """Number of characteristics including price (K+1 if price_in_diffs)."""
        return self.K + 1 if self.price_in_diffs else self.K


def get_omega_structure(
    transformer: OmegaTransformer,
    J: int,
    J_max: Optional[int] = None
) -> OmegaStructure:
    """
    Extract the structure of omega from an OmegaTransformer.

    Parameters
    ----------
    transformer : OmegaTransformer
        Fitted omega transformer
    J : int
        Number of inside goods in the market
    J_max : int, optional
        Maximum number of inside goods across all markets.
        If None, uses transformer.J_max_ if available, else J.

    Returns
    -------
    structure : OmegaStructure
        Structure describing omega indices
    """
    K = transformer.n_characteristics_
    price_in_diffs = transformer.price_in_diffs
    include_prices = transformer.include_prices
    include_shares = transformer.include_shares
    share_rep = transformer.share_representation

    # If J_max not provided, use transformer's J_max_ if available, else J
    if J_max is None:
        J_max = getattr(transformer, 'J_max_', J)

    idx = 0

    # 1. Own price (if price_in_diffs=False)
    own_price_index = None
    if include_prices and not price_in_diffs:
        own_price_index = idx
        idx += 1

    # 2. Shares -- use J_max for column counts (matches padding in transform)
    share_start = idx
    s0_index = None
    if include_shares:
        if share_rep == 'all':
            share_end = idx + J_max
            idx = share_end
            s0_index = idx
            idx += 1
        elif share_rep == 'inner':
            share_end = idx + J_max
            idx = share_end
        elif share_rep == 'others':
            share_end = idx + J_max - 1
            idx = share_end
    else:
        share_end = share_start

    # 3. Own characteristics (x_aug if price_in_diffs, else x) -- not affected by J_max
    own_char_start = idx
    if include_prices and price_in_diffs:
        # x_aug = [price, x^(2)]
        own_char_end = idx + K + 1
    else:
        own_char_end = idx + K
    idx = own_char_end

    # 4. Characteristic differences -- use J_max for column counts
    diff_start = idx
    if include_prices and price_in_diffs:
        diff_end = idx + J_max * (K + 1)
    else:
        diff_end = idx + J_max * K

    return OmegaStructure(
        J=J,
        J_max=J_max,
        K=K,
        share_start=share_start,
        share_end=share_end,
        s0_index=s0_index,
        own_price_index=own_price_index,
        own_char_start=own_char_start,
        own_char_end=own_char_end,
        diff_start=diff_start,
        diff_end=diff_end,
        price_in_diffs=price_in_diffs,
        share_representation=share_rep
    )


class OwnPriceElasticity:
    """
    Compute own-price elasticity ε_{jj} and its Gateaux derivative.

    The own-price elasticity functional is:
        ε_{jj}(γ) = (p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj}

    This class uses ANALYTICAL derivatives from the gamma estimator to compute
    the Jacobian matrices Γᵖ and Γˢ, avoiding numerical differentiation.

    Parameters
    ----------
    transformer : OmegaTransformer
        Fitted transformer for omega construction
    product_id : int or str, optional
        Product identifier for single-product mode
    regularize_gateaux : bool, default=False
        Whether to regularize A when computing Gateaux derivatives for ADMLIV.
        When False (recommended), ill-conditioned markets should be excluded
        from M computation to avoid bias in Riesz representer estimation.
    cond_threshold : float, default=100.0
        Condition number threshold for A matrix. If cond(A) > threshold,
        Tikhonov regularization is applied: A_reg = A + reg_lambda * I
    reg_lambda : float, default=0.1
        Regularization parameter for Tikhonov regularization when A is
        ill-conditioned.

    Examples
    --------
    >>> # Setup
    >>> transformer = OmegaTransformer(price_in_diffs=True, share_representation='all')
    >>> transformer.fit(x, market_ids, price=price, shares=shares)
    >>>
    >>> # Create elasticity moment
    >>> moment = OwnPriceElasticity(transformer, product_id=0)
    >>>
    >>> # Compute elasticity for a market
    >>> eps_jj = moment.compute_market_elasticity(gamma_hat, data, market_id=0)
    >>>
    >>> # Compute average elasticity
    >>> avg_eps = moment.compute(gamma_hat, data)
    """

    def __init__(
        self,
        transformer: OmegaTransformer,
        product_id: Optional[Union[int, str]] = None,
        regularize: bool = True,
        regularize_gateaux: bool = False,
        cond_threshold: float = 100.0,
        reg_lambda: float = 0.1
    ):
        self.transformer = transformer
        self.product_id = product_id
        self.regularize = regularize
        self.regularize_gateaux = regularize_gateaux
        self.cond_threshold = cond_threshold
        self.reg_lambda = reg_lambda

    def _regularize_matrix(self, A: NDArray) -> NDArray:
        """
        Apply adaptive Tikhonov regularization if enabled and A is ill-conditioned.

        The regularization parameter λ is chosen adaptively to ensure the
        smallest eigenvalue of A_reg is at least reg_lambda (default 0.1).
        This handles cases where A has negative eigenvalues.

        Parameters
        ----------
        A : NDArray, shape (J, J)
            The matrix A = L - Γˢ

        Returns
        -------
        A_reg : NDArray, shape (J, J)
            Regularized matrix A + λI if regularize=True and ill-conditioned,
            else A unchanged
        """
        if not self.regularize:
            return A

        cond = np.linalg.cond(A)
        if cond > self.cond_threshold:
            J = A.shape[0]
            # Compute minimum eigenvalue to determine required regularization
            eigvals = np.linalg.eigvals(A).real
            lambda_min = eigvals.min()

            # Choose λ to ensure smallest eigenvalue is at least reg_lambda
            # If λ_min < 0, we need λ > |λ_min| + reg_lambda
            # If λ_min > 0 but small, we need λ = reg_lambda - λ_min (if positive)
            if lambda_min < self.reg_lambda:
                adaptive_lambda = self.reg_lambda - lambda_min + 1e-4  # small buffer
            else:
                adaptive_lambda = self.reg_lambda

            # Choose λ propoertionate to the average eigenvalue
            # adaptive_lambda = self.reg_lambda * np.trace(A) / J  

            A_reg = A + adaptive_lambda * np.eye(J)
            return A_reg
        return A

    def _get_omega_jacobian_price(
        self,
        structure: OmegaStructure,
        j_local: int
    ) -> NDArray:
        """
        Compute ∂ω_j/∂p for all prices in the market.

        Returns matrix of shape (d_omega, J) where column k gives ∂ω_j/∂p_k.

        Parameters
        ----------
        structure : OmegaStructure
            Omega structure for this market
        j_local : int
            Local index of product j within the market

        Returns
        -------
        jacobian : NDArray, shape (d_omega, J)
            Jacobian matrix ∂ω_j/∂p
        """
        J = structure.J
        K = structure.K
        d_omega = structure.diff_end

        jacobian = np.zeros((d_omega, J))

        if structure.price_in_diffs:
            # Own price is in own_char (first position after shares)
            own_p_idx = structure.own_char_start
            jacobian[own_p_idx, j_local] = 1.0

            # Price differences: (p_j - p_k) for k = 0, ..., J-1
            # Located at diff_start + k * (K+1) for each k
            for k in range(J):
                diff_price_idx = structure.diff_start + k * (K + 1)
                if k == j_local:
                    # ∂(p_j - p_j)/∂p_k = 0 for any k (self-diff)
                    pass
                else:
                    # ∂(p_j - p_k)/∂p_j = 1
                    jacobian[diff_price_idx, j_local] = 1.0
                    # ∂(p_j - p_k)/∂p_k = -1
                    jacobian[diff_price_idx, k] = -1.0
        else:
            # Own price is at own_price_index
            if structure.own_price_index is not None:
                jacobian[structure.own_price_index, j_local] = 1.0

        return jacobian

    def _get_omega_jacobian_share(
        self,
        structure: OmegaStructure,
        j_local: int
    ) -> NDArray:
        """
        Compute ∂ω_j/∂s for all shares in the market.

        Returns matrix of shape (d_omega, J) where column k gives ∂ω_j/∂s_k.
        Note: We don't include ∂/∂s_0 directly since s_0 = 1 - Σ s_k.

        Parameters
        ----------
        structure : OmegaStructure
            Omega structure for this market
        j_local : int
            Local index of product j within the market

        Returns
        -------
        jacobian : NDArray, shape (d_omega, J)
            Jacobian matrix ∂ω_j/∂s
        """
        J = structure.J
        d_omega = structure.diff_end

        jacobian = np.zeros((d_omega, J))

        # Shares in omega: ∂s_i/∂s_k = δ_{ik}
        if structure.share_representation == 'all':
            for k in range(J):
                jacobian[structure.share_start + k, k] = 1.0
            # Outside good share: ∂s_0/∂s_k = -1 (since s_0 = 1 - Σ s_i)
            if structure.s0_index is not None:
                for k in range(J):
                    jacobian[structure.s0_index, k] = -1.0
        elif structure.share_representation == 'inner':
            for k in range(J):
                jacobian[structure.share_start + k, k] = 1.0
        elif structure.share_representation == 'others':
            # Only other products' shares (excluding own)
            # Need to figure out the mapping
            col = 0
            for k in range(J):
                if k != j_local:
                    jacobian[structure.share_start + col, k] = 1.0
                    col += 1

        return jacobian

    def _compute_gamma_jacobians(
        self,
        gamma: GammaEstimatorWithDerivative,
        omega_t: NDArray,
        structure: OmegaStructure
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute Γᵖ and Γˢ matrices for a market using analytical derivatives.

        Γᵖ_{jk} = ∂γ(ω_j)/∂p_k = Σ_i (∂γ/∂ω_i)(∂ω_i/∂p_k)
        Γˢ_{jk} = ∂γ(ω_j)/∂s_k = Σ_i (∂γ/∂ω_i)(∂ω_i/∂s_k)

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator with predict_derivative method
        omega_t : NDArray, shape (J, d_omega)
            Omega values for all products in market t
        structure : OmegaStructure
            Omega structure for this market

        Returns
        -------
        Gamma_p : NDArray, shape (J, J)
            Price Jacobian matrix
        Gamma_s : NDArray, shape (J, J)
            Share Jacobian matrix
        """
        J = structure.J

        # Get ∂γ/∂ω for all products: shape (J, d_omega)
        grad_gamma = gamma.predict_derivative(omega_t)

        Gamma_p = np.zeros((J, J))
        Gamma_s = np.zeros((J, J))

        for j in range(J):
            # Get Jacobians for product j's omega
            jac_p = self._get_omega_jacobian_price(structure, j)  # (d_omega, J)
            jac_s = self._get_omega_jacobian_share(structure, j)  # (d_omega, J)

            # Chain rule: ∂γ_j/∂p_k = grad_gamma[j] @ jac_p[:, k]
            Gamma_p[j, :] = grad_gamma[j] @ jac_p
            Gamma_s[j, :] = grad_gamma[j] @ jac_s

        return Gamma_p, Gamma_s

    def _compute_all_basis_jacobians(
        self,
        featurizer: FeaturizerWithDerivative,
        omega_t: NDArray,
        structure: OmegaStructure
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute Zᵖ and Zˢ matrices for ALL basis functions efficiently.

        Returns 3D arrays where Z_p[k, j, m] = ∂d_k(ω_j)/∂p_m.

        Parameters
        ----------
        featurizer : FeaturizerWithDerivative
            Fitted featurizer with transform_derivative method
        omega_t : NDArray, shape (J, d_omega)
            Omega values for all products in market t
        structure : OmegaStructure
            Omega structure for this market

        Returns
        -------
        Z_p_all : NDArray, shape (n_basis, J, J)
            Price Jacobian matrices for all basis functions
        Z_s_all : NDArray, shape (n_basis, J, J)
            Share Jacobian matrices for all basis functions
        """
        J = structure.J
        d_omega = structure.diff_end

        # Get number of basis functions
        n_basis = featurizer.n_features_out_

        # Precompute all Jacobians for each product
        jac_p_all = np.zeros((J, d_omega, J))  # [j, i, m] = ∂ω_{j,i}/∂p_m
        jac_s_all = np.zeros((J, d_omega, J))  # [j, i, m] = ∂ω_{j,i}/∂s_m
        for j in range(J):
            jac_p_all[j] = self._get_omega_jacobian_price(structure, j)
            jac_s_all[j] = self._get_omega_jacobian_share(structure, j)

        # Precompute all basis derivatives w.r.t. each omega dimension
        # grad_basis[i, j, k] = ∂d_k(ω_j)/∂ω_i
        if hasattr(featurizer, 'transform_derivative_all'):
            grad_basis = featurizer.transform_derivative_all(omega_t)
        else:
            grad_basis = np.zeros((d_omega, J, n_basis))
            for i in range(d_omega):
                grad_basis[i] = featurizer.transform_derivative(omega_t, wrt=i)

        # Compute Z matrices for all basis functions via einsum
        # Want: Z_p_all[k, j, m] = sum_i grad_basis[i, j, k] * jac_p_all[j, i, m]
        Z_p_all = np.einsum('ijk,jil->kjl', grad_basis, jac_p_all)
        Z_s_all = np.einsum('ijk,jil->kjl', grad_basis, jac_s_all)

        return Z_p_all, Z_s_all

    def _compute_L_matrix(
        self,
        shares: NDArray,
        s0: float,
        min_share: float = 1e-10
    ) -> NDArray:
        """
        Compute the L matrix: L = D_s^{-1} + (1/s_0) 11'.

        Parameters
        ----------
        shares : NDArray, shape (J,)
            Inside good shares
        s0 : float
            Outside good share
        min_share : float, default=1e-10
            Minimum share value for numerical stability

        Returns
        -------
        L : NDArray, shape (J, J)
            The L matrix
        """
        J = len(shares)

        # Numerical stability: clip very small shares
        if np.any(shares < min_share):
            warnings.warn(
                f"Some shares are very small (min={shares.min():.2e}). "
                f"Clipping to {min_share:.0e} for numerical stability.",
                RuntimeWarning
            )
        shares_safe = np.maximum(shares, min_share)
        s0_safe = max(s0, min_share)

        # D_s^{-1} = diag(1/s_1, ..., 1/s_J)
        D_s_inv = np.diag(1.0 / shares_safe)

        # (1/s_0) 11'
        ones_outer = np.ones((J, J)) / s0_safe

        L = D_s_inv + ones_outer
        return L

    def compute_market_elasticity(
        self,
        gamma: GammaEstimatorWithDerivative,
        data: RawData,
        market_id,
        product_id: Optional[Union[int, str]] = None,
        return_components: bool = False,
        omega_t: Optional[NDArray] = None
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Compute own-price elasticity for a specific product in a specific market.

        ε_{jj} = (p_j/s_j) [(L - Γˢ)⁻¹ Γᵖ]_{jj}

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator with predict_derivative method
        data : RawData
            Raw panel data
        market_id : int or str
            Market identifier
        product_id : int or str, optional
            Product identifier. Uses self.product_id if None.
        return_components : bool, default=False
            If True, also return intermediate matrices (L, Gamma_p, Gamma_s, A)
        omega_t : NDArray, shape (J, d_omega), optional
            Pre-computed omega for this market. If None, transforms from raw data.

        Returns
        -------
        elasticity : float
            Own-price elasticity ε_{jj}
        components : Dict (if return_components=True)
            Dictionary with L, Gamma_p, Gamma_s, A, A_inv_Gamma_p
        """
        pid = product_id if product_id is not None else self.product_id
        if pid is None:
            raise ValueError("product_id must be specified")

        # Extract market data
        mask = data.get_market_mask(market_id)
        price_t = data.price[mask]
        shares_t = data.shares[mask]

        J = len(price_t)
        j_local = data.get_local_index(market_id, pid)

        # Compute outside good share
        s0 = 1.0 - shares_t.sum()
        if s0 <= 0:
            raise ValueError(f"Outside good share s0={s0} <= 0 in market {market_id}")

        # Use pre-computed omega or transform from raw data
        if omega_t is None:
            x2_t = data.x2[mask]
            market_ids_t = data.market_ids[mask]
            omega_t = self.transformer.transform(
                x2_t, market_ids_t, price=price_t, shares=shares_t
            )

        # Get omega structure
        structure = get_omega_structure(self.transformer, J)

        # Compute Jacobian matrices using analytical derivatives
        Gamma_p, Gamma_s = self._compute_gamma_jacobians(gamma, omega_t, structure)

        # Compute L matrix
        L = self._compute_L_matrix(shares_t, s0)

        # A = L - Γˢ
        A = L - Gamma_s

        # Apply Tikhonov regularization if A is ill-conditioned
        A_reg = self._regularize_matrix(A)

        # Solve A_reg x = Γᵖ for x = A_reg⁻¹ Γᵖ
        try:
            A_inv_Gamma_p = np.linalg.solve(A_reg, Gamma_p)
        except np.linalg.LinAlgError:
            # Matrix is still singular - use pseudoinverse with warning
            A_inv_Gamma_p = np.linalg.lstsq(A_reg, Gamma_p, rcond=None)[0]

        # Elasticity: ε_{jj} = (p_j/s_j) * [A⁻¹ Γᵖ]_{jj}
        p_j = price_t[j_local]
        s_j = shares_t[j_local]
        elasticity = (p_j / s_j) * A_inv_Gamma_p[j_local, j_local]

        if return_components:
            components = {
                'L': L,
                'Gamma_p': Gamma_p,
                'Gamma_s': Gamma_s,
                'A': A,
                'A_reg': A_reg,
                'A_inv_Gamma_p': A_inv_Gamma_p,
                'p_j': p_j,
                's_j': s_j,
                's0': s0
            }
            return elasticity, components

        return elasticity

    def compute_market_basis_gateaux(
        self,
        gamma: GammaEstimatorWithDerivative,
        featurizer: FeaturizerWithDerivative,
        data: RawData,
        market_id,
        product_id: Optional[Union[int, str]] = None,
        cond_threshold: Optional[float] = None,
        omega_t: Optional[NDArray] = None
    ) -> Optional[NDArray]:
        """
        Compute Gateaux derivative for ALL basis functions in a market using featurizer.

        D_γ ε_{jj}[d_k] = (p_j/s_j) [(A⁻¹ Zᵖ_k)_{jj} + (A⁻¹ Zˢ_k A⁻¹ Γᵖ)_{jj}]

        This is more efficient than calling compute_market_gateaux for each basis
        since it computes shared quantities (A, Γᵖ) only once.

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator (for computing A and Γᵖ)
        featurizer : FeaturizerWithDerivative
            Fitted featurizer with transform_derivative method
        data : RawData
            Raw panel data
        market_id : int or str
            Market identifier
        product_id : int or str, optional
            Product identifier. Uses self.product_id if None.
        cond_threshold : float, optional
            If provided, check cond(A) and return None if above threshold.
            This avoids a separate condition-check pass over markets.
        omega_t : NDArray, shape (J, d_omega), optional
            Pre-computed omega for this market. If None, transforms from raw data.

        Returns
        -------
        gateaux : NDArray, shape (n_basis,), or None
            Gateaux derivative D_γ ε_{jj}[d_k] for each basis function k.
            Returns None if cond_threshold is set and cond(A) exceeds it.
        """
        pid = product_id if product_id is not None else self.product_id
        if pid is None:
            raise ValueError("product_id must be specified")

        # Extract market data
        mask = data.get_market_mask(market_id)
        price_t = data.price[mask]
        shares_t = data.shares[mask]

        J = len(price_t)
        j_local = data.get_local_index(market_id, pid)

        # Compute outside good share
        s0 = 1.0 - shares_t.sum()

        # Use pre-computed omega or transform from raw data
        if omega_t is None:
            x2_t = data.x2[mask]
            market_ids_t = data.market_ids[mask]
            omega_t = self.transformer.transform(
                x2_t, market_ids_t, price=price_t, shares=shares_t
            )

        # Get omega structure
        structure = get_omega_structure(self.transformer, J)

        # Compute Jacobians for gamma (shared across all basis functions)
        Gamma_p, Gamma_s = self._compute_gamma_jacobians(gamma, omega_t, structure)

        # Compute L and A matrices
        L = self._compute_L_matrix(shares_t, s0)
        A = L - Gamma_s

        # Early exit if ill-conditioned (avoids separate pre-filtering pass)
        if cond_threshold is not None:
            if np.linalg.cond(A) > cond_threshold:
                return None

        # Apply regularization only if regularize_gateaux is True
        # NOTE: For ADMLIV, it's better to NOT regularize Gateaux derivatives
        # and instead exclude ill-conditioned markets from M computation
        if self.regularize_gateaux:
            A_work = self._regularize_matrix(A)
        else:
            A_work = A

        # Compute Jacobians for ALL basis functions
        Z_p_all, Z_s_all = self._compute_all_basis_jacobians(featurizer, omega_t, structure)
        n_basis = Z_p_all.shape[0]

        # Use LU decomposition for efficient multiple solves
        # This computes the factorization once and reuses it
        try:
            lu, piv = lu_factor(A_work)
            use_lu = True
        except np.linalg.LinAlgError:
            # Matrix is still singular - fall back to lstsq
            use_lu = False
            warnings.warn(
                f"Matrix A is singular in market {market_id}. Using least squares.",
                RuntimeWarning
            )

        # Solve A⁻¹ Γᵖ (shared across all basis functions)
        if use_lu:
            A_inv_Gamma_p = lu_solve((lu, piv), Gamma_p)
        else:
            A_inv_Gamma_p = np.linalg.lstsq(A_work, Gamma_p, rcond=None)[0]

        # Compute Gateaux for each basis using the precomputed LU factorization
        p_j = price_t[j_local]
        s_j = shares_t[j_local]
        prefactor = p_j / s_j

        gateaux = np.zeros(n_basis)
        for k in range(n_basis):
            # Solve A⁻¹ Zᵖ_k and A⁻¹ Zˢ_k using precomputed LU
            if use_lu:
                A_inv_Z_p_k = lu_solve((lu, piv), Z_p_all[k])
                A_inv_Z_s_k = lu_solve((lu, piv), Z_s_all[k])
            else:
                A_inv_Z_p_k = np.linalg.lstsq(A_work, Z_p_all[k], rcond=None)[0]
                A_inv_Z_s_k = np.linalg.lstsq(A_work, Z_s_all[k], rcond=None)[0]

            # Term 1: (A⁻¹ Zᵖ_k)_{jj}
            term1 = A_inv_Z_p_k[j_local, j_local]

            # Term 2: (A⁻¹ Zˢ_k A⁻¹ Γᵖ)_{jj}
            A_inv_Z_s_A_inv_Gamma_p = A_inv_Z_s_k @ A_inv_Gamma_p
            term2 = A_inv_Z_s_A_inv_Gamma_p[j_local, j_local]

            gateaux[k] = prefactor * (term1 + term2)

        return gateaux

    def compute_market_condition_number(
        self,
        gamma: GammaEstimatorWithDerivative,
        data: RawData,
        market_id,
        product_id: Optional[Union[int, str]] = None
    ) -> float:
        """
        Compute the condition number of A = L - Γˢ for a specific market.

        Used to identify ill-conditioned markets that should be excluded
        from Gateaux derivative computation in ADMLIV.

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator
        data : RawData
            Raw panel data
        market_id : int or str
            Market identifier
        product_id : int or str, optional
            Product identifier. Uses self.product_id if None.

        Returns
        -------
        cond : float
            Condition number of A matrix
        """
        pid = product_id if product_id is not None else self.product_id

        # Extract market data
        mask = data.get_market_mask(market_id)
        price_t = data.price[mask]
        shares_t = data.shares[mask]
        x2_t = data.x2[mask]
        market_ids_t = data.market_ids[mask]

        J = len(price_t)

        # Compute outside good share
        s0 = 1.0 - shares_t.sum()

        # Transform to omega
        omega_t = self.transformer.transform(
            x2_t, market_ids_t, price=price_t, shares=shares_t
        )

        # Get omega structure
        structure = get_omega_structure(self.transformer, J)

        # Compute Jacobian matrices
        _, Gamma_s = self._compute_gamma_jacobians(gamma, omega_t, structure)

        # Compute L and A matrices
        L = self._compute_L_matrix(shares_t, s0)
        A = L - Gamma_s

        return np.linalg.cond(A)

    def get_well_conditioned_markets(
        self,
        gamma: GammaEstimatorWithDerivative,
        data: RawData,
        product_id: Optional[Union[int, str]] = None,
        threshold: Optional[float] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Get markets where A is well-conditioned (below threshold).

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator
        data : RawData
            Raw panel data
        product_id : int or str, optional
            Product identifier. Uses self.product_id if None.
        threshold : float, optional
            Condition number threshold. Uses self.cond_threshold if None.

        Returns
        -------
        well_conditioned_markets : NDArray
            Market IDs where cond(A) <= threshold
        condition_numbers : NDArray
            Condition numbers for all markets (for diagnostics)
        """
        pid = product_id if product_id is not None else self.product_id
        if pid is None:
            raise ValueError("product_id must be specified")

        thresh = threshold if threshold is not None else self.cond_threshold
        markets = data.get_markets_with_product(pid)

        if len(markets) == 0:
            return np.array([]), np.array([])

        cond_numbers = np.array([
            self.compute_market_condition_number(gamma, data, t, pid)
            for t in markets
        ])

        mask = cond_numbers <= thresh
        return markets[mask], cond_numbers

    def compute(
        self,
        gamma: GammaEstimatorWithDerivative,
        data: RawData,
        product_id: Optional[Union[int, str]] = None,
        return_markets: bool = False,
        omega: Optional[NDArray] = None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Compute own-price elasticity for each market where product exists.

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator with predict_derivative method
        data : RawData
            Raw panel data
        product_id : int or str, optional
            Product identifier. Uses self.product_id if None.
        return_markets : bool, default=False
            If True, also return the market_ids.
        omega : NDArray, shape (n, d_omega), optional
            Pre-computed omega for the full panel (same row order as data).
            If provided, per-market omega is sliced from this array instead
            of re-transforming from raw data.

        Returns
        -------
        elasticities : NDArray, shape (T_j,)
            Own-price elasticity for each market where product j exists
        markets : NDArray (if return_markets=True)
            Market identifiers
        """
        pid = product_id if product_id is not None else self.product_id
        if pid is None:
            raise ValueError("product_id must be specified")

        markets = data.get_markets_with_product(pid)

        if len(markets) == 0:
            if return_markets:
                return np.array([]), np.array([])
            return np.array([])

        elasticities = np.array([
            self.compute_market_elasticity(
                gamma, data, t, pid,
                omega_t=omega[data.get_market_mask(t)] if omega is not None else None
            )
            for t in markets
        ])

        if return_markets:
            return elasticities, markets
        return elasticities

    def compute_basis_gateaux(
        self,
        gamma: GammaEstimatorWithDerivative,
        featurizer: FeaturizerWithDerivative,
        data: RawData,
        product_id: Optional[Union[int, str]] = None,
        return_markets: bool = False,
        exclude_ill_conditioned: bool = False,
        cond_threshold: Optional[float] = None,
        omega: Optional[NDArray] = None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Compute Gateaux derivative for each basis function using a featurizer.

        Used in PGMM for Riesz representer estimation. Uses analytical derivatives
        from the featurizer's transform_derivative method.

        When exclude_ill_conditioned=True, the condition check is performed inline
        during the Gateaux computation (single pass over markets) rather than in a
        separate pre-filtering pass.

        Parameters
        ----------
        gamma : GammaEstimatorWithDerivative
            Fitted gamma estimator with predict_derivative method
        featurizer : FeaturizerWithDerivative
            Fitted featurizer with transform_derivative method.
            Must support featurizer.transform(omega) -> (n, n_basis)
            and featurizer.transform_derivative(omega, wrt=i) -> (n, n_basis)
        data : RawData
            Raw panel data
        product_id : int or str, optional
            Product identifier. Uses self.product_id if None.
        return_markets : bool, default=False
            If True, also return the market_ids.
        exclude_ill_conditioned : bool, default=False
            If True, exclude markets where A matrix is ill-conditioned.
            This is recommended for ADMLIV to avoid biased Riesz estimation.
        cond_threshold : float, optional
            Condition number threshold for excluding markets.
            Uses self.cond_threshold if None.
        omega : NDArray, shape (n, d_omega), optional
            Pre-computed omega for the full panel (same row order as data).
            If provided, per-market omega is sliced from this array instead
            of re-transforming from raw data.

        Returns
        -------
        M : NDArray, shape (T_j, n_basis)
            M[t, k] = D_γ ε_{jj,t}[d_k] for market t and basis k
        markets : NDArray (if return_markets=True)
            Market identifiers (may be fewer than all markets if excluding)
        """
        pid = product_id if product_id is not None else self.product_id
        if pid is None:
            raise ValueError("product_id must be specified")

        all_markets = data.get_markets_with_product(pid)

        if len(all_markets) == 0:
            if return_markets:
                return np.array([]).reshape(0, 0), np.array([])
            return np.array([]).reshape(0, 0)

        thresh = cond_threshold if cond_threshold is not None else self.cond_threshold

        # Single pass: compute Gateaux and filter ill-conditioned inline
        M_list = []
        kept_markets = []
        for t in all_markets:
            omega_t = omega[data.get_market_mask(t)] if omega is not None else None
            gateaux_t = self.compute_market_basis_gateaux(
                gamma, featurizer, data, t, pid,
                cond_threshold=thresh if exclude_ill_conditioned else None,
                omega_t=omega_t,
            )
            if gateaux_t is not None:
                M_list.append(gateaux_t)
                kept_markets.append(t)

        if len(M_list) == 0:
            if return_markets:
                return np.array([]).reshape(0, 0), np.array([])
            return np.array([]).reshape(0, 0)

        M = np.vstack(M_list)
        markets = np.array(kept_markets)

        if return_markets:
            return M, markets
        return M
