# simulations/demand_model/utils/em_basis_featurizer.py

"""
Empirical Moment Basis Featurizer for Nonparametric Demand Estimation.

Compresses rival information into share-weighted polynomial moments of the
empirical distribution, then applies a flexible second-stage expansion.

Mathematical Specification
--------------------------
For product j, multi-index (a_0, a_1, ..., a_{D-1}):

    m^j = sum_{k != j} s_k^{a_0} * prod_{d} Delta_{jkd}^{a_d}

where s_k = rival share, Delta_{jkd} = x_{jd} - x_{kd}.

Multi-index constraints (B_n):
    sum(a) = n, n >= 2, a_0 > 0, a_0 < n  (when shares present).

First-order extension:
    When min_moment_order=1, include ALL order-1 partitions unfiltered.

Two-layer pipeline:
    raw moments -> moment_featurizer (default: PolyTransform(degree=2))
"""

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone

from .own_price_elasticity import OmegaStructure
from admliv.utils.featurizers import PolyTransform


class EMBasisFeaturizer(BaseEstimator, TransformerMixin):
    """
    Empirical moment basis featurizer with analytical derivatives.

    Two-layer pipeline:
        1. Compute raw share-weighted moments of characteristic differences
        2. Apply a second-stage featurizer (default: PolyTransform) to raw moments

    Parameters
    ----------
    max_moment_order : int, default=2
        Maximum order of empirical moments to compute.
    min_moment_order : int, default=2
        Minimum order. Set to 1 to include first-order moments (unfiltered).
    include_bias : bool, default=True
        Whether to prepend a constant column.
    moment_featurizer : estimator or None
        Second-stage featurizer applied to raw moments. Must have
        transform_derivative(X, wrt) method. Default: PolyTransform(degree=2).
    """

    def __init__(
        self,
        max_moment_order: int = 2,
        min_moment_order: int = 2,
        include_bias: bool = True,
        moment_featurizer=None,
        omega_structure: Optional[OmegaStructure] = None,
    ):
        self.max_moment_order = max_moment_order
        self.min_moment_order = min_moment_order
        self.include_bias = include_bias
        self.moment_featurizer = moment_featurizer
        self.omega_structure = omega_structure

    def fit(self, X, y=None, *, omega_structure: Optional[OmegaStructure] = None) -> 'EMBasisFeaturizer':
        """
        Extract layout from OmegaStructure and set up multi-indices.

        Parameters
        ----------
        X : array-like
            Ignored (for API compatibility).
        y : ignored
        omega_structure : OmegaStructure, optional
            Describes the column layout of omega vectors.
            If None, uses the omega_structure from __init__.

        Returns
        -------
        self
        """
        if omega_structure is None:
            omega_structure = self.omega_structure
        if omega_structure is None:
            raise ValueError("omega_structure must be provided either in __init__ or fit()")
        self.J_ = omega_structure.J
        self.J_max_ = omega_structure.J_max
        # Compute D from diff block size (uses J_max, since diff block is padded)
        self.D_ = (omega_structure.diff_end - omega_structure.diff_start) // omega_structure.J_max
        self.share_start_ = omega_structure.share_start
        self.share_end_ = omega_structure.share_end
        self.diff_start_ = omega_structure.diff_start
        self.has_shares_ = (omega_structure.share_end > omega_structure.share_start)

        # Generate multi-indices
        self.multi_indices_ = self._generate_multi_indices()
        self.n_raw_moments_ = len(self.multi_indices_)

        if self.n_raw_moments_ == 0:
            raise ValueError(
                f"No valid multi-indices generated for orders "
                f"[{self.min_moment_order}, {self.max_moment_order}] with D={self.D_}"
            )

        # Setup moment featurizer
        if self.moment_featurizer is None:
            self.moment_featurizer_ = PolyTransform(degree=2, include_bias=False)
        else:
            self.moment_featurizer_ = clone(self.moment_featurizer)

        if not hasattr(self.moment_featurizer_, 'transform_derivative'):
            raise ValueError("moment_featurizer must have transform_derivative method")

        # Fit on dummy data to learn output dimensions
        dummy = np.zeros((1, self.n_raw_moments_))
        self.moment_featurizer_.fit(dummy)

        n_poly = self.moment_featurizer_.n_features_out_
        self.n_features_out_ = n_poly + (1 if self.include_bias else 0)

        return self

    def transform(self, X: NDArray) -> NDArray:
        """
        Transform omega to EM basis features.

        Parameters
        ----------
        X : NDArray, shape (n, d_omega)
            Omega vectors (typically one market: (J, d_omega)).

        Returns
        -------
        basis : NDArray, shape (n, n_features_out_)
        """
        raw_moments = self._compute_raw_moments(X)
        basis = self.moment_featurizer_.transform(raw_moments)

        if self.include_bias:
            basis = np.c_[np.ones(X.shape[0]), basis]

        return basis

    def transform_derivative(self, X: NDArray, wrt: int = 0) -> NDArray:
        """
        Compute derivative of basis features w.r.t. X[:, wrt].

        Uses chain rule:
            d(basis)/d(omega_i) = sum_l d(poly)/d(m_l) * d(m_l)/d(omega_i)

        Parameters
        ----------
        X : NDArray, shape (n, d_omega)
        wrt : int
            Column index in omega to differentiate w.r.t.

        Returns
        -------
        d_basis : NDArray, shape (n, n_features_out_)
        """
        n = X.shape[0]
        raw_moments = self._compute_raw_moments(X)
        d_moments = self._compute_raw_moment_derivatives(X, wrt)

        # Chain through moment_featurizer
        n_poly = self.moment_featurizer_.n_features_out_
        d_basis = np.zeros((n, n_poly))

        for l in range(self.n_raw_moments_):
            dpoly_dl = self.moment_featurizer_.transform_derivative(raw_moments, wrt=l)
            d_basis += dpoly_dl * d_moments[:, l:l+1]

        if self.include_bias:
            d_basis = np.c_[np.zeros(n), d_basis]

        return d_basis

    def transform_derivative_all(self, X: NDArray) -> NDArray:
        """
        Compute derivatives of basis features w.r.t. ALL omega dimensions at once.

        Much more efficient than calling transform_derivative in a loop because:
        - Raw moments are computed once (not d_omega times)
        - PolyTransform derivatives are computed once per raw moment (not d_omega times)

        Parameters
        ----------
        X : NDArray, shape (n, d_omega)

        Returns
        -------
        d_basis_all : NDArray, shape (d_omega, n, n_features_out_)
            d_basis_all[i, :, :] = d(basis)/d(omega_i)
        """
        n, d_omega = X.shape
        n_poly = self.moment_featurizer_.n_features_out_
        bias_offset = 1 if self.include_bias else 0

        # Compute raw moments ONCE
        raw_moments = self._compute_raw_moments(X)

        # Precompute ALL poly derivatives w.r.t. each raw moment ONCE
        # dpoly_all[l, n, p] = d(poly_p)/d(m_l) evaluated at raw_moments
        dpoly_all = np.zeros((self.n_raw_moments_, n, n_poly))
        for l in range(self.n_raw_moments_):
            dpoly_all[l] = self.moment_featurizer_.transform_derivative(raw_moments, wrt=l)

        # For each omega dimension, compute raw moment derivatives and chain rule
        result = np.zeros((d_omega, n, self.n_features_out_))
        for i in range(d_omega):
            d_moments_i = self._compute_raw_moment_derivatives(X, wrt=i)
            # Chain rule: sum_l dpoly_all[l,n,p] * d_moments_i[n,l] -> [n, p]
            result[i, :, bias_offset:] = np.einsum('lnp,nl->np', dpoly_all, d_moments_i)

        return result

    # ------------------------------------------------------------------ #
    #  Internal methods                                                    #
    # ------------------------------------------------------------------ #

    def _generate_multi_indices(self) -> List[Tuple[int, ...]]:
        """
        Generate all moment multi-index tuples.

        With shares: tuples of length D+1 where index 0 is the share exponent.
        Without shares: tuples of length D.

        For order >= 2 with shares: a_0 > 0 AND a_0 < l.
        For order == 1: no filtering (all partitions included).
        """
        D = self.D_
        multi_indices = []

        for l in range(self.min_moment_order, self.max_moment_order + 1):
            n_vars = D + 1 if self.has_shares_ else D
            partitions = self._partitions_of_order(l, n_vars)

            if self.has_shares_ and l >= 2:
                partitions = [p for p in partitions if p[0] > 0 and p[0] < l]

            multi_indices.extend(partitions)

        return multi_indices

    @staticmethod
    def _partitions_of_order(order: int, n_vars: int) -> List[Tuple[int, ...]]:
        """Generate all non-negative integer tuples of length n_vars summing to order."""
        if n_vars == 1:
            return [(order,)]
        partitions = []
        for first in range(order + 1):
            for rest in EMBasisFeaturizer._partitions_of_order(order - first, n_vars - 1):
                partitions.append((first,) + rest)
        return partitions

    def _parse_omega(self, X: NDArray):
        """
        Extract shares and diffs from omega.

        Returns
        -------
        shares : NDArray, shape (n, J_max) or None
            Share of each inside good (padded to J_max width).
        diffs_3d : NDArray, shape (n, J_max, D)
            Characteristic differences reshaped to 3D (padded to J_max).
        """
        n = X.shape[0]
        J_max = self.J_max_
        D = self.D_

        if self.has_shares_:
            shares = X[:, self.share_start_:self.share_start_ + J_max]
        else:
            shares = None

        diffs_flat = X[:, self.diff_start_:self.diff_start_ + J_max * D]
        diffs_3d = diffs_flat.reshape(n, J_max, D)

        return shares, diffs_3d

    def _compute_raw_moments(self, X: NDArray) -> NDArray:
        """
        Compute raw empirical moments for each row.

        Called per-market by own_price_elasticity.py: X has shape (J_t, d_omega)
        where row j IS product j. Rival k's diff for product j is
        diffs_3d[j, k, :]. Self-rival k=j has all-zero diffs.

        Parameters
        ----------
        X : NDArray, shape (n, d_omega)

        Returns
        -------
        moments : NDArray, shape (n, n_raw_moments_)
        """
        n = X.shape[0]
        J_max = self.J_max_

        shares, diffs_3d = self._parse_omega(X)

        # Build exclude mask: positions to EXCLUDE from sums.
        # When called per-market (n <= J_max), n == J_t (actual products).
        # Exclude: (1) self-rival k==j, (2) padded positions k >= J_t.
        if n <= J_max:
            J_t = n
            exclude_mask = np.zeros((J_t, J_max), dtype=bool)
            # Self-rival: k == j
            for j in range(J_t):
                exclude_mask[j, j] = True
            # Padded positions: k >= J_t
            if J_t < J_max:
                exclude_mask[:, J_t:] = True
        else:
            exclude_mask = None

        moments = np.zeros((n, self.n_raw_moments_))

        for m_idx, mi in enumerate(self.multi_indices_):
            if self.has_shares_:
                a_0 = mi[0]
                char_exps = mi[1:]
            else:
                a_0 = 0
                char_exps = mi

            # Start with share factor: (n, J_max)
            if a_0 > 0 and shares is not None:
                term = shares ** a_0
            else:
                term = np.ones((n, J_max))

            # Multiply by char diff powers
            for d, a_d in enumerate(char_exps):
                if a_d > 0:
                    term = term * (diffs_3d[:, :, d] ** a_d)

            # Mask excluded positions (self-rival + padding)
            if exclude_mask is not None:
                term = term.copy()
                term[exclude_mask] = 0.0

            # Use sums (not averages) to avoid scale attenuation in PGMM
            moments[:, m_idx] = term.sum(axis=1)

        return moments

    def _compute_raw_moment_derivatives(self, X: NDArray, wrt: int) -> NDArray:
        """
        Compute derivative of raw moments w.r.t. X[:, wrt].

        Determines which block wrt falls in (shares, diffs, or neither)
        and dispatches accordingly.

        Parameters
        ----------
        X : NDArray, shape (n, d_omega)
        wrt : int
            Column index in omega.

        Returns
        -------
        d_moments : NDArray, shape (n, n_raw_moments_)
        """
        n = X.shape[0]
        J_max = self.J_max_
        D = self.D_

        # Share block (J_max wide)
        if self.has_shares_ and self.share_start_ <= wrt < self.share_start_ + J_max:
            k_prime = wrt - self.share_start_
            return self._share_derivative(X, k_prime)

        # Diff block (J_max * D wide)
        diff_end = self.diff_start_ + J_max * D
        if self.diff_start_ <= wrt < diff_end:
            offset = wrt - self.diff_start_
            k_prime = offset // D
            d_prime = offset % D
            return self._diff_derivative(X, k_prime, d_prime)

        # Own price/chars or s0 — moments don't depend on these
        return np.zeros((n, self.n_raw_moments_))

    def _share_derivative(self, X: NDArray, k_prime: int) -> NDArray:
        """
        Derivative of raw moments w.r.t. share of rival k'.

        dm/d(s_{k'}) = a_0 * s_{k'}^{a_0-1} * prod_d Delta_{j,k',d}^{a_d}

        Zero when a_0 == 0 or k' == j (self-rival) or k' is padded.
        """
        n = X.shape[0]
        J_max = self.J_max_

        # If k_prime is a padded rival (k' >= J_t), derivative is zero
        if n <= J_max and k_prime >= n:
            return np.zeros((n, self.n_raw_moments_))

        shares, diffs_3d = self._parse_omega(X)

        # Self-mask: row k_prime is the self-rival
        is_self = np.zeros(n, dtype=bool)
        if n <= J_max and k_prime < n:
            is_self[k_prime] = True

        d_moments = np.zeros((n, self.n_raw_moments_))

        for m_idx, mi in enumerate(self.multi_indices_):
            if not self.has_shares_:
                continue

            a_0 = mi[0]
            char_exps = mi[1:]

            if a_0 == 0:
                continue

            # a_0 * s_{k'}^{a_0 - 1}
            if a_0 == 1:
                deriv = np.ones(n)
            else:
                deriv = a_0 * (shares[:, k_prime] ** (a_0 - 1))

            # prod_d Delta_{j, k', d}^{a_d}
            for d, a_d in enumerate(char_exps):
                if a_d > 0:
                    deriv = deriv * (diffs_3d[:, k_prime, d] ** a_d)

            deriv[is_self] = 0.0
            d_moments[:, m_idx] = deriv

        return d_moments

    def _diff_derivative(self, X: NDArray, k_prime: int, d_prime: int) -> NDArray:
        """
        Derivative of raw moments w.r.t. Delta_{j, k', d'}.

        dm/d(Delta_{j,k',d'}) = a_{d'} * s_{k'}^{a_0}
                                * Delta_{j,k',d'}^{a_{d'}-1}
                                * prod_{d!=d'} Delta_{j,k',d}^{a_d}

        Zero when a_{d'} == 0 or k' == j or k' is padded.
        """
        n = X.shape[0]
        J_max = self.J_max_

        # If k_prime is a padded rival (k' >= J_t), derivative is zero
        if n <= J_max and k_prime >= n:
            return np.zeros((n, self.n_raw_moments_))

        shares, diffs_3d = self._parse_omega(X)

        is_self = np.zeros(n, dtype=bool)
        if n <= J_max and k_prime < n:
            is_self[k_prime] = True

        d_moments = np.zeros((n, self.n_raw_moments_))

        for m_idx, mi in enumerate(self.multi_indices_):
            if self.has_shares_:
                a_0 = mi[0]
                char_exps = mi[1:]
            else:
                a_0 = 0
                char_exps = mi

            a_dprime = char_exps[d_prime]
            if a_dprime == 0:
                continue

            # Share factor: s_{k'}^{a_0}
            if a_0 > 0 and shares is not None:
                deriv = shares[:, k_prime] ** a_0
            else:
                deriv = np.ones(n)

            # a_{d'} * Delta_{k', d'}^{a_{d'} - 1}
            deriv = deriv * a_dprime
            if a_dprime > 1:
                deriv = deriv * (diffs_3d[:, k_prime, d_prime] ** (a_dprime - 1))

            # prod_{d != d'} Delta_{k', d}^{a_d}
            for d, a_d in enumerate(char_exps):
                if d != d_prime and a_d > 0:
                    deriv = deriv * (diffs_3d[:, k_prime, d] ** a_d)

            deriv[is_self] = 0.0
            d_moments[:, m_idx] = deriv

        return d_moments
