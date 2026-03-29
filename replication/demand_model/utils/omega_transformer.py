# admliv/simulations/conditional_demand/utils/omega_transformer.py

"""
Omega Space Transformer for Nonparametric Demand Estimation.

This module provides utilities to transform raw panel data (prices, characteristics, 
shares) into the omega space with product characteristic differences.

The omega space consists of:
1. Shares of all products in the market s = (s_0, s_1, ..., s_J)  
2. Characteristic differences (including price): x_j - x_k for all k in the market (including the outside good)

Mathematical Background:
-----------------------
The demand function is:
    gamma(omega_jt) where omega_jt = (p_jt, x_jt^(2), s_{-j,t})

For computational purposes, we represent omega as:
    omega_j = [s_1, ..., s_J, (x_j - x_1), ..., (x_j - x_J)], 
    where x_j = [p_j, x_j^(2)]

This representation:
- Captures within-market variation
- Enables differentiation w.r.t. own price (first coordinate)
- Preserves the conditional demand interpretation
"""

from typing import Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class OmegaTransformer(BaseEstimator):
    """
    Transform raw panel data to omega space for demand estimation.
    
    Omega space construction depends on `price_in_diffs`:
    
    - If price_in_diffs=True (default):
      omega_jt = [shares_t, (p_j - p_k, x_j - x_k) for all k]
      Price enters through characteristic differences.
      
    - If price_in_diffs=False:
      omega_jt = [p_j, shares_t, (x_j - x_k) for all k]
      Price is prepended as the first element.
    
    Parameters
    ----------
    price_in_diffs : bool, default=True
        If True, price is included in characteristic differences (augment x 
        with price, then compute diffs). 
        If False, price is prepended as the first column of omega.
    include_prices : bool, default=True
        Whether to include prices in omega.
    include_shares : bool, default=True
        Whether to include market shares in omega.
    share_representation : str, default='all'
        How to represent shares:
        - 'all': Include all shares (s_0, s_1, ..., s_J), including the outside good
        - 'inner': Iclude all inner product share (s_1, ..., s_J)
        - 'others': Include only other pinnner roducts' shares (excluding own and outside good)
    use_jax : bool, default=False
        Whether to use JAX for computation (enables autodiff compatibility).
    
    Attributes
    ----------
    n_products_per_market_ : int or array
        Number of products per market (fitted)
    n_characteristics_ : int
        Number of characteristics K (fitted)
    n_markets_ : int
        Number of markets T (fitted)
    omega_dim_ : int
        Dimension of omega vector (fitted)
    price_index_ : int or None
        Index of own price in omega. If price_in_diffs=True, this is None
        (price is distributed across diffs). If price_in_diffs=False, this is 0.
    
    Examples
    --------
    >>> # Price in diffs 
    >>> transformer = OmegaTransformer(price_in_diffs=True)
    >>> omega = transformer.fit_transform(x, market_ids, price=price, shares=shares)

    >>> # Price as first column
    >>> transformer = OmegaTransformer(price_in_diffs=False)
    >>> omega = transformer.fit_transform(x, market_ids, price=price, shares=shares)
    >>> # omega[:, 0] is own price

    >>> # Without prices and shares (characteristics only for instruments)
    >>> transformer = OmegaTransformer(include_prices=False, include_shares=False)
    >>> omega = transformer.fit_transform(x, market_ids)
    """
    
    def __init__(
        self,
        price_in_diffs: bool = True,
        include_prices: bool = True,
        include_shares: bool = True,
        share_representation: str = 'all',
        use_jax: bool = False
    ):
        self.price_in_diffs = price_in_diffs
        self.include_prices = include_prices
        self.include_shares = include_shares
        self.share_representation = share_representation
        self.use_jax = use_jax and JAX_AVAILABLE
    
    def fit(
        self,
        x: NDArray[np.float64],
        market_ids: NDArray[np.int64],
        price: Optional[NDArray[np.float64]] = None,
        shares: Optional[NDArray[np.float64]] = None
    ) -> 'OmegaTransformer':
        """
        Fit the transformer to learn data dimensions.

        Parameters
        ----------
        x : NDArray, shape (n, K)
            Product characteristics (required)
        market_ids : NDArray, shape (n,)
            Market identifiers (required)
        price : NDArray, shape (n,) or (n, 1), optional
            Product prices (required if include_prices=True)
        shares : NDArray, shape (n,) or (n, 1), optional
            Market shares (required if include_shares=True)

        Returns
        -------
        self : OmegaTransformer

        Raises
        ------
        ValueError
            If required arguments are missing based on transformer configuration
        """
        # Validate inputs match configuration
        if self.include_prices and price is None:
            raise ValueError(
                "include_prices=True but price argument not provided. "
                "Either set include_prices=False or provide price array."
            )
        if self.include_shares and shares is None:
            raise ValueError(
                "include_shares=True but shares argument not provided. "
                "Either set include_shares=False or provide shares array."
            )

        # Ensure 2D
        x = np.atleast_2d(x.reshape(-1, 1)) if x.ndim == 1 else x
        if price is not None:
            price = np.atleast_2d(price.reshape(-1, 1)) if price.ndim == 1 else price
        if shares is not None:
            shares = np.atleast_2d(shares.reshape(-1, 1)) if shares.ndim == 1 else shares
        
        self.n_characteristics_ = x.shape[1]
        self.unique_markets_ = np.unique(market_ids)
        self.n_markets_ = len(self.unique_markets_)
        
        # Products per market
        products_per_market = [np.sum(market_ids == t) for t in self.unique_markets_]
        if len(set(products_per_market)) == 1:
            self.n_products_per_market_ = products_per_market[0]
            self.balanced_panel_ = True
            self.J_max_ = self.n_products_per_market_
        else:
            self.n_products_per_market_ = np.array(products_per_market)
            self.balanced_panel_ = False
            self.J_max_ = max(products_per_market)

        # Compute omega dimension using J_max
        J = self.J_max_
        K = self.n_characteristics_
        
        # Dimension calculation
        dim = 0
        
        # Shares
        if self.include_shares:
            if self.share_representation == 'all':
                dim += J + 1
            elif self.share_representation == 'inner':
                dim += J 
            elif self.share_representation == 'others':
                dim += J - 1
        
        # Own characteristics and characteristic differences
        if self.include_prices:
            if self.price_in_diffs:
                # Price enters through diffs: augmented characteristics (K+1)
                # own_chars (K+1) + diffs (J * (K+1)) = (J+1) * (K+1)
                dim += (J + 1) * (K + 1)
                self.price_index_ = None  # Price is distributed across diffs
            else:
                # Price prepended separately
                dim += 1  # Own price
                dim += (J + 1) * K  # own_chars (K) + diffs (J * K)
                self.price_index_ = 0  # Price is first column
        elif K > 0:
            # No prices: own_chars (K) + diffs (J * K) = (J+1) * K
            dim += (J + 1) * K
            self.price_index_ = None

        self.omega_dim_ = dim
        
        return self
    
    def transform(
        self,
        x: NDArray[np.float64],
        market_ids: NDArray[np.int64],
        price: Optional[NDArray[np.float64]] = None,
        shares: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Transform raw data to omega space.

        Parameters
        ----------
        x : NDArray, shape (n, K)
            Product characteristics (required)
        market_ids : NDArray, shape (n,)
            Market identifiers (required)
        price : NDArray, shape (n,) or (n, 1), optional
            Product prices (required if include_prices=True)
        shares : NDArray, shape (n,) or (n, 1), optional
            Market shares (required if include_shares=True)

        Returns
        -------
        omega : NDArray, shape (n, omega_dim)
            Transformed data in omega space

        Raises
        ------
        ValueError
            If required arguments are missing based on transformer configuration
        """
        # Validate inputs match configuration
        if self.include_prices and price is None:
            raise ValueError(
                "include_prices=True but price argument not provided. "
                "Either set include_prices=False or provide price array."
            )
        if self.include_shares and shares is None:
            raise ValueError(
                "include_shares=True but shares argument not provided. "
                "Either set include_shares=False or provide shares array."
            )

        # Ensure proper shapes
        if price is not None:
            price = price.reshape(-1, 1) if price.ndim == 1 else price
        if shares is not None:
            shares = shares.reshape(-1, 1) if shares.ndim == 1 else shares
        
        # Pre-allocate output array and fill in-place to preserve input row order
        n = len(x)
        omega = np.zeros((n, self.omega_dim_))

        unique_markets = np.unique(market_ids)
        for t in unique_markets:
            mask = market_ids == t
            omega_t = self._transform_market(
                x[mask],
                price_t=price[mask] if price is not None else None,
                shares_t=shares[mask] if shares is not None else None
            )
            omega[mask] = omega_t

        return omega
    
    def _transform_market(
        self,
        x_t: NDArray,
        price_t: Optional[NDArray] = None,
        shares_t: Optional[NDArray] = None
    ) -> NDArray:
        """
        Transform data for a single market.

        Returns omega_t of shape (J_t, omega_dim_) where omega_dim_ is based
        on J_max. For markets with J_t < J_max, columns corresponding to
        absent rivals are zero-padded.
        """
        J_t = len(x_t)
        J_max = self.J_max_
        K = self.n_characteristics_

        components = []

        # 1. Price handling depends on price_in_diffs
        if self.include_prices and not self.price_in_diffs and price_t is not None:
            # Price prepended as first column
            components.append(price_t.reshape(-1, 1))

        # 2. Shares (padded to J_max width)
        if self.include_shares and shares_t is not None:
            if self.share_representation == 'all':
                shares_mat = np.zeros((J_t, J_max))
                shares_mat[:, :J_t] = np.tile(shares_t.T, (J_t, 1))
                components.append(shares_mat)
                # Outside good share
                s0 = 1.0 - shares_t.sum()
                components.append(np.full((J_t, 1), s0))
            elif self.share_representation == 'inner':
                shares_mat = np.zeros((J_t, J_max))
                shares_mat[:, :J_t] = np.tile(shares_t.T, (J_t, 1))
                components.append(shares_mat)
            elif self.share_representation == 'others':
                shares_full = np.tile(shares_t.T, (J_t, 1))
                others_mask = ~np.eye(J_t, dtype=bool)
                shares_others_local = shares_full[others_mask].reshape(J_t, J_t - 1)
                shares_others = np.zeros((J_t, J_max - 1))
                shares_others[:, :J_t - 1] = shares_others_local
                components.append(shares_others)

        # 3. Characteristic differences (padded to J_max width)
        if self.include_prices and self.price_in_diffs and price_t is not None:
            x_aug_t = np.c_[price_t.reshape(-1, 1), x_t]
            K_aug = K + 1
            # Own characteristics (not padded -- always K_aug wide)
            components.append(x_aug_t)
            # Diffs: (J_t, J_t * K_aug) -> pad to (J_t, J_max * K_aug)
            char_diffs_local = self._compute_char_differences(x_aug_t)
            char_diffs = np.zeros((J_t, J_max * K_aug))
            char_diffs[:, :J_t * K_aug] = char_diffs_local
            components.append(char_diffs)
        elif K > 0:
            components.append(x_t)
            char_diffs_local = self._compute_char_differences(x_t)
            char_diffs = np.zeros((J_t, J_max * K))
            char_diffs[:, :J_t * K] = char_diffs_local
            components.append(char_diffs)

        omega_t = np.hstack(components)
        return omega_t
    
    def _compute_char_differences(self, x_t: NDArray) -> NDArray:
        """
        Compute characteristic differences within a market.
        
        For each product j, computes (x_j - x_k) for all k in the market.
        
        Parameters
        ----------
        x_t : NDArray, shape (J_t, K)
            Characteristics for products in market t
        
        Returns
        -------
        diffs : NDArray, shape (J_t, J_t * K)
            Flattened characteristic differences
        """
        J_t, K = x_t.shape
        
        # Compute all pairwise differences
        # x_j - x_k for all (j, k) pairs
        # Result: (J_t, J_t, K) array where [j, k, :] = x_j - x_k
        diffs_3d = x_t[:, np.newaxis, :] - x_t[np.newaxis, :, :]
        
        # Reshape to (J_t, J_t * K)
        diffs = diffs_3d.reshape(J_t, J_t * K)
        
        return diffs
    
    def fit_transform(
        self,
        x: NDArray[np.float64],
        market_ids: NDArray[np.int64],
        price: Optional[NDArray[np.float64]] = None,
        shares: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(x, market_ids, price=price, shares=shares).transform(
            x, market_ids, price=price, shares=shares
        )
    
    def get_price_index(self) -> Optional[int]:
        """
        Return the index of own price in omega.
        
        Returns None if price_in_diffs=True (price distributed across diffs).
        Returns 0 if price_in_diffs=False (price is first column).
        """
        return self.price_index_
    
    def get_feature_names(self) -> list:
        """Return descriptive names for omega features."""
        names = []
        J = self.J_max_
        K = self.n_characteristics_
        
        # Price (if not in diffs)
        if not self.price_in_diffs:
            names.append('price_own')
        
        # Shares
        if self.include_shares:
            if self.share_representation == 'all':
                names.extend([f's_{k}' for k in range(J)])
            elif self.share_representation == 'others':
                names.extend([f's_other_{k}' for k in range(J - 1)])
            elif self.share_representation == 'outside':
                names.append('s_0')
        
        # Characteristic differences
        if self.price_in_diffs:
            # Augmented: price + K characteristics
            for k in range(J):
                names.append(f'p_diff_{k}')
                for d in range(K):
                    names.append(f'x_{d}_diff_{k}')
        else:
            # Just K characteristics
            for k in range(J):
                for d in range(K):
                    names.append(f'x_{d}_diff_{k}')
        
        return names


class OmegaTransformerJAX(OmegaTransformer):
    """
    JAX-compatible version of OmegaTransformer.
    
    This version uses JAX arrays and operations, enabling automatic
    differentiation through the transformation.
    """
    
    def __init__(self, **kwargs):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for OmegaTransformerJAX")
        kwargs['use_jax'] = True
        super().__init__(**kwargs)
    
    def _compute_char_differences(self, x_t: NDArray) -> NDArray:
        """JAX-compatible characteristic differences."""
        x_t = jnp.array(x_t)
        J_t, K = x_t.shape
        
        # Efficient vectorized computation
        # tile: repeat x_t J_t times horizontally
        # repeat: repeat each row J_t times
        tiled = jnp.tile(x_t.T, J_t).T
        repeated = jnp.repeat(x_t, J_t, axis=0)
        
        diffs_flat = tiled - repeated  # (J_t^2, K)
        diffs = diffs_flat.reshape(J_t, J_t * K)
        
        return np.array(diffs)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_omega(
    x: NDArray,
    market_ids: NDArray,
    price: Optional[NDArray] = None,
    shares: Optional[NDArray] = None,
    price_in_diffs: bool = True,
    include_prices: bool = True,
    include_shares: bool = True
) -> Tuple[NDArray, Optional[int]]:
    """
    Convenience function to create omega from raw data.

    Parameters
    ----------
    x : NDArray
        Product characteristics (required)
    market_ids : NDArray
        Market identifiers (required)
    price : NDArray, optional
        Product prices (required if include_prices=True)
    shares : NDArray, optional
        Market shares (required if include_shares=True)
    price_in_diffs : bool, default=True
        If True, price enters through characteristic differences.
        If False, price is prepended as first column.
    include_prices : bool, default=True
        Whether to include prices in omega
    include_shares : bool, default=True
        Whether to include shares in omega

    Returns
    -------
    omega : NDArray
        Transformed omega space data
    price_index : int or None
        Index of price in omega (0 if price_in_diffs=False, None otherwise)
    """
    transformer = OmegaTransformer(
        price_in_diffs=price_in_diffs,
        include_prices=include_prices,
        include_shares=include_shares
    )
    omega = transformer.fit_transform(x, market_ids, price=price, shares=shares)
    price_index = transformer.get_price_index()

    return omega, price_index


def omega_to_dict(
    omega: NDArray,
    market_ids: NDArray,
    price_index: int = 0
) -> Dict[str, NDArray]:
    """
    Package omega and metadata into a dictionary for moment computation.
    
    Returns
    -------
    W : Dict
        Dictionary with keys: 'omega', 'X' (alias), 'market_ids', 'price_index'
    """
    return {
        'omega': omega,
        'X': omega,  # Alias for compatibility
        'market_ids': market_ids,
        'price_index': price_index
    }