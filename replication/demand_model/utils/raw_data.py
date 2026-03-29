# simulations/demand_model/utils/raw_data.py

"""
Raw Panel Data Container for Demand Estimation.

This module provides the RawData class, a container for panel data used in
semiparametric demand estimation with efficient indexing and caching.
"""

from typing import Optional, Dict
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class RawData:
    """
    Container for raw panel data with efficient indexing.

    The data structure follows the estimation equation:
        log(s_jt / s_0t) = x_jt^(1) + γ(ω_jt) + ξ_jt

    where:
    - x^(1) is the special regressor to form the linear index x_jt^(1) + ξ_jt
    - x^(2) are characteristics used in omega construction
    - w are optional instruments that go through omega_iv_transformer
      (e.g., cost shifters that should be differenced like characteristics)
    - w_external are optional instruments appended after omega_iv transformation
      (e.g., Hausman IVs that should NOT be differenced)

    Attributes
    ----------
    price : NDArray, shape (n,)
        Product prices
    x1 : NDArray, shape (n,)
        Special regressor x^(1) used in constructing the linear index x_jt^(1) + ξ_jt
    x2 : NDArray, shape (n, K2)
        Characteristics used in omega construction
    shares : NDArray, shape (n,)
        Market shares
    market_ids : NDArray, shape (n,)
        Market identifiers
    w : NDArray, shape (n, K_w), optional
        Instruments that go through omega_iv_transformer.
        These are differenced like characteristics in omega_iv construction.
    w_external : NDArray, shape (n, K_we), optional
        Instruments appended after omega_iv transformation.
        These bypass the transformer and are concatenated to omega_iv directly.
    product_ids : NDArray, shape (n,), optional
        Product identifiers (e.g., brand names, UPCs).
        Required for unbalanced panels where products vary across markets.
        If None, assumes balanced panel with products indexed 0, 1, ..., J-1
        within each market.
    """
    price: NDArray
    x1: NDArray
    x2: NDArray
    shares: NDArray
    market_ids: NDArray
    w: Optional[NDArray] = None
    w_external: Optional[NDArray] = None
    product_ids: Optional[NDArray] = None

    # Cached indices (computed lazily)
    _unique_markets: NDArray = field(default=None, repr=False, compare=False)
    _market_masks: Dict = field(default_factory=dict, repr=False, compare=False)
    _market_slices: Dict = field(default_factory=dict, repr=False, compare=False)
    _product_market_map: Dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        """Ensure consistent shapes and validate data."""
        self.price = np.asarray(self.price).flatten()
        self.shares = np.asarray(self.shares).flatten()
        self.market_ids = np.asarray(self.market_ids).flatten()

        # Handle x1 shape - ensure (n,)
        self.x1 = np.asarray(self.x1).flatten()

        # Handle x2 shape - ensure (n, K2)
        self.x2 = np.asarray(self.x2)
        if self.x2.ndim == 1:
            self.x2 = self.x2[:, np.newaxis]

        # Handle w shape if provided
        if self.w is not None:
            self.w = np.asarray(self.w)
            if self.w.ndim == 1:
                self.w = self.w[:, np.newaxis]

        # Handle w_external shape if provided
        if self.w_external is not None:
            self.w_external = np.asarray(self.w_external)
            if self.w_external.ndim == 1:
                self.w_external = self.w_external[:, np.newaxis]

        if self.product_ids is not None:
            self.product_ids = np.asarray(self.product_ids).flatten()

        # Validate shapes
        n = len(self.price)
        if len(self.shares) != n:
            raise ValueError(f"shares length {len(self.shares)} != price length {n}")
        if len(self.market_ids) != n:
            raise ValueError(f"market_ids length {len(self.market_ids)} != price length {n}")
        if len(self.x1) != n:
            raise ValueError(f"x1 length {len(self.x1)} != price length {n}")
        if self.x2.shape[0] != n:
            raise ValueError(f"x2 shape {self.x2.shape} incompatible with n={n}")
        if self.w is not None and self.w.shape[0] != n:
            raise ValueError(f"w shape {self.w.shape} incompatible with n={n}")
        if self.w_external is not None and self.w_external.shape[0] != n:
            raise ValueError(f"w_external shape {self.w_external.shape} incompatible with n={n}")
        if self.product_ids is not None and len(self.product_ids) != n:
            raise ValueError(f"product_ids length {len(self.product_ids)} != price length {n}")

        # Reset caches
        self._unique_markets = None
        self._market_masks = {}
        self._market_slices = {}
        self._product_market_map = {}

    @property
    def has_instruments(self) -> bool:
        """Whether transformer instruments w are available."""
        return self.w is not None

    @property
    def has_external_instruments(self) -> bool:
        """Whether external instruments w_external (appended after transform) are available."""
        return self.w_external is not None

    @property
    def n_obs(self) -> int:
        return len(self.price)

    @property
    def n_characteristics(self) -> int:
        """Number of characteristics in x2."""
        return self.x2.shape[1]

    @property
    def n_markets(self) -> int:
        return len(self.unique_markets)

    @property
    def unique_markets(self) -> NDArray:
        """Cached unique market IDs."""
        if self._unique_markets is None:
            self._unique_markets = np.unique(self.market_ids)
        return self._unique_markets

    @property
    def unique_products(self) -> NDArray:
        """Return unique product identifiers."""
        if self.product_ids is None:
            # Balanced panel: infer from first market
            first_market = self.unique_markets[0]
            J = self.get_market_size(first_market)
            return np.arange(J)
        return np.unique(self.product_ids)

    @property
    def n_products(self) -> int:
        """Number of unique products."""
        return len(self.unique_products)

    def get_market_mask(self, market_id) -> NDArray:
        """Get boolean mask for a market (cached)."""
        if market_id not in self._market_masks:
            self._market_masks[market_id] = self.market_ids == market_id
        return self._market_masks[market_id]

    def get_market_size(self, market_id) -> int:
        """Get number of products in a market."""
        return self.get_market_mask(market_id).sum()

    def get_markets_with_product(self, product_id) -> NDArray:
        """Return market_ids where the specified product exists (cached)."""
        if product_id not in self._product_market_map:
            if self.product_ids is None:
                # Balanced panel: product exists in all markets
                self._product_market_map[product_id] = self.unique_markets
            else:
                mask = self.product_ids == product_id
                self._product_market_map[product_id] = np.unique(self.market_ids[mask])
        return self._product_market_map[product_id]

    def get_local_index(self, market_id, product_id) -> int:
        """
        Get the local index of a product within a market.

        Returns the row index (0-indexed) of the product within the market.
        """
        if self.product_ids is None:
            # Balanced panel: product_id IS the local index
            return int(product_id)

        mask = self.get_market_mask(market_id)
        market_product_ids = self.product_ids[mask]
        local_idx = np.where(market_product_ids == product_id)[0]

        if len(local_idx) == 0:
            raise ValueError(f"Product {product_id} not found in market {market_id}")

        return int(local_idx[0])

    def get_market_data(self, market_id) -> tuple:
        """
        Extract all data for a single market.

        Returns
        -------
        price_t, x2_t, shares_t, market_ids_t : tuple of NDArrays
            Returns x2 (characteristics for omega), not x1.
            Returns copies to allow safe modification.
        """
        mask = self.get_market_mask(market_id)
        return (
            self.price[mask].copy(),
            self.x2[mask].copy(),
            self.shares[mask].copy(),
            self.market_ids[mask].copy()
        )

    def subset(self, idx: NDArray) -> 'RawData':
        """Create a subset of the data using observation indices."""
        return RawData(
            price=self.price[idx],
            x1=self.x1[idx],
            x2=self.x2[idx],
            shares=self.shares[idx],
            market_ids=self.market_ids[idx],
            w=self.w[idx] if self.w is not None else None,
            w_external=self.w_external[idx] if self.w_external is not None else None,
            product_ids=self.product_ids[idx] if self.product_ids is not None else None
        )

    def subset_markets(self, market_ids: NDArray) -> 'RawData':
        """Create a subset containing only specified markets."""
        mask = np.isin(self.market_ids, market_ids)
        return self.subset(np.where(mask)[0])
