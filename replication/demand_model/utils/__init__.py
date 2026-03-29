# replication/demand_model/utils/__init__.py

"""
Utility functions for demand elasticity simulations.

This module provides:
- Input space transformer to create characteristic differences
- Own-price elasticity functional with analytical derivatives
- Modified PGMM and ADMLIV classes for panel data with elasticity estimation
"""

from .raw_data import RawData
from .omega_transformer import OmegaTransformer, create_omega, omega_to_dict
from .pgmm_elasticity import PGMMElasticity, PGMMElasticityControl
from .admliv_elasticity import (
    ADMLIVElasticity,
    ADMLIVElasticityControl,
    ADMLIVElasticityResult,
    MarketKFold,
)
from .own_price_elasticity import (
    OwnPriceElasticity,
    OmegaStructure,
    get_omega_structure,
)

__all__ = [
    'RawData',
    'OmegaTransformer',
    'create_omega',
    'omega_to_dict',
    'PGMMElasticity',
    'PGMMElasticityControl',
    'ADMLIVElasticity',
    'ADMLIVElasticityControl',
    'ADMLIVElasticityResult',
    'MarketKFold',
    'OwnPriceElasticity',
    'OmegaStructure',
    'get_omega_structure',
]
