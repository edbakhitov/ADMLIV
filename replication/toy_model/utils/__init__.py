# simulations/toy_model/utils/__init__.py

"""
Utility functions and estimators for toy model simulations.

This module provides:
- Analytical Riesz representer computation for average derivatives
- DML estimator with analytical RR and cross-fitting
"""

from .rr_ad_analytical import _rr_analytical_ad, AnalyticalRieszRepresenter
from .dml_analytical_ad import DMLAnalyticalAD

__all__ = [
    '_rr_analytical_ad',
    'AnalyticalRieszRepresenter',
    'DMLAnalyticalAD',
]
