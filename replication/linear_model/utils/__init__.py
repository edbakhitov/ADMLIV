# simulations/linear_model/utils/__init__.py

"""
Utility functions and estimators for linear model simulations.

This module provides:
- RMD Lasso estimator for high-dimensional linear regression
- Cross-validated RMD Lasso for automatic penalty selection
"""

from .rmd_lasso import RMDLasso
from .rmd_lasso_cv import RMDLassoCV
from .control import RMDControl, RMDCVControl

__all__ = [
    'RMDLasso',
    'RMDLassoCV',
    'RMDControl',
    'RMDCVControl',
]
