# admliv/estimators/__init__.py
"""
MLIV Estimators for ADML.

This module provides various estimators for estimating the structural
function gamma(X) in instrumental variables settings.

Estimators
----------
NpivSieveEstimator : 2SLS with sieve basis expansion
DoubleLassoEstimator : High-dimensional IV with Lasso
KIVEstimator : Kernel Instrumental Variables

Adapters
--------
MLIVDictAdapter : Adapter for Dict interface compatibility
make_dict_compatible : Convenience function to create adapters

All estimators follow the same interface:
- fit(W): Fit the estimator, where W is a Dict with keys 'X', 'Z', 'Y'
- predict(X): Return predictions gamma(X)
- __call__(X): Alias for predict, allows use as callable

See Also
--------
admliv.moments : Moment function classes for ADML
admliv.core : PGMM and RMD estimators for Riesz representers
"""

from .base import BaseMLIVEstimator
from .sieve import NpivSieveEstimator, DoubleLassoEstimator
from .kiv import KIVEstimator

__all__ = [
    'BaseMLIVEstimator',
    'NpivSieveEstimator',
    'DoubleLassoEstimator',
    'KIVEstimator',
]