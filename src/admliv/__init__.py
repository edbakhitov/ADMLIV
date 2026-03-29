# admliv/__init__.py
"""
ADMLIV: Automatic Debiased Machine Learning for Instrumental Variables

This package provides tools for debiased estimation of functionals in IV settings
using machine learning methods.

Main Components
---------------
Core Estimators:
    - ADMLIV: Main debiased ML-IV estimator with cross-fitting
    - PGMM: Penalized GMM for Riesz representer estimation
    - PGMMCV: Cross-validated PGMM
    - PGMMLinearIV: PGMM for linear IV regression
    - PGMMLinearIVCV: Cross-validated linear IV PGMM

MLIV Estimators:
    - NpivSieveEstimator: 2SLS with sieve basis expansion
    - DoubleLassoEstimator: High-dimensional IV with Double Lasso
    - KIVEstimator: Kernel IV estimator

Moment Functions:
    - WeightedAverage: Weighted average functional
    - AveragePolicyEffect: Average treatment effect under policy
    - WeightedAverageDerivative: Weighted average of derivatives

Control Classes:
    - ADMLIVControl: Control parameters for ADMLIV
    - PGMMControl, PGMMCVControl: Control for PGMM estimators
    - RMDControl, RMDCVControl: Control for RMD estimators

Examples
--------
Basic ADMLIV estimation:

>>> from admliv import ADMLIV, ADMLIVControl
>>> from admliv.estimators import DoubleLassoEstimator
>>> from admliv.moments import WeightedAverage
>>> from admliv.utils.featurizers import CoordinatePolyTransform
>>>
>>> # Setup
>>> x_feat = CoordinatePolyTransform(degree=2)
>>> z_feat = CoordinatePolyTransform(degree=2)
>>> moment = WeightedAverage()
>>>
>>> # MLIV estimator factory
>>> def mliv_factory():
...     return DoubleLassoEstimator(
...         x_featurizer=CoordinatePolyTransform(degree=2),
...         z_featurizer=CoordinatePolyTransform(degree=2)
...     )
>>>
>>> # Fit ADMLIV
>>> control = ADMLIVControl(n_folds=5, verbose=True)
>>> admliv = ADMLIV(
...     mliv_estimator=mliv_factory,
...     x_featurizer=x_feat,
...     z_featurizer=z_feat,
...     control=control
... )
>>> result = admliv.fit(W, moment, weight_func=lambda x: np.ones(x.shape[0]))
>>> print(result.summary())
"""

# Main ADMLIV interface
from .main import (
    ADMLIV,
    ADMLIVControl,
    ADMLIVResult,
    fit_admliv,
)

# Core estimators
from .core import (
    PGMM,
    PGMMCV,
    PGMMLinearIV,
    PGMMLinearIVCV,
    PGMMControl,
    PGMMCVControl,
)

# MLIV estimators
from .estimators import (
    BaseMLIVEstimator,
    NpivSieveEstimator,
    DoubleLassoEstimator,
    KIVEstimator,
)

# Moment functions
from .moments import (
    BaseMoment,
    WeightedAverage,
    AveragePolicyEffect,
    WeightedAverageDerivative,
)

__version__ = "0.1.0"

__all__ = [
    # Main ADMLIV
    'ADMLIV',
    'ADMLIVControl',
    'ADMLIVResult',
    'fit_admliv',
    # PGMM estimators
    'PGMM',
    'PGMMCV',
    'PGMMLinearIV',
    'PGMMLinearIVCV',
    # Control classes
    'PGMMControl',
    'PGMMCVControl',
    # MLIV estimators
    'BaseMLIVEstimator',
    'NpivSieveEstimator',
    'DoubleLassoEstimator',
    'KIVEstimator',
    # Moments
    'BaseMoment',
    'WeightedAverage',
    'AveragePolicyEffect',
    'WeightedAverageDerivative',
]
