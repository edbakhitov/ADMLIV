# ADMLIV: Overview

## What is ADMLIV?

ADMLIV (Automatic Debiased Machine Learning for Instrumental Variables) is a Python package for performing valid statistical inference on functionals of machine learning instrumental variables estimators.

In many applications, we estimate a structural function $\gamma(x) = E[Y|X=x]$ using instruments $Z$ via machine learning methods (kernel methods, LASSO, neural networks, etc.). While these ML estimators converge slowly to the truth, ADMLIV enables $\sqrt{n}$-consistent estimation and valid confidence intervals for *functionals* $\theta(\gamma) = E[m(W;\gamma)]$ by automatically constructing a debiasing correction.

The methodology is based on the paper *"Penalized GMM Framework for Inference on Functionals of Nonparametric Instrumental Variable Estimators"* by Edvard Bakhitov (2026).

## Installation

```bash
# From source
git clone https://github.com/ebakhitov/admliv.git
cd admliv
pip install -e .

# With all optional dependencies
pip install -e ".[dev,notebooks]"
```

### Requirements

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- scikit-learn >= 1.0
- Numba >= 0.55
- Joblib >= 1.0
- Pandas >= 1.3

## Quick Example

```python
import numpy as np
from admliv import ADMLIV, ADMLIVControl
from admliv.estimators import KIVEstimator
from admliv.moments import WeightedAverageDerivative
from admliv.utils.featurizers import CoordinatePolyTransform

# Generate data
n = 1000
z = np.random.randn(n, 5)
x = z[:, :3] + np.random.randn(n, 3) * 0.3
y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + np.random.randn(n) * 0.5
W = {'Y': y, 'X': x, 'Z': z}

# ADMLIV with KIV first stage
control = ADMLIVControl(n_folds=5, verbose=True)
estimator = ADMLIV(
    mliv_estimator=lambda: KIVEstimator(),
    x_featurizer=CoordinatePolyTransform(degree=3),
    z_featurizer=CoordinatePolyTransform(degree=3),
    control=control,
)

# Estimate weighted average derivative
moment = WeightedAverageDerivative(wrt=0)
result = estimator.fit(W, moment, weight_func=lambda x: np.ones(x.shape[0]))
print(result.summary())
```

## Package Structure

```
src/admliv/
├── core/          # PGMM algorithm for Riesz representer estimation
├── estimators/    # First-stage MLIV estimators (KIV, Double Lasso, Sieve)
├── moments/       # Functional/moment definitions
├── main/          # Main ADMLIV class with cross-fitting
└── utils/         # Featurizers (polynomial, spline, etc.)
```
