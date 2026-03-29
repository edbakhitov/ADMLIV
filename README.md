# ADMLIV: Automatic Debiased Machine Learning for Instrumental Variables

A Python package for debiased estimation and inference on functionals of machine learning instrumental variables (MLIV) estimators. ADMLIV provides valid confidence intervals and hypothesis tests for a wide class of functionals&mdash;including weighted averages, average derivatives, and demand elasticities&mdash;without requiring strong assumptions on the MLIV function classes.

The methodology is based on the paper *"Penalized GMM Framework for Inference on Functionals of Nonparametric Instrumental Variable Estimators"* by Edvard Bakhitov (2026).

## Installation

```bash
# From source
git clone https://github.com/ebakhitov/admliv.git
cd admliv
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With notebook dependencies
pip install -e ".[notebooks]"
```

## Quick Start

```python
import numpy as np
from admliv import ADMLIV, ADMLIVControl
from admliv.estimators import DoubleLassoEstimator
from admliv.moments import WeightedAverage
from admliv.utils.featurizers import CoordinatePolyTransform

# Data: W = {'Y': y, 'X': x, 'Z': z}
n = 500
z = np.random.randn(n, 5)
x = z[:, :3] + 0.5 * np.random.randn(n, 3)
y = x @ np.array([1.0, 0.5, -0.5]) + np.random.randn(n)
W = {'Y': y, 'X': x, 'Z': z}

# First-stage MLIV estimator factory
def mliv_factory():
    return DoubleLassoEstimator(
        x_featurizer=CoordinatePolyTransform(degree=2),
        z_featurizer=CoordinatePolyTransform(degree=2),
    )

# Set up ADMLIV
control = ADMLIVControl(n_folds=5)
estimator = ADMLIV(
    mliv_estimator=mliv_factory,
    x_featurizer=CoordinatePolyTransform(degree=2),
    z_featurizer=CoordinatePolyTransform(degree=2),
    control=control,
)

# Estimate a weighted average functional
moment = WeightedAverage()
result = estimator.fit(W, moment, weight_func=lambda x: np.ones(x.shape[0]))
print(result.summary())
```

## Key Components

### MLIV Estimators
- **`KIVEstimator`** &mdash; Kernel Instrumental Variables
- **`DoubleLassoEstimator`** &mdash; High-dimensional IV with Double Lasso
- **`NpivSieveEstimator`** &mdash; 2SLS with sieve basis expansion

Other first-stage estimators (e.g., DeepIV, DeepGMM, Adversarial GMM, etc.) can be integrated by implementing the `BaseMLIVEstimator` interface.

### Moment Functions (Functionals)
- **`WeightedAverage`** &mdash; $E[w(X)\gamma(X)]$
- **`WeightedAverageDerivative`** &mdash; $E[w(X)\partial\gamma(X)/\partial x_j]$
- **`AveragePolicyEffect`** &mdash; $E[\gamma(g(X)) - \gamma(X)]$
- **`SquaredGammaAverage`** &mdash; $E[\gamma(X)^2]$

Custom functionals can be added by subclassing `BaseMoment`.

### Core Algorithms
- **`PGMM`** &mdash; Penalized GMM for Riesz representer estimation with LASSO penalty
- **`PGMMCV`** &mdash; Cross-validated penalty selection
- **`PGMMLinearIV` / `PGMMLinearIVCV`** &mdash; Specialized PGMM for linear IV regression

### Featurizers
- `CoordinatePolyTransform`, `PolyTransform` &mdash; Polynomial basis expansions
- `BsplineTransform` &mdash; B-spline basis
- `TrigPolyTransform` &mdash; Trigonometric polynomial basis
- `HermitePolyTransform` &mdash; Hermite polynomial basis

## Examples

See the `examples/` directory for Jupyter notebooks:

1. **`01_hd_linear_regression.ipynb`** &mdash; PGMM for high-dimensional linear regression
2. **`02_hd_linear_iv.ipynb`** &mdash; PGMM for high-dimensional linear IV
3. **`03_demand_estimation.ipynb`** &mdash; Demand elasticity estimation with ADMLIV

## Replication

The `replication/` directory contains Monte Carlo simulation scripts from the paper. See [`replication/README.md`](replication/README.md) for instructions.

## Documentation

See the [`docs/`](docs/) directory for detailed documentation:

- **[Overview](docs/overview.md)** &mdash; Package description and installation
- **[Methodology](docs/methodology.md)** &mdash; Mathematical background
- **[API Reference](docs/api/)** &mdash; Detailed class and method documentation

## Citation

```bibtex
@article{bakhitov2026pgmm,
  title={Penalized GMM Framework for Inference on Functionals of Nonparametric Instrumental Variable Estimators},
  author={Bakhitov, Edvard},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
