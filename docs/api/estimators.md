# API: MLIV Estimators

All estimators follow the interface defined by `BaseMLIVEstimator`:

```python
class BaseMLIVEstimator:
    def fit(self, W: Dict) -> self: ...
    def predict(self, X: NDArray) -> NDArray: ...
```

where `W` is a dict with keys `'X'`, `'Z'`, `'Y'`.

---

## `KIVEstimator`

Kernel Instrumental Variables estimator using Gaussian RBF kernels.

```python
from admliv.estimators import KIVEstimator

kiv = KIVEstimator(bandwidth_scale=1.0, bandwidth_method='std')
kiv.fit(W)
predictions = kiv.predict(X_new)
```

**Parameters:**
- `bandwidth_scale`: Multiplier applied to the computed bandwidth (default: 1.0).
- `bandwidth_method`: How to compute kernel bandwidths. `'median'` uses median interpoint distance (default); `'std'` uses column-wise standard deviation.
- `split_frac`: Fraction of data for first split (default: 0.5).
- `lam_bounds`: Search bounds for first-stage regularization (default: `(1e-6, 10.0)`).
- `xi_bounds`: Search bounds for second-stage regularization (default: `(1e-6, 10.0)`).
- `bandwidth_subsample`: Max points for median bandwidth computation (default: 1000).

---

## `DoubleLassoEstimator`

High-dimensional IV estimator using LASSO in both first and second stages.

```python
from admliv.estimators import DoubleLassoEstimator
from admliv.utils.featurizers import CoordinatePolyTransform

dl = DoubleLassoEstimator(
    x_featurizer=CoordinatePolyTransform(degree=2),
    z_featurizer=CoordinatePolyTransform(degree=2),
)
dl.fit(W)
predictions = dl.predict(X_new)
```

**Parameters:**
- `x_featurizer`: Featurizer for endogenous variable $X$.
- `z_featurizer`: Featurizer for instruments $Z$.
- `control`: `DoubleLassoControl` with settings for both first and second stages. Each stage is configured independently via `LassoStageControl` (alpha, CV settings, max iterations, tolerance). Default: no CV on first stage, CV on second stage.

---

## `NpivSieveEstimator`

Standard 2SLS with sieve basis expansion (no penalization).

```python
from admliv.estimators import NpivSieveEstimator

sieve = NpivSieveEstimator(
    x_featurizer=CoordinatePolyTransform(degree=3),
    z_featurizer=CoordinatePolyTransform(degree=3),
)
```
