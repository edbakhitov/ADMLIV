# API: PGMM

## `PGMM`

Penalized Generalized Method of Moments for Riesz representer estimation.

```python
from admliv.core import PGMM, PGMMControl
```

### Constructor

```python
PGMM(
    x_featurizer: TransformerMixin,
    z_featurizer: TransformerMixin,
    lambda_: Optional[float] = None,
    adaptive: bool = True,
    Omega: Optional[NDArray] = None,
    control: PGMMControl = PGMMControl(),
    verbose: bool = True,
)
```

**Parameters:**
- `x_featurizer`: Basis expansion for endogenous variable $X$.
- `z_featurizer`: Basis expansion for instrument $Z$.
- `lambda_`: Penalty value. If `None`, computed as `c * sqrt(log(p) / n)`.
- `adaptive`: Whether to use adaptive LASSO weights.
- `Omega`: Optional weighting matrix for GMM criterion.
- `control`: `PGMMControl` with algorithm settings.

### `fit(W, moment, **moment_kwargs)`

Fit the PGMM estimator.

**Parameters:**
- `W`: Data dictionary with keys `'X'`, `'Z'`, `'Y'`.
- `moment`: `BaseMoment` instance defining the functional.
- `**moment_kwargs`: Additional arguments passed to `moment.compute_all_basis()`.

**Returns:** self

### `predict(Z)`

Predict $\hat{\alpha}(Z) = b(Z)'\hat{\rho}$ for new instrument values.

**Parameters:**
- `Z`: Instrument values, shape `(n, d_z)`.

**Returns:** Predictions, shape `(n,)`.

---

## `PGMMControl`

```python
PGMMControl(
    maxiter: int = 5000,
    optTol: float = 1e-5,
    zeroThreshold: float = 1e-6,
    intercept_penalty: float = 0.1,
    c: float = 0.01,
    adaptive_threshold: float = 1e-10,
    adaptive_max_weight: float = 1e10,
)
```

---

## `PGMMCV`

Cross-validated PGMM with automatic penalty selection.

```python
from admliv.core import PGMMCV, PGMMCVControl
```

---

## `PGMMLinearIV` / `PGMMLinearIVCV`

Specialized PGMM for high-dimensional linear IV regression, where the Riesz representer has a known linear structure. These directly estimate coefficients in linear IV models.
