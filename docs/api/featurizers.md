# API: Featurizers

Featurizers transform raw input variables into basis function expansions used by PGMM and MLIV estimators.

All featurizers implement:

```python
class Featurizer:
    def fit(self, X: NDArray) -> self: ...
    def transform(self, X: NDArray) -> NDArray: ...
    def transform_derivative(self, X: NDArray, wrt: int) -> NDArray: ...
```

---

## `CoordinatePolyTransform`

Coordinate-wise polynomial features (no cross-terms). Produces $1, x_1, x_1^2, \ldots, x_d, x_d^2, \ldots$

```python
from admliv.utils.featurizers import CoordinatePolyTransform
feat = CoordinatePolyTransform(degree=3, include_bias=True)
```

**Parameters:**
- `degree`: Maximum polynomial degree.
- `include_bias`: Whether to include a constant column (default: True).

---

## `PolyTransform`

Full polynomial features with cross-terms up to given degree.

```python
from admliv.utils.featurizers import PolyTransform
feat = PolyTransform(degree=2, include_bias=True)
```

---

## `BsplineTransform`

B-spline basis functions.

```python
from admliv.utils.featurizers import BsplineTransform
feat = BsplineTransform(n_knots=5, degree=3)
```

---

## `TrigPolyTransform`

Trigonometric polynomial basis: $\sin(kx), \cos(kx)$ for $k = 1, \ldots, K$.

```python
from admliv.utils.featurizers import TrigPolyTransform
feat = TrigPolyTransform(degree=5)
```

---

## `HermitePolyTransform`

Hermite polynomial basis (useful for Gaussian-distributed data).

```python
from admliv.utils.featurizers import HermitePolyTransform
feat = HermitePolyTransform(degree=5)
```

---

## `PairwiseInteractionTransform`

Produces pairwise interaction terms between input features.

```python
from admliv.utils.featurizers import PairwiseInteractionTransform
feat = PairwiseInteractionTransform()
```
