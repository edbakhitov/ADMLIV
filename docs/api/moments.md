# API: Moment Functions

All moments implement the `BaseMoment` interface:

```python
class BaseMoment:
    def compute(self, gamma_hat, data) -> NDArray: ...
    def compute_basis_gateaux(self, gamma_hat, featurizer, data) -> NDArray: ...
```

- `compute`: Evaluates $m(W; \hat{\gamma})$ for each observation.
- `compute_basis_gateaux`: Computes the Gateaux derivative $D_\gamma\theta[d_k]$ for each basis function $d_k$.

---

## `WeightedAverage`

$$\theta(\gamma) = E[w(X)\gamma(X)]$$

A linear functional. The Gateaux derivative is $D_\gamma\theta[d] = E[w(X)d(X)]$.

```python
from admliv.moments import WeightedAverage
moment = WeightedAverage()
```

---

## `WeightedAverageDerivative`

$$\theta(\gamma) = E\left[w(X)\frac{\partial\gamma(X)}{\partial x_j}\right]$$

A linear functional targeting the average partial derivative. Requires the MLIV estimator to support `predict_derivative()`.

```python
from admliv.moments import WeightedAverageDerivative
moment = WeightedAverageDerivative(wrt=0)  # derivative w.r.t. first coordinate
```

---

## `AveragePolicyEffect`

$$\theta(\gamma) = E[\gamma(g(X)) - \gamma(X)]$$

A linear functional for evaluating the average effect of a counterfactual covariate shift $x \mapsto g(x)$.

```python
from admliv.moments import AveragePolicyEffect
moment = AveragePolicyEffect()
# Use with: estimator.fit(W, moment, policy_func=lambda x: x + 0.1)
```

---

## `SquaredGammaAverage`

$$\theta(\gamma) = E[\gamma(X)^2]$$

A **nonlinear** functional. Requires double cross-fitting (minimum 3 folds) and does not support `use_cv_for_pgmm=True`.

```python
from admliv.moments import SquaredGammaAverage
moment = SquaredGammaAverage()
```
