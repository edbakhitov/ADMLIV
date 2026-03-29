# API: ADMLIV

## `ADMLIV`

Main estimator class implementing cross-fitted debiased ML-IV estimation.

```python
from admliv import ADMLIV, ADMLIVControl
```

### Constructor

```python
ADMLIV(
    mliv_estimator: Callable[[], BaseMLIVEstimator],
    x_featurizer: Featurizer,
    z_featurizer: Featurizer,
    control: ADMLIVControl = ADMLIVControl(),
)
```

**Parameters:**
- `mliv_estimator`: A factory function (callable with no arguments) that returns a fresh MLIV estimator instance. Called once per fold.
- `x_featurizer`: Featurizer for the endogenous variable $X$ (used in PGMM basis).
- `z_featurizer`: Featurizer for the instrument $Z$ (used in PGMM basis).
- `control`: Configuration object controlling cross-fitting, PGMM, and inference settings.

### `fit(W, moment, weight_func=None)`

Fit the ADMLIV estimator.

**Parameters:**
- `W`: Dict with keys `'X'`, `'Z'`, `'Y'` (and optionally other variables).
- `moment`: A `BaseMoment` instance defining the functional of interest.
- `weight_func`: Optional callable `f(X) -> weights` for weighted functionals.

**Returns:** `ADMLIVResult`

---

## `ADMLIVControl`

Configuration dataclass for ADMLIV.

```python
ADMLIVControl(
    n_folds: int = 5,
    random_state: int = 42,
    pgmm_control: Optional[PGMMControl] = None,
    use_cv_for_pgmm: bool = False,
    use_adaptive_pgmm: bool = True,
    confidence_level: float = 0.95,
    verbose: bool = True,
)
```

**Key Parameters:**
- `n_folds`: Number of cross-fitting folds (minimum 3 for nonlinear functionals).
- `pgmm_control`: `PGMMControl` or `PGMMCVControl` instance. Controls PGMM penalty (`c`), adaptive weights, convergence, etc. If `None`, uses defaults.
- `use_adaptive_pgmm`: Whether to use adaptive LASSO weights in PGMM.
- `use_cv_for_pgmm`: Cross-validate the penalty (not supported for nonlinear functionals).
- `confidence_level`: Confidence level for CIs (default: 0.95).

---

## `ADMLIVResult`

Result container returned by `ADMLIV.fit()`.

**Attributes:**
- `theta_debiased`: Debiased estimate $\hat{\theta}_{\text{ADMLIV}}$
- `theta_plugin`: Plug-in (uncorrected) estimate $\hat{\theta}_{\text{PI}}$
- `se_debiased`: Standard error of the debiased estimate
- `se_plugin`: Standard error of the plug-in estimate
- `variance_debiased`, `variance_plugin`: Variance estimates
- `ci_lower`, `ci_upper`: Confidence interval bounds (debiased)
- `ci_lower_plugin`, `ci_upper_plugin`: Confidence interval bounds (plug-in)
- `influence_functions`: Influence function values for each observation
- `n_samples`, `n_folds`: Sample size and number of folds
- `fold_estimates`: Per-fold diagnostics

**Methods:**
- `summary()`: Return a formatted summary of results.

---

## `fit_admliv`

Convenience function for one-shot estimation.

```python
result = fit_admliv(W, moment, mliv_factory, x_feat, z_feat, control)
```
