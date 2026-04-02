# Methodology

## Problem Setting

Consider the nonparametric instrumental variables (NPIV) model:

$$Y = \gamma_0(X) + \varepsilon, \quad E[\varepsilon | Z] = 0$$

where $Y$ is the outcome, $X$ is the endogenous variable, $Z$ is the instrument, and $\gamma_0$ is the unknown structural function. We estimate $\gamma_0$ using a machine learning IV (MLIV) estimator $\hat{\gamma}$.

**Goal**: Estimate and perform inference on a functional $\theta(\gamma_0) = E[m(W; \gamma_0)]$ where $m$ is a known moment function.

## Debiasing Formula

The plug-in estimator $\hat{\theta}_{\text{PI}} = \frac{1}{n}\sum_i m(W_i; \hat{\gamma})$ is generally biased due to the slow convergence of $\hat{\gamma}$. ADMLIV corrects this bias using the **Riesz representer** $\alpha_0(Z)$:

$$\hat{\theta}_{\text{ADMLIV}} = \frac{1}{n}\sum_{i=1}^{n}\left[m(W_i; \hat{\gamma}_{-\ell(i)}) + \hat{\alpha}(Z_i)\left(Y_i - \hat{\gamma}_{-\ell(i)}(X_i)\right)\right]$$

where:
- $\hat{\gamma}_{-\ell}$ is the MLIV estimator trained on data excluding fold $\ell$
- $\hat{\alpha}$ is the estimated Riesz representer (debiasing correction)
- The second term corrects for the bias in the plug-in estimator

## Riesz Representer

The Riesz representer $\alpha_0$ satisfies:

$$E[\alpha_0(Z) \cdot d(X)] = D_\gamma \theta[d] \quad \text{for all } d \in \mathcal{G}$$

where $D_\gamma \theta[d]$ is the Gateaux derivative of the functional $\theta$ in direction $d$. This is estimated via **Penalized GMM (PGMM)**.

## PGMM Algorithm

PGMM estimates $\alpha$ by solving a penalized moment condition:

$$\hat{\rho} = \arg\min_\rho \left\| M - G'\rho \right\|_\Omega^2 + \lambda \|\rho\|_1$$

where:
- $G$ is the Gram matrix of instruments evaluated at basis functions
- $M$ is the moment vector (Gateaux derivative evaluated at basis functions)
- $\Omega$ is a weighting matrix
- $\lambda$ is the LASSO penalty parameter

The penalty rate is $\lambda = c \cdot \sqrt{\log(p) / n}$ for linear functionals and $\lambda = c \cdot \sqrt{\log(p)} \cdot n^{-1/4}$ for nonlinear functionals.

**Two-stage adaptive estimation**: After initial estimation, adaptive weights $w_j = 1/|\tilde{\rho}_j|$ refine the penalty to achieve oracle properties.

## Cross-Fitting

**Single cross-fitting** (linear functionals): Split data into $K$ folds. For each fold $k$, train $\hat{\gamma}_{-k}$ and $\hat{\alpha}_{-k}$ on the remaining folds, then evaluate on fold $k$.

**Double cross-fitting** (nonlinear functionals): When the moment $m(W;\gamma)$ depends on $\gamma$, the Gateaux derivative $M$ also depends on $\gamma$. Double cross-fitting ensures each component is computed on held-out data:

For test fold $k$:
1. For each fold $\ell \neq k$: train $\hat{\gamma}_{k,\ell}$ on folds $\notin \{k, \ell\}$, compute $M[\ell]$ using $\hat{\gamma}_{k,\ell}$
2. Fit PGMM on folds $\neq k$ using the assembled $M$
3. Evaluate the debiased estimate on fold $k$

## Asymptotic Theory

Under regularity conditions, the ADMLIV estimator is asymptotically normal:

$$\sqrt{n}(\hat{\theta}_{\text{ADMLIV}} - \theta_0) \xrightarrow{d} N(0, \sigma^2)$$

where $\sigma^2 = \text{Var}[m(W;\gamma_0) + \alpha_0(Z)(Y - \gamma_0(X))]$. The variance is consistently estimated by the sample variance of the influence function, yielding valid confidence intervals.

## Extensibility

ADMLIV is designed as a modular framework:
- **Any MLIV estimator** can be used as the first stage by implementing the `BaseMLIVEstimator` interface
- **Any functional** can be targeted by subclassing `BaseMoment` and providing `compute()` and `compute_basis_gateaux()` methods
- **Any basis expansion** can be used by implementing the featurizer interface with `transform()` and `transform_derivative()` methods

## Reference

Bakhitov, E. (2026). *Penalized GMM Framework for Inference on Functionals of Nonparametric Instrumental Variable Estimators*. [[arXiv]](https://arxiv.org/abs/2603.29889)
