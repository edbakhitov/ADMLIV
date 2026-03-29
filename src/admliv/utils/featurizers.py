# admliv/utils/featurizers.py

import numpy as np
from itertools import combinations
from scipy.special import eval_hermitenorm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import SplineTransformer


class SimpleFeaturizer(BaseEstimator, TransformerMixin):
    """
    Simple featurizer that adds intercept to X.
    Transforms X -> [1, X] (adds intercept column)
    """
    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias
        
    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self.n_features_out_ = self.n_features_in_ + (1 if self.include_bias else 0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.include_bias:
            return np.c_[np.ones(X.shape[0]), X]
        return X
        

class BsplineTransform(BaseEstimator, TransformerMixin):
    """
    B-spline basis expansion with optional pairwise interactions.
    """
    def __init__(
        self,
        degree: int = 3,
        n_knots: int = 5,
        include_bias: bool = True,
        pairwise_interactions: bool = False,
        knots: str = 'quantile',
        extrapolation: str = 'constant'
    ):
        """
        Parameters
        ----------
        degree : int, default=3
            Degree of the B-spline
        n_knots : int, default=5
            Number of interior knots for each feature
        include_bias : bool, default=True
            If True, includes a bias (constant) term in the spline basis
        pairwise_interactions : bool, default=False
            If True, includes pairwise interactions of raw features
        knots : str, default='quantile'
            Strategy for placing knots: 'quantile' or 'uniform'
        extrapolation : str, default='constant'
            Extrapolation strategy: 'constant', 'continue', 'linear', or 'error'
        """
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias
        self.pairwise_interactions = pairwise_interactions
        self.knots = knots
        self.extrapolation = extrapolation

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        
        self.spline_transformers_ = []
        
        for i in range(self.n_features_in_):
            spline = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=self.include_bias,
                knots=self.knots,
                extrapolation=self.extrapolation
            )
            spline.fit(X[:, i:i+1])
            self.spline_transformers_.append(spline)

        # Number of spline basis functions
        n_splines = self.n_knots + self.degree - 1 - (1 - self.include_bias)
        n_output_features = self.n_features_in_ * n_splines

        if self.pairwise_interactions:
            n_interactions = self.n_features_in_ * (self.n_features_in_ - 1) // 2
        else:
            n_interactions = 0
        
        self.n_output_features_ = n_output_features + n_interactions
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        
        spline_features = [
            spline.transform(X[:, i:i+1]) 
            for i, spline in enumerate(self.spline_transformers_)
        ]
        X_splined = np.hstack(spline_features)
        
        if self.pairwise_interactions:
            interactions = []
            for i, j in combinations(range(X.shape[1]), 2):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                interactions.append(interaction)
            X_interactions = np.hstack(interactions) if interactions else np.empty((X.shape[0], 0))
            return np.hstack([X_splined, X_interactions])
        else:
            return X_splined

    # TODO: Implement transform_derivative for B-splines
    # B-spline derivatives can be computed using scipy.interpolate.BSpline.derivative()
    # This requires accessing the internal knot vectors and coefficients from
    # sklearn's SplineTransformer, which is not straightforward.
    # For now, JAX or numerical differentiation will be used as fallback.


class TrigPolyTransform(TransformerMixin, BaseEstimator):
    """
    Univariate trigonometric polynomial basis.
    
    Generates features of the form: (cos(2*k*pi*X), sin(2*k*pi*X))
    for k = 1, ..., degree.
    """
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        pairwise_interactions: bool = False
    ):
        """
        Parameters
        ----------
        degree : int, default=2
            Polynomial degree
        include_bias : bool, default=True
            If True, includes a bias (constant) term
        pairwise_interactions : bool, default=False
            If True, includes pairwise interactions of raw features
        """
        self.degree = degree
        self.include_bias = include_bias
        self.pairwise_interactions = pairwise_interactions

    def fit(self, X: np.ndarray, y=None):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_features_out_ = self.degree * n_features * 2 + int(self.include_bias)
        if self.pairwise_interactions:
            self.interaction_indices_ = list(combinations(range(n_features), 2))
            self.n_features_out_ += len(self.interaction_indices_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError("X shape does not match training shape")

        XTP = []
        for j in range(n_features):
            XTP.append(np.hstack([
                np.c_[
                    np.cos(2 * (k + 1) * np.pi * X[:, j]),
                    np.sin(2 * (k + 1) * np.pi * X[:, j])
                ] for k in range(self.degree)
            ]))
        XTP = np.hstack(XTP)

        if self.pairwise_interactions:
            interaction_terms = np.empty((n_samples, len(self.interaction_indices_)))
            for idx, (i, j) in enumerate(self.interaction_indices_):
                interaction_terms[:, idx] = X[:, i] * X[:, j]
            XTP = np.hstack([XTP, interaction_terms])

        if self.include_bias:
            return np.c_[np.ones(n_samples), XTP]
        else:
            return XTP

    def transform_derivative(self, X: np.ndarray, wrt: int = 0) -> np.ndarray:
        """
        Compute derivative of transform output w.r.t. X[:, wrt].
        
        For cos(2*k*pi*x), derivative is -2*k*pi*sin(2*k*pi*x)
        For sin(2*k*pi*x), derivative is 2*k*pi*cos(2*k*pi*x)
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
        wrt : int, default=0
            Index of feature to differentiate with respect to
        
        Returns
        -------
        derivative : np.ndarray, shape (n_samples, n_output_features)
            Derivative of each output feature w.r.t. X[:, wrt]
        """
        n_samples, n_features = X.shape
        if wrt >= n_features:
            raise ValueError("wrt index exceeds number of features")
        
        derivs = []
        
        if self.include_bias:
            derivs.append(np.zeros((n_samples, 1)))
        
        # Derivatives of trigonometric features
        for j in range(n_features):
            if j == wrt:
                # d/dx[cos(2*k*pi*x)] = -2*k*pi*sin(2*k*pi*x)
                # d/dx[sin(2*k*pi*x)] = 2*k*pi*cos(2*k*pi*x)
                dXTP = []
                for k in range(self.degree):
                    freq = 2 * (k + 1) * np.pi
                    dXTP.append(np.c_[
                        -freq * np.sin(freq * X[:, j]),
                        freq * np.cos(freq * X[:, j])
                    ])
                derivs.append(np.hstack(dXTP))
            else:
                # Derivative w.r.t. different variable is zero
                derivs.append(np.zeros((n_samples, self.degree * 2)))
        
        # Derivatives of pairwise interaction terms
        if self.pairwise_interactions:
            interaction_terms = []
            for (i, j) in self.interaction_indices_:
                if i == wrt:
                    interaction_terms.append(X[:, j].reshape(-1, 1))
                elif j == wrt:
                    interaction_terms.append(X[:, i].reshape(-1, 1))
                else:
                    interaction_terms.append(np.zeros((n_samples, 1)))
            derivs.append(np.hstack(interaction_terms))
        
        return np.hstack(derivs)


class HermitePolyTransform(TransformerMixin, BaseEstimator):
    """
    Univariate probabilist's (normalized) Hermite polynomial basis.
    """
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        pairwise_interactions: bool = False
    ):
        """
        Parameters
        ----------
        degree : int, default=2
            Polynomial degree
        include_bias : bool, default=True
            If True, includes a bias (constant) term
        pairwise_interactions : bool, default=False
            If True, includes pairwise interactions of raw features
        """
        self.degree = degree
        self.include_bias = include_bias
        self.pairwise_interactions = pairwise_interactions

    def fit(self, X: np.ndarray, y=None):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_features_out_ = self.degree * n_features + int(self.include_bias)
        if self.pairwise_interactions:
            self.interaction_indices_ = list(combinations(range(n_features), 2))
            self.n_features_out_ += len(self.interaction_indices_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError("X shape does not match training shape")

        XHP = []
        for j in range(n_features):
            XHP.append(np.hstack([
                eval_hermitenorm(k + 1, X[:, j]).reshape(-1, 1) 
                for k in range(self.degree)
            ]))
        XHP = np.hstack(XHP)
            
        if self.pairwise_interactions:
            interaction_terms = np.empty((n_samples, len(self.interaction_indices_)))
            for idx, (i, j) in enumerate(self.interaction_indices_):
                interaction_terms[:, idx] = X[:, i] * X[:, j]
            XHP = np.hstack([XHP, interaction_terms])

        if self.include_bias:
            return np.c_[np.ones(n_samples), XHP]
        else:
            return XHP
            
    def transform_derivative(self, X: np.ndarray, wrt: int = 0) -> np.ndarray:
        """
        Compute derivative of transform output w.r.t. X[:, wrt].
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
        wrt : int, default=0
            Index of feature to differentiate with respect to
        
        Returns
        -------
        derivative : np.ndarray, shape (n_samples, n_output_features)
            Derivative of each output feature w.r.t. X[:, wrt]
        """
        n_samples, n_features = X.shape
        if wrt >= n_features:
            raise ValueError("wrt index exceeds number of features")

        derivs = []

        if self.include_bias:
            derivs.append(np.zeros((n_samples, 1)))

        # Derivatives of Hermite features
        for j in range(n_features):
            if j == wrt:
                dXHP = [
                    ((k + 1) * eval_hermitenorm(k, X[:, j])).reshape(-1, 1)
                    for k in range(self.degree)
                ]
            else:
                dXHP = [np.zeros((n_samples, 1)) for _ in range(self.degree)]
            derivs.append(np.hstack(dXHP))

        # Derivatives of pairwise interaction terms
        if self.pairwise_interactions:
            interaction_terms = []
            for (i, j) in self.interaction_indices_:
                if i == wrt:
                    interaction_terms.append(X[:, j].reshape(-1, 1))
                elif j == wrt:
                    interaction_terms.append(X[:, i].reshape(-1, 1))
                else:
                    interaction_terms.append(np.zeros((n_samples, 1)))
            derivs.append(np.hstack(interaction_terms))

        return np.hstack(derivs)


class CoordinatePolyTransform(TransformerMixin, BaseEstimator):
    """
    Univariate polynomial basis (coordinate-wise powers).
    """
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        pairwise_interactions: bool = False
    ):
        """
        Parameters
        ----------
        degree : int, default=2
            Polynomial degree
        include_bias : bool, default=True
            If True, includes a bias (constant) term
        pairwise_interactions : bool, default=False
            If True, includes pairwise interactions of raw features
        """
        self.degree = degree
        self.include_bias = include_bias
        self.pairwise_interactions = pairwise_interactions

    def fit(self, X: np.ndarray, y=None):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_features_out_ = self.degree * n_features + int(self.include_bias)
        if self.pairwise_interactions:
            self.interaction_indices_ = list(combinations(range(n_features), 2))
            self.n_features_out_ += len(self.interaction_indices_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to polynomial basis. JAX-compatible."""
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError("X shape does not match training shape")

        # Detect if we're using JAX arrays
        try:
            import jax.numpy as jnp
            is_jax = isinstance(X, jnp.ndarray)
            xp = jnp if is_jax else np
        except ImportError:
            is_jax = False
            xp = np

        XCP = []
        for d in range(1, self.degree + 1):
            XCP.append(X ** d)
        XCP = xp.hstack(XCP) if len(XCP) > 1 else XCP[0]

        if self.pairwise_interactions:
            interaction_terms = []
            for i, j in self.interaction_indices_:
                interaction_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
            if interaction_terms:
                interaction_block = xp.hstack(interaction_terms)
                XCP = xp.hstack([XCP, interaction_block])

        if self.include_bias:
            ones = xp.ones((n_samples, 1))
            return xp.hstack([ones, XCP])
        else:
            return XCP

    def transform_derivative(self, X: np.ndarray, wrt: int = 0) -> np.ndarray:
        """
        Compute derivative of transform output w.r.t. X[:, wrt].
        
        For polynomial x^d, derivative is d*x^(d-1)
        
        The output column order matches transform():
        [bias (if included), X^1, X^2, ..., X^degree, interactions (if included)]
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
        wrt : int, default=0
            Index of feature to differentiate with respect to
        
        Returns
        -------
        derivative : np.ndarray, shape (n_samples, n_output_features)
            Derivative of each output feature w.r.t. X[:, wrt]
        """
        n_samples, n_features = X.shape
        if wrt >= n_features:
            raise ValueError("wrt index exceeds number of features")
        
        derivs = []
        
        if self.include_bias:
            derivs.append(np.zeros((n_samples, 1)))
        
        # Derivatives organized by degree (same as transform)
        # For each degree d, we have all features [X[0]^d, X[1]^d, ...]
        for d in range(1, self.degree + 1):
            for j in range(n_features):
                if j == wrt:
                    # d/dX[wrt] of X[j]^d
                    if d == 1:
                        derivs.append(np.ones((n_samples, 1)))
                    else:
                        derivs.append((d * X[:, j] ** (d - 1)).reshape(-1, 1))
                else:
                    # Derivative w.r.t. different variable is zero
                    derivs.append(np.zeros((n_samples, 1)))
        
        # Derivatives of pairwise interaction terms
        if self.pairwise_interactions:
            for (i, j) in self.interaction_indices_:
                if i == wrt:
                    derivs.append(X[:, j].reshape(-1, 1))
                elif j == wrt:
                    derivs.append(X[:, i].reshape(-1, 1))
                else:
                    derivs.append(np.zeros((n_samples, 1)))
        
        return np.hstack(derivs)


class PolyTransform(TransformerMixin, BaseEstimator):
    """
    Full polynomial basis with all interactions up to a given degree.

    Generates all monomials x1^a1 * x2^a2 * ... * xn^an where a1 + a2 + ... + an <= degree.
    Similar to sklearn's PolynomialFeatures but with analytical transform_derivative.

    Examples
    --------
    For degree=2, n_features=2 (x1, x2):
        [1, x1, x2, x1², x1*x2, x2²]

    For degree=3, n_features=2:
        [1, x1, x2, x1², x1*x2, x2², x1³, x1²*x2, x1*x2², x2³]
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False
    ):
        """
        Parameters
        ----------
        degree : int, default=2
            Maximum degree of polynomial features
        include_bias : bool, default=True
            If True, includes a bias (constant) term
        interaction_only : bool, default=False
            If True, only interaction features are produced (no x^2, x^3, etc.)
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def _generate_powers(self, n_features: int, degree: int):
        """
        Generate all multi-indices (a1, ..., an) with sum <= degree.

        Returns list of tuples, each tuple has length n_features.
        Ordered by total degree, then lexicographically.
        """
        from itertools import combinations_with_replacement

        powers = []

        # Start from degree 0 if including bias, else degree 1
        start_deg = 0 if self.include_bias else 1

        for d in range(start_deg, degree + 1):
            # Generate all combinations of indices that sum to d
            for combo in combinations_with_replacement(range(n_features), d):
                # Convert to power tuple
                power = [0] * n_features
                for idx in combo:
                    power[idx] += 1

                # If interaction_only, skip terms with any power > 1
                if self.interaction_only and any(p > 1 for p in power):
                    continue

                powers.append(tuple(power))

        return powers

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Generate all power tuples
        self.powers_ = self._generate_powers(n_features, self.degree)
        self.n_features_out_ = len(self.powers_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to polynomial features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_output_features)
            Polynomial features
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, expected {self.n_features_in_}"
            )

        # Compute each monomial
        X_poly = np.empty((n_samples, len(self.powers_)))

        for col, power in enumerate(self.powers_):
            # Compute x1^a1 * x2^a2 * ... * xn^an
            term = np.ones(n_samples)
            for j, p in enumerate(power):
                if p > 0:
                    term = term * (X[:, j] ** p)
            X_poly[:, col] = term

        return X_poly

    def transform_derivative(self, X: np.ndarray, wrt: int = 0) -> np.ndarray:
        """
        Compute derivative of polynomial features w.r.t. X[:, wrt].

        For monomial x1^a1 * ... * xn^an, derivative w.r.t. x_i is:
            a_i * x1^a1 * ... * x_i^(a_i - 1) * ... * xn^an  if a_i > 0
            0  if a_i == 0

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
        wrt : int, default=0
            Index of feature to differentiate with respect to

        Returns
        -------
        derivative : np.ndarray, shape (n_samples, n_output_features)
            Derivative of each output feature w.r.t. X[:, wrt]
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape

        if wrt >= n_features:
            raise ValueError(f"wrt={wrt} exceeds number of features {n_features}")

        derivs = np.empty((n_samples, len(self.powers_)))

        for col, power in enumerate(self.powers_):
            a_i = power[wrt]  # Power of variable we're differentiating w.r.t.

            if a_i == 0:
                # Derivative is zero if wrt variable doesn't appear
                derivs[:, col] = 0.0
            else:
                # Derivative: a_i * x1^a1 * ... * x_i^(a_i-1) * ... * xn^an
                term = np.ones(n_samples) * a_i
                for j, p in enumerate(power):
                    if j == wrt:
                        # Reduce power by 1
                        if p > 1:
                            term = term * (X[:, j] ** (p - 1))
                        # If p == 1, x^0 = 1, so no multiplication needed
                    elif p > 0:
                        term = term * (X[:, j] ** p)
                derivs[:, col] = term

        return derivs


class PairwiseInteractionTransform(TransformerMixin, BaseEstimator):
    """
    Pairwise interactions between raw features.
    """
    def __init__(self, include_bias: bool = True):
        """
        Parameters
        ----------
        include_bias : bool, default=True
            If True, includes a bias (constant) term
        """
        self.include_bias = include_bias

    def fit(self, X: np.ndarray, y=None):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.pair_indices_ = list(combinations(range(n_features), 2))
        self.n_features_out_ = len(self.pair_indices_) + int(self.include_bias)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError("X shape does not match training shape")

        interactions = np.empty((n_samples, len(self.pair_indices_)), dtype=X.dtype)
        for idx, (i, j) in enumerate(self.pair_indices_):
            interactions[:, idx] = X[:, i] * X[:, j]

        if self.include_bias:
            return np.c_[np.ones(n_samples), interactions]
        else:
            return interactions