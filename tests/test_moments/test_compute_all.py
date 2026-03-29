# tests/test_moments/test_compute_all.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from admliv.moments import WeightedAverage, WeightedAverageDerivative, AveragePolicyEffect
from admliv.utils.featurizers import (
    CoordinatePolyTransform, 
    HermitePolyTransform,
    TrigPolyTransform,
    PairwiseInteractionTransform
)


class TestWeightedAverageComputeAllBasis:
    """Test WeightedAverage.compute_all_basis with various featurizers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        self.d_x = 3
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_polynomial_featurizer(self):
        """Test with polynomial featurizer."""
        moment = WeightedAverage()
        featurizer = CoordinatePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Compute using compute_all_basis
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func)
        
        # Expected: M[i, j] = w(X_i) * d_j(X_i) = 1 * d_j(X_i)
        Wx_expected = featurizer.transform(self.X)
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, Wx_expected, rtol=1e-10)
    
    def test_hermite_featurizer(self):
        """Test with Hermite polynomial featurizer."""
        moment = WeightedAverage()
        featurizer = HermitePolyTransform(degree=3, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return X[:, 0]  # Weight by first column
        
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func)
        
        # Expected: M[i, j] = X[i, 0] * d_j(X_i)
        weights = self.X[:, 0].reshape(-1, 1)
        Wx = featurizer.transform(self.X)
        expected = weights * Wx
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_quadratic_weights(self):
        """Test with quadratic weights w(X) = sum(X^2)."""
        moment = WeightedAverage()
        featurizer = CoordinatePolyTransform(degree=2, include_bias=False)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.sum(X**2, axis=1)
        
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func)
        
        # Expected
        weights = np.sum(self.X**2, axis=1).reshape(-1, 1)
        Wx = featurizer.transform(self.X)
        expected = weights * Wx
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_consistency_with_compute(self):
        """Test that compute_all_basis is consistent with compute for each basis."""
        moment = WeightedAverage()
        featurizer = CoordinatePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Compute using compute_all_basis
        M_all = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func)
        
        # Compute using individual compute calls
        k = M_all.shape[1]
        M_individual = np.zeros_like(M_all)
        for j in range(k):
            def gamma_j(X_input):
                return featurizer.transform(X_input)[:, j:j+1]
            
            M_individual[:, j] = moment.compute(gamma_j, self.W, weight_func=weight_func)
        
        # Should match
        np.testing.assert_allclose(M_all, M_individual, rtol=1e-10)


class TestWeightedAverageDerivativeComputeAllBasis:
    """Test WeightedAverageDerivative.compute_all_basis with various methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 50  # Smaller for JAX tests
        self.d_x = 2
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_polynomial_analytical_derivative(self):
        """Test analytical derivatives for polynomial featurizer."""
        moment = WeightedAverageDerivative(use_jax=False)
        featurizer = CoordinatePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Derivative w.r.t. first variable
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func, deriv_index=0)
        
        # Expected: derivatives of [1, X[0], X[1], X[0]^2, X[1]^2]
        # w.r.t. X[0] are: [0, 1, 0, 2*X[0], 0]
        expected = np.c_[
            np.zeros(self.n),           # derivative of constant
            np.ones(self.n),             # derivative of X[0]
            np.zeros(self.n),            # derivative of X[1]
            2 * self.X[:, 0],            # derivative of X[0]^2
            np.zeros(self.n)             # derivative of X[1]^2
        ]
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_hermite_analytical_derivative(self):
        """Test analytical derivatives for Hermite polynomial featurizer."""
        moment = WeightedAverageDerivative(use_jax=False)
        featurizer = HermitePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Derivative w.r.t. second variable
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func, deriv_index=1)
        
        # Use featurizer's transform_derivative
        dWx_expected = featurizer.transform_derivative(self.X, wrt=1)
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, dWx_expected, rtol=1e-10)
    
    def test_trig_analytical_derivative(self):
        """Test analytical derivatives for trigonometric featurizer."""
        moment = WeightedAverageDerivative(use_jax=False)
        featurizer = TrigPolyTransform(degree=1, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Derivative w.r.t. first variable
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func, deriv_index=0)
        
        # Expected: derivatives of [1, cos(2*pi*X[0]), sin(2*pi*X[0]), cos(2*pi*X[1]), sin(2*pi*X[1])]
        # w.r.t. X[0] are: [0, -2*pi*sin(2*pi*X[0]), 2*pi*cos(2*pi*X[0]), 0, 0]
        freq = 2 * np.pi
        expected = np.c_[
            np.zeros(self.n),                    # derivative of constant
            -freq * np.sin(freq * self.X[:, 0]), # derivative of cos
            freq * np.cos(freq * self.X[:, 0]),  # derivative of sin
            np.zeros(self.n),                    # derivative of cos(X[1])
            np.zeros(self.n)                     # derivative of sin(X[1])
        ]
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-9)
    
    def test_numerical_derivative_fallback(self):
        """Test numerical derivative for featurizer without transform_derivative."""
        moment = WeightedAverageDerivative(use_jax=False)
        # PairwiseInteractionTransform doesn't have transform_derivative
        featurizer = PairwiseInteractionTransform(include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func, deriv_index=0)
        
        # Expected: derivatives of [1, X[0]*X[1]] w.r.t. X[0] are [0, X[1]]
        expected = np.c_[
            np.zeros(self.n),
            self.X[:, 1]
        ]
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-4, atol=1e-6)
    
    def test_with_weights(self):
        """Test derivatives with non-uniform weights."""
        moment = WeightedAverageDerivative(use_jax=False)
        featurizer = CoordinatePolyTransform(degree=1, include_bias=False)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return X[:, 0]
        
        M = moment.compute_all_basis(featurizer, self.W, weight_func=weight_func, deriv_index=0)
        
        # Expected: w(X) * d/dX[0] of [X[0], X[1]] = X[0] * [1, 0]
        weights = self.X[:, 0].reshape(-1, 1)
        derivatives = np.c_[np.ones(self.n), np.zeros(self.n)]
        expected = weights * derivatives
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_numerical_vs_analytical_polynomial(self):
        """Test that numerical and analytical derivatives agree for polynomials."""
        
        # Polynomial featurizer has analytical derivatives
        featurizer = CoordinatePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Compute with analytical method (default when transform_derivative exists)
        moment_analytical = WeightedAverageDerivative(use_jax=False)
        M_analytical = moment_analytical.compute_all_basis(
            featurizer, self.W, weight_func=weight_func, deriv_index=0
        )
        
        # Numerical derivative for comparison
        moment_numerical = WeightedAverageDerivative(use_jax=False)
        M_numerical = moment_numerical._compute_featurizer_derivative_numerical(
            featurizer, self.X, deriv_index=0
        )
        
        # Should be close (numerical is approximate)
        assert M_analytical.shape == M_numerical.shape
        np.testing.assert_allclose(M_analytical, M_numerical, rtol=1e-4, atol=1e-6)
    
    def test_numerical_vs_analytical_hermite(self):
        """Test that numerical and analytical derivatives agree for Hermite polynomials."""
        
        featurizer = HermitePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Analytical (via transform_derivative)
        moment_analytical = WeightedAverageDerivative(use_jax=False)
        M_analytical = moment_analytical.compute_all_basis(
            featurizer, self.W, weight_func=weight_func, deriv_index=1
        )
        
        # Numerical derivative for comparison
        moment_numerical = WeightedAverageDerivative(use_jax=False)
        M_numerical = moment_numerical._compute_featurizer_derivative_numerical(
            featurizer, self.X, deriv_index=1
        )
        
        # Should match closely
        assert M_analytical.shape == M_numerical.shape
        np.testing.assert_allclose(M_analytical, M_numerical, rtol=1e-5, atol=1e-7)
    
    def test_numerical_vs_analytical_trig(self):
        """Test that numerical and analytical derivatives agree for trigonometric polynomials."""
        
        featurizer = TrigPolyTransform(degree=1, include_bias=True)
        featurizer.fit(self.X)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Analytical
        moment_analytical = WeightedAverageDerivative(use_jax=False)
        M_analytical = moment_analytical.compute_all_basis(
            featurizer, self.W, weight_func=weight_func, deriv_index=0
        )
        
        # Numerical
        moment_numerical = WeightedAverageDerivative(use_jax=True)
        M_numerical = moment_numerical._compute_featurizer_derivative_numerical(
            featurizer, self.X, deriv_index=0
        )
        
        # Should match (JAX is exact for smooth functions)
        assert M_analytical.shape == M_numerical.shape
        np.testing.assert_allclose(M_analytical, M_numerical, rtol=1e-5, atol=1e-7)


class TestAveragePolicyEffectComputeAllBasis:
    """Test AveragePolicyEffect.compute_all_basis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        self.d_x = 3
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_additive_policy(self):
        """Test with additive policy h(X) = X + delta."""
        moment = AveragePolicyEffect()
        featurizer = CoordinatePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        delta = np.array([0.5, 0.0, 0.0])
        def policy_func(X):
            return X + delta
        
        M = moment.compute_all_basis(featurizer, self.W, policy_func=policy_func)
        
        # Expected: d_j(X + delta) - d_j(X)
        X_policy = self.X + delta
        Wx_policy = featurizer.transform(X_policy)
        Wx_original = featurizer.transform(self.X)
        expected = Wx_policy - Wx_original
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_multiplicative_policy(self):
        """Test with multiplicative policy h(X) = alpha * X."""
        moment = AveragePolicyEffect()
        featurizer = CoordinatePolyTransform(degree=2, include_bias=False)
        featurizer.fit(self.X)
        
        alpha = 2.0
        def policy_func(X):
            return alpha * X
        
        M = moment.compute_all_basis(featurizer, self.W, policy_func=policy_func)
        
        # Expected: for [X, X^2], we get [2X - X, 4X^2 - X^2] = [X, 3X^2]
        # More generally: d_j(2X) - d_j(X)
        X_policy = alpha * self.X
        Wx_policy = featurizer.transform(X_policy)
        Wx_original = featurizer.transform(self.X)
        expected = Wx_policy - Wx_original
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_identity_policy_returns_zero(self):
        """Test that identity policy gives zero effect."""
        moment = AveragePolicyEffect()
        featurizer = HermitePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        def policy_func(X):
            return X
        
        M = moment.compute_all_basis(featurizer, self.W, policy_func=policy_func)
        
        # Expected: all zeros
        expected = np.zeros((self.n, featurizer.n_features_out_))
        
        assert M.shape == (self.n, featurizer.n_features_out_)
        np.testing.assert_allclose(M, expected, rtol=1e-10)
    
    def test_consistency_with_compute(self):
        """Test consistency between compute_all_basis and compute."""
        moment = AveragePolicyEffect()
        featurizer = CoordinatePolyTransform(degree=2, include_bias=True)
        featurizer.fit(self.X)
        
        delta = np.array([1.0, 0.0, 0.0])
        def policy_func(X):
            return X + delta
        
        # Compute using compute_all_basis
        M_all = moment.compute_all_basis(featurizer, self.W, policy_func=policy_func)
        
        # Compute using individual compute calls
        k = M_all.shape[1]
        M_individual = np.zeros_like(M_all)
        for j in range(k):
            def gamma_j(X_input):
                return featurizer.transform(X_input)[:, j:j+1]
            
            M_individual[:, j] = moment.compute(gamma_j, self.W, policy_func=policy_func)
        
        # Should match
        np.testing.assert_allclose(M_all, M_individual, rtol=1e-10)


class TestComputeAllBasisShapes:
    """Test that compute_all_basis returns correct shapes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 50
        self.d_x = 2
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_weighted_average_shape(self):
        """Test output shape for WeightedAverage."""
        moment = WeightedAverage()
        
        for degree in [1, 2, 3]:
            featurizer = CoordinatePolyTransform(degree=degree, include_bias=True)
            featurizer.fit(self.X)
            
            M = moment.compute_all_basis(
                featurizer, self.W, 
                weight_func=lambda X: np.ones(X.shape[0])
            )
            
            expected_shape = (self.n, featurizer.n_features_out_)
            assert M.shape == expected_shape
    
    def test_weighted_derivative_shape(self):
        """Test output shape for WeightedAverageDerivative."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        for degree in [1, 2]:
            featurizer = HermitePolyTransform(degree=degree, include_bias=True)
            featurizer.fit(self.X)
            
            M = moment.compute_all_basis(
                featurizer, self.W,
                weight_func=lambda X: np.ones(X.shape[0]),
                deriv_index=0
            )
            
            expected_shape = (self.n, featurizer.n_features_out_)
            assert M.shape == expected_shape
    
    def test_policy_effect_shape(self):
        """Test output shape for AveragePolicyEffect."""
        moment = AveragePolicyEffect()
        
        for degree in [1, 2, 3]:
            featurizer = CoordinatePolyTransform(degree=degree, include_bias=False)
            featurizer.fit(self.X)
            
            M = moment.compute_all_basis(
                featurizer, self.W,
                policy_func=lambda X: X + 0.1
            )
            
            expected_shape = (self.n, featurizer.n_features_out_)
            assert M.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])