# tests/test_moments/test_compute.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from admliv.moments import WeightedAverage, WeightedAverageDerivative, AveragePolicyEffect

class TestWeightedAverage:
    """Tests for WeightedAverage.compute moment function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        self.d_x = 3
        
        # Create toy data
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_compute_uniform_weights(self):
        """Test with uniform weights (w(X) = 1)."""
        moment = WeightedAverage()
        
        # Define toy gamma: linear function gamma(X) = X @ beta
        beta = np.array([1.0, 2.0, 3.0])
        def gamma(X):
            return (X @ beta).reshape(-1, 1)
        
        # Define uniform weight function
        def weight_func(X):
            return np.ones(X.shape[0])
        
        result = moment.compute(gamma, self.W, weight_func=weight_func)
        
        # Expected: gamma(X) = X @ beta
        expected = (self.X @ beta)
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_compute_quadratic_weights(self):
        """Test with quadratic weights (w(X) = X'X for each row)."""
        moment = WeightedAverage()
        
        # Define toy gamma
        def gamma(X):
            return np.sum(X, axis=1, keepdims=True)
        
        # Define quadratic weight function
        def weight_func(X):
            return np.sum(X**2, axis=1)
        
        result = moment.compute(gamma, self.W, weight_func=weight_func)
        
        # Expected: w(X) * gamma(X)
        weights = np.sum(self.X**2, axis=1)
        gamma_vals = np.sum(self.X, axis=1)
        expected = weights * gamma_vals
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_compute_with_first_column_weight(self):
        """Test with weight as first column of X."""
        moment = WeightedAverage()
        
        # Define toy gamma
        def gamma(X):
            return X[:, 1:2]  # Second column
        
        # Define weight as first column
        def weight_func(X):
            return X[:, 0]
        
        result = moment.compute(gamma, self.W, weight_func=weight_func)
        
        # Expected: X[:, 0] * X[:, 1]
        expected = self.X[:, 0] * self.X[:, 1]
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_missing_weight_func_raises_error(self):
        """Test that missing weight_func raises ValueError."""
        moment = WeightedAverage()
        
        def gamma(X):
            return np.ones((X.shape[0], 1))
        
        with pytest.raises(ValueError, match="requires 'weight_func'"):
            moment.compute(gamma, self.W)
    
    def test_dim_property(self):
        """Test that dim property returns 1."""
        moment = WeightedAverage()
        assert moment.dim == 1
    
    def test_mean_approximates_parameter(self):
        """Test that E[m(W, gamma)] approximates true parameter."""
        moment = WeightedAverage()
        
        # True parameter: theta = E[X_1 * X_2]
        true_theta = np.mean(self.X[:, 0] * self.X[:, 1])
        
        def gamma(X):
            return X[:, 1:2]
        
        def weight_func(X):
            return X[:, 0]
        
        result = moment.compute(gamma, self.W, weight_func=weight_func)
        estimated_theta = np.mean(result)
        
        np.testing.assert_allclose(estimated_theta, true_theta, rtol=1e-10)


class TestWeightedAverageDerivative:
    """Tests for WeightedAverageDerivative.compute moment function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        self.d_x = 3
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_compute_linear_gamma_numerical(self):
        """Test derivative of linear gamma using numerical differentiation."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        # Define linear gamma: gamma(X) = beta' X
        beta = np.array([1.0, 2.0, 3.0])
        def gamma(X):
            return (X @ beta).reshape(-1, 1)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Derivative w.r.t. X[:, 0] should be beta[0] = 1.0
        result = moment.compute(gamma, self.W, weight_func=weight_func, deriv_index=0)
        
        expected = np.ones(self.n) * beta[0]
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_compute_quadratic_gamma_numerical(self):
        """Test derivative of quadratic gamma."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        # Define quadratic gamma: gamma(X) = X_0^2
        def gamma(X):
            return (X[:, 0]**2).reshape(-1, 1)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Derivative w.r.t. X[:, 0] should be 2*X[:, 0]
        result = moment.compute(gamma, self.W, weight_func=weight_func, deriv_index=0)
        
        expected = 2 * self.X[:, 0]
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_compute_with_weights(self):
        """Test derivative with non-uniform weights."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        # gamma(X) = X_1
        def gamma(X):
            return X[:, 1:2]
        
        # Weight by X_0
        def weight_func(X):
            return X[:, 0]
        
        # Derivative of X_1 w.r.t. X_1 is 1
        result = moment.compute(gamma, self.W, weight_func=weight_func, deriv_index=1)
        
        expected = self.X[:, 0] * 1.0
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_derivative_of_different_indices(self):
        """Test derivatives w.r.t. different indices."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        # gamma(X) = X_0 + 2*X_1 + 3*X_2
        def gamma(X):
            return (X[:, 0] + 2*X[:, 1] + 3*X[:, 2]).reshape(-1, 1)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Test each derivative
        for idx, expected_coef in enumerate([1.0, 2.0, 3.0]):
            result = moment.compute(gamma, self.W, weight_func=weight_func, deriv_index=idx)
            expected = np.ones(self.n) * expected_coef
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_missing_weight_func_raises_error(self):
        """Test that missing weight_func raises ValueError."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        def gamma(X):
            return np.ones((X.shape[0], 1))
        
        with pytest.raises(ValueError, match="requires 'weight_func'"):
            moment.compute(gamma, self.W, deriv_index=0)
    
    def test_missing_deriv_index_raises_error(self):
        """Test that missing deriv_index raises ValueError."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        def gamma(X):
            return np.ones((X.shape[0], 1))
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        with pytest.raises(ValueError, match="requires 'deriv_index'"):
            moment.compute(gamma, self.W, weight_func=weight_func)
    
    def test_invalid_deriv_index_raises_error(self):
        """Test that invalid deriv_index raises ValueError."""
        moment = WeightedAverageDerivative(use_jax=False)
        
        def gamma(X):
            return np.ones((X.shape[0], 1))
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        with pytest.raises(ValueError, match="deriv_index must be in"):
            moment.compute(gamma, self.W, weight_func=weight_func, deriv_index=10)
    
    def test_dim_property(self):
        """Test that dim property returns 1."""
        moment = WeightedAverageDerivative(use_jax=False)
        assert moment.dim == 1


# Optional: JAX tests (only if JAX is available)
class TestWeightedAverageDerivativeJAX:
    """Tests for JAX-based derivatives."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pytest.importorskip("jax")
        np.random.seed(42)
        self.n = 100
        self.d_x = 3
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_jax_vs_numerical_linear(self):
        """Test that JAX and numerical derivatives agree for linear function."""
        moment_jax = WeightedAverageDerivative(use_jax=True)
        moment_numerical = WeightedAverageDerivative(use_jax=False)
        
        beta = np.array([1.0, 2.0, 3.0])
        def gamma(X):
            return (X @ beta).reshape(-1, 1)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        result_jax = moment_jax.compute(gamma, self.W, weight_func=weight_func, deriv_index=0)
        result_numerical = moment_numerical.compute(gamma, self.W, weight_func=weight_func, deriv_index=0)
        
        np.testing.assert_allclose(result_jax, result_numerical, rtol=1e-4, atol=1e-6)
    
    def test_jax_vs_numerical_quadratic(self):
        """Test that JAX and numerical derivatives agree for quadratic function."""
        moment_jax = WeightedAverageDerivative(use_jax=True)
        moment_numerical = WeightedAverageDerivative(use_jax=False)
        
        def gamma(X):
            return (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        for idx in range(2):
            result_jax = moment_jax.compute(gamma, self.W, weight_func=weight_func, deriv_index=idx)
            result_numerical = moment_numerical.compute(gamma, self.W, weight_func=weight_func, deriv_index=idx)
            
            np.testing.assert_allclose(result_jax, result_numerical, rtol=1e-3, atol=1e-5)


class TestAveragePolicyEffect:
    """Tests for AveragePolicyEffect.compute moment function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        self.d_x = 3
        
        self.X = np.random.randn(self.n, self.d_x)
        self.Y = np.random.randn(self.n, 1)
        self.Z = np.random.randn(self.n, 2)
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_compute_additive_policy(self):
        """Test with additive policy: h(X) = X + delta."""
        moment = AveragePolicyEffect()
        
        # Define linear gamma
        beta = np.array([1.0, 2.0, 3.0])
        def gamma(X):
            return (X @ beta).reshape(-1, 1)
        
        # Define additive policy
        delta = np.array([0.5, 0.0, 0.0])
        def policy_func(X):
            return X + delta
        
        result = moment.compute(gamma, self.W, policy_func=policy_func)
        
        # Expected: gamma(X + delta) - gamma(X) = beta' delta
        expected = np.ones(self.n) * (beta @ delta)
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_compute_multiplicative_policy(self):
        """Test with multiplicative policy: h(X) = alpha * X."""
        moment = AveragePolicyEffect()
        
        # Define quadratic gamma: gamma(X) = sum(X^2)
        def gamma(X):
            return np.sum(X**2, axis=1, keepdims=True)
        
        # Define multiplicative policy
        alpha = 2.0
        def policy_func(X):
            return alpha * X
        
        result = moment.compute(gamma, self.W, policy_func=policy_func)
        
        # Expected: sum((alpha*X)^2) - sum(X^2) = (alpha^2 - 1) * sum(X^2)
        expected = (alpha**2 - 1) * np.sum(self.X**2, axis=1)
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_compute_first_column_shift(self):
        """Test policy that only shifts first column."""
        moment = AveragePolicyEffect()
        
        # gamma(X) = X[:, 0]
        def gamma(X):
            return X[:, 0:1]
        
        # Policy: shift first column by 1
        def policy_func(X):
            X_new = X.copy()
            X_new[:, 0] += 1.0
            return X_new
        
        result = moment.compute(gamma, self.W, policy_func=policy_func)
        
        # Expected: (X[:, 0] + 1) - X[:, 0] = 1
        expected = np.ones(self.n)
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_compute_identity_policy(self):
        """Test that identity policy gives zero effect."""
        moment = AveragePolicyEffect()
        
        def gamma(X):
            return np.sum(X, axis=1, keepdims=True)
        
        # Identity policy
        def policy_func(X):
            return X
        
        result = moment.compute(gamma, self.W, policy_func=policy_func)
        
        # Expected: gamma(X) - gamma(X) = 0
        expected = np.zeros(self.n)
        
        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_missing_policy_func_raises_error(self):
        """Test that missing policy_func raises ValueError."""
        moment = AveragePolicyEffect()
        
        def gamma(X):
            return np.ones((X.shape[0], 1))
        
        with pytest.raises(ValueError, match="requires 'policy_func'"):
            moment.compute(gamma, self.W)
    
    def test_dim_property(self):
        """Test that dim property returns 1."""
        moment = AveragePolicyEffect()
        assert moment.dim == 1
    
    def test_nonlinear_gamma_and_policy(self):
        """Test with nonlinear gamma and policy."""
        moment = AveragePolicyEffect()
        
        # Nonlinear gamma: gamma(X) = exp(X[:, 0])
        def gamma(X):
            return np.exp(X[:, 0:1])
        
        # Policy: double the first column
        def policy_func(X):
            X_new = X.copy()
            X_new[:, 0] *= 2.0
            return X_new
        
        result = moment.compute(gamma, self.W, policy_func=policy_func)
        
        # Expected: exp(2*X[:, 0]) - exp(X[:, 0])
        expected = np.exp(2 * self.X[:, 0]) - np.exp(self.X[:, 0])

        assert result.shape == (self.n,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])