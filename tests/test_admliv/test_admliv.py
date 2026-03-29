# tests/test_admliv/test_admliv.py
"""
Comprehensive tests for ADMLIV class and fit_admliv convenience function.

Tests cover:
- Basic ADMLIV functionality
- Different MLIV estimators (DoubleLasso, NPIV, KIV)
- Convenience function fit_admliv
- Control parameters
- Error handling
- Results validation
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
from numpy.testing import assert_allclose

from admliv import (
    ADMLIV,
    ADMLIVControl,
    ADMLIVResult,
    fit_admliv,
)
from admliv.estimators import (
    DoubleLassoEstimator,
    NpivSieveEstimator,
    KIVEstimator,
)
from admliv.moments import WeightedAverageDerivative
from admliv.utils.featurizers import CoordinatePolyTransform, SimpleFeaturizer


@pytest.fixture
def simple_iv_data():
    """
    Generate simple IV data for testing.

    Model:
        Y = X * beta + eps
        X = Z * pi + v

    Where E[eps | Z] = 0 but E[eps | X] != 0 (endogeneity)
    """
    np.random.seed(42)
    n = 200

    # True parameters
    beta = 2.0
    pi = 1.5
    rho = 0.5  # Endogeneity strength

    # Generate instrument
    Z = np.random.randn(n, 1)

    # Generate errors with correlation
    eps = np.random.randn(n)
    v = rho * eps + np.sqrt(1 - rho**2) * np.random.randn(n)

    # Generate endogenous variable
    X = Z * pi + v.reshape(-1, 1)

    # Generate outcome
    Y = (X * beta).flatten() + eps

    # True average treatment effect (ATE) is beta
    # For WeightedAverageDerivative with uniform weights, this estimates E[dY/dX] = beta
    true_ate = beta

    return {
        'Y': Y,
        'X': X,
        'Z': Z,
        'true_ate': true_ate,
        'n': n
    }


class TestADMLIVBasic:
    """Test basic ADMLIV functionality."""

    def test_admliv_initialization(self):
        """Test ADMLIV can be initialized with different configurations."""
        # With factory function
        def mliv_factory():
            return DoubleLassoEstimator()

        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2)
        )
        assert admliv is not None

        # With estimator instance
        admliv2 = ADMLIV(
            mliv_estimator=DoubleLassoEstimator(),
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2)
        )
        assert admliv2 is not None

    def test_admliv_control_defaults(self):
        """Test ADMLIVControl default values."""
        control = ADMLIVControl()
        assert control.n_folds == 5
        assert control.random_state == 42
        assert control.confidence_level == 0.95
        assert control.verbose is True

    def test_admliv_control_validation(self):
        """Test ADMLIVControl parameter validation."""
        # Invalid n_folds
        with pytest.raises(ValueError, match="n_folds must be greater than 1"):
            ADMLIVControl(n_folds=1)

        # Invalid confidence_level
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            ADMLIVControl(confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            ADMLIVControl(confidence_level=0.0)


class TestADMLIVWithDoubleLasso:
    """Test ADMLIV with DoubleLasso estimator."""

    def test_fit_with_double_lasso(self, simple_iv_data):
        """Test basic fit with DoubleLasso estimator."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        # Define MLIV estimator factory
        def mliv_factory():
            return DoubleLassoEstimator()

        # Define moment function (weighted average derivative = ATE)
        moment = WeightedAverageDerivative()

        # Define weight function (uniform weights for ATE)
        def weight_func(X):
            return np.ones(X.shape[0])

        # Fit ADMLIV
        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Check result type
        assert isinstance(result, ADMLIVResult)

        # Check result attributes exist
        assert hasattr(result, 'theta_debiased')
        assert hasattr(result, 'theta_plugin')
        assert hasattr(result, 'se_debiased')
        assert hasattr(result, 'se_plugin')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 'confidence_level')

        # Check estimates are reasonable (within 3 standard errors)
        true_ate = simple_iv_data['true_ate']
        assert abs(result.theta_debiased - true_ate) < 3 * result.se_debiased

        # Check confidence interval contains true value
        assert result.ci_lower < true_ate < result.ci_upper

        # Check confidence level is correct
        assert result.confidence_level == 0.95

    def test_custom_confidence_level(self, simple_iv_data):
        """Test ADMLIV with custom confidence level."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        # Use 99% confidence level
        control = ADMLIVControl(n_folds=3, confidence_level=0.99, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Check confidence level
        assert result.confidence_level == 0.99

        # 99% CI should be wider than 95% CI
        # (ci_upper - ci_lower) should be larger for 99%
        ci_width_99 = result.ci_upper - result.ci_lower

        # Fit again with 95%
        control_95 = ADMLIVControl(n_folds=3, confidence_level=0.95, verbose=False)
        admliv_95 = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control_95
        )
        result_95 = admliv_95.fit(W, moment, weight_func=weight_func, deriv_index=0)
        ci_width_95 = result_95.ci_upper - result_95.ci_lower

        assert ci_width_99 > ci_width_95


class TestADMLIVWithNpivSieve:
    """Test ADMLIV with NPIV Sieve estimator."""

    def test_fit_with_npiv_sieve(self, simple_iv_data):
        """Test ADMLIV with NpivSieveEstimator."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        # Use polynomial featurizers for sieve basis
        # Note: For MLIV we use lower degree for X than Z (standard in NPIV)
        # But for PGMM we need dim(d(X)) >= dim(b(Z)), so use same degree
        mliv_x_feat = CoordinatePolyTransform(degree=2)
        mliv_z_feat = CoordinatePolyTransform(degree=3)

        pgmm_feat = CoordinatePolyTransform(degree=3)

        def mliv_factory():
            return NpivSieveEstimator(
                x_featurizer=mliv_x_feat,
                z_featurizer=mliv_z_feat
            )

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=pgmm_feat,
            z_featurizer=pgmm_feat,
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Check result is valid
        assert isinstance(result, ADMLIVResult)
        assert not np.isnan(result.theta_debiased)
        assert not np.isnan(result.se_debiased)
        assert result.se_debiased > 0


class TestADMLIVWithKIV:
    """Test ADMLIV with KIV estimator."""

    def test_fit_with_kiv(self, simple_iv_data):
        """Test ADMLIV with KIVEstimator."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return KIVEstimator(bandwidth_scale=1.0, verbose=False)

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Check result is valid
        assert isinstance(result, ADMLIVResult)
        assert not np.isnan(result.theta_debiased)
        assert not np.isnan(result.se_debiased)


class TestFitAdmlivConvenienceFunction:
    """Test fit_admliv convenience function."""

    def test_fit_admliv_basic(self, simple_iv_data):
        """Test fit_admliv convenience function."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        # Use convenience function
        result = fit_admliv(
            W=W,
            mliv_estimator=mliv_factory,
            moment=moment,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            weight_func=weight_func,
            deriv_index=0,
            n_folds=3,
            verbose=False
        )

        assert isinstance(result, ADMLIVResult)
        assert not np.isnan(result.theta_debiased)

    def test_fit_admliv_with_custom_params(self, simple_iv_data):
        """Test fit_admliv with custom n_folds parameter."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        result = fit_admliv(
            W=W,
            mliv_estimator=mliv_factory,
            moment=moment,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            weight_func=weight_func,
            deriv_index=0,
            n_folds=4,
            verbose=False
        )

        assert result.n_folds == 4
        assert isinstance(result, ADMLIVResult)


class TestADMLIVResult:
    """Test ADMLIVResult functionality."""

    def test_result_repr(self, simple_iv_data):
        """Test ADMLIVResult __repr__ method."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Check repr contains key information
        repr_str = repr(result)
        assert 'ADMLIVResult' in repr_str
        assert 'theta_debiased' in repr_str
        assert '95% CI' in repr_str
        assert 'n_samples' in repr_str

    def test_result_summary(self, simple_iv_data):
        """Test ADMLIVResult summary method."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Check summary returns a string
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'Debiased' in summary or 'theta' in summary
        assert 'Confidence Interval' in summary


class TestADMLIVEdgeCases:
    """Test ADMLIV edge cases and error handling."""

    def test_admliv_with_small_sample(self):
        """Test ADMLIV handles small sample sizes."""
        np.random.seed(42)
        n = 50  # Small sample

        Z = np.random.randn(n, 1)
        X = Z + 0.5 * np.random.randn(n, 1)
        Y = 2 * X.flatten() + np.random.randn(n)

        W = {'Y': Y, 'X': X, 'Z': Z}

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        # Should still work with small sample
        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)
        assert isinstance(result, ADMLIVResult)

    def test_admliv_reproducibility(self, simple_iv_data):
        """Test ADMLIV gives same results with same random state."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        # Fit twice with same random state
        control1 = ADMLIVControl(n_folds=3, random_state=123, verbose=False)
        admliv1 = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control1
        )
        result1 = admliv1.fit(W, moment, weight_func=weight_func, deriv_index=0)

        control2 = ADMLIVControl(n_folds=3, random_state=123, verbose=False)
        admliv2 = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control2
        )
        result2 = admliv2.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Results should be identical
        assert_allclose(result1.theta_debiased, result2.theta_debiased)
        assert_allclose(result1.se_debiased, result2.se_debiased)


class TestADMLIVStatisticalProperties:
    """Test statistical properties of ADMLIV estimates."""

    def test_debiased_vs_plugin(self, simple_iv_data):
        """Test that debiased estimate has properties expected."""
        W = {
            'Y': simple_iv_data['Y'],
            'X': simple_iv_data['X'],
            'Z': simple_iv_data['Z']
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        moment = WeightedAverageDerivative()
        weight_func = lambda X: np.ones(X.shape[0])

        control = ADMLIVControl(n_folds=5, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        # Both estimates should be finite
        assert np.isfinite(result.theta_debiased)
        assert np.isfinite(result.theta_plugin)

        # Standard errors should be positive
        assert result.se_debiased > 0
        assert result.se_plugin > 0

        # Confidence interval should be valid
        assert result.ci_lower < result.ci_upper
        assert result.ci_lower < result.theta_debiased < result.ci_upper


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
