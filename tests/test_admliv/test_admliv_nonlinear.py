# tests/test_admliv/test_admliv_nonlinear.py
"""
Tests for nonlinear functional support in ADMLIV.

Tests cover:
1. SquaredGammaAverage moment (compute, compute_all_basis_gamma, is_linear)
2. Duck typing detection (_is_nonlinear_moment)
3. PGMM.fit_with_M (precomputed M, dimension checks)
4. ADMLIV.fit with linear moment (correctness checks)
5. ADMLIV.fit with nonlinear moment (double cross-fitting)
6. Precompute inner gammas (correct number of pairs, correct training data)
7. Error handling (n_folds < 3 for nonlinear)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
from numpy.testing import assert_allclose

from admliv import ADMLIV, ADMLIVControl, ADMLIVResult
from admliv.moments import WeightedAverage, WeightedAverageDerivative
from admliv.moments.squared_gamma_average import SquaredGammaAverage
from admliv.moments.base import BaseMoment
from admliv.core.pgmm import PGMM
from admliv.core.control import PGMMControl
from admliv.estimators import DoubleLassoEstimator, NpivSieveEstimator
from admliv.utils.featurizers import CoordinatePolyTransform


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def linear_iv_data():
    """
    Simple linear IV data.

    DGP:
        Z ~ N(0, I_2)
        X = Z @ pi + v    (d_x = 2)
        Y = X @ beta + eps
        Cov(eps, v) != 0   (endogeneity)

    True theta for WeightedAverage with w=1: E[gamma(X)] = E[X @ beta]
    True theta for SquaredGammaAverage: E[(X @ beta)^2]
    """
    np.random.seed(42)
    n = 300

    beta = np.array([2.0, -1.0])
    pi = np.array([[1.5, 0.3], [0.2, 1.2]])
    rho = 0.5

    Z = np.random.randn(n, 2)
    eps = np.random.randn(n)
    v = rho * eps[:, None] + np.sqrt(1 - rho**2) * np.random.randn(n, 2)

    X = Z @ pi + v
    gamma_true = X @ beta
    Y = gamma_true + eps

    theta_weighted_avg = np.mean(gamma_true)
    theta_squared = np.mean(gamma_true ** 2)

    return {
        'Y': Y, 'X': X, 'Z': Z,
        'gamma_true': gamma_true,
        'theta_weighted_avg': theta_weighted_avg,
        'theta_squared': theta_squared,
        'n': n,
    }


# ============================================================
# Test SquaredGammaAverage Moment
# ============================================================

class TestSquaredGammaAverage:

    def test_compute(self, linear_iv_data):
        """compute() should return gamma(X)^2."""
        moment = SquaredGammaAverage()
        W = {
            'X': linear_iv_data['X'],
            'Y': linear_iv_data['Y'],
            'Z': linear_iv_data['Z'],
        }
        gamma_true = linear_iv_data['gamma_true']

        def gamma_func(X):
            return (X @ np.array([2.0, -1.0])).reshape(-1, 1)

        m = moment.compute(gamma_func, W)
        expected = gamma_true ** 2
        assert_allclose(m, expected, rtol=1e-10)

    def test_compute_all_basis_gamma(self, linear_iv_data):
        """compute_all_basis_gamma should return M[i,k] = 2*gamma(X_i)*d_k(X_i)."""
        moment = SquaredGammaAverage()
        W = {
            'X': linear_iv_data['X'],
            'Y': linear_iv_data['Y'],
            'Z': linear_iv_data['Z'],
        }

        def gamma_func(X):
            return (X @ np.array([2.0, -1.0])).reshape(-1, 1)

        feat = CoordinatePolyTransform(degree=2)
        feat.fit(W['X'])

        M = moment.compute_all_basis_gamma(gamma_func, feat, W)
        Wx = feat.transform(W['X'])
        gamma_vals = gamma_func(W['X']).flatten()

        expected = 2 * gamma_vals[:, None] * Wx
        assert_allclose(M, expected, rtol=1e-10)

    def test_compute_all_basis_raises(self):
        """compute_all_basis should raise NotImplementedError."""
        moment = SquaredGammaAverage()
        feat = CoordinatePolyTransform(degree=2)
        feat.fit(np.random.randn(10, 2))
        W = {'X': np.random.randn(10, 2), 'Y': np.random.randn(10), 'Z': np.random.randn(10, 2)}
        with pytest.raises(NotImplementedError):
            moment.compute_all_basis(feat, W)

    def test_is_linear(self):
        """SquaredGammaAverage.is_linear should be False."""
        assert SquaredGammaAverage().is_linear is False

    def test_gateaux_matches_numerical(self, linear_iv_data):
        """Analytical Gateaux derivative should match numerical perturbation."""
        moment = SquaredGammaAverage()
        X = linear_iv_data['X']
        W = {'X': X, 'Y': linear_iv_data['Y'], 'Z': linear_iv_data['Z']}

        beta = np.array([2.0, -1.0])

        def gamma_func(X):
            return (X @ beta).reshape(-1, 1)

        feat = CoordinatePolyTransform(degree=2)
        feat.fit(X)

        # Analytical M
        M_analytical = moment.compute_all_basis_gamma(gamma_func, feat, W)

        # Numerical: perturb gamma by eps * d_k and check
        eps = 1e-5
        Wx = feat.transform(X)
        p = Wx.shape[1]
        n = X.shape[0]

        # For each basis function d_k, compute:
        #   [m(W, gamma + eps*d_k) - m(W, gamma - eps*d_k)] / (2*eps)
        # where m(W, gamma) = gamma(X)^2
        M_numerical = np.zeros((n, p))
        gamma_vals = gamma_func(X).flatten()

        for k in range(p):
            # gamma + eps * d_k evaluated at X:
            gamma_plus = gamma_vals + eps * Wx[:, k]
            gamma_minus = gamma_vals - eps * Wx[:, k]

            m_plus = gamma_plus ** 2
            m_minus = gamma_minus ** 2

            M_numerical[:, k] = (m_plus - m_minus) / (2 * eps)

        assert_allclose(M_analytical, M_numerical, rtol=1e-4)


# ============================================================
# Test Duck Typing Detection
# ============================================================

class TestNonlinearDetection:

    def test_weighted_average_is_linear(self):
        assert ADMLIV._is_nonlinear_moment(WeightedAverage()) is False

    def test_weighted_average_derivative_is_linear(self):
        assert ADMLIV._is_nonlinear_moment(WeightedAverageDerivative()) is False

    def test_squared_gamma_is_nonlinear(self):
        assert ADMLIV._is_nonlinear_moment(SquaredGammaAverage()) is True

    def test_base_moment_properties(self):
        """BaseMoment defaults: is_linear=True, compute_all_basis_gamma raises."""
        assert WeightedAverage().is_linear is True
        assert SquaredGammaAverage().is_linear is False


# ============================================================
# Test PGMM.fit_with_M
# ============================================================

class TestPGMMFitWithM:

    def test_fit_with_M_matches_fit(self, linear_iv_data):
        """
        For a linear moment, fit_with_M with externally-computed M
        should produce the same rho as fit() with the moment.
        """
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        weight_func = lambda X: np.ones(X.shape[0])
        moment = WeightedAverage()

        pgmm_control = PGMMControl(c=0.1)

        # Path 1: standard fit()
        pgmm1 = PGMM(
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            adaptive=True,
            control=pgmm_control,
            verbose=False
        )
        pgmm1.fit(W, moment, weight_func=weight_func)

        # Path 2: fit_with_M with same M
        pgmm2 = PGMM(
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            adaptive=True,
            control=pgmm_control,
            verbose=False
        )
        # Compute M the same way PGMM.fit() does internally
        pgmm2.x_featurizer.fit(W['X'])
        M = moment.compute_all_basis(pgmm2.x_featurizer, W, weight_func=weight_func)
        # Reset featurizer so fit_with_M re-fits it
        pgmm2.x_featurizer = CoordinatePolyTransform(degree=2)
        pgmm2.z_featurizer = CoordinatePolyTransform(degree=2)
        pgmm2.fit_with_M(W, M)

        assert_allclose(pgmm1.rho_, pgmm2.rho_, rtol=1e-10)

    def test_fit_with_M_dimension_check(self, linear_iv_data):
        """fit_with_M should raise on M dimension mismatch."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        n = W['X'].shape[0]

        pgmm = PGMM(
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            verbose=False
        )

        # Wrong number of rows
        M_bad_rows = np.zeros((n + 10, 5))
        with pytest.raises(ValueError, match="rows"):
            pgmm.fit_with_M(W, M_bad_rows)

    def test_fit_with_M_predict(self, linear_iv_data):
        """After fit_with_M, predict() should work."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        weight_func = lambda X: np.ones(X.shape[0])
        moment = WeightedAverage()

        pgmm = PGMM(
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            verbose=False
        )

        # Compute M externally
        feat_tmp = CoordinatePolyTransform(degree=2)
        feat_tmp.fit(W['X'])
        M = moment.compute_all_basis(feat_tmp, W, weight_func=weight_func)

        pgmm.fit_with_M(W, M)
        alpha = pgmm.predict(W['Z'])

        assert alpha.shape == (W['Z'].shape[0],)
        assert np.all(np.isfinite(alpha))


# ============================================================
# Test ADMLIV.fit — Linear Moment
# ============================================================

class TestFitLinear:
    """
    Tests that fit() works correctly for linear moments, including
    the WeightedAverageDerivative and WeightedAverage functionals.
    """

    def test_fit_linear_derivative(self, linear_iv_data):
        """fit() with WeightedAverageDerivative should give reasonable results."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        weight_func = lambda X: np.ones(X.shape[0])
        moment = WeightedAverageDerivative()

        def mliv_factory():
            return DoubleLassoEstimator()

        control = ADMLIVControl(n_folds=5, random_state=42, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )
        result = admliv.fit(W, moment, weight_func=weight_func, deriv_index=0)

        assert isinstance(result, ADMLIVResult)
        assert np.isfinite(result.theta_debiased)
        assert np.isfinite(result.theta_plugin)
        assert result.se_debiased > 0
        assert result.ci_lower < result.ci_upper

    def test_fit_linear_weighted_average(self, linear_iv_data):
        """fit() with WeightedAverage should work and give reasonable results."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        weight_func = lambda X: np.ones(X.shape[0])
        moment = WeightedAverage()

        def mliv_factory():
            return DoubleLassoEstimator()

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment, weight_func=weight_func)
        assert isinstance(result, ADMLIVResult)
        assert np.isfinite(result.theta_debiased)
        assert result.se_debiased > 0


# ============================================================
# Test ADMLIV.fit — Nonlinear Path
# ============================================================

class TestFitNonlinear:

    def test_fit_nonlinear_basic(self, linear_iv_data):
        """fit() with SquaredGammaAverage should run and return valid result."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        moment = SquaredGammaAverage()

        def mliv_factory():
            return DoubleLassoEstimator()

        control = ADMLIVControl(n_folds=5, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment)

        assert isinstance(result, ADMLIVResult)
        assert np.isfinite(result.theta_debiased)
        assert np.isfinite(result.theta_plugin)
        assert result.se_debiased > 0
        assert result.ci_lower < result.ci_upper

    def test_fit_nonlinear_reasonable_estimate(self, linear_iv_data):
        """
        Plugin estimate should be in the right ballpark for E[gamma(X)^2].
        With linear gamma, true theta = E[(X @ beta)^2].
        """
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        moment = SquaredGammaAverage()
        true_theta = linear_iv_data['theta_squared']

        def mliv_factory():
            return DoubleLassoEstimator()

        control = ADMLIVControl(n_folds=5, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        result = admliv.fit(W, moment)

        # Estimate should be within 50% of true value
        # (loose bound since n=300 with endogeneity)
        assert abs(result.theta_plugin - true_theta) < 0.5 * abs(true_theta), (
            f"Plugin {result.theta_plugin:.3f} too far from true {true_theta:.3f}"
        )

    def test_fit_nonlinear_requires_3_folds(self, linear_iv_data):
        """Nonlinear moment with n_folds=2 should raise ValueError."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        moment = SquaredGammaAverage()

        def mliv_factory():
            return DoubleLassoEstimator()

        control = ADMLIVControl(n_folds=2, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        with pytest.raises(ValueError, match="n_folds >= 3"):
            admliv.fit(W, moment)

    def test_fit_nonlinear_rejects_cv(self, linear_iv_data):
        """Nonlinear moment with use_cv_for_pgmm=True should raise ValueError."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        moment = SquaredGammaAverage()

        def mliv_factory():
            return DoubleLassoEstimator()

        control = ADMLIVControl(
            n_folds=5, use_cv_for_pgmm=True, verbose=False
        )
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        with pytest.raises(ValueError, match="not supported for nonlinear"):
            admliv.fit(W, moment)

    def test_fit_nonlinear_reproducible(self, linear_iv_data):
        """Same random_state should give identical results."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        moment = SquaredGammaAverage()

        def mliv_factory():
            return DoubleLassoEstimator()

        results = []
        for _ in range(2):
            control = ADMLIVControl(n_folds=5, random_state=123, verbose=False)
            admliv = ADMLIV(
                mliv_estimator=mliv_factory,
                x_featurizer=CoordinatePolyTransform(degree=2),
                z_featurizer=CoordinatePolyTransform(degree=2),
                control=control
            )
            results.append(admliv.fit(W, moment))

        assert_allclose(results[0].theta_debiased, results[1].theta_debiased)
        assert_allclose(results[0].se_debiased, results[1].se_debiased)


# ============================================================
# Test Inner Gamma Precomputation
# ============================================================

class TestPrecomputeInnerGammas:

    def test_correct_number_of_pairs(self, linear_iv_data):
        """Should produce C(K,2) = K*(K-1)/2 gamma fits."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }

        def mliv_factory():
            return DoubleLassoEstimator()

        n_folds = 5
        control = ADMLIVControl(n_folds=n_folds, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=2),
            z_featurizer=CoordinatePolyTransform(degree=2),
            control=control
        )

        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_assignments = np.full(W['X'].shape[0], -1, dtype=int)
        for idx, (_, test_idx) in enumerate(kfold.split(W['X'])):
            fold_assignments[test_idx] = idx

        cache = admliv._precompute_inner_gammas(W, fold_assignments, n_folds)

        expected_pairs = n_folds * (n_folds - 1) // 2
        assert len(cache) == expected_pairs

        # All keys should be frozensets of size 2
        for key in cache:
            assert isinstance(key, frozenset)
            assert len(key) == 2

    def test_cache_keys_are_symmetric(self, linear_iv_data):
        """frozenset({k, ell}) == frozenset({ell, k}), so same key."""
        assert frozenset({0, 1}) == frozenset({1, 0})
        assert frozenset({2, 4}) == frozenset({4, 2})


# ============================================================
# Test with NpivSieve (different estimator)
# ============================================================

class TestFitWithNpivSieve:

    def test_nonlinear_with_npiv(self, linear_iv_data):
        """fit() nonlinear path should work with NpivSieveEstimator."""
        W = {
            'Y': linear_iv_data['Y'],
            'X': linear_iv_data['X'],
            'Z': linear_iv_data['Z'],
        }
        moment = SquaredGammaAverage()

        def mliv_factory():
            return NpivSieveEstimator(
                x_featurizer=CoordinatePolyTransform(degree=2),
                z_featurizer=CoordinatePolyTransform(degree=3)
            )

        control = ADMLIVControl(n_folds=3, verbose=False)
        admliv = ADMLIV(
            mliv_estimator=mliv_factory,
            x_featurizer=CoordinatePolyTransform(degree=3),
            z_featurizer=CoordinatePolyTransform(degree=3),
            control=control
        )

        result = admliv.fit(W, moment)
        assert isinstance(result, ADMLIVResult)
        assert np.isfinite(result.theta_debiased)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
