# tests/test_pgmm/test_pgmm_linear_iv_cv.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from admliv.core.pgmm_linear_iv_cv import PGMMLinearIVCV
from admliv.core.pgmm_linear_iv import PGMMLinearIV
from admliv.core.control import PGMMCVControl, PGMMControl
from admliv.utils.featurizers import SimpleFeaturizer


class TestPGMMLinearIVCVBasic:
    """Basic tests for PGMMLinearIVCV estimator."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 2
        self.d_z = 2

        # Generate simple IV data
        self.Z = np.random.randn(self.n, self.d_z)
        self.X = self.Z + 0.5 * np.random.randn(self.n, self.d_x)
        self.beta_true = np.array([1.0, 0.5])
        self.Y = self.X @ self.beta_true + np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_pgmm_linear_iv_cv_initialization(self):
        """Test PGMMLinearIVCV initialization."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False
        )

        assert pgmm_cv.x_featurizer is x_feat
        assert pgmm_cv.z_featurizer is z_feat
        assert pgmm_cv.adaptive == False
        assert pgmm_cv.control.n_folds == 3
        assert len(pgmm_cv.control.c_vec) == 3

    def test_pgmm_linear_iv_cv_fit_basic(self):
        """Test basic PGMMLinearIVCV fitting."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        # Fit should work
        pgmm_cv.fit(self.W)

        # Check that rho is estimated
        # SimpleFeaturizer adds intercept, so dimension is d_x + 1
        assert hasattr(pgmm_cv, 'rho_')
        assert pgmm_cv.rho_ is not None
        assert pgmm_cv.rho_.shape[0] == self.d_x + 1
        assert pgmm_cv.is_fitted_ == True

    def test_pgmm_linear_iv_cv_get_rho(self):
        """Test getting structural coefficients."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        rho = pgmm_cv.get_rho()

        # Check shape and properties
        # SimpleFeaturizer adds intercept, so dimension is d_x + 1
        assert rho.shape[0] == self.d_x + 1
        assert isinstance(rho, np.ndarray)


class TestPGMMLinearIVCVCrossValidation:
    """Test cross-validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 2
        self.d_z = 3

        self.Z = np.random.randn(self.n, self.d_z)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :self.d_x] + 0.5 * eps[:, np.newaxis] + 0.3 * np.random.randn(self.n, self.d_x)
        true_rho = np.array([1.0, 0.5])
        self.Y = self.X @ true_rho + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_cv_selects_best_c(self):
        """Test that CV selects best c from grid."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        c_vec = np.array([0.5, 1.0, 1.5, 2.0])
        control = PGMMCVControl(
            n_folds=3,
            c_vec=c_vec
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        # Best c should be from the grid
        assert pgmm_cv.best_c_ in c_vec

        # Best c should minimize CV score
        best_idx = np.argmin(pgmm_cv.cv_scores_)
        assert pgmm_cv.best_c_ == c_vec[best_idx]

    def test_cv_scores_shape(self):
        """Test that CV scores have correct shape."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        c_vec = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        control = PGMMCVControl(
            n_folds=3,
            c_vec=c_vec
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        # CV scores should have same length as c_vec
        assert len(pgmm_cv.cv_scores_) == len(c_vec)
        assert len(pgmm_cv.cv_scores_std_) == len(c_vec)

        # All scores should be non-negative
        assert np.all(pgmm_cv.cv_scores_ >= 0)
        assert np.all(pgmm_cv.cv_scores_std_ >= 0)

    def test_cv_lambda_computation(self):
        """Test that best lambda is computed correctly."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        c_vec = np.array([0.5, 1.0, 2.0])
        control = PGMMCVControl(
            n_folds=3,
            c_vec=c_vec
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        # Check lambda formula: lambda = c * sqrt(log(q) / n) where q = dim(b(Z))
        q = pgmm_cv.n_features_z_
        n = pgmm_cv.n_samples_
        expected_lambda = pgmm_cv.best_c_ * np.sqrt(np.log(q) / n)

        assert pgmm_cv.best_lambda_ is not None
        assert np.abs(pgmm_cv.best_lambda_ - expected_lambda) < 1e-10


class TestPGMMLinearIVCVRefit:
    """Test refit functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 2
        self.d_z = 3

        self.Z = np.random.randn(self.n, self.d_z)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :self.d_x] + 0.5 * eps[:, np.newaxis] + 0.3 * np.random.randn(self.n, self.d_x)
        true_rho = np.array([1.0, 0.5])
        self.Y = self.X @ true_rho + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_refit_true_creates_best_estimator(self):
        """Test that refit=True creates best_estimator_."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        # Should have best_estimator_
        assert hasattr(pgmm_cv, 'best_estimator_')
        assert isinstance(pgmm_cv.best_estimator_, PGMMLinearIV)
        assert pgmm_cv.best_estimator_.is_fitted_ == True

    def test_refit_false_no_get_rho(self):
        """Test that refit=False prevents getting rho."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=False
        )

        pgmm_cv.fit(self.W)

        # Should raise error when trying to get rho
        with pytest.raises(ValueError, match="refit=False"):
            pgmm_cv.get_rho()

    def test_refit_consistency_with_pgmm_linear_iv(self):
        """Test that PGMMLinearIVCV with refit gives same results as PGMMLinearIV with fixed c."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        # Fit PGMMLinearIV with c=1.0
        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=PGMMControl(c=1.0),
            verbose=False
        )
        pgmm.fit(self.W)

        # Fit PGMMLinearIVCV with only c=1.0 in grid
        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=PGMMCVControl(
                n_folds=3,
                c_vec=np.array([1.0]),
                random_state=42
            ),
            verbose=False,
            refit=True
        )
        pgmm_cv.fit(self.W)

        # Best c should be 1.0
        assert pgmm_cv.best_c_ == 1.0

        # Rho estimates should be very close (same c, same full data after refit)
        # Tight tolerance to allow for small numerical differences in SGD algo
        np.testing.assert_array_almost_equal(
            pgmm_cv.rho_,
            pgmm.rho_,
            decimal=6,
            err_msg="PGMMLinearIVCV with single c should match PGMMLinearIV with same c"
        )


class TestPGMMLinearIVCVAdaptive:
    """Test adaptive PGMMLinearIVCV with two-step estimation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 3
        self.d_z = 5

        # Sparse true model
        self.Z = np.random.randn(self.n, self.d_z)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :self.d_x] + 0.5 * eps[:, np.newaxis] + 0.3 * np.random.randn(self.n, self.d_x)
        # Only first coefficient is non-zero
        true_rho = np.array([1.0, 0.0, 0.0])
        self.Y = self.X @ true_rho + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_adaptive_pgmm_linear_iv_cv(self):
        """Test adaptive PGMMLinearIVCV estimation."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=True,
            control=control,
            verbose=False,
            refit=True
        )

        # Fit with adaptive estimation
        pgmm_cv.fit(self.W)

        # Check that estimation completed
        assert hasattr(pgmm_cv, 'rho_')
        assert pgmm_cv.is_fitted_ == True

        # Adaptive should produce sparse solution
        n_nonzero = np.count_nonzero(pgmm_cv.rho_)
        n_total = len(pgmm_cv.rho_)
        assert n_nonzero < n_total  # Should have some zeros due to L1 penalty


class TestPGMMLinearIVCVControl:
    """Test control parameters for PGMMLinearIVCV."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 3)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :2] + 0.5 * eps[:, np.newaxis] + np.random.randn(self.n, 2)
        self.Y = self.X[:, 0] + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_custom_control_parameters(self):
        """Test PGMMLinearIVCV with custom control parameters."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            maxiter=1000,
            optTol=1e-4,
            zeroThreshold=1e-5,
            n_folds=5,
            c_vec=np.array([0.5, 1.0, 1.5, 2.0]),
            random_state=123
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        assert pgmm_cv.control.maxiter == 1000
        assert pgmm_cv.control.optTol == 1e-4
        assert pgmm_cv.control.n_folds == 5
        assert pgmm_cv.control.random_state == 123
        assert pgmm_cv.is_fitted_ == True

    def test_default_c_vec(self):
        """Test that default c_vec is used when None."""
        control = PGMMCVControl(c_vec=None)

        # Should have default grid
        assert control.c_vec is not None
        assert len(control.c_vec) > 0
        assert np.all(control.c_vec > 0)

    def test_invalid_n_folds(self):
        """Test that invalid n_folds raises error."""
        with pytest.raises(ValueError, match="n_folds must be greater than 1"):
            control = PGMMCVControl(n_folds=1)


class TestPGMMLinearIVCVCriterion:
    """Test GMM criterion computation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 3)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :2] + 0.5 * eps[:, np.newaxis] + np.random.randn(self.n, 2)
        self.Y = self.X[:, 0] + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_compute_criterion(self):
        """Test GMM criterion computation."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        # Compute criterion
        criterion = pgmm_cv.compute_criterion(self.W)

        assert isinstance(criterion, float)
        assert criterion >= 0  # Criterion should be non-negative


class TestPGMMLinearIVCVComparison:
    """Test comparison between PGMMLinearIVCV and PGMMLinearIV."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200

        self.Z = np.random.randn(self.n, 4)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :3] + 0.3 * eps[:, np.newaxis] + 0.2 * np.random.randn(self.n, 3)
        self.Y = self.X[:, 0] + 0.5 * self.X[:, 1] + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_cv_can_improve_over_default(self):
        """Test that CV can potentially improve over default c=1.0."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        # Fit PGMMLinearIV with default c=1.0
        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=PGMMControl(c=1.0),
            verbose=False
        )
        pgmm.fit(self.W)
        criterion_pgmm = pgmm.compute_criterion(self.W)

        # Fit PGMMLinearIVCV with grid including 1.0
        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=PGMMCVControl(
                n_folds=3,
                c_vec=np.array([0.1, 0.5, 1.0, 2.0, 5.0])
            ),
            verbose=False,
            refit=True
        )
        pgmm_cv.fit(self.W)
        criterion_pgmm_cv = pgmm_cv.compute_criterion(self.W)

        # PGMMLinearIVCV should not be worse (it can select c=1.0 if that's best)
        assert criterion_pgmm_cv <= criterion_pgmm + 1e-6  # Small tolerance for numerical differences


class TestPGMMLinearIVCVEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 3)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :2] + 0.5 * eps[:, np.newaxis] + np.random.randn(self.n, 2)
        self.Y = self.X[:, 0] + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_get_rho_before_fit(self):
        """Test that getting rho before fitting raises error."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )

        with pytest.raises(ValueError, match="must be fitted"):
            pgmm_cv.get_rho()

    def test_underidentification_error(self):
        """Test that under-identification raises error."""
        # More features in X than Z -> under-identified
        np.random.seed(42)
        n = 100
        Z = np.random.randn(n, 2)  # 2 instruments
        X = np.random.randn(n, 4)  # 4 endogenous variables
        Y = np.random.randn(n, 1)

        W = {'Y': Y, 'X': X, 'Z': Z}

        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0])
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False
        )

        # Should raise ValueError during CV (in one of the folds)
        with pytest.raises(ValueError, match="Under-identified"):
            pgmm_cv.fit(W)


class TestPGMMLinearIVCVReproducibility:
    """Test reproducibility with random_state."""

    def setup_method(self):
        """Set up test fixtures."""
        self.n = 200
        self.d_x = 2
        self.d_z = 3

        np.random.seed(42)
        self.Z = np.random.randn(self.n, self.d_z)
        eps = np.random.randn(self.n)
        self.X = self.Z[:, :self.d_x] + 0.5 * eps[:, np.newaxis] + 0.3 * np.random.randn(self.n, self.d_x)
        true_rho = np.array([1.0, 0.5])
        self.Y = self.X @ true_rho + eps

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_random_state_reproducibility(self):
        """Test that same random_state gives same results."""
        x_feat = SimpleFeaturizer()
        z_feat = SimpleFeaturizer()

        # First run
        control1 = PGMMCVControl(
            n_folds=5,
            c_vec=np.array([0.5, 1.0, 2.0]),
            random_state=123
        )

        pgmm_cv1 = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control1,
            verbose=False,
            refit=True
        )
        pgmm_cv1.fit(self.W)

        # Second run with same random_state
        control2 = PGMMCVControl(
            n_folds=5,
            c_vec=np.array([0.5, 1.0, 2.0]),
            random_state=123
        )

        pgmm_cv2 = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control2,
            verbose=False,
            refit=True
        )
        pgmm_cv2.fit(self.W)

        # Should give identical results
        assert pgmm_cv1.best_c_ == pgmm_cv2.best_c_
        np.testing.assert_array_almost_equal(pgmm_cv1.cv_scores_, pgmm_cv2.cv_scores_)
        np.testing.assert_array_almost_equal(pgmm_cv1.rho_, pgmm_cv2.rho_)


class TestPGMMLinearIVCVRecovery:
    """Test parameter recovery in simple cases."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 500  # Larger sample for better recovery
        self.d_x = 2
        self.d_z = 3  # Overidentified

        # Generate data with strong instruments
        self.Z = np.random.randn(self.n, self.d_z)
        self.X = self.Z[:, :self.d_x] + 0.1 * np.random.randn(self.n, self.d_x)
        self.beta_true = np.array([1.0, -0.5])
        self.Y = self.X @ self.beta_true + 0.1 * np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_parameter_recovery_no_penalty(self):
        """Test that true parameters are recovered with low penalty."""
        x_feat = SimpleFeaturizer(include_bias=False)
        z_feat = SimpleFeaturizer(include_bias=False)

        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.1, 0.5, 1.0]),  # Small penalties for unbiased recovery
            random_state=42
        )

        pgmm_cv = PGMMLinearIVCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )

        pgmm_cv.fit(self.W)

        # Should recover true parameters approximately
        np.testing.assert_array_almost_equal(
            pgmm_cv.rho_,
            self.beta_true,
            decimal=2,  # Loose tolerance due to estimation noise
            err_msg="Should approximately recover true parameters"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
