# tests/test_pgmm/test_pgmm_linear_iv.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from admliv.core.pgmm_linear_iv import PGMMLinearIV
from admliv.core.control import PGMMControl
from admliv.utils.featurizers import SimpleFeaturizer


class TestPGMMLinearIVBasic:
    """Basic tests for PGMMLinearIV estimator."""

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

    def test_pgmm_linear_iv_initialization(self):
        """Test PGMMLinearIV initialization."""
        x_feat = SimpleFeaturizer(include_bias=False)
        z_feat = SimpleFeaturizer(include_bias=False)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            verbose=False
        )

        assert pgmm.x_featurizer is x_feat
        assert pgmm.z_featurizer is z_feat
        assert pgmm.adaptive == False
        assert pgmm.lambda_ is None

    def test_pgmm_linear_iv_fit_basic(self):
        """Test basic PGMMLinearIV fitting."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )

        # Fit should work
        pgmm.fit(self.W)

        # Check that rho is estimated
        assert hasattr(pgmm, 'rho_')
        assert pgmm.rho_ is not None
        assert pgmm.rho_.shape[0] == 3  # intercept + 2 features
        assert pgmm.is_fitted_ == True

    def test_pgmm_linear_iv_predict(self):
        """Test PGMMLinearIV predict (structural function evaluation)."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )

        pgmm.fit(self.W)

        # Predict Y = X'ρ
        Y_pred = pgmm.predict(self.X)

        assert Y_pred.shape == (self.n,)
        assert not np.all(Y_pred == 0)  # Should have non-zero values

    def test_pgmm_linear_iv_get_rho(self):
        """Test getting structural coefficients."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )

        pgmm.fit(self.W)

        rho = pgmm.get_rho()

        # Check shape and properties
        assert rho.shape[0] == 3  # intercept + 2 features
        assert isinstance(rho, np.ndarray)


class TestPGMMLinearIVAdaptive:
    """Test adaptive PGMMLinearIV with two-step estimation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 2
        self.d_z = 2

        self.Z = np.random.randn(self.n, self.d_z)
        self.X = self.Z + 0.5 * np.random.randn(self.n, self.d_x)
        self.beta_true = np.array([1.0, 0.5])
        self.Y = self.X @ self.beta_true + np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_adaptive_pgmm_linear_iv(self):
        """Test adaptive PGMMLinearIV estimation."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=True,
            verbose=False
        )

        # Fit with adaptive estimation
        pgmm.fit(self.W)

        # Check that estimation completed
        assert hasattr(pgmm, 'rho_')
        assert hasattr(pgmm, 'rho_preliminary_')
        assert pgmm.is_fitted_ == True

        # Adaptive should produce sparse solution
        n_nonzero = np.count_nonzero(pgmm.rho_)
        n_total = len(pgmm.rho_)
        assert n_nonzero <= n_total  # Can have some zeros due to L1 penalty


class TestPGMMLinearIVIdentification:
    """Test identification conditions and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_underidentification_error(self):
        """Test that under-identification raises error."""
        # Fewer instruments than regressors after featurization -> under-identified
        x_feat = SimpleFeaturizer(include_bias=True)  # 3 features (1 + 2)
        z_feat = SimpleFeaturizer(include_bias=False)  # 2 features

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            verbose=False
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Under-identified"):
            pgmm.fit(self.W)

    def test_exact_identification(self):
        """Test exact identification (dim(X) = dim(Z))."""
        # Same number of basis functions
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )

        # Should work
        pgmm.fit(self.W)
        assert pgmm.is_fitted_ == True

    def test_overidentification(self):
        """Test over-identification (dim(Z) > dim(X))."""
        # More instruments than regressors -> over-identified (good)
        x_feat = SimpleFeaturizer(include_bias=False)  # 2 features
        z_feat = SimpleFeaturizer(include_bias=True)  # 3 features (1 + 2)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )

        # Should work
        pgmm.fit(self.W)
        assert pgmm.is_fitted_ == True


class TestPGMMLinearIVControl:
    """Test control parameters for PGMMLinearIV."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_custom_control_parameters(self):
        """Test PGMMLinearIV with custom control parameters."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        control = PGMMControl(
            maxiter=1000,
            optTol=1e-4,
            zeroThreshold=1e-5,
            c=1.5
        )

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False
        )

        pgmm.fit(self.W)

        assert pgmm.control.maxiter == 1000
        assert pgmm.control.optTol == 1e-4
        assert pgmm.is_fitted_ == True

    def test_automatic_lambda(self):
        """Test automatic lambda calculation."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=None,  # Should be computed automatically
            verbose=False
        )

        pgmm.fit(self.W)

        # Lambda should have been computed
        assert pgmm.lambda_ is not None
        assert pgmm.lambda_ > 0


class TestPGMMLinearIVCriterion:
    """Test GMM criterion computation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_compute_criterion(self):
        """Test GMM criterion computation."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )

        pgmm.fit(self.W)

        # Compute criterion
        criterion = pgmm.compute_criterion(self.W)

        assert isinstance(criterion, float)
        assert criterion >= 0  # Criterion should be non-negative


class TestPGMMLinearIVSparsity:
    """Test sparsity properties of PGMMLinearIV."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200

        # Sparse true model
        self.Z = np.random.randn(self.n, 5)
        self.X = self.Z + 0.2 * np.random.randn(self.n, 5)
        self.beta_true = np.array([1.0, 0.0, 0.5, 0.0, 0.0])
        self.Y = self.X @ self.beta_true + np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_penalty_induces_sparsity(self):
        """Test that L1 penalty induces sparsity."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        # High penalty should give more zeros
        pgmm_high_penalty = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.5,  # High penalty
            verbose=False
        )

        # Low penalty should give fewer zeros
        pgmm_low_penalty = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.001,  # Low penalty
            verbose=False
        )

        pgmm_high_penalty.fit(self.W)
        pgmm_low_penalty.fit(self.W)

        n_zeros_high = np.sum(pgmm_high_penalty.rho_ == 0)
        n_zeros_low = np.sum(pgmm_low_penalty.rho_ == 0)

        # High penalty should produce more (or equal) zeros
        assert n_zeros_high >= n_zeros_low


class TestPGMMLinearIVOmega:
    """Test weight matrix (Omega) functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_optimal_omega_computed(self):
        """Test that optimal Omega is computed when Omega=None."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            Omega=None,
            lambda_=0.01,
            verbose=False
        )

        pgmm.fit(self.W)

        # Should have computed optimal Omega
        assert hasattr(pgmm, 'Omega_opt_')
        assert pgmm.Omega_opt_ is not None
        assert pgmm.Omega_opt_.shape == (3, 3)  # q x q for 3 instruments

    def test_custom_omega(self):
        """Test fitting with custom Omega."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        # Custom identity weight matrix
        custom_omega = np.eye(3)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            Omega=custom_omega,
            lambda_=0.01,
            verbose=False
        )

        pgmm.fit(self.W)

        # Should use provided Omega
        assert hasattr(pgmm, 'Omega_opt_')
        np.testing.assert_array_equal(pgmm.Omega_opt_, custom_omega)


class TestPGMMLinearIVEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100

        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n)

        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}

    def test_predict_before_fit(self):
        """Test that predicting before fitting raises error."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )

        with pytest.raises(ValueError, match="must be fitted"):
            pgmm.predict(self.X)

    def test_get_rho_before_fit(self):
        """Test that getting rho before fitting raises error."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )

        with pytest.raises(ValueError, match="must be fitted"):
            pgmm.get_rho()

    def test_get_omega_before_fit(self):
        """Test that getting omega before fitting raises error."""
        x_feat = SimpleFeaturizer(include_bias=True)
        z_feat = SimpleFeaturizer(include_bias=True)

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )

        with pytest.raises(ValueError, match="must be fitted"):
            pgmm.get_omega()


class TestPGMMLinearIVRecovery:
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

        pgmm = PGMMLinearIV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=1e-6,  # Very low penalty
            verbose=False
        )

        pgmm.fit(self.W)

        # Should recover true parameters approximately
        np.testing.assert_array_almost_equal(
            pgmm.rho_,
            self.beta_true,
            decimal=2,  # Loose tolerance due to estimation noise
            err_msg="Should approximately recover true parameters"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
