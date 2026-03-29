# tests/test_pgmm/test_pgmm.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from admliv.core.pgmm import PGMM
from admliv.core.control import PGMMControl
from admliv.moments import WeightedAverage
from admliv.utils.featurizers import CoordinatePolyTransform


class TestPGMMBasic:
    """Basic tests for PGMM estimator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 2
        self.d_z = 2
        
        # Generate simple data
        self.Z = np.random.randn(self.n, self.d_z)
        self.X = self.Z + 0.5 * np.random.randn(self.n, self.d_x)
        self.Y = self.X[:, 0] + 0.5 * self.X[:, 1] + np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_pgmm_initialization(self):
        """Test PGMM initialization."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            verbose=False
        )
        
        assert pgmm.x_featurizer is x_feat
        assert pgmm.z_featurizer is z_feat
        assert pgmm.adaptive == False
        assert pgmm.lambda_ is None
    
    def test_pgmm_fit_basic(self):
        """Test basic PGMM fitting."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Fit should work
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        # Check that rho is estimated
        assert hasattr(pgmm, 'rho_')
        assert pgmm.rho_ is not None
        assert pgmm.rho_.shape[0] == z_feat.fit(self.Z).n_features_out_
        assert pgmm.is_fitted_ == True
    
    def test_pgmm_predict(self):
        """Test PGMM predict (Riesz representer evaluation)."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        # Predict alpha(Z)
        alpha = pgmm.predict(self.Z)
        
        assert alpha.shape == (self.n,)
        assert not np.all(alpha == 0)  # Should have non-zero values
    
    def test_pgmm_get_rho(self):
        """Test getting Riesz representer coefficients."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        rho = pgmm.get_rho()
        
        # Check shape and properties
        assert rho.shape[0] == z_feat.n_features_out_
        assert isinstance(rho, np.ndarray)


class TestPGMMAdaptive:
    """Test adaptive PGMM with two-step estimation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        self.d_x = 2
        self.d_z = 2
        
        self.Z = np.random.randn(self.n, self.d_z)
        self.X = self.Z + 0.5 * np.random.randn(self.n, self.d_x)
        self.Y = self.X[:, 0] + 0.5 * self.X[:, 1] + np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_adaptive_pgmm(self):
        """Test adaptive PGMM estimation."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=True,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Fit with adaptive estimation
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        # Check that estimation completed
        assert hasattr(pgmm, 'rho_')
        assert pgmm.is_fitted_ == True
        
        # Adaptive should produce sparse solution
        n_nonzero = np.count_nonzero(pgmm.rho_)
        n_total = len(pgmm.rho_)
        assert n_nonzero < n_total  # Should have some zeros due to L1 penalty


class TestPGMMIdentification:
    """Test identification conditions and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        
        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_underidentification_error(self):
        """Test that under-identification raises error."""
        # More basis functions in Z than X -> under-identified
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)  # 3 features
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)  # 5 features
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Under-identified"):
            pgmm.fit(self.W, moment, weight_func=weight_func)
    
    def test_exact_identification(self):
        """Test exact identification (dim(X) = dim(Z))."""
        # Same number of basis functions
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Should work
        pgmm.fit(self.W, moment, weight_func=weight_func)
        assert pgmm.is_fitted_ == True
    
    def test_overidentification(self):
        """Test over-identification (dim(X) > dim(Z))."""
        # More basis functions in X than Z -> over-identified (good)
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)  # 5 features
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)  # 3 features
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Should work
        pgmm.fit(self.W, moment, weight_func=weight_func)
        assert pgmm.is_fitted_ == True


class TestPGMMControl:
    """Test control parameters for PGMM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        
        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_custom_control_parameters(self):
        """Test PGMM with custom control parameters."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMControl(
            maxiter=1000,
            optTol=1e-4,
            zeroThreshold=1e-5,
            c=1.5
        )
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        assert pgmm.control.maxiter == 1000
        assert pgmm.control.optTol == 1e-4
        assert pgmm.is_fitted_ == True
    
    def test_automatic_lambda(self):
        """Test automatic lambda calculation."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=None,  # Should be computed automatically
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        # Lambda should have been computed
        assert pgmm.lambda_ is not None
        assert pgmm.lambda_ > 0


class TestPGMMCriterion:
    """Test GMM criterion computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        
        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_compute_criterion(self):
        """Test GMM criterion computation."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.01,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        # Compute criterion
        criterion = pgmm.compute_criterion(self.W, moment, weight_func=weight_func)
        
        assert isinstance(criterion, float)
        assert criterion >= 0  # Criterion should be non-negative


class TestPGMMSparsity:
    """Test sparsity properties of PGMM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        
        self.Z = np.random.randn(self.n, 3)
        self.X = self.Z + 0.2 * np.random.randn(self.n, 3)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_penalty_induces_sparsity(self):
        """Test that L1 penalty induces sparsity."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        # High penalty should give more zeros
        pgmm_high_penalty = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.5,  # High penalty
            verbose=False
        )
        
        # Low penalty should give fewer zeros
        pgmm_low_penalty = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            lambda_=0.001,  # Low penalty
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_high_penalty.fit(self.W, moment, weight_func=weight_func)
        pgmm_low_penalty.fit(self.W, moment, weight_func=weight_func)
        
        n_zeros_high = np.sum(pgmm_high_penalty.rho_ == 0)
        n_zeros_low = np.sum(pgmm_low_penalty.rho_ == 0)
        
        # High penalty should produce more zeros
        assert n_zeros_high >= n_zeros_low


class TestPGMMEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        
        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_predict_before_fit(self):
        """Test that predicting before fitting raises error."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        
        pgmm_cv = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="must be fitted"):
            pgmm_cv.predict(self.Z)
    
    def test_get_rho_before_fit(self):
        """Test that getting rho before fitting raises error."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        
        pgmm_cv = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="must be fitted"):
            pgmm_cv.get_rho()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])