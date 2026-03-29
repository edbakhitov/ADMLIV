# tests/test_pgmm/test_pgmm_cv.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from admliv.core.pgmm_cv import PGMMCV
from admliv.core.pgmm import PGMM
from admliv.core.control import PGMMCVControl, PGMMControl
from admliv.moments import WeightedAverage
from admliv.utils.featurizers import CoordinatePolyTransform


class TestPGMMCVBasic:
    """Basic tests for PGMMCV estimator."""
    
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
    
    def test_pgmm_cv_initialization(self):
        """Test PGMMCV initialization."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
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
    
    def test_pgmm_cv_fit_basic(self):
        """Test basic PGMMCV fitting."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Fit should work
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Check that rho is estimated
        assert hasattr(pgmm_cv, 'rho_')
        assert pgmm_cv.rho_ is not None
        assert pgmm_cv.rho_.shape[0] == z_feat.fit(self.Z).n_features_out_
        assert pgmm_cv.is_fitted_ == True
    
    def test_pgmm_cv_predict(self):
        """Test PGMMCV predict (Riesz representer evaluation)."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Predict alpha(Z)
        alpha = pgmm_cv.predict(self.Z)
        
        assert alpha.shape == (self.n,)
        assert not np.all(alpha == 0)  # Should have non-zero values
    
    def test_pgmm_cv_get_rho(self):
        """Test getting Riesz representer coefficients."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        rho = pgmm_cv.get_rho()

        # Check shape and properties
        # rho has dimension n_features_z (the dimension of b(Z))
        assert rho.shape[0] == pgmm_cv.n_features_z_
        assert isinstance(rho, np.ndarray)


class TestPGMMCVCrossValidation:
    """Test cross-validation functionality."""
    
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
    
    def test_cv_selects_best_c(self):
        """Test that CV selects best c from grid."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        c_vec = np.array([0.5, 1.0, 1.5, 2.0])
        control = PGMMCVControl(
            n_folds=3,
            c_vec=c_vec
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Best c should be from the grid
        assert pgmm_cv.best_c_ in c_vec
        
        # Best c should minimize CV score
        best_idx = np.argmin(pgmm_cv.cv_scores_)
        assert pgmm_cv.best_c_ == c_vec[best_idx]
    
    def test_cv_scores_shape(self):
        """Test that CV scores have correct shape."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        c_vec = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        control = PGMMCVControl(
            n_folds=3,
            c_vec=c_vec
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # CV scores should have same length as c_vec
        assert len(pgmm_cv.cv_scores_) == len(c_vec)
        assert len(pgmm_cv.cv_scores_std_) == len(c_vec)
        
        # All scores should be non-negative
        assert np.all(pgmm_cv.cv_scores_ >= 0)
        assert np.all(pgmm_cv.cv_scores_std_ >= 0)
    
    def test_cv_lambda_computation(self):
        """Test that best lambda is computed correctly."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        c_vec = np.array([0.5, 1.0, 2.0])
        control = PGMMCVControl(
            n_folds=3,
            c_vec=c_vec
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Check lambda formula: lambda = c * sqrt(log(p) / n)
        p = pgmm_cv.n_features_z_
        n = pgmm_cv.n_samples_
        expected_lambda = pgmm_cv.best_c_ * np.sqrt(np.log(p) / n)
        
        assert pgmm_cv.best_lambda_ is not None
        assert np.abs(pgmm_cv.best_lambda_ - expected_lambda) < 1e-10


class TestPGMMCVRefit:
    """Test refit functionality."""
    
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
    
    def test_refit_true_creates_best_estimator(self):
        """Test that refit=True creates best_estimator_."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Should have best_estimator_
        assert hasattr(pgmm_cv, 'best_estimator_')
        assert isinstance(pgmm_cv.best_estimator_, PGMM)
        assert pgmm_cv.best_estimator_.is_fitted_ == True
    
    def test_refit_false_no_predict(self):
        """Test that refit=False prevents prediction."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Should raise error when trying to predict
        with pytest.raises(ValueError, match="refit=False"):
            pgmm_cv.predict(self.Z)
    
    def test_refit_consistency_with_pgmm(self):
        """Test that PGMMCV with refit gives same results as PGMM with fixed c."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Fit PGMM with c=1.0
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=PGMMControl(c=1.0),
            verbose=False
        )
        pgmm.fit(self.W, moment, weight_func=weight_func)
        
        # Fit PGMMCV with only c=1.0 in grid
        pgmm_cv = PGMMCV(
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
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Best c should be 1.0
        assert pgmm_cv.best_c_ == 1.0
        
        # Rho estimates should be very close (same c, same full data after refit)
        # Tight tolerance to allow for small numerical differences in SGD algo
        np.testing.assert_array_almost_equal(
            pgmm_cv.rho_, 
            pgmm.rho_,
            decimal=6,
            err_msg="PGMMCV with single c should match PGMM with same c"
        )


class TestPGMMCVAdaptive:
    """Test adaptive PGMMCV with two-step estimation."""
    
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
    
    def test_adaptive_pgmm_cv(self):
        """Test adaptive PGMMCV estimation."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=True,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Fit with adaptive estimation
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Check that estimation completed
        assert hasattr(pgmm_cv, 'rho_')
        assert pgmm_cv.is_fitted_ == True
        
        # Adaptive should produce sparse solution
        n_nonzero = np.count_nonzero(pgmm_cv.rho_)
        n_total = len(pgmm_cv.rho_)
        assert n_nonzero < n_total  # Should have some zeros due to L1 penalty


class TestPGMMCVControl:
    """Test control parameters for PGMMCV."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 100
        
        self.Z = np.random.randn(self.n, 2)
        self.X = self.Z + np.random.randn(self.n, 2)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_custom_control_parameters(self):
        """Test PGMMCV with custom control parameters."""
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=1, include_bias=True)
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            maxiter=1000,
            optTol=1e-4,
            zeroThreshold=1e-5,
            n_folds=5,
            c_vec=np.array([0.5, 1.0, 1.5, 2.0]),
            random_state=123
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
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


class TestPGMMCVCriterion:
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
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0, 2.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False,
            refit=True
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        
        # Compute criterion
        criterion = pgmm_cv.compute_criterion(self.W, moment, weight_func=weight_func)
        
        assert isinstance(criterion, float)
        assert criterion >= 0  # Criterion should be non-negative


class TestPGMMCVComparison:
    """Test comparison between PGMMCV and PGMM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 200
        
        self.Z = np.random.randn(self.n, 3)
        self.X = self.Z + 0.2 * np.random.randn(self.n, 3)
        self.Y = np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_cv_can_improve_over_default(self):
        """Test that CV can potentially improve over default c=1.0."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Fit PGMM with default c=1.0
        pgmm = PGMM(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=PGMMControl(c=1.0),
            verbose=False
        )
        pgmm.fit(self.W, moment, weight_func=weight_func)
        criterion_pgmm = pgmm.compute_criterion(self.W, moment, weight_func=weight_func)
        
        # Fit PGMMCV with grid including 1.0
        pgmm_cv = PGMMCV(
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
        pgmm_cv.fit(self.W, moment, weight_func=weight_func)
        criterion_pgmm_cv = pgmm_cv.compute_criterion(self.W, moment, weight_func=weight_func)
        
        # PGMMCV should not be worse (it can select c=1.0 if that's best)
        assert criterion_pgmm_cv <= criterion_pgmm + 1e-6  # Small tolerance for numerical differences


class TestPGMMCVEdgeCases:
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
        
        pgmm_cv = PGMMCV(
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
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="must be fitted"):
            pgmm_cv.get_rho()
    
    def test_underidentification_error(self):
        """Test that under-identification raises error."""
        # More basis functions in Z than X -> under-identified
        x_feat = CoordinatePolyTransform(degree=1, include_bias=True)  # 3 features
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)  # 5 features
        moment = WeightedAverage()
        
        control = PGMMCVControl(
            n_folds=3,
            c_vec=np.array([0.5, 1.0])
        )
        
        pgmm_cv = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control,
            verbose=False
        )
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # Should raise ValueError during CV (in one of the folds)
        with pytest.raises(ValueError, match="Under-identified"):
            pgmm_cv.fit(self.W, moment, weight_func=weight_func)


class TestPGMMCVReproducibility:
    """Test reproducibility with random_state."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n = 200
        self.d_x = 2
        self.d_z = 2
        
        np.random.seed(42)
        self.Z = np.random.randn(self.n, self.d_z)
        self.X = self.Z + 0.5 * np.random.randn(self.n, self.d_x)
        self.Y = self.X[:, 0] + 0.5 * self.X[:, 1] + np.random.randn(self.n, 1)
        
        self.W = {'Y': self.Y, 'X': self.X, 'Z': self.Z}
    
    def test_random_state_reproducibility(self):
        """Test that same random_state gives same results."""
        x_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        z_feat = CoordinatePolyTransform(degree=2, include_bias=True)
        moment = WeightedAverage()
        
        def weight_func(X):
            return np.ones(X.shape[0])
        
        # First run
        control1 = PGMMCVControl(
            n_folds=5,
            c_vec=np.array([0.5, 1.0, 2.0]),
            random_state=123
        )
        
        pgmm_cv1 = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control1,
            verbose=False,
            refit=True
        )
        pgmm_cv1.fit(self.W, moment, weight_func=weight_func)
        
        # Second run with same random_state
        control2 = PGMMCVControl(
            n_folds=5,
            c_vec=np.array([0.5, 1.0, 2.0]),
            random_state=123
        )
        
        pgmm_cv2 = PGMMCV(
            x_featurizer=x_feat,
            z_featurizer=z_feat,
            adaptive=False,
            control=control2,
            verbose=False,
            refit=True
        )
        pgmm_cv2.fit(self.W, moment, weight_func=weight_func)
        
        # Should give identical results
        assert pgmm_cv1.best_c_ == pgmm_cv2.best_c_
        np.testing.assert_array_almost_equal(pgmm_cv1.cv_scores_, pgmm_cv2.cv_scores_)
        np.testing.assert_array_almost_equal(pgmm_cv1.rho_, pgmm_cv2.rho_)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])