# admliv/core/pgmm_cv.py

from dataclasses import replace
from typing import Dict, Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from .control import PGMMCVControl
from .pgmm import PGMM
from ..moments.base import BaseMoment


class PGMMCV(BaseEstimator):
    """
    Cross-validated Penalized GMM estimator for Riesz representer.
    
    Performs K-fold cross-validation to select the optimal penalty parameter c
    from a grid of values. The penalty is lambda = c * sqrt(log(p) / n).
    
    This class wraps the PGMM estimator and adds automatic tuning of the
    penalty parameter via cross-validation.
    
    Parameters
    ----------
    x_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion d(X)
    z_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion b(Z)
    adaptive : bool, default=True
        If True, uses adaptive weights based on preliminary estimation
    Omega : Optional[NDArray], default=None
        Weight matrix for GMM criterion. If None, uses identity for preliminary,
        then optimal diagonal matrix for adaptive step
    control : Optional[PGMMCVControl], default=None
        Control parameters for CV and optimization
    verbose : bool, default=True
        If True, prints CV progress
    refit : bool, default=True
        If True, refit on full data with best c. If False, use average across folds.
    
    Attributes
    ----------
    best_c_ : float
        Best penalty parameter selected by CV
    best_lambda_ : float
        Best penalty value (c * sqrt(log(p) / n))
    cv_scores_ : NDArray
        Cross-validation scores for each c value (shape: n_c)
    cv_scores_std_ : NDArray
        Standard deviation of CV scores across folds (shape: n_c)
    best_estimator_ : PGMM
        PGMM estimator fitted with best c on full data (if refit=True)
    is_fitted_ : bool
        Whether the estimator has been fitted
    """
    
    def __init__(
        self,
        x_featurizer: TransformerMixin,
        z_featurizer: TransformerMixin,
        adaptive: bool = True,
        Omega: Optional[NDArray[np.float64]] = None,
        control: Optional[PGMMCVControl] = None,
        verbose: bool = True,
        refit: bool = True
    ):
        self.x_featurizer = x_featurizer
        self.z_featurizer = z_featurizer
        self.adaptive = adaptive
        self.Omega = Omega
        self.control = control if control is not None else PGMMCVControl()
        self.verbose = verbose
        self.refit = refit
        self.is_fitted_ = False
        
    def fit(
        self,
        W: Dict[str, NDArray[np.float64]],
        moment: BaseMoment,
        **moment_kwargs
    ):
        """
        Fit PGMM with cross-validated penalty parameter selection.
        
        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys 'Y', 'X', 'Z'
        moment : BaseMoment
            Moment function instance
        **moment_kwargs : dict
            Additional arguments passed to moment.compute_all_basis()
        
        Returns
        -------
        self : PGMMCV
            Fitted estimator
        """
        X = W['X']
        n = X.shape[0]
        
        # Setup cross-validation
        kfold = KFold(
            n_splits=self.control.n_folds,
            shuffle=True,
            random_state=self.control.random_state
        )
        
        c_vec = self.control.c_vec
        n_c = len(c_vec)
        
        if self.verbose:
            print("="*70)
            print("Cross-Validated PGMM Estimation")
            print("="*70)
            print(f"Number of folds: {self.control.n_folds}")
            print(f"Penalty grid size: {n_c}")
            print(f"c values: {c_vec}")
            print("="*70)
        
        # Run cross-validation
        if self.control.n_jobs == 1:
            # Sequential execution
            cv_results = [
                self._fit_single_c(c, W, moment, kfold, moment_kwargs)
                for c in c_vec
            ]
        else:
            # Parallel execution
            cv_results = Parallel(n_jobs=self.control.n_jobs)(
                delayed(self._fit_single_c)(c, W, moment, kfold, moment_kwargs)
                for c in c_vec
            )
        
        # Extract scores
        cv_scores = np.array([score for score, _ in cv_results])
        cv_scores_std = np.array([std for _, std in cv_results])
        
        # Select best c (minimize GMM criterion)
        best_idx = np.argmin(cv_scores)
        best_c = c_vec[best_idx]
        
        # Store results
        self.best_c_ = best_c
        self.cv_scores_ = cv_scores
        self.cv_scores_std_ = cv_scores_std
        
        # Compute best lambda
        # Fit featurizers to get dimensions
        self.x_featurizer.fit(X)
        Wx = self.x_featurizer.transform(X)
        q = Wx.shape[1]
        self.best_lambda_ = best_c * np.sqrt(np.log(q) / n)
        
        if self.verbose:
            print("\nCross-Validation Results:")
            print("-" * 70)
            for i, c in enumerate(c_vec):
                marker = " <-- BEST" if i == best_idx else ""
                print(f"c = {c:6.3f}: CV score = {cv_scores[i]:10.6f} "
                      f"(+/- {cv_scores_std[i]:8.6f}){marker}")
            print("-" * 70)
            print(f"Best c: {best_c:.4f}")
            print(f"Best lambda: {self.best_lambda_:.6f}")
            print("="*70)
        
        # Refit on full data with best c
        if self.refit:
            if self.verbose:
                print("\nRefitting on full data with best c...")
            
            self.best_estimator_ = PGMM(
                x_featurizer=clone(self.x_featurizer),
                z_featurizer=clone(self.z_featurizer),
                lambda_=self.best_lambda_,
                adaptive=self.adaptive,
                Omega=self.Omega,
                control=self.control,  # PGMMCVControl inherits from PGMMControl
                verbose=self.verbose
            )
            self.best_estimator_.fit(W, moment, **moment_kwargs)
            
            # Store for convenience
            self.rho_ = self.best_estimator_.rho_
            self.n_samples_ = self.best_estimator_.n_samples_
            self.n_features_x_ = self.best_estimator_.n_features_x_
            self.n_features_z_ = self.best_estimator_.n_features_z_
        else:
            # Use average model from CV folds (not recommended, but available)
            if self.verbose:
                print("\nWarning: refit=False. Using average across folds.")
                print("This is not recommended. Consider setting refit=True.")
        
        self.is_fitted_ = True
        return self
    
    def _fit_single_c(
        self,
        c: float,
        W: Dict[str, NDArray[np.float64]],
        moment: BaseMoment,
        kfold: KFold,
        moment_kwargs: dict
    ) -> Tuple[float, float]:
        """
        Fit PGMM for a single c value using K-fold CV.
        
        Parameters
        ----------
        c : float
            Penalty parameter multiplier
        W : Dict[str, NDArray]
            Full data dictionary
        moment : BaseMoment
            Moment function
        kfold : KFold
            Cross-validation splitter
        moment_kwargs : dict
            Additional moment arguments
        
        Returns
        -------
        mean_score : float
            Mean CV score across folds
        std_score : float
            Standard deviation of CV scores across folds
        """
        Y = W['Y']
        X = W['X']
        Z = W['Z']
        
        fold_scores = []

        for train_idx, val_idx in kfold.split(X):
            # Split data
            W_train = {
                'Y': Y[train_idx],
                'X': X[train_idx],
                'Z': Z[train_idx]
            }
            W_val = {
                'Y': Y[val_idx],
                'X': X[val_idx],
                'Z': Z[val_idx]
            }

            # Create PGMM instance for this fold with its own control copy
            fold_control = replace(self.control, c=c)
            pgmm_fold = PGMM(
                # use clone() to ensure that featurizers are fitted on each data fold
                # to avoid data leakage
                x_featurizer=clone(self.x_featurizer),
                z_featurizer=clone(self.z_featurizer),
                lambda_=None,  # Will be computed inside fit
                adaptive=self.adaptive,
                Omega=self.Omega,
                control=fold_control,
                verbose=False  # Quiet for CV
            )
            
            # Fit on training data
            pgmm_fold.fit(W_train, moment, **moment_kwargs)
            
            # Evaluate on validation fold using GMM criterion
            # The criterion is: psi' Omega psi / 2
            # score = pgmm_fold.compute_criterion(W_val, moment, **moment_kwargs)

            # Compute optimal Omega from validation data
            Y_val = W_val['Y']
            X_val = W_val['X']
            Wx_val = pgmm_fold.x_featurizer.transform(X_val)
            Wz_val = pgmm_fold.z_featurizer.transform(W_val['Z'])

            psi_val = pgmm_fold._compute_orthogonal_moment(
                Y_val, X_val, Wx_val, Wz_val, moment, moment_kwargs, pgmm_fold.rho_
            )
            Omega_val = PGMM._compute_optimal_weight_matrix(psi_val)

            # Compute criterion
            score = pgmm_fold.compute_criterion(W_val, moment, Omega_val, **moment_kwargs)
            
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        if self.verbose:
            print(f"  c = {c:.3f}: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return mean_score, std_score
    
    def predict(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute Riesz representer alpha(Z) = b(Z)' rho using best model.
        
        Parameters
        ----------
        Z : NDArray, shape (n, d_z)
            Instrumental variables
        
        Returns
        -------
        alpha : NDArray, shape (n,)
            Riesz representer values
        """
        if not self.is_fitted_:
            raise ValueError("PGMMCV must be fitted before prediction")
        
        if not self.refit:
            raise ValueError(
                "Cannot predict when refit=False. "
                "Set refit=True to fit on full data with best c."
            )
        
        return self.best_estimator_.predict(Z)
    
    def get_rho(self) -> NDArray[np.float64]:
        """
        Get the estimated Riesz representer coefficients from best model.
        
        Returns
        -------
        rho : NDArray, shape (dim(b),)
            Estimated coefficients
        """
        if not self.is_fitted_:
            raise ValueError("PGMMCV must be fitted before accessing rho")
        
        if not self.refit:
            raise ValueError(
                "Cannot get rho when refit=False. "
                "Set refit=True to fit on full data with best c."
            )
        
        return self.best_estimator_.get_rho()
    
    def compute_criterion(
        self,
        W: Dict[str, NDArray[np.float64]],
        moment: BaseMoment,
        Omega: Optional[NDArray[np.float64]] = None,
        **moment_kwargs
    ) -> float:
        """
        Compute GMM criterion using best model.
        
        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary
        moment : BaseMoment
            Moment function
        Omega : Optional[NDArray], default=None
            Weight matrix. If None, uses identity
        **moment_kwargs : dict
            Additional moment arguments
        
        Returns
        -------
        criterion : float
            GMM criterion value
        """
        if not self.is_fitted_:
            raise ValueError("PGMMCV must be fitted before computing criterion")
        
        if not self.refit:
            raise ValueError(
                "Cannot compute criterion when refit=False. "
                "Set refit=True to fit on full data with best c."
            )
        
        return self.best_estimator_.compute_criterion(W, moment, Omega, **moment_kwargs)
    
    def plot_cv_results(self, figsize=(10, 6)):
        """
        Plot cross-validation results.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if not self.is_fitted_:
            raise ValueError("PGMMCV must be fitted before plotting")
        
        try:
            import matplotlib.pyplot as plt  # noqa: E402
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        c_vec = self.control.c_vec
        scores = self.cv_scores_
        scores_std = self.cv_scores_std_
        
        # Plot with error bars
        ax.errorbar(
            c_vec, scores, yerr=scores_std,
            marker='o', linestyle='-', capsize=5,
            label='CV Score'
        )
        
        # Mark best c
        best_idx = np.argmin(scores)
        ax.axvline(
            self.best_c_, color='red', linestyle='--',
            label=f'Best c = {self.best_c_:.3f}'
        )
        ax.plot(
            self.best_c_, scores[best_idx],
            'r*', markersize=15, label='Selected'
        )
        
        ax.set_xlabel('Penalty parameter c', fontsize=12)
        ax.set_ylabel('CV Score (GMM Criterion)', fontsize=12)
        ax.set_title('PGMM Cross-Validation Results', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax