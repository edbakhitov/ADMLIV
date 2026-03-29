# admliv/core/pgmm_linear_iv_cv.py

"""
Cross-validated PGMM for High-Dimensional Linear IV Regression.

Adds cross-validation for penalty parameter selection to PGMMLinearIV.
"""

from dataclasses import replace
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from .control import PGMMCVControl
from .pgmm_linear_iv import PGMMLinearIV
from admliv.utils.featurizers import SimpleFeaturizer


class PGMMLinearIVCV(BaseEstimator):
    """
    Cross-validated Penalized GMM estimator for Linear IV Regression.

    Performs K-fold cross-validation to select the optimal penalty parameter c
    from a grid of values. The penalty is lambda = c * sqrt(log(q) / n).

    This class wraps the PGMMLinearIV estimator and adds automatic tuning of the
    penalty parameter via cross-validation. Unlike PGMMCV which estimates Riesz
    representer coefficients, this estimates structural coefficients ρ directly
    using standard IV moment conditions.

    Parameters
    ----------
    x_featurizer : TransformerMixin, default=SimpleFeaturizer
        Sklearn-style transformer for basis expansion b(X) of regressors
    z_featurizer : TransformerMixin, default=SimpleFeaturizer
        Sklearn-style transformer for basis expansion b(Z) of instruments
    adaptive : bool, default=True
        If True, uses adaptive weights based on preliminary estimation
    Omega : Optional[NDArray], default=None
        Weight matrix for GMM criterion (q × q). If None, uses identity for preliminary,
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
        Best penalty value (c * sqrt(log(q) / n))
    cv_scores_ : NDArray
        Cross-validation scores for each c value (shape: n_c)
    cv_scores_std_ : NDArray
        Standard deviation of CV scores across folds (shape: n_c)
    best_estimator_ : PGMMLinearIV
        PGMMLinearIV estimator fitted with best c on full data (if refit=True)
    is_fitted_ : bool
        Whether the estimator has been fitted
    """

    def __init__(
        self,
        x_featurizer: TransformerMixin = SimpleFeaturizer,
        z_featurizer: TransformerMixin = SimpleFeaturizer,
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
        W: Dict[str, NDArray[np.float64]]
    ):
        """
        Fit PGMMLinearIV with cross-validated penalty parameter selection.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary with keys:
            - 'Y': outcome, shape (n,) or (n, 1)
            - 'X': endogenous regressors, shape (n, d_x)
            - 'Z': instruments, shape (n, d_z)

        Returns
        -------
        self : PGMMLinearIVCV
            Fitted estimator
        """
        X = W['X']
        Z = W['Z']
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
            print("Cross-Validated PGMM Linear IV Estimation")
            print("="*70)
            print(f"Number of folds: {self.control.n_folds}")
            print(f"Penalty grid size: {n_c}")
            print(f"c values: {c_vec}")
            print("="*70)

        # Run cross-validation
        if self.control.n_jobs == 1:
            # Sequential execution
            cv_results = [
                self._fit_single_c(c, W, kfold)
                for c in c_vec
            ]
        else:
            # Parallel execution
            cv_results = Parallel(n_jobs=self.control.n_jobs)(
                delayed(self._fit_single_c)(c, W, kfold)
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
        self.z_featurizer.fit(Z)
        Wz = self.z_featurizer.transform(Z)
        q = Wz.shape[1]  # Use p (number of instruments) for linear IV
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

            self.best_estimator_ = PGMMLinearIV(
                x_featurizer=clone(self.x_featurizer),
                z_featurizer=clone(self.z_featurizer),
                lambda_=self.best_lambda_,
                adaptive=self.adaptive,
                Omega=self.Omega,
                control=self.control,  # PGMMCVControl inherits from PGMMControl
                verbose=self.verbose
            )
            self.best_estimator_.fit(W)

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
        kfold: KFold
    ) -> Tuple[float, float]:
        """
        Fit PGMMLinearIV for a single c value using K-fold CV.

        Parameters
        ----------
        c : float
            Penalty parameter multiplier
        W : Dict[str, NDArray]
            Full data dictionary
        kfold : KFold
            Cross-validation splitter

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

            # Create PGMMLinearIV instance for this fold with its own control copy
            fold_control = replace(self.control, c=c)
            pgmm_fold = PGMMLinearIV(
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
            pgmm_fold.fit(W_train)

            # Compute optimal Omega from validation data
            Y_val = W_val['Y']
            X_val = W_val['X']
            Z_val = W_val['Z']
            Wx_val = pgmm_fold.x_featurizer.transform(X_val)
            Wz_val = pgmm_fold.z_featurizer.transform(Z_val)

            # Compute moment conditions: psi_i = (Y_i - X_i'ρ) * Z_i
            residual_val = Y_val - Wx_val @ pgmm_fold.rho_
            psi_val = residual_val[:, np.newaxis] * Wz_val

            Omega_val = PGMMLinearIV._compute_optimal_weight_matrix(psi_val)

            # Compute criterion on validation fold
            score = pgmm_fold.compute_criterion(W_val, Omega_val)

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        if self.verbose:
            print(f"  c = {c:.3f}: {mean_score:.6f} (+/- {std_score:.6f})")

        return mean_score, std_score

    def get_rho(self) -> NDArray[np.float64]:
        """
        Get the estimated structural coefficients from best model.

        Returns
        -------
        rho : NDArray, shape (dim(b),)
            Estimated structural coefficients
        """
        if not self.is_fitted_:
            raise ValueError("PGMMLinearIVCV must be fitted before accessing rho")

        if not self.refit:
            raise ValueError(
                "Cannot get rho when refit=False. "
                "Set refit=True to fit on full data with best c."
            )

        return self.best_estimator_.get_rho()

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute prediction Y_hat = b(X)' rho using best model.

        Parameters
        ----------
        X : NDArray, shape (n, d_x)
            Endogenous regressors

        Returns
        -------
        Y_hat : NDArray, shape (n,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("PGMMLinearIVCV must be fitted before prediction")

        if not self.refit:
            raise ValueError(
                "Cannot predict when refit=False. "
                "Set refit=True to fit on full data with best c."
            )

        return self.best_estimator_.predict(X)

    def compute_criterion(
        self,
        W: Dict[str, NDArray[np.float64]],
        Omega: Optional[NDArray[np.float64]] = None
    ) -> float:
        """
        Compute GMM criterion using best model.

        Parameters
        ----------
        W : Dict[str, NDArray]
            Data dictionary
        Omega : Optional[NDArray], default=None
            Weight matrix. If None, uses identity

        Returns
        -------
        criterion : float
            GMM criterion value
        """
        if not self.is_fitted_:
            raise ValueError("PGMMLinearIVCV must be fitted before computing criterion")

        if not self.refit:
            raise ValueError(
                "Cannot compute criterion when refit=False. "
                "Set refit=True to fit on full data with best c."
            )

        return self.best_estimator_.compute_criterion(W, Omega)

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
            raise ValueError("PGMMLinearIVCV must be fitted before plotting")

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
        ax.set_title('PGMM Linear IV Cross-Validation Results', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax
