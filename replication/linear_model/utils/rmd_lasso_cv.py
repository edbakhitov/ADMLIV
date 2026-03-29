# simulations/linear_model/utils/rmd_lasso_cv.py

"""
RMD Lasso with Cross-Validation for penalty parameter selection.

Implements the cross-validation procedure from CNS paper Section A.1.2.

Reference:
    Chernozhukov, V., Newey, W. K., & Singh, R. (2022).
    Automatic debiased machine learning of causal and structural effects.
    Econometrica.
"""

from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import KFold

from admliv.moments.base import BaseMoment
from .control import RMDCVControl
from .rmd_lasso import RMDLasso


class RMDLassoCV(RMDLasso):
    """
    RMD Lasso with K-fold cross-validation for penalty parameter selection.
    
    Inherits from RMDLasso and adds cross-validation for selecting the
    penalty parameter c. Implements the CV procedure from CNS Section A.1.2:
    
        CV(c) = sum_l sum_{i in I_l} [-2*m(W_i, b)'*rho_l(c) + (b(X_i)'*rho_l(c))^2]
    
    Parameters
    ----------
    x_featurizer : TransformerMixin
        Sklearn-style transformer for basis expansion b(X)
    control : Optional[RMDCVControl], default=None
        Control parameters including CV settings
    verbose : bool, default=True
        If True, prints progress
    
    Attributes
    ----------
    rho_ : NDArray
        Estimated coefficients using best c
    best_c_ : float
        Best penalty parameter from CV
    cv_results_ : dict
        Cross-validation results for each c value
    is_fitted_ : bool
        Whether the estimator has been fitted
    """
    
    def __init__(
        self,
        x_featurizer: TransformerMixin,
        control: Optional[RMDCVControl] = None,
        verbose: bool = True
    ):
        control = control if control is not None else RMDCVControl()
        super().__init__(x_featurizer, control, verbose)
    
    def _compute_cv_criterion(
        self,
        B_test: NDArray[np.float64],
        M_test: NDArray[np.float64],
        rho: NDArray[np.float64]
    ) -> float:
        """
        Compute cross-validation criterion from CNS section A.1.2.
        
        CV = sum_i [-2 * m(W_i, b)' @ rho + (b(X_i)' @ rho)^2]
        
        Parameters
        ----------
        B_test : array (n_test, p)
            Test basis matrix
        M_test : array (n_test, p)
            Test moment matrix
        rho : array (p,)
            Coefficient estimate
            
        Returns
        -------
        cv_value : float
            Cross-validation criterion value
        """
        term1 = -2 * np.sum(M_test @ rho)
        alpha_test = B_test @ rho
        term2 = np.sum(alpha_test ** 2)
        return term1 + term2
    
    def _fit_fold(
        self,
        Y_train: NDArray[np.float64],
        X_train: NDArray[np.float64],
        moment: BaseMoment,
        c: float,
        moment_kwargs: dict
    ) -> NDArray[np.float64]:
        """
        Fit RMD Lasso on training fold data.
        
        Parameters
        ----------
        Y_train : array (n_train,)
            Training outcomes
        X_train : array (n_train, p_x)
            Training features
        moment : BaseMoment
            Moment class
        c : float
            Penalty scaling constant
        moment_kwargs : dict
            Additional moment arguments
            
        Returns
        -------
        rho_hat : array (p,)
            Estimated coefficients
        """
        n_train = len(Y_train)
        p_x = X_train.shape[1] if X_train.ndim > 1 else 1
        
        # Clone and fit featurizer
        feat = clone(self.x_featurizer)
        feat.fit(X_train)
        B_train = feat.transform(X_train)
        p = B_train.shape[1]
        
        # Compute moments
        W_train = {'Y': Y_train, 'X': X_train}
        M_full = moment.compute_all_basis(feat, W_train, **moment_kwargs)
        M_hat = M_full.mean(axis=0)
        G_hat = B_train.T @ B_train / n_train
        
        # Initialize
        rho_init = self._low_dim_init(G_hat, M_hat, p_x, p)
        
        # Fit with specified c
        return self._fit_with_c(B_train, M_full, M_hat, G_hat, rho_init, c)
    
    def _compute_cv_criterion_fold(
        self,
        Y_train: NDArray[np.float64],
        X_train: NDArray[np.float64],
        Y_test: NDArray[np.float64],
        X_test: NDArray[np.float64],
        moment: BaseMoment,
        c: float,
        moment_kwargs: dict
    ) -> float:
        """
        Compute CV criterion for one fold and one c value.
        
        Parameters
        ----------
        Y_train, X_train : arrays
            Training data
        Y_test, X_test : arrays
            Test data
        moment : BaseMoment
            Moment class
        c : float
            Penalty scaling constant
        moment_kwargs : dict
            Additional moment arguments
            
        Returns
        -------
        cv_value : float
            CV criterion value for this fold
        """
        # Fit on training data
        rho_hat = self._fit_fold(Y_train, X_train, moment, c, moment_kwargs)
        
        # Evaluate on test data using training featurizer
        feat = clone(self.x_featurizer)
        feat.fit(X_train)
        B_test = feat.transform(X_test)
        
        W_test = {'Y': Y_test, 'X': X_test}
        M_test = moment.compute_all_basis(feat, W_test, **moment_kwargs)
        
        return self._compute_cv_criterion(B_test, M_test, rho_hat)
    
    def fit(
        self,
        W: Dict[str, NDArray[np.float64]],
        moment: BaseMoment,
        **moment_kwargs
    ):
        """
        Fit RMD Lasso with cross-validation for penalty selection.
        
        Parameters
        ----------
        W : dict
            Data dictionary with keys 'Y', 'X'
        moment : BaseMoment
            Moment class implementing compute_all_basis
        **moment_kwargs
            Additional arguments for moment computation
            
        Returns
        -------
        self
        """
        Y = W['Y']
        X = W['X']
        n = len(Y)
        
        self.n_samples_ = n
        
        if self.verbose:
            print("=" * 60)
            print("RMD Lasso CV (CNS method)")
            print("=" * 60)
            print(f"n = {n}")
            print(f"c values: {self.control.c_vec}")
            print(f"K-fold: {self.control.n_folds}")
        
        # Setup K-fold CV
        kf = KFold(n_splits=self.control.n_folds, shuffle=False)
        fold_indices = list(kf.split(Y))
        
        # Compute CV criterion for each c value
        cv_results = {c: [] for c in self.control.c_vec}
        
        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            X_train, X_test = X[train_idx], X[test_idx]
            
            if self.verbose:
                print(f"\nFold {fold_idx + 1}/{self.control.n_folds}")
            
            for c in self.control.c_vec:
                cv_value = self._compute_cv_criterion_fold(
                    Y_train, X_train, Y_test, X_test,
                    moment, c, moment_kwargs
                )
                cv_results[c].append(cv_value)
        
        # Average across folds
        cv_means = {c: np.mean(cv_results[c]) for c in self.control.c_vec}
        cv_stds = {c: np.std(cv_results[c]) for c in self.control.c_vec}
        
        # Select best c (minimum CV criterion)
        best_c = min(cv_means, key=cv_means.get)
        
        if self.verbose:
            print("\n" + "-" * 40)
            print("CV Results:")
            for c in self.control.c_vec:
                marker = " <-- BEST" if c == best_c else ""
                print(f"  c = {c:.4f}: CV = {cv_means[c]:.4f} (std = {cv_stds[c]:.4f}){marker}")
            print("-" * 40)
        
        self.cv_results_ = {
            'c_vec': self.control.c_vec,
            'cv_means': cv_means,
            'cv_stds': cv_stds,
            'cv_raw': cv_results
        }
        self.best_c_ = best_c
        
        # Refit on full data with best c
        if self.control.refit:
            if self.verbose:
                print(f"\nRefitting on full data with c = {best_c:.4f}")
            
            # Use parent class machinery for final fit
            p_x = X.shape[1] if X.ndim > 1 else 1
            self.x_featurizer.fit(X)
            B = self.x_featurizer.transform(X)
            p = B.shape[1]
            
            self.n_features_ = p
            
            M_full = moment.compute_all_basis(self.x_featurizer, W, **moment_kwargs)
            M_hat = M_full.mean(axis=0)
            G_hat = B.T @ B / n
            
            self._B = B
            self._M_full = M_full
            self._M_hat = M_hat
            self._G_hat = G_hat
            
            rho_init = self._low_dim_init(G_hat, M_hat, p_x, p)
            self.rho_ = self._fit_with_c(B, M_full, M_hat, G_hat, rho_init, best_c)
            
            if self.verbose:
                print(f"Non-zero coefficients: {np.count_nonzero(self.rho_)}")
        
        self.is_fitted_ = True
        
        if self.verbose:
            print("=" * 60)
        
        return self
    
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
            raise ValueError("RMDLassoCV must be fitted before plotting")

        try:
            import matplotlib.pyplot as plt  # noqa: E402
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        c_vec = self.cv_results_['c_vec']
        cv_means = np.array([self.cv_results_['cv_means'][c] for c in c_vec])
        cv_stds = np.array([self.cv_results_['cv_stds'][c] for c in c_vec])

        # Plot with error bars
        ax.errorbar(
            c_vec, cv_means, yerr=cv_stds,
            marker='o', linestyle='-', capsize=5,
            label='CV Score'
        )

        # Mark best c
        best_idx = c_vec.index(self.best_c_)
        ax.axvline(
            self.best_c_, color='red', linestyle='--',
            label=f'Best c = {self.best_c_:.3f}'
        )
        ax.plot(
            self.best_c_, cv_means[best_idx],
            'r*', markersize=15, label='Selected'
        )

        ax.set_xlabel('Penalty parameter c', fontsize=12)
        ax.set_ylabel('CV Criterion', fontsize=12)
        ax.set_title('RMD Lasso Cross-Validation Results', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax