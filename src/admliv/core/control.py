# admliv/core/control.py

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PGMMControl:
    """
    Control parameters for PGMM estimation.

    Parameters
    ----------
    maxiter : int, default=4000
        Maximum number of coordinate descent iterations
    optTol : float, default=1e-5
        Convergence tolerance for coordinate descent
    zeroThreshold : float, default=1e-6
        Threshold below which coefficients are set to zero
    intercept_penalty : float, default=0.1
        Penalty multiplier for intercept term (should be small)
    c : float, default=1.0
        Penalty parameter multiplier for lambda = c * sqrt(log(p) / n)
    adaptive_threshold : float, default=1e-10
        Threshold for adaptive weights calculation (numerical stability)
    adaptive_max_weight : float, default=1e10
        Maximum value for adaptive weights (numerical stability)
    check_frequency : int, default=5
        How often to check inactive coordinates in active set strategy
    buffer_factor : float, default=1.1
        Buffer for KKT violation check in active set (larger = more conservative)
    """
    maxiter: int = 5000
    optTol: float = 1e-5
    zeroThreshold: float = 1e-6
    intercept_penalty: float = 0.1
    c: float = 0.01
    adaptive_threshold: float = 1e-10
    adaptive_max_weight: float = 1e10
    check_frequency: int = 5
    buffer_factor: float = 1.1

    def __post_init__(self):
        """Validate control parameters."""
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.optTol <= 0:
            raise ValueError("optTol must be positive")
        if self.zeroThreshold < 0:
            raise ValueError("zeroThreshold must be non-negative")
        if self.intercept_penalty < 0:
            raise ValueError("intercept_penalty must be non-negative")
        if self.c <= 0:
            raise ValueError("c must be positive")
        if self.check_frequency <= 0:
            raise ValueError("check_frequency must be positive")
        if self.buffer_factor <= 0:
            raise ValueError("buffer_factor must be positive")


@dataclass
class PGMMCVControl(PGMMControl):
    """
    Control parameters for cross-validated PGMM.

    Inherits all parameters from PGMMControl and adds:

    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-validation folds
    c_vec : Optional[np.ndarray], default=None
        Grid of c values to search over. If None, uses default grid.
    n_jobs : int, default=1
        Number of parallel jobs for cross-validation
    random_state : int, default=42
        Random seed for reproducibility

    Notes
    -----
    The inherited parameter `c` is not used in cross-validation. Instead,
    `c_vec` specifies the grid of values to search over. When creating a
    PGMMCVControl instance, you can omit the `c` parameter entirely - it will
    use the default value but will be ignored during cross-validation.
    """
    n_folds: int = 5
    c_vec: Optional[np.ndarray] = None
    n_jobs: int = 1
    random_state: int = 42

    def __post_init__(self):
        """Validate CV-specific parameters and set defaults."""
        # Note: We still call super().__post_init__() to validate other inherited
        # parameters (maxiter, optTol, etc.), but the `c` validation is irrelevant
        # for CV since c_vec is used instead
        super().__post_init__()

        if self.n_folds <= 1:
            raise ValueError("n_folds must be greater than 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")

        # Default grid for c if not provided
        if self.c_vec is None:
            self.c_vec = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

        # Validate c_vec values
        if not isinstance(self.c_vec, np.ndarray):
            self.c_vec = np.array(self.c_vec)
        if len(self.c_vec) == 0:
            raise ValueError("c_vec must contain at least one value")
        if np.any(self.c_vec <= 0):
            raise ValueError("All values in c_vec must be positive")
