# simulations/linear_model/utils/control.py

"""Control parameters for RMD Lasso estimation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RMDControl:
    """
    Control parameters for RMD Lasso estimation.

    Parameters
    ----------
    maxiter : int
        Maximum iterations for coordinate descent
    optTol : float
        Convergence tolerance for coordinate descent
    zeroThreshold : float
        Threshold below which coefficients are set to zero
    intercept_penalty : float (c3 in CNS paper)
        Penalty multiplier for intercept (typically small, e.g., 0.1)
    c : float
        Penalty scaling constant (c1 in CNS paper)
    alpha : float
        Significance level for penalty formula (c2 in CNS paper)
    max_outer_iter : int
        Maximum iterations for outer normalization loop
    outer_tol : float
        Convergence tolerance for outer loop
    D_LB : float
        Lower bound for diagonal entries of D
    D_add : float
        Additive stabilization for D (CNS uses 0.2)
    normalize : bool
        Whether to use iterative normalization
    low_dim_divisor : int
        Divisor for low-dimensional initialization (p0 = p_x // low_dim_divisor)
        CNS uses 40
    """
    maxiter: int = 1000
    optTol: float = 1e-5
    zeroThreshold: float = 1e-6
    intercept_penalty: float = 0.1
    c: float = 1.0
    alpha: float = 0.1
    max_outer_iter: int = 100
    outer_tol: float = 1e-6
    D_LB: float = 0.0
    D_add: float = 0.2
    normalize: bool = True
    low_dim_divisor: int = 40

    def __post_init__(self):
        """Validate control parameters."""
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.optTol <= 0:
            raise ValueError("optTol must be positive")
        if self.zeroThreshold < 0:
            raise ValueError("zeroThreshold must be non-negative")
        if self.c <= 0:
            raise ValueError("c must be positive")
        if self.low_dim_divisor <= 0:
            raise ValueError("low_dim_divisor must be positive")


@dataclass
class RMDCVControl(RMDControl):
    """
    Control parameters for RMD Lasso with cross-validation.

    Additional parameters beyond RMDControl:
    ----------
    n_folds : int
        Number of cross-validation folds
    c_vec : List[float]
        Grid of c values to search over (CNS paper: {5/4, 1, 3/4, 1/2})
    refit : bool
        Whether to refit on full data with best c
    """
    n_folds: int = 5
    c_vec: List[float] = field(default_factory=lambda: [1.25, 1.0, 0.75, 0.5])
    refit: bool = True

    def __post_init__(self):
        """Validate CV-specific parameters and set defaults."""
        # Note: We still call super().__post_init__() to validate other inherited
        # parameters (maxiter, optTol, etc.), but the `c` validation is irrelevant
        # for CV since c_vec is used instead
        super().__post_init__()
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if len(self.c_vec) == 0:
            raise ValueError("c_vec must not be empty")
