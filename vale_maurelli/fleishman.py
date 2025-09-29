"""Class to generate a non-normal Fleishman sample with 
known mean, variance, skewness and (excess) kurtosis.
"""

import warnings

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import optimize

from .errors import FitError

class FleishmanGenerator:
    """Generate a non-normal Fleishman sample with known 
    mean, variance, skewness and (excess) kurtosis.

    This is based on Fleishman method, which creates a cubic polynomial
    with a normal seed sample, which is non-normal.

    Y = a + bX + cX^2 + dX^3

    The trick is to tune the four polynomial coefficient (a, b, c, d) such that
    the resulting non-normal sample (Y) has the desired mean, var, skew & ekurt.

    Attributes
    ----------
    fitted : bool
        If True, the generator has been fitted.

    fl_coeffs : dict
        Dictionary containing the fitted Fleishman coefficients.

    Parameters
    ----------
    mean : float, optional
        Mean of the non-normal distribution.
        Default: 0.

    var : float, optional
        Variance of the non-normal distribution.
        Default: 1.

    skew : float, optional
        Skewness of the non-normal distribution.
        Default: 0.

    ekurt : float, optional
        Excess kurtosis of the non-normal distribution.
        Default: 0.

    seed: int, optional
        Random number generator seed value.
        Default: 42

    Raises
    ------
    FitError
        Error to be raised whenever a class instance is called to 
        generate a sample without the generator having been fit yet.
    """
    def __init__(self, mean: float = 0., var: float = 1., skew: float = 0., ekurt: float = 0., seed: int = 42) -> None:    
        self.mean = mean
        self.var = var
        self.skew = skew
        self.ekurt = ekurt
        self.seed = seed
        
        # Feasibility condition for the existence of solutions
        # ! Check correctness of this expression!
        # TODO Define a mechanism to check that the values of skew and ekurt are always consistent.
        ekurt_thresh = -1.13168 + 1.58837 * self.skew**2
        if self.ekurt < ekurt_thresh:
            warnings.warn(
                f'For the Fleishman method to work with:\n' + 
                f'\tmean: {self.mean:.2f}\n\tvar:  {self.var:.2f}\n\tskew: {self.skew:.2f}\n' + 
                f'The value of [ekurt] must be >= [{ekurt_thresh:.4f}]\nUsing [ekurt] threshold value.')
            self.ekurt = ekurt_thresh

        self.__rng = np.random.default_rng(seed=self.seed)
        self.__fitted = False
        self.__fl_coeffs = None
        return

    @property
    def fitted(self) -> bool:
        return self.__fitted
    
    @property
    def fl_coeffs(self) -> dict:
        return self.__fl_coeffs

    def __check_fit(self) -> None:
        """If the generator has not been fit yet, raise a FitError."""
        if not self.__fitted:
            message = """The Fleishman generator has not been fit yet."""
            raise FitError(message=message, error_code=0)
        return
    
    def __fl_func(self, x: tuple[float]) -> float:
        """
        Define a real function which will have roots iff the coefficients
        give the desired values of skewness and excess kurtosis.
        """
        # ? May there be a better strategy to find the solutions?
        b, c, d = x

        f1 = (b**2) + 6 * (b * d) + 2 * (c**2) + 15 * (d**2) - 1
        f2 = 2 * c * ((b**2) + 24 * (b * d) + 105 * (d**2) + 2) - self.skew
        f3 = 24 * (
              (b * d)
            + (c**2) * (1 + (b**2) + 28 * (b * d))
            + (d**2) * (12 + 48 * (b * d) + 141 * (c**2) + 225 * (d**2))
        ) - self.ekurt

        return f1**2 + f2**2 + f3**2

    def __fl_ic(self) -> tuple[float]:
        """Initial condition estimate of the Fleishman coefficients."""
        # ? Why are these values as they are?
        # TODO Allow for multiple ic for grid search.
        b0 = (
              0.95357
            - 0.05679 * self.ekurt
            + 0.03520 * self.skew**2
            + 0.00133 * self.ekurt**2
        )
        c0 = 0.10007 * self.skew + 0.00844 * self.skew**3
        d0 = 0.30978 - 0.31655 * b0

        return (b0, c0, d0)
    
    def reset_sampling(self, seed: int = 42) -> None:
        self.__check_fit()
        self.seed = seed
        self.__rng = np.random.default_rng(seed=self.seed)
        return
    
    def fit(self, method: str = 'nelder-mead', maxiter: int = 256, converge: float = 1e-10, verbose: int = 0) -> None:
        """Fit the generator to compute Fleishman coefficients.

        Parameters
        ----------
        method : str, optional
            Optimization method used to fit the generator.
            Default: 'nelder-mead'
        maxiter : int, optional
            Maximum number of iterations to iterations 
            for the optimization method to converge.
            Default: 256
        converge : float, optional
            Optimization method convergence threshold. 
            Default: 1e-10
        verbose : int, optional
            Verbosity level of the optimizer.
            Default: 0
        """
        # TODO Allow for multiple ic for grid search.
        x0 = self.__fl_ic()

        # TODO Allow for higher optimizer customization.
        # TODO Define a mechanism to find and manage multiple solutions.
        optimize_results = optimize.minimize(
            lambda x: self.__fl_func(x), x0=x0, method=method,  # 'nelder-mead' 'BFGS'
            options={'maxiter': maxiter, 'xatol': converge, 'disp': verbose})
        
        b, c, d = optimize_results.x

        self.__fitted = True
        self.__fl_coeffs = {'a': -c, 'b': b, 'c': c, 'd': d}
        return

    def sample(self, size: int = 1) -> NDArray[float64]:
        """Generate a non-normal Fleishman sample.

        Parameters
        ----------
        size : int, optional
            Size of the generated sample.
            Default: 1

        Returns
        -------
        sample : NDArray[float64]
            The generated non-normal Fleishman sample.
        """
        self.__check_fit()

        b, c, d = self.__fl_coeffs['b'], self.__fl_coeffs['c'], self.__fl_coeffs['d']

        sample = self.__rng.normal(size=size)
        # Generate the field from the Fleishman polynomial.
        # Then scale it by the std and mean.
        sample = (-1 * c + sample * (b + sample * (c + sample * d))) * np.sqrt(self.var) + self.mean
        if size == 1:
            return sample[0]
        return sample
     