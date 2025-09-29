"""Class to generate non-normal Vale & Maurelli sample with 
known mean, covariance, skewness and (excess) kurtosis.
"""

import warnings

import cloudpickle
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from scipy import stats

from . import fleishman
from . import mvsk
from . import matrix
from . import utils
from .errors import NormalizationError, FitError, LoadError

class ValeMaurelliSynthesizer():
    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=False, 
                 verbose=False, random_state=None):
        self.metadata = metadata
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.verbose = verbose
        self.random_state = random_state

        self.columns = list(self.metadata.columns.keys())
        self.nvar = len(self.columns)

        self._min_max_data_values = None

        self.__fitted = False
        self.__stats = None
        self.__fl_coeffs = None
        self.__int_corr = None
        self.__rng = None
        return

    @property
    def fitted(self) -> bool:
        return self.__fitted
    
    @property
    def stats(self):
        return self.__stats
    
    @property
    def fl_coeffs(self):
        return self.__fl_coeffs
    
    @property
    def int_corr(self):
        return self.__int_corr

    def __check_fit(self):
        """If the generator has not been fit yet, raise a FitError."""
        if not self.__fitted:
            message = """The ValeMaurelli synthesizer has not been fit yet."""
            raise FitError(message=message, error_code=0)
    
    def __check_positive_definite(self, A, msg=None):
        if not matrix.isPD(A):
            raise ValueError(msg)
        
    def _compute_min_max_data_values(self, data):
        self._min_max_data_values = data.agg(['min', 'max'])

    def _preprocess_data(self, data):
        data_prep = utils.normalize(data)
        data_prep = utils.to_array(data_prep)
        return data_prep
        
    def _compute_data_statistics(self, data):
        mean, cov, var, corr, skew, ekurt = mvsk.mvsk(
            data, ddof=1, bias=False, axis=1, 
            shrink_method='basic', shrinkage=1e-5, 
            to_tuple=True)

        self.__stats = {'mean': mean, 'cov': cov, 'var': var, 
                        'corr': corr, 'skew': skew, 'ekurt': ekurt}

        msg = 'Input covariance matrix must be positive-definite.'
        self.__check_positive_definite(cov, msg=msg)

        msg = 'Input correlation matrix must be positive-definite.'
        self.__check_positive_definite(corr, msg=msg)
        
    def _compute_fleishman_coefficients(self, **kwargs):
        fl_coeffs = []
        for i in range(self.nvar):
            ff = fleishman.FleishmanGenerator(
                mean=self.__stats['mean'][i], var=self.__stats['var'][i], 
                skew=self.__stats['skew'][i], ekurt=self.__stats['ekurt'][i])
            ff.fit(**kwargs)
            fl_coeffs.append(ff.fl_coeffs)
        self.__fl_coeffs = fl_coeffs
    
    def _compute_interm_corr_coeff(self, flc1, flc2, corr, thrs=1e-10):
        """Compute intermediate correlation for Vale & Maurelli transformation, 
        given the coeffients of two Fleishman polynomials as input.

        Parameters
        ----------
        fl_coeffs : tuple
            Tuple containing two dictionaries, which contain the Fleishman 
            coefficients of the two variables to be simulated. Each dictionary  
            must be formatted as follows:
            {
                  'a': a
                , 'b': b
                , 'c': c
                , 'd': d
            }
        corr : float
            Original correlation of the two variables to be simulated.
        thrs : float
            Threshold under which the imaginary part of the roots of the 
            transformation polynomial is to be dropped.
                
        Returns
        -------
        ro : float
            Intermediate correlation coefficient for Vale & Maurelli transformation.
        """
        # Extract Fleishman's coefficients
        b1, c1, d1 = flc1['b'], flc1['c'], flc1['d']
        b2, c2, d2 = flc2['b'], flc2['c'], flc2['d']

        coef = [-1.*corr, b1*b2 + 3.*b1*d2 + 3.*d1*b2 + 9.*d1*d2, 2.*c1*c2, 6.*d1*d2]
        p = Polynomial(coef=coef)

        # Find all real roots up to a specified threshold
        # TODO: Find a better method to manage the threshold parameter
        # ro = np.roots(p)
        ro = p.roots()
        ro = ro.real[abs(ro.imag) < thrs]
        ro = [r for r in ro if np.abs(r) <= 1.]

        # Check result
        # TODO Find a better method to manage results. In particular
        # TODO evaluate the effectiveness of the numerical solution
        if len(ro) == 0: # If no valid solution exist, use default correlation
            warnings.warn('No valid solution to compute intermediate correlation exists. Using original correlation as default.')
            ro = corr
        else:
            if len(ro) > 1: # If there are more real roots, warn the user and take the first one
                warnings.warn('There are more than one real roots with abs less than one.')
            ro = min(ro)
        return ro
    
    def _compute_intermediate_correlation_matrix(self):
        int_corr = np.ones((self.nvar, self.nvar))
        for i in range(self.nvar):
            for j in range(i + 1, self.nvar):
                int_corr[i, j] = self._compute_interm_corr_coeff(
                    self.__fl_coeffs[i], self.__fl_coeffs[j], self.__stats['corr'][i, j])
                int_corr[j, i] = int_corr[i, j]
        # If the intermediate correlation matrix is not positive semi-definite, 
        # redefine it as the nearest positive semi-definite matrix
        if not matrix.isPD(int_corr):
            int_corr = matrix.nearestPD(int_corr)
        # TODO Implement other regularization methods, including shrinkage.
        self.__int_corr = int_corr

    def _generate_synthetic_sample(self, num_rows):
        multvar_norm_sample = self.__rng.rvs(size=num_rows)
        if num_rows == 1:
            multvar_norm_sample = np.expand_dims(multvar_norm_sample, axis=0)
        synthetic_sample = np.zeros((num_rows, self.nvar))
        for i in range(self.nvar):
            N = multvar_norm_sample[:, i]
            b, c, d = self.fl_coeffs[i]['b'], self.fl_coeffs[i]['c'], self.fl_coeffs[i]['d']
            synthetic_sample[:, i] = (-1 * c + N * (b + N * (c + N * d))) * np.sqrt(self.__stats['var'][i]) + self.__stats['mean'][i]
        return synthetic_sample
    
    def _postprocess_data(self, data):
        data_post = pd.DataFrame(data, columns=self.columns)
        data_post = utils.inv_normalize(data_post, min_max_values=self._min_max_data_values)
        if self.enforce_min_max_values:
            data_post = utils.transform_range(
                data_post, target_range=self._min_max_data_values)
        return data_post
    
    def reset_sampling(self, random_state=None):
        self.__check_fit()
        if random_state is not None:
            self.random_state = random_state
        self.__rng = stats.multivariate_normal(
            mean=np.zeros((self.nvar,)), cov=self.__int_corr, 
            allow_singular=True, seed=self.random_state)

    def fit(self, data):
        self._compute_min_max_data_values(data)
        data_prep = self._preprocess_data(data)
        self._compute_data_statistics(data_prep)
        self._compute_fleishman_coefficients()
        self._compute_intermediate_correlation_matrix()
        self.__rng = stats.multivariate_normal(
            mean=np.zeros((self.nvar,)), cov=self.__int_corr, 
            allow_singular=True, seed=self.random_state)
        self.__fitted = True

    def sample(self, num_rows=1):
        self.__check_fit()
        synthetic_data = self._generate_synthetic_sample(num_rows)
        synthetic_data = self._postprocess_data(synthetic_data)
        return synthetic_data
    
    def save(self, filepath):
        """Save this model instance to the given path using cloudpickle.

        Parameters
        ----------
        filepath (str):
            Path where the synthesizer instance will be serialized.

        """
        with open(filepath, 'wb') as output:
            cloudpickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a single-table synthesizer from a given path.

        Parameters
        ----------
        filepath (str):
            A string describing the filepath of your saved synthesizer.

        Returns
        -------
        SingleTableSynthesizer:
            The loaded synthesizer.
        """
        with open(filepath, 'rb') as f:
            try:
                synthesizer = cloudpickle.load(f)
            except RuntimeError as e:
                err_msg = (
                    'Attempting to deserialize object on a CUDA device but '
                    'torch.cuda.is_available() is False. If you are running on a CPU-only machine,'
                    " please use torch.load with map_location=torch.device('cpu') "
                    'to map your storages to the CPU.'
                )
                if str(e) == err_msg:
                    raise LoadError(
                        'This synthesizer was created on a machine with GPU but the current '
                        'machine is CPU-only. This feature is currently unsupported. We recommend'
                        ' sampling on the same GPU-enabled machine.'
                    )
                raise e

        return synthesizer
