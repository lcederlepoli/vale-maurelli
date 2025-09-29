from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from . import fleishman
from . import matrix
from . import utils

def mvsk(a, ddof=1, bias=False, axis=None, shrink_method=None, shrinkage=.1, to_tuple=False) -> Union[dict, tuple]:
    """Compute mean, variance, skewness, and (excess) kurtosis of a 
    given sample of data.
    
    If the input dimension is greater than one, compute also 
    covariance and correlation matrices.

    Parameters
    ----------
    a : array_like
        Statistical sample of data.
    ddof : int
       Degrees of freedom to estimate variance. Input value for
       scipy.stats.tvar.
    bias : bool
        Bias to estimate skewness and kurtosis. Input value for
        scipy.stats.skew and scipy.stats.kurtosis.
    axis : int (0, 1)
        Axis to compute statistics along.
    shrink_method : string
        Shrink method to apply.
    shrinkage : float
        Shrinkage parameter.
    to_tuple : bool
        If True, return output formatted as a tuple.
            
    Returns
    -------
    mvsk_stats : dict, tuple
        Dictionary containing the statistics computed on the input 
        data sample.
        {
            'mean': mean
         [, 'cov' covariance]
          , 'var': variance
         [, 'cov' covariance]
          , 'skew': skewness
          , 'ekurt': (excess) kurtosis
        }
        If parameter to_tuple is True, output is converted to a tuple 
        with the same ordering.
    """
    # Convert input to numpy ndarray of shape (n_variables, n_samples)
    a = utils.to_array(a)
    
    # Compute statistics depending on the size of input
    if a.ndim == 1:
        mvsk_stats = {
            'mean': stats.tmean(a, axis=axis)
          , 'var': stats.tvar(a, ddof=ddof)
          , 'skew': stats.skew(a, bias=bias)
          , 'ekurt': stats.kurtosis(a, bias=bias)
        }
    else:
        # Compute covariance matrix. If the numpy Maximum Likelihood estimate is not 
        # positive-definite, use scikit-learn basic shrinkage method to provide a 
        # better estimate.
        # TODO: implement Ledoit-Wolf shrinkage and Oracle approximating shrinkage
        # TODO: consider implementing the following:
        # TODO: - Sparse inverse covariance
        # TODO: - Robust Covariance Estimation
        cov = np.cov(a, ddof=ddof)
        if shrink_method is not None:
            cov = matrix.shrink_matrix(cov, method=shrink_method, shrinkage=shrinkage)
        var = np.diag(cov)
        if not np.all(var) > 0.:
            msg = 'Not all variances are positive. Try using a shrinkage method or change shrinking parameter.'
            raise ValueError(msg)

        # Compute correlation matrix. Same methodology as for the covariance matrix.
        corr = np.corrcoef(a)
        if shrink_method is not None:
            corr = matrix.shrink_matrix(corr, is_corr=True, shrinkage=shrinkage)

        mvsk_stats = {
            'mean': stats.tmean(a, axis=axis)
          , 'cov': cov
          , 'var': var
          , 'corr': corr
          , 'skew': stats.skew(a, bias=bias, axis=axis)
          , 'ekurt': stats.kurtosis(a, bias=bias, axis=axis)
        }

    if to_tuple:
        mvsk_stats = tuple(mvsk_stats.values())
    return mvsk_stats

def mvsk_describe(df: pd.DataFrame, ddof=1, bias=False, pivot=False) -> pd.DataFrame:
    """Compute mean, variance, skewness, and (excess) kurtosis of the 
    columns of a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        Statistical sample of data.
    ddof : int
       Degrees of freedom to estimate variance. Input value for
       scipy.stats.tvar.
    bias : bool
        Bias to estimate skewness and kurtosis. Input value for
        scipy.stats.skew and scipy.stats.kurtosis.
    pivot : bool
        If True, transpose the output.
            
    Returns
    -------
    mvsk_df : pandas DataFrame
        Pandas DataFrame containing mean, variance, skewness, and (excess) 
        kurtosis of the columns of the input DataFrame. If pivot = False, 
        each row represents a column of the input DataFrame. 
    """
    # Check input type
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input data must be formatted as a pandas DataFrame.')

    # Compute statistics for each column
    mean = pd.DataFrame(df.apply(lambda x: stats.tmean(x)), columns=['mean'])
    var = pd.DataFrame(df.apply(lambda x: stats.tvar(x, ddof=ddof)), columns=['var'])
    skew = pd.DataFrame(df.apply(lambda x: stats.skew(x, bias=bias)), columns=['skew'])
    ekurt = pd.DataFrame(df.apply(lambda x: stats.kurtosis(x, bias=bias)), columns=['ekurt'])

    mvsk_df = pd.concat([mean, var, skew, ekurt], axis=1)

    # Pivot the result
    if pivot:
        mvsk_df = mvsk_df.T
    return mvsk_df

def mvsk_compare(orig: pd.DataFrame, synth: pd.DataFrame, side_by_side=False, ddof=1, bias=False, pivot=False) -> pd.DataFrame:
    """
    """
    # Check if the input DataFrames have the same columns in the same order
    if not orig.columns.tolist() == synth.columns.tolist():
        raise ValueError('Input DataFrames must have the same column names in the same order.')
    mvsk_orig = mvsk_describe(orig, ddof=ddof, bias=bias, pivot=pivot)
    mvsk_synth = mvsk_describe(synth, ddof=ddof, bias=bias, pivot=pivot)
    mvsk = pd.concat([mvsk_orig, mvsk_synth], axis=1)
    # Create multi-index
    if side_by_side:
        cols = mvsk.columns.tolist()
        cols = [0, 4, 1, 5, 2, 6, 3, 7]
        mvsk = mvsk.iloc[:, cols]
        multi_index = pd.MultiIndex.from_tuples([(col, src) for col in mvsk_orig.columns for src in ['Original', 'Synthetic']])
        mvsk.columns = multi_index
    else:
        multi_index = pd.MultiIndex.from_tuples([(src, col) for src in ['Original', 'Synthetic'] for col in mvsk_orig.columns])
        mvsk.columns = multi_index
    return mvsk

def fl_coeff(a) -> dict:
    """Compute Fleishman's coefficients of a given statistical sample.

    Parameters
    ----------
    a : array_like, dict
        If array-like, the input is the statistical sample of interest.
        If dictionary, it must contain unbiased mean, variance, skewness, 
        and (excess) kurtosis of the statistical sample. In this case the
        input must be formatted as follows:
        {
            'mean': mean
          , 'var': variance
          , 'skew': skewness
          , 'ekurt': (excess) kurtosis
        }
            
    Returns
    -------
    ff_coeff : dict
        Dictionary containing coefficients of the Fleishman's polynomial 
        Y = a + b*X + c*X**2 + d*X**3, formatted as follows
        {
            'a': a
          , 'b': b
          , 'c': c
          , 'd': d
        }
    """
    # Check input type
    # TO-DO: enhance type check!
    if not isinstance(a, dict):
        a = np.array(a)
        if not isinstance(a, np.ndarray):
            raise TypeError('Input must be either a numpy array or a dictionary.')
        a = mvsk(a)

    # Initialize Fleishman instance and compute Fleishman's coefficients
    ff = fleishman.FleishmanGenerator(mean=a['mean'], var=a['var'], skew=a['skew'], ekurt=a['ekurt'])
    ff.fit()
    fl_coeffs = ff.fl_coeffs
    return fl_coeffs
