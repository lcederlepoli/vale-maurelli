import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

def to_array(a: ArrayLike) -> NDArray:
    """Convert input data to NumPy NDArray, where 
    each row represents a feature.

    Parameters
    ----------
    a : ArrayLike
        Statistical sample of data.
            
    Returns
    -------
    arr : NDArray
        Sample data formatted as numpy NDArray.
    """  
    if isinstance(a, pd.Series):
        a = a.values
    elif isinstance(a, pd.DataFrame) and a.shape[1] == 1:
        a = a.values.reshape(-1)
    elif isinstance(a, pd.DataFrame) and a.shape[1] > 1:
        a = a.values.T
    # If input is a list or tuple, covert to numpy array
    arr = np.asarray(a)
    return arr

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes input Pandas DataFrame."""
    return (df - df.min()) / (df.max() - df.min())

def inv_normalize(df: pd.DataFrame, min_max_values: pd.DataFrame) -> pd.DataFrame:
    """Inverts normalization of input Pandas DataFrame."""
    min, max = min_max_values.loc['min'], min_max_values.loc['max']
    return df * (max - min) + min

def transform_range(df: pd.DataFrame, target_range: pd.DataFrame, 
                    input_range: pd.DataFrame = None) -> pd.DataFrame:
    if input_range is None:
        input_range = df.agg(['min', 'max'])
    input_min, input_max = input_range.loc['min'], input_range.loc['max']
    target_min, target_max = target_range.loc['min'], target_range.loc['max']
    return (df - input_min) / (input_max - input_min) * (target_max - target_min) + target_min
