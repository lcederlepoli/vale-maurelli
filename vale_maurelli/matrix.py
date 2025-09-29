import numpy as np
import sklearn.covariance as skcov

def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B) -> bool:
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def shrink_matrix(a, is_corr=False, method='basic', shrinkage=.1):
    """
    Shrink input matrix to convert it to positive definite.

    Parameters
    ----------
    a : 2-D array_like, of shape (N, N)
        Matrix to be shrunk.
    method : string
       Shrinkage method to apply.
    shrinkage : float
        Shrinkage parameter.
            
    Returns
    -------
    a_shrk : 2-D array_like, of shape (N, N)
        Shrunk matrix.
    """
    # TODO: Check input type
    # TODO: Check input shape

    # TODO: implement Ledoit-Wolf shrinkage and Oracle approximating shrinkage
    # TODO: consider implementing the following:
    # - Sparse inverse covariance
    # - Robust Covariance Estimation
    if not isPD(a):
        a = skcov.shrunk_covariance(a, shrinkage=shrinkage)
        if is_corr:
            np.fill_diagonal(a, 1.)
            # Check bounds of correlation matrix
            # TODO: Test and implement normalization.
            if np.abs(a).max() > 1.:
                raise ValueError('Correlation matrix is not correctly normalized.')
        if not isPD(a):
            raise ValueError('Resulting covariance matrix is not positive definite. Try using a different value for parameter shrinkage.')
    return a
