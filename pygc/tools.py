########################################################################################
# Module with auxiliary functions
########################################################################################
import numpy as np


def xcorr(x, y, maxlags):
    '''
        Estimate the auto (x=y) or cross (x!=y) correlation between two signals
        Inputs:
        > x : Signal x of size [Number of variables, Number of observations].
        > y : Signal y of size [Number of variables, Number of observations].
        > maxlag : maximum number of lags for the correlations.
        Outputs:
        > lag : Lags of the correlations function (> 0).
        > Rxx : Correlation function.
    '''
    Nvars = x.shape[0]
    N     = x.shape[1]
    lags  = np.arange(0, maxlags)
    Rxx   = np.zeros([lags.shape[0], Nvars, Nvars])
    for k in lags:
        Rxx[k, :, :] = np.matmul(x[:, 0:N-k], y[:, k:].T) / N
    return lags, Rxx


def PlusOperator(g, m, fs, freq):
    """Plus-operator for Wilson factorization.

    Replaces the original element-wise double-loop over (i, j) with a single
    batch FFT/IFFT along the frequency axis (axis 2).
    """
    N    = freq.shape[0] - 1
    freq_len = len(freq)

    # Batch IFFT over all (i,j) pairs simultaneously
    gam  = np.fft.ifft(g, axis=2)           # (m, m, 2N)
    gamp = gam.copy()
    gamp[:, :, 0]        = np.triu(0.5 * gam[:, :, 0])
    gamp[:, :, freq_len:] = 0

    # Batch FFT back
    return np.fft.fft(gamp, axis=2)         # (m, m, 2N)


def demean(X, norm=False):
    n, m, N = X.shape
    U = np.ones([1, N * m])
    Y = np.swapaxes(X, 1, 2).reshape((n, N * m))
    Y = Y - np.matmul(Y.mean(axis=1)[:, None], U)
    if norm:
        Y = Y / np.matmul(Y.std(axis=1)[:, None], U)
    return np.swapaxes(Y.reshape([n, N, m]), 1, 2)


def rdet(A):
    if not np.shape(A):
        return A
    return np.linalg.det(A)
