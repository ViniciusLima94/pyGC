########################################################################################
# Module with functions for parametric estimation of GC
########################################################################################
import numpy as np
import scipy.linalg


def YuleWalker(X, m):
    """Estimate the VAR model coefficients by solving the Yule-Walker equations.

    Parameters
    ----------
    X : ndarray (Nvars, N) — multi-channel time series.
    m : int — model order.

    Returns
    -------
    AR_yw  : ndarray (m, Nvars, Nvars) — AR coefficient matrices.
    eps_yw : ndarray (Nvars, Nvars)    — residual covariance.
    """
    Nvars = X.shape[0]
    N     = X.shape[1]

    # Build design matrix A as column-stack of lagged observations
    b = X.T[m:]                         # (N-m, Nvars) — response
    A = np.hstack([X.T[m - i - 1:N - i - 1, :] for i in range(m)])  # (N-m, Nvars*m)

    r      = np.matmul(A.T, b) / N
    R      = np.matmul(A.T, A) / N
    AR_yw  = np.matmul(scipy.linalg.inv(R).T, r).T
    AR_yw  = AR_yw.T.reshape((m, Nvars, Nvars))

    # Residual covariance via lagged cross-correlations
    # R_{-k}[i,j] = E[X_i(t-k) X_j(t)] = X[:, :N-k] @ X[:, k:].T / N
    eps_yw = X @ X.T / N
    for i in range(m):
        k = i + 1
        R_neg_k = X[:, :N - k] @ X[:, k:].T / N
        eps_yw -= AR_yw[i].T @ R_neg_k

    return AR_yw, eps_yw


def compute_transfer_function(AR, sigma, f, Fs):
    """Compute the transfer function H(f) and cross-spectrum S(f) from VAR coefficients.

    All loops over frequency bins are replaced with a single einsum + batch
    matrix inversion, giving a large speedup for many frequency bins.

    Parameters
    ----------
    AR    : ndarray (m, Nvars, Nvars) — AR coefficients from YuleWalker.
    sigma : ndarray (Nvars, Nvars)    — noise covariance.
    f     : ndarray (n_freq,)         — frequency axis (Hz).
    Fs    : float                     — sampling rate (Hz).

    Returns
    -------
    H : ndarray (Nvars, Nvars, n_freq) — transfer function.
    S : ndarray (Nvars, Nvars, n_freq) — cross-spectral matrix.
    """
    m     = AR.shape[0]
    Nvars = AR.shape[1]
    n_freq = f.shape[0]

    # Phase factors: comp[lag, freq] = exp(-j * 2*pi * lag * f / Fs)
    lags = np.arange(1, m + 1)                        # (m,)
    comp = np.exp(-1j * 2 * np.pi
                  * lags[:, np.newaxis] / Fs
                  * f[np.newaxis, :])                  # (m, n_freq)

    # H_correction[v1, v2, freq] = sum_lag AR[lag].T[v1,v2] * comp[lag, freq]
    AR_T        = AR.transpose(0, 2, 1)                # (m, Nvars, Nvars)
    H_correction = np.einsum('lf,lvw->vwf', comp, AR_T)   # (Nvars, Nvars, n_freq)

    # H = I - H_correction  (shape: Nvars, Nvars, n_freq)
    H = np.eye(Nvars)[:, :, np.newaxis] - H_correction

    # Batch matrix inversion across all frequencies
    H_batch = H.transpose(2, 0, 1)                    # (n_freq, Nvars, Nvars)
    H_batch = np.linalg.inv(H_batch)                  # (n_freq, Nvars, Nvars)

    # S = H @ sigma @ H.conj().T  (batch)
    S_batch = H_batch @ sigma @ np.conj(H_batch).swapaxes(-1, -2)

    H = H_batch.transpose(1, 2, 0)                    # (Nvars, Nvars, n_freq)
    S = S_batch.transpose(1, 2, 0)                    # (Nvars, Nvars, n_freq)
    return H, S
