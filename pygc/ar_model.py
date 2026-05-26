import numpy as np


def ar_model_dhamala(N=5000, Trials=10, Fs=200, C=0.2, t_start=0, t_stop=None, cov=None):
    """
    Bivariate AR model from Dhamala et al. (2008).

    Y drives X with coupling strength C in the interval [t_start, t_stop].
    GC(Y->X) > 0, GC(X->Y) ≈ 0.

    Parameters
    ----------
    N      : int   — number of time samples per trial.
    Trials : int   — number of independent trials.
    Fs     : float — sampling rate (Hz).
    C      : float — coupling coefficient from Y to X.
    t_start, t_stop : float — time interval (seconds) during which coupling is active.
    cov    : ndarray, shape (2,2) — noise covariance matrix.

    Returns
    -------
    Z : ndarray, shape (2, Trials, N)
    """
    if cov is None:
        cov = np.eye(2)

    T = N / Fs
    time = np.linspace(0, T, N)

    X = np.random.random([Trials, N])
    Y = np.random.random([Trials, N])

    def _active(t):
        if t_stop is None:
            return t >= t_start
        return (t >= t_start) & (t <= t_stop)

    for i in range(Trials):
        E = np.random.multivariate_normal(np.zeros(2), cov, size=(N,))
        for t in range(2, N):
            X[i, t] = (0.55 * X[i, t - 1] - 0.8 * X[i, t - 2]
                       + _active(time[t]) * C * Y[i, t - 1] + E[t, 0])
            Y[i, t] = 0.55 * Y[i, t - 1] - 0.8 * Y[i, t - 2] + E[t, 1]

    Z = np.zeros([2, Trials, N])
    Z[0] = X
    Z[1] = Y
    return Z


def ar_model_baccala(nvars=5, N=1000, ntrials=1):
    """
    Five-variable AR model from Baccalá & Sameshima (2001).

    Connectivity: X1 → X2, X1 → X3, X1 → X4, X1 → X5 (via X4).

    Parameters
    ----------
    nvars   : int — number of variables (must be 5).
    N       : int — number of time samples per trial.
    ntrials : int — number of trials.

    Returns
    -------
    Y : ndarray, shape (nvars, N, ntrials)
    """
    Y = np.random.uniform(size=(nvars, N, ntrials))
    w = np.random.normal(0, 1, size=(nvars, N, ntrials))

    for i in range(ntrials):
        for t in range(5, N):
            Y[0, t, i] = (0.95 * np.sqrt(2.0) * Y[0, t - 1, i]
                          - 0.9025 * Y[0, t - 2, i] + w[0, t, i])
            Y[1, t, i] = 0.5 * Y[0, t - 2, i] + w[1, t, i]
            Y[2, t, i] = -0.4 * Y[0, t - 3, i] + w[2, t, i]
            Y[3, t, i] = (-0.5 * Y[0, t - 2, i]
                          + 0.25 * np.sqrt(2.0) * Y[3, t - 1, i]
                          + 0.25 * np.sqrt(2.0) * Y[4, t - 1, i] + w[3, t, i])
            Y[4, t, i] = (-0.25 * np.sqrt(2.0) * Y[3, t - 1, i]
                          + 0.25 * np.sqrt(2.0) * Y[4, t - 1, i] + w[4, t, i])
    return Y
