import numpy as np
from joblib import Parallel, delayed
from .tools import rdet


def covgc_time(X, dt, lag, t0, n_jobs=1):
    """Covariance-based Granger Causality (time-domain, single trial).

    Parameters
    ----------
    X      : ndarray (nSo, nTi) — data, sources × time points.
    dt     : int — window duration in samples.
    lag    : int — number of lag samples.
    t0     : int — zero-time index.
    n_jobs : int — joblib workers for parallel pair computation (-1 = all cores).

    Returns
    -------
    GC : ndarray (nPairs, 3) — [GC(x->y), GC(y->x), GC(x.y)] per pair.
    """
    X  = np.array(X)
    dt = int(dt)
    t0 = int(t0)

    nSo, _ = X.shape

    # Build index matrix
    ind_t = np.empty((dt, lag + 1), dtype=int)
    for i in range(dt):
        for j in range(lag + 1):
            ind_t[i, j] = (t0 - lag) + i + (lag - j)

    nPairs = int(nSo * (nSo - 1) // 2)

    def _pair_gc(i, j):
        x = X[i, ind_t]   # (dt, lag+1)
        y = X[j, ind_t]

        # Conditional entropies
        det_yi1  = rdet(np.cov(y.T))
        det_yi   = rdet(np.cov(y[:, 1:].T))
        Hycy     = np.log(det_yi1) - np.log(det_yi)

        det_xi1  = rdet(np.cov(x.T))
        det_xi   = rdet(np.cov(x[:, 1:].T))
        Hxcx     = np.log(det_xi1) - np.log(det_xi)

        det_yxi1 = rdet(np.cov(np.column_stack((y, x[:, 1:])).T))
        det_yxi  = rdet(np.cov(np.column_stack((y[:, 1:], x[:, 1:])).T))
        Hycx     = np.log(det_yxi1) - np.log(det_yxi)

        det_xyi1  = rdet(np.cov(np.column_stack((x, y[:, 1:])).T))
        Hxcy      = np.log(det_xyi1) - np.log(det_yxi)

        det_xyji1 = rdet(np.cov(np.column_stack((x, y)).T))
        Hxxcyy    = np.log(det_xyji1) - np.log(det_yxi)

        return np.array([Hycy - Hycx, Hxcx - Hxcy, Hycx + Hxcy - Hxxcyy])

    pairs = [(i, j) for i in range(nSo) for j in range(i + 1, nSo)]
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_pair_gc)(i, j) for i, j in pairs
    )

    return np.array(results)     # (nPairs, 3)
