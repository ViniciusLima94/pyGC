########################################################################################
# Unified Pipeline(metric).fit(data) API over the GC estimators in granger.py
########################################################################################
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from .granger import (
    _VALID_BACKENDS,
    _bivariate_gc_from_HZ,
    _compute_csd,
    _get_factorization_fn,
    conditional_granger_causality,
    conditional_spec_granger_causality,
)

_VALID_METRICS = ('gc', 'spectral_gc', 'conditional', 'conditional_spectral')


class Pipeline:
    """Compute an all-to-all Granger Causality estimate over multi-node data.

    Parameters
    ----------
    metric          : {'gc', 'spectral_gc', 'conditional', 'conditional_spectral'}
                      'gc'/'spectral_gc'    : bivariate, frequency-resolved GC for
                                               every node pair (via granger_causality).
                      'conditional'         : time-domain conditional GC matrix
                                               (via conditional_granger_causality).
                      'conditional_spectral': frequency-resolved conditional GC
                                               (via conditional_spec_granger_causality).
    fs              : float — sampling rate (Hz).
    spectral_method : {'fourier', 'morlet', 'welch', 'multitaper'} — spectral estimator.
    backend         : {'numpy', 'jax'} — Wilson factorization backend.
    n_jobs          : int — joblib parallelism for 'numpy'-backend pairwise/conditional
                      computations (-1 = all cores). Ignored by the 'jax' backend on
                      the bivariate path, which reuses a single JIT-compiled kernel.
    Niterations     : int — maximum Wilson factorization iterations.
    tol             : float — convergence tolerance.
    verbose         : bool — print factorization progress.
    spectral_params : dict or None — extra kwargs for the spectral estimator.
    ensure_stability: bool — regularize near-singular spectral matrices.
    """

    def __init__(self, metric='gc', fs=None, spectral_method='fourier', backend='numpy',
                 n_jobs=1, Niterations=100, tol=1e-12, verbose=False,
                 spectral_params=None, ensure_stability=True):
        if metric not in _VALID_METRICS:
            raise ValueError(f"Unknown metric {metric!r}. Choose from {_VALID_METRICS}.")
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}.")
        if fs is None:
            raise ValueError("fs (sampling rate) must be provided.")

        self.metric = metric
        self.fs = fs
        self.spectral_method = spectral_method
        self.backend = backend
        self.n_jobs = n_jobs
        self.Niterations = Niterations
        self.tol = tol
        self.verbose = verbose
        self.spectral_params = spectral_params
        self.ensure_stability = ensure_stability
        self.result_ = None

    def fit(self, data):
        """Compute the connectivity estimate.

        Parameters
        ----------
        data : ndarray — (nodes, time) for a single trial, or (trials, nodes, time).

        Returns
        -------
        result : xr.DataArray — dims ('edge', 'freq') for spectral metrics, or
                 ('edge',) for 'conditional'. The 'edge' coordinate holds strings
                 "Node_i->Node_j" for every ordered pair i != j.
        """
        if self.metric in ('gc', 'spectral_gc'):
            result = self._fit_bivariate(data)
        elif self.metric == 'conditional':
            result = self._fit_conditional(data)
        else:  # 'conditional_spectral'
            result = self._fit_conditional_spectral(data)

        self.result_ = result
        return result

    def _fit_bivariate(self, X):
        nvars = X.shape[-2]
        S, f = _compute_csd(X, self.fs, self.spectral_method, self.spectral_params)
        factorize = _get_factorization_fn(self.backend)
        pairs = [(i, j) for i in range(nvars) for j in range(i + 1, nvars)]

        def _pair(i, j):
            S_pair = S[np.ix_([i, j], [i, j])]
            _, H, Z = factorize(S_pair, f, self.fs, self.Niterations, self.tol,
                                 False, self.ensure_stability)
            Ix2y, Iy2x, _ = _bivariate_gc_from_HZ(S_pair, H, Z)
            return i, j, Ix2y, Iy2x

        if self.backend == 'numpy':
            results = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(_pair)(i, j) for i, j in pairs
            )
        else:  # 'jax' — shared (2, 2, n_freq) shape, JIT compiles once and is reused
            results = [_pair(i, j) for i, j in pairs]

        edges = []
        values = np.zeros((2 * len(pairs), len(f)))
        for k, (i, j, Ix2y, Iy2x) in enumerate(results):
            edges.append(f"Node_{i}->Node_{j}")
            values[2 * k] = Ix2y
            edges.append(f"Node_{j}->Node_{i}")
            values[2 * k + 1] = Iy2x

        return xr.DataArray(values, dims=('edge', 'freq'),
                             coords={'edge': edges, 'freq': f})

    def _fit_conditional(self, X):
        F = conditional_granger_causality(
            X, self.fs, spectral_method=self.spectral_method, backend=self.backend,
            Niterations=self.Niterations, tol=self.tol, verbose=self.verbose,
            n_jobs=self.n_jobs, spectral_params=self.spectral_params,
            ensure_stability=self.ensure_stability,
        )
        nvars = F.shape[0]
        edges = []
        values = []
        for i in range(nvars):          # target
            for j in range(nvars):      # source
                if i == j:
                    continue
                edges.append(f"Node_{j}->Node_{i}")  # F[i, j] = GC(j -> i)
                values.append(F[i, j])

        return xr.DataArray(np.array(values), dims=('edge',), coords={'edge': edges})

    def _fit_conditional_spectral(self, X):
        GC, f = conditional_spec_granger_causality(
            X, self.fs, spectral_method=self.spectral_method, backend=self.backend,
            Niterations=self.Niterations, tol=self.tol, verbose=self.verbose,
            n_jobs=self.n_jobs, spectral_params=self.spectral_params,
            ensure_stability=self.ensure_stability,
        )
        nvars = GC.shape[0]
        edges = []
        rows = []
        for j in range(nvars):          # source
            for i in range(nvars):      # target
                if i == j:
                    continue
                edges.append(f"Node_{j}->Node_{i}")  # GC[j, i, :] = source j -> target i
                rows.append(GC[j, i, :])

        return xr.DataArray(np.array(rows), dims=('edge', 'freq'),
                             coords={'edge': edges, 'freq': f})
