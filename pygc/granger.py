########################################################################################
# Module with functions to compute GC
########################################################################################
import logging
from typing import Callable
import numpy as np
from joblib import Parallel, delayed
from itertools import combinations
from .output import build_granger_dataset, build_conditional_gc_dataset, build_conditional_spec_gc_dataset

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ('numpy', 'jax')
_VALID_SPECTRAL_METHODS = ('fourier', 'morlet', 'welch', 'multitaper')
_VALID_MODELS = ('nonparametric', 'parametric')


def _get_factorization_fn(backend: str) -> Callable[..., tuple]:
    """Return the Wilson factorization callable for *backend* ('numpy' or 'jax')."""
    if backend == 'numpy':
        from .non_parametric import wilson_factorization
        return wilson_factorization
    if backend == 'jax':
        from ._jax_backend import wilson_factorization_jax, JAX_AVAILABLE
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. Install it with: pip install jax"
            )
        return wilson_factorization_jax
    raise ValueError(f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}.")


def _compute_csd(X, fs, spectral_method, spectral_params):
    """Compute cross-spectral matrix and frequency axis from raw data.

    Parameters
    ----------
    X               : ndarray
                      'fourier'/'morlet' : (nvars, N)
                      'welch'            : (trials, nvars, N) or (nvars, N) for 1 trial
    fs              : float — sampling rate.
    spectral_method : str — one of _VALID_SPECTRAL_METHODS.
    spectral_params : dict or None — extra kwargs forwarded to the spectral estimator.
                      'morlet' requires 'freqs'; supports 'n_cycles' (default 7.0).
                      'welch'  supports 'window', 'nfft', 'scaling', 'n_jobs'.

    Returns
    -------
    S : ndarray (nvars, nvars, n_freq) — cross-spectral matrix.
    f : ndarray (n_freq,) — frequency axis (Hz).
    """
    params = spectral_params or {}

    if spectral_method == 'fourier':
        from .spectral_analysis import compute_freq, csd_fourier
        if X.ndim == 2:
            X = X[np.newaxis]   # treat as 1 trial: (1, nvars, N)
        if X.ndim != 3:
            raise ValueError("'fourier' expects X of shape (nvars, N) or (trials, nvars, N).")
        trials, nvars, N = X.shape
        f = compute_freq(N, fs)
        S = np.zeros((nvars, nvars, len(f)), dtype=complex)
        for i in range(nvars):
            for j in range(nvars):
                for t in range(trials):
                    S[i, j] += csd_fourier(X[t, i], X[t, j], f, fs)
                S[i, j] /= trials
        return S, f

    if spectral_method == 'morlet':
        from .spectral_analysis import morlet_csd
        if X.ndim == 2:
            X = X[np.newaxis]   # treat as 1 trial: (1, nvars, N)
        if X.ndim != 3:
            raise ValueError("'morlet' expects X of shape (nvars, N) or (trials, nvars, N).")
        freqs = params.get('freqs')
        if freqs is None:
            raise ValueError("spectral_params must include 'freqs' for 'morlet'.")
        n_cycles = params.get('n_cycles', 7.0)
        trials, nvars, _ = X.shape
        f = np.asarray(freqs, dtype=float)
        S = np.zeros((nvars, nvars, len(f)), dtype=complex)
        for i in range(nvars):
            for j in range(nvars):
                for t in range(trials):
                    csd_tf = morlet_csd(X[t, i], X[t, j], freqs=f, Fs=fs, n_cycles=n_cycles)
                    S[i, j] += np.mean(csd_tf, axis=0)
                S[i, j] /= trials
        return S, f

    if spectral_method == 'welch':
        from .spectral_analysis import welch_spectrum
        from scipy import signal as _scipy_signal
        if X.ndim == 2:
            X = X[np.newaxis]
        window   = params.get('window', 'hann')
        nperseg  = params.get('nperseg', None)
        nfft     = params.get('nfft', None)
        scaling  = params.get('scaling', 'density')
        n_jobs   = params.get('n_jobs', 1)
        S = welch_spectrum(data=X, fs=fs, window=window, nperseg=nperseg,
                           nfft=nfft, scaling=scaling, n_jobs=n_jobs)
        # scipy.signal.csd uses conj(X)*Y; conjugate to match csd_fourier's X*conj(Y)
        S = np.conj(S)
        f, _ = _scipy_signal.csd(X[0, 0], X[0, 0], fs,
                                  window=window, nperseg=nperseg, nfft=nfft,
                                  scaling=scaling)
        return S, f

    if spectral_method == 'multitaper':
        from .spectral_analysis import multitaper_spectrum
        if X.ndim == 2:
            X = X[np.newaxis]
        if X.ndim != 3:
            raise ValueError("'multitaper' expects X of shape (nvars, N) or (trials, nvars, N).")
        S, f = multitaper_spectrum(
            data=X, fs=fs,
            bandwidth=params.get('bandwidth', None),
            adaptive=params.get('adaptive', False),
            low_bias=params.get('low_bias', True),
            fmin=params.get('fmin', 0),
            fmax=params.get('fmax', np.inf),
            n_fft=params.get('n_fft', None),
            n_jobs=params.get('n_jobs', 1),
        )
        return S, f

    raise ValueError(
        f"Unknown spectral_method {spectral_method!r}. Choose from {_VALID_SPECTRAL_METHODS}."
    )


def _full_estimate(X, fs, model, spectral_method, backend, Niterations, tol,
                    verbose, spectral_params, ensure_stability, order, f):
    if model == 'nonparametric':
        S, f = _compute_csd(X, fs, spectral_method, spectral_params)
        factorize = _get_factorization_fn(backend)
        _, H, Z = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)
        return S, H, Z, f

    if model == 'parametric':
        from .parametric import YuleWalker, YuleWalker_multitrial, compute_transfer_function
        if order is None:
            raise ValueError("`order` (VAR model order) is required when model='parametric'.")

        if X.ndim == 2:
            N = X.shape[1]
            AR, eps = YuleWalker(X, order)
        elif X.ndim == 3:
            N = X.shape[2]
            AR, eps = YuleWalker_multitrial(X, order)
        else:
            raise ValueError(
                "model='parametric' expects X of shape (nvars, N) or (Ntrials, nvars, N); "
                f"got ndim={X.ndim}."
            )

        if f is None:
            from .spectral_analysis import compute_freq
            f = compute_freq(N, fs)
        H, S = compute_transfer_function(AR, eps, f, fs)
        Z = eps
        return S, H, Z, f

    raise ValueError(f"Unknown model {model!r}. Choose from {_VALID_MODELS}.")


def _reduced_estimate(X_full, S_full, j, fs, model, spectral_method, backend,
                       Niterations, tol, spectral_params, ensure_stability,
                       order, f):
    if model == 'nonparametric':
        S_aux = np.delete(np.delete(S_full, j, 0), j, 1)
        factorize = _get_factorization_fn(backend)
        _, H, Z = factorize(S_aux, f, fs, Niterations, tol, verbose=False,
                             ensure_stability=ensure_stability)
        return H, np.diag(Z)

    if model == 'parametric':
        from .parametric import YuleWalker, YuleWalker_multitrial, compute_transfer_function
        # channel axis is 0 for single-trial (nvars, N), 1 for multi-trial (Ntrials, nvars, N)
        channel_axis = 0 if X_full.ndim == 2 else 1
        X_aux = np.delete(X_full, j, axis=channel_axis)
        if X_full.ndim == 2:
            AR_j, eps_j = YuleWalker(X_aux, order)
        else:
            AR_j, eps_j = YuleWalker_multitrial(X_aux, order)
        H, _ = compute_transfer_function(AR_j, eps_j, f, fs)
        return H, np.diag(eps_j)

    raise ValueError(f"Unknown model {model!r}. Choose from {_VALID_MODELS}.")


def _directional_gc(H_ii, Z_ii, Z_jj, Z_ij, H_ij):
    """One directional term of Geweke's frequency-domain GC decomposition."""
    intrinsic = H_ii * Z_ii * np.conj(H_ii)
    causal = H_ij * (Z_jj - Z_ij ** 2 / Z_ii) * np.conj(H_ij)
    return np.log((intrinsic + causal) / intrinsic)


def _gc(S, H, Z, x_s, x_t):
    """Pairwise frequency-domain Granger Causality from a Wilson factorization.

    Pure post-factorization step: takes the already-estimated cross-spectral
    density and its factorization and computes GC spectra for one channel
    pair. Knows nothing about how S/H/Z were produced.

    Parameters
    ----------
    S    : ndarray, shape (n_channels, n_channels, n_freq) — cross-spectral density.
    H    : ndarray, shape (n_channels, n_channels, n_freq) — transfer function.
    Z    : ndarray, shape (n_channels, n_channels) — innovations (noise) covariance.
    x_s  : int — source channel index.
    x_t  : int — target channel index.

    Returns
    -------
    Ix2y, Iy2x, Ixy : ndarray (n_freq,) each — real-valued GC spectra.
    """
    if x_s == x_t:
        raise ValueError(f"x_s and x_t must differ, got x_s=x_t={x_s}")
    n = S.shape[0]
    if not (0 <= x_s < n and 0 <= x_t < n):
        raise ValueError(f"x_s={x_s}, x_t={x_t} out of bounds for {n} channels")

    Hxx, Hxy = H[x_s, x_s], H[x_s, x_t]
    Hyx, Hyy = H[x_t, x_s], H[x_t, x_t]
    Zxx, Zxy = Z[x_s, x_s], Z[x_s, x_t]
    Zyx, Zyy = Z[x_t, x_s], Z[x_t, x_t]

    # Wilson-rotated transfer functions removing instantaneous mixing
    Hxx_tilda = Hxx + (Zxy / Zxx) * Hxy
    Hyy_circf = Hyy + (Zyx / Zyy) * Hyx

    Ix2y = _directional_gc(Hyy_circf, Zyy, Zxx, Zyx, Hyx)
    Iy2x = _directional_gc(Hxx_tilda, Zxx, Zyy, Zxy, Hxy)

    intrinsic_x = (Hxx_tilda * Zxx * np.conj(Hxx_tilda)).real
    intrinsic_y = (Hyy_circf * Zyy * np.conj(Hyy_circf)).real
    det_S = np.linalg.det(S.transpose(2, 0, 1)).real
    Ixy = np.log(intrinsic_x * intrinsic_y / det_S)

    return Ix2y.real, Iy2x.real, Ixy.real


def _gc_pairs(S, H, Z, pairs, n_jobs=-1, backend='loky'):
    """Compute GC for a list of channel pairs in parallel via joblib.

    Parameters
    ----------
    S, H, Z : see `_gc`.
    pairs   : list of (x_s, x_t) index tuples.
    n_jobs  : int — passed to joblib.Parallel (-1 = all cores).
    backend : str — joblib backend ('loky', 'threading', ...).

    Returns
    -------
    results : list of (Ix2y, Iy2x, Ixy) tuples, aligned with `pairs`.
    """
    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_gc)(S, H, Z, i, j) for i, j in pairs
    )


def granger_causality(X, fs, pairs=None, model='nonparametric', order=None, f=None,
                       spectral_method='fourier', backend='numpy', Niterations=100,
                       tol=1e-12, verbose=False, spectral_params=None,
                       ensure_stability=True, n_jobs=-1, joblib_backend='loky'):
    """Frequency-domain Granger Causality for one or more channel pairs.

    Parameters
    ----------
    ... (existing params unchanged) ...
    model  : {'nonparametric', 'parametric'} — estimation family for S, H, Z.
             'nonparametric' uses `_compute_csd` + Wilson factorization
             (existing behaviour, default). 'parametric' fits a VAR model via
             Yule-Walker and derives H, S analytically — `spectral_method`,
             `backend`, `Niterations`, `tol`, `ensure_stability` are ignored
             in this case.
    order  : int or None — VAR model order, required when model='parametric'.
    f      : ndarray or None — frequency axis to evaluate on when
             model='parametric'. If None, derived from data length via
             `compute_freq`. Ignored when model='nonparametric' (f comes
             from the spectral estimator instead).

    Returns
    -------
    ds : xarray.Dataset — see `build_granger_dataset`.
    """
    S, H, Z, f = _full_estimate(X, fs, model, spectral_method, backend, Niterations,
                                 tol, verbose, spectral_params, ensure_stability, order, f)

    if pairs is None:
        pairs = list(combinations(range(S.shape[0]), 2))
    pairs = list(pairs)

    if len(pairs) == 1:
        i, j = pairs[0]
        results = [_gc(S, H, Z, i, j)]
    else:
        results = _gc_pairs(S, H, Z, pairs, n_jobs=n_jobs, backend=joblib_backend)

    Ix2y, Iy2x, Ixy = (np.stack(vals) for vals in zip(*results))

    return build_granger_dataset(Ix2y, Iy2x, Ixy, pairs, f)


def conditional_granger_causality(X, fs, targets=None, channel_names=None,
                                   model='nonparametric', order=None, f=None,
                                   spectral_method='fourier', backend='numpy',
                                   Niterations=100, tol=1e-12, verbose=True,
                                   n_jobs=1, spectral_params=None,
                                   ensure_stability=True):
    """Conditional Granger Causality (time-domain summary).

    Parameters
    ----------
    ... (existing params unchanged) ...
    model  : {'nonparametric', 'parametric'} — see `granger_causality`. For
             'parametric', each reduced model is refit from raw X with the
             target channel removed (slicing AR coefficients would not give
             the correct reduced VAR model), so `n_jobs` parallelism still
             applies but is comparatively more expensive per target.
    order  : int or None — VAR model order, required when model='parametric'.
    f      : ndarray or None — frequency axis for the parametric transfer
             function evaluation; not otherwise used by this function's
             output (F is frequency-independent) but affects numerical
             conditioning of `compute_transfer_function`. If None, derived
             from data length.

    Returns
    -------
    ds : xarray.Dataset — see `build_conditional_gc_dataset`.
    """
    S, _, Znew, f = _full_estimate(X, fs, model, spectral_method, backend, Niterations,
                                    tol, verbose, spectral_params, ensure_stability, order, f)

    nvars = S.shape[0] 
    targets = range(nvars) if targets is None else list(targets)
    channel_names = list(channel_names) if channel_names is not None else list(range(nvars))
    if len(channel_names) != nvars:
        raise ValueError(f"channel_names has length {len(channel_names)}, expected {nvars}")

    LSIG = np.log(np.diag(Znew))

    def _reduced(j):
        _, Z_diag_j = _reduced_estimate(X, S, j, fs, model, spectral_method, backend,
                                         Niterations, tol, spectral_params,
                                         ensure_stability, order, f)
        return j, np.log(Z_diag_j)

    results: list = Parallel(n_jobs=n_jobs, prefer='threads')(  # type: ignore[assignment]
        delayed(_reduced)(j) for j in targets
    )

    F = np.zeros([nvars, nvars])
    for j, LSIGj in results:
        j0 = np.concatenate((np.arange(0, j), np.arange(j + 1, nvars)))
        for ii, i in enumerate(j0):
            F[i, j] = LSIGj[ii] - LSIG[i]
    return build_conditional_gc_dataset(F, channel_names)


def conditional_spec_granger_causality(X, fs, targets=None, channel_names=None,
                                        model='nonparametric', order=None, f=None,
                                        spectral_method='fourier', backend='numpy',
                                        Niterations=100, tol=1e-12, verbose=True,
                                        n_jobs=1, spectral_params=None,
                                        ensure_stability=True):
    """Conditional spectral Granger Causality.

    Parameters
    ----------
    ... (existing params unchanged) ...
    model  : {'nonparametric', 'parametric'} — see `granger_causality`.
    order  : int or None — VAR model order, required when model='parametric'.
    f      : ndarray or None — frequency axis, required to be consistent
             between full and reduced models. If None, derived from data
             length for model='parametric'.

    Returns
    -------
    ds : xarray.Dataset — see `build_conditional_spec_gc_dataset`.
    """
    S, Hnew, Znew, f = _full_estimate(X, fs, model, spectral_method, backend, Niterations,
                                       tol, verbose, spectral_params, ensure_stability, order, f)

    nvars = S.shape[0]
    targets = range(nvars) if targets is None else list(targets)
    channel_names = list(channel_names) if channel_names is not None else list(range(nvars))
    if len(channel_names) != nvars:
        raise ValueError(f"channel_names has length {len(channel_names)}, expected {nvars}")

    def _reduced(j):
        Hij, SIGj = _reduced_estimate(X, S, j, fs, model, spectral_method, backend,
                                       Niterations, tol, spectral_params,
                                       ensure_stability, order, f)
        return j, Hij, SIGj

    results: list = Parallel(n_jobs=n_jobs, prefer='threads')(  # type: ignore[assignment]
        delayed(_reduced)(j) for j in targets
    )

    GC = np.zeros([nvars, nvars, len(f)])
    for j, Hij, SIGj in results:
        logger.debug('j = %d', j)
        j0 = np.concatenate((np.arange(0, j), np.arange(j + 1, nvars)))

        G = np.zeros([nvars, nvars, len(f)], dtype=complex)
        for i in range(len(f)):
            aux = np.insert(Hij[:, :, i], j, np.zeros(nvars - 1), axis=1)
            aux = np.insert(aux, j, np.zeros(nvars), axis=0)
            G[:, :, i] = aux
        G[j, j, :] = 1

        G_T    = G.transpose(2, 0, 1)
        Hnew_T = Hnew.transpose(2, 0, 1)
        Q_T    = np.linalg.inv(G_T) @ Hnew_T

        for ii, i in enumerate(j0):
            div = Q_T[:, i, i] * Znew[i, i] * np.conj(Q_T[:, i, i])
            GC[j, i, :] = np.log(SIGj[ii] / np.abs(div))

    return build_conditional_spec_gc_dataset(GC, f, channel_names)