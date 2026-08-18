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


def _directional_gc(H_ii, Z_ii, Z_jj, Z_ij, H_ij):
    """One directional term of Geweke's frequency-domain GC decomposition.

    Ref: Geweke (1982); Dhamala, Rangarajan & Ding (2008).
    """
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


def granger_causality(X, fs, pairs=None, spectral_method='fourier',
                       backend='numpy', Niterations=100, tol=1e-12,
                       verbose=False, spectral_params=None,
                       ensure_stability=True, n_jobs=-1, joblib_backend='loky'):
    """Frequency-domain Granger Causality for one or more channel pairs.

    Parameters
    ----------
    X               : ndarray — raw signal data.
                      'fourier'/'morlet' : (n_channels, N)
                      'welch'            : (trials, n_channels, N) or (n_channels, N) for 1 trial
    fs              : float — sampling rate (Hz).
    pairs           : list/tuple of (x_s, x_t) index pairs, or None.
                      If None: all unordered channel pairs are used.
                      `_gc` raises if any pair has equal or out-of-bounds indices.
    spectral_method : {'fourier', 'morlet', 'welch', 'multitaper'} — spectral estimator.
    backend         : {'numpy', 'jax'} — Wilson factorization backend.
    Niterations     : int — maximum factorization iterations.
    tol             : float — convergence tolerance.
    verbose         : bool — print factorization progress.
    spectral_params : dict or None — extra kwargs for the spectral estimator.
    ensure_stability: bool — enforce a stable spectral factorization.
    n_jobs          : int — joblib workers (-1 = all cores). Skipped
                      entirely when len(pairs) == 1.
    joblib_backend  : str — joblib backend ('loky', 'threading', ...).

    Returns
    -------
    Ix2y  : ndarray, shape (n_pairs, n_freq) — Ix2y[k] = GC pairs[k][0] -> pairs[k][1].
    Iy2x  : ndarray, shape (n_pairs, n_freq) — Iy2x[k] = GC pairs[k][1] -> pairs[k][0].
    Ixy   : ndarray, shape (n_pairs, n_freq) — instantaneous causality per pair.
    pairs : list of (x_s, x_t) tuples, in the order computed.
    f     : ndarray (n_freq,) — frequency axis (Hz).
    """
    S, f = _compute_csd(X, fs, spectral_method, spectral_params)
    factorize = _get_factorization_fn(backend)
    _, H, Z = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)

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


def conditional_granger_causality(X, fs, targets=None, channel_names=None, spectral_method='fourier',
                                   backend='numpy', Niterations=100, tol=1e-12,
                                   verbose=True, n_jobs=1, spectral_params=None,
                                   ensure_stability=True):
    """Conditional Granger Causality (time-domain summary).

    A reduced-model Wilson factorization is run per target `j`, yielding
    F[i, j] for every i != j in that model at once — there is no per-pair
    shortcut, so `targets` (not `pairs`) is the right unit of selection here.
    Reduced-model factorizations are parallelised when n_jobs > 1.

    Parameters
    ----------
    X               : ndarray — raw signal data.
                      'fourier'/'morlet' : (nvars, N)
                      'welch'            : (trials, nvars, N) or (nvars, N) for 1 trial
    fs              : float — sampling rate (Hz).
    targets         : list/tuple of int, or None — target channel indices `j`
                      to compute reduced models for. If None, all channels
                      are used (equivalent to `range(nvars)`). Rows F[:, j]
                      are only populated for j in `targets`; all other rows
                      stay zero.
    spectral_method : {'fourier', 'morlet', 'welch', 'multitaper'} — spectral estimator.
    backend         : {'numpy', 'jax'} — Wilson factorization backend.
    Niterations     : int.
    tol             : float.
    verbose         : bool — passed to full-model factorization (reduced models are silent).
    n_jobs          : int — joblib parallelism (-1 = all cores).
    spectral_params : dict or None — extra kwargs for the spectral estimator.

    Returns
    -------
    F : ndarray (nvars, nvars) — conditional GC matrix.
    """
    S, f = _compute_csd(X, fs, spectral_method, spectral_params)
    factorize = _get_factorization_fn(backend)

    nvars = S.shape[0]
    targets = range(nvars) if targets is None else list(targets)

    _, _, Znew = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)
    LSIG = np.log(np.diag(Znew))

    def _reduced(j):
        S_aux = np.delete(np.delete(S, j, 0), j, 1)
        _, _, Zij = factorize(S_aux, f, fs, Niterations, tol, verbose=False)
        return j, np.log(np.diag(Zij))

    results: list = Parallel(n_jobs=n_jobs, prefer='threads')(  # type: ignore[assignment]
        delayed(_reduced)(j) for j in targets
    )

    F = np.zeros([nvars, nvars])
    for j, LSIGj in results:
        j0 = np.concatenate((np.arange(0, j), np.arange(j + 1, nvars)))
        for ii, i in enumerate(j0):
            F[i, j] = LSIGj[ii] - LSIG[i]
    return build_conditional_gc_dataset(F, channel_names)


def conditional_spec_granger_causality(X, fs, targets=None, channel_names=None, spectral_method='fourier',
                                        backend='numpy', Niterations=100, tol=1e-12,
                                        verbose=True, n_jobs=1, spectral_params=None,
                                        ensure_stability=True):
    """Conditional spectral Granger Causality.

    A reduced-model Wilson factorization is run per target `j`, yielding
    GC[j, i, :] for every i != j in that model at once — there is no
    per-pair shortcut, so `targets` (not `pairs`) is the right unit of
    selection here. Reduced-model factorizations are parallelised when
    n_jobs > 1.

    Parameters
    ----------
    X               : ndarray — raw signal data.
                      'fourier'/'morlet' : (nvars, N)
                      'welch'            : (trials, nvars, N) or (nvars, N) for 1 trial
    fs              : float — sampling rate (Hz).
    targets         : list/tuple of int, or None — target channel indices `j`
                      to compute reduced models for. If None, all channels
                      are used (equivalent to `range(nvars)`). Slices
                      GC[j, :, :] are only populated for j in `targets`;
                      all other slices stay zero.
    spectral_method : {'fourier', 'morlet', 'welch', 'multitaper'} — spectral estimator.
    backend         : {'numpy', 'jax'} — Wilson factorization backend.
    Niterations     : int.
    tol             : float.
    verbose         : bool — passed to full-model factorization (reduced models are silent).
    n_jobs          : int — joblib parallelism (-1 = all cores).
    spectral_params : dict or None — extra kwargs for the spectral estimator.

    Returns
    -------
    GC : ndarray (nvars, nvars, n_freq) — spectral GC matrix.
    """
    S, f = _compute_csd(X, fs, spectral_method, spectral_params)
    factorize = _get_factorization_fn(backend)

    nvars = S.shape[0]
    targets = range(nvars) if targets is None else list(targets)

    _, Hnew, Znew = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)

    def _reduced(j):
        S_aux = np.delete(np.delete(S, j, 0), j, 1)
        _, Hij, Zij = factorize(S_aux, f, fs, Niterations, tol, verbose=False)
        return j, Hij, np.diag(Zij)

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