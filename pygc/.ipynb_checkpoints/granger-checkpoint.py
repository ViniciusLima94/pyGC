########################################################################################
# Module with functions to compute GC
########################################################################################
import logging
from typing import Callable
import numpy as np
from joblib import Parallel, delayed

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


def granger_causality(X, fs, spectral_method='fourier', backend='numpy',
                      Niterations=100, tol=1e-12, verbose=False,
                      spectral_params=None, ensure_stability=True):
    """Bivariate frequency-domain Granger Causality.

    Parameters
    ----------
    X               : ndarray — raw signal data.
                      'fourier'/'morlet' : (2, N)
                      'welch'            : (trials, 2, N) or (2, N) for 1 trial
    fs              : float — sampling rate (Hz).
    spectral_method : {'fourier', 'morlet', 'welch', 'multitaper'} — spectral estimator.
    backend         : {'numpy', 'jax'} — Wilson factorization backend.
    Niterations     : int — maximum factorization iterations.
    tol             : float — convergence tolerance.
    verbose         : bool — print factorization progress.
    spectral_params : dict or None — extra kwargs for the spectral estimator.

    Returns
    -------
    Ix2y, Iy2x, Ixy : ndarray (n_freq,) each — GC spectra (real).
    """
    S, f = _compute_csd(X, fs, spectral_method, spectral_params)
    factorize = _get_factorization_fn(backend)
    _, H, Z = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)

    Hxx = H[0, 0, :]
    Hxy = H[0, 1, :]
    Hyx = H[1, 0, :]
    Hyy = H[1, 1, :]

    Hxx_tilda  = Hxx + (Z[0, 1] / Z[0, 0]) * Hxy
    Hyy_circf  = Hyy + (Z[1, 0] / Z[1, 1]) * Hyx

    Ix2y = np.log(
        (Hyy_circf * Z[1, 1] * np.conj(Hyy_circf)
         + Hyx * (Z[0, 0] - Z[1, 0] ** 2 / Z[1, 1]) * np.conj(Hyx))
        / (Hyy_circf * Z[1, 1] * np.conj(Hyy_circf))
    )
    Iy2x = np.log(
        (Hxx_tilda * Z[0, 0] * np.conj(Hxx_tilda)
         + Hxy * (Z[1, 1] - Z[0, 1] ** 2 / Z[0, 0]) * np.conj(Hxy))
        / (Hxx_tilda * Z[0, 0] * np.conj(Hxx_tilda))
    )

    det_S = np.linalg.det(S.transpose(2, 0, 1))
    Ixy = np.log(
        (Hxx_tilda * Z[0, 0] * np.conj(Hxx_tilda)).real
        * (Hyy_circf * Z[1, 1] * np.conj(Hyy_circf)).real
        / det_S.real
    ).real

    return Ix2y.real, Iy2x.real, Ixy, f


def conditional_granger_causality(X, fs, spectral_method='fourier', backend='numpy',
                                   Niterations=100, tol=1e-12, verbose=True,
                                   n_jobs=1, spectral_params=None):
    """Conditional Granger Causality (time-domain summary).

    Wilson factorizations for each reduced model are run in parallel when
    n_jobs > 1.

    Parameters
    ----------
    X               : ndarray — raw signal data.
                      'fourier'/'morlet' : (nvars, N)
                      'welch'            : (trials, nvars, N) or (nvars, N) for 1 trial
    fs              : float — sampling rate (Hz).
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
    _, _, Znew = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)
    LSIG = np.log(np.diag(Znew))

    def _reduced(j):
        S_aux = np.delete(np.delete(S, j, 0), j, 1)
        _, _, Zij = factorize(S_aux, f, fs, Niterations, tol, verbose=False)
        return j, np.log(np.diag(Zij))

    results: list = Parallel(n_jobs=n_jobs, prefer='threads')(  # type: ignore[assignment]
        delayed(_reduced)(j) for j in range(nvars)
    )

    F = np.zeros([nvars, nvars])
    for j, LSIGj in results:
        j0 = np.concatenate((np.arange(0, j), np.arange(j + 1, nvars)))
        for ii, i in enumerate(j0):
            F[i, j] = LSIGj[ii] - LSIG[i]
    return F


def conditional_spec_granger_causality(X, fs, spectral_method='fourier', backend='numpy',
                                        Niterations=100, tol=1e-12, verbose=True,
                                        n_jobs=1, spectral_params=None):
    """Conditional spectral Granger Causality.

    Reduced-model factorizations are parallelised when n_jobs > 1.

    Parameters
    ----------
    X               : ndarray — raw signal data.
                      'fourier'/'morlet' : (nvars, N)
                      'welch'            : (trials, nvars, N) or (nvars, N) for 1 trial
    fs              : float — sampling rate (Hz).
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
    _, Hnew, Znew = factorize(S, f, fs, Niterations, tol, verbose, ensure_stability)

    def _reduced(j):
        S_aux = np.delete(np.delete(S, j, 0), j, 1)
        _, Hij, Zij = factorize(S_aux, f, fs, Niterations, tol, verbose=False)
        return j, Hij, np.diag(Zij)

    results: list = Parallel(n_jobs=n_jobs, prefer='threads')(  # type: ignore[assignment]
        delayed(_reduced)(j) for j in range(nvars)
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

    return GC, f
