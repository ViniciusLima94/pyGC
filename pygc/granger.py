########################################################################################
# Module with functions to compute GC
########################################################################################
import logging
import numpy as np
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def granger_causality(S, H, Z):
    """Bivariate frequency-domain Granger Causality.

    Parameters
    ----------
    S : ndarray (2, 2, N) — cross-spectral matrix.
    H : ndarray (2, 2, N) — transfer function.
    Z : ndarray (2, 2)    — noise covariance.

    Returns
    -------
    Ix2y, Iy2x, Ixy : ndarray (N,) each — GC spectra (real).
    """
    N   = H.shape[2]

    Hxx = H[0, 0, :]
    Hxy = H[0, 1, :]
    Hyx = H[1, 0, :]
    Hyy = H[1, 1, :]

    Hxx_tilda  = Hxx + (Z[0, 1] / Z[0, 0]) * Hxy
    Hyy_circf  = Hyy + (Z[1, 0] / Z[1, 1]) * Hyx

    Syy = (Hyy_circf * Z[1, 1] * np.conj(Hyy_circf)
           + Hyx * (Z[0, 0] - Z[1, 0] ** 2 / Z[1, 1]) * np.conj(Hyx))
    Sxx = (Hxx_tilda * Z[0, 0] * np.conj(Hxx_tilda)
           + Hxy * (Z[1, 1] - Z[0, 1] ** 2 / Z[0, 0]) * np.conj(Hxy))

    Ix2y = np.log(Syy / (Hyy_circf * Z[1, 1] * np.conj(Hyy_circf)))
    Iy2x = np.log(Sxx / (Hxx_tilda * Z[0, 0] * np.conj(Hxx_tilda)))

    # Vectorised det over all frequencies (replaces Python loop)
    det_S = np.linalg.det(S.transpose(2, 0, 1))       # (N,) batch det
    Ixy = np.log(
        (Hxx_tilda * Z[0, 0] * np.conj(Hxx_tilda)).real
        * (Hyy_circf * Z[1, 1] * np.conj(Hyy_circf)).real
        / det_S.real
    ).real

    return Ix2y.real, Iy2x.real, Ixy


def conditional_granger_causality(S, f, fs, Niterations=100, tol=1e-12,
                                   verbose=True, n_jobs=1):
    """Conditional Granger Causality (time-domain summary).

    Wilson factorizations for each reduced model are run in parallel when
    n_jobs > 1.

    Parameters
    ----------
    S           : ndarray (nvars, nvars, N) — cross-spectral matrix.
    f           : ndarray (N,) — frequency axis.
    fs          : float — sampling rate.
    Niterations : int.
    tol         : float.
    verbose     : bool — passed to full-model Wilson (reduced models are silent).
    n_jobs      : int — joblib parallelism (-1 = all cores).

    Returns
    -------
    F : ndarray (nvars, nvars) — conditional GC matrix.
    """
    from .non_parametric import wilson_factorization

    nvars = S.shape[0]
    _, _, Znew = wilson_factorization(S, f, fs, Niterations, tol, verbose)
    LSIG = np.log(np.diag(Znew))

    def _reduced(j):
        S_aux = np.delete(np.delete(S, j, 0), j, 1)
        _, _, Zij = wilson_factorization(S_aux, f, fs, Niterations, tol, verbose=False)
        return j, np.log(np.diag(Zij))

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_reduced)(j) for j in range(nvars)
    )

    F = np.zeros([nvars, nvars])
    for j, LSIGj in results:
        j0 = np.concatenate((np.arange(0, j), np.arange(j + 1, nvars)))
        for ii, i in enumerate(j0):
            F[i, j] = LSIGj[ii] - LSIG[i]
    return F


def conditional_spec_granger_causality(S, f, fs, Niterations=100, tol=1e-12,
                                        verbose=True, n_jobs=1):
    """Conditional spectral Granger Causality.

    Reduced-model Wilson factorizations are parallelised when n_jobs > 1.

    Parameters
    ----------
    (same as conditional_granger_causality)

    Returns
    -------
    GC : ndarray (nvars, nvars, N) — spectral GC matrix.
    """
    from .non_parametric import wilson_factorization

    nvars = S.shape[0]
    _, Hnew, Znew = wilson_factorization(S, f, fs, Niterations, tol, verbose)
    SIG = np.diag(Znew)

    def _reduced(j):
        S_aux = np.delete(np.delete(S, j, 0), j, 1)
        _, Hij, Zij = wilson_factorization(S_aux, f, fs, Niterations, tol, verbose=False)
        return j, Hij, np.diag(Zij)

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_reduced)(j) for j in range(nvars)
    )

    GC = np.zeros([nvars, nvars, len(f)])
    for j, Hij, SIGj in results:
        logger.debug('j = %d', j)
        j0 = np.concatenate((np.arange(0, j), np.arange(j + 1, nvars)))

        # Build padded transfer-function G (nvars × nvars × N)
        G = np.zeros([nvars, nvars, len(f)], dtype=complex)
        for i in range(len(f)):
            aux = np.insert(Hij[:, :, i], j, np.zeros(nvars - 1), axis=1)
            aux = np.insert(aux, j, np.zeros(nvars), axis=0)
            G[:, :, i] = aux
        G[j, j, :] = 1

        # Q = G^{-1} @ Hnew — batch inversion
        G_T   = G.transpose(2, 0, 1)                  # (N, nvars, nvars)
        Hnew_T = Hnew.transpose(2, 0, 1)              # (N, nvars, nvars)
        Q_T   = np.linalg.inv(G_T) @ Hnew_T          # (N, nvars, nvars)

        for ii, i in enumerate(j0):
            div = Q_T[:, i, i] * Znew[i, i] * np.conj(Q_T[:, i, i])
            GC[j, i, :] = np.log(SIGj[ii] / np.abs(div))

    return GC
