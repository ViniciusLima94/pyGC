"""
Optional JAX backend for pygc.

Provides a JIT-compiled Wilson spectral factorization that uses XLA kernels
(CPU or GPU) for batched matrix operations.  All loops — including the
convergence-checked iteration — are compiled with ``jax.lax.while_loop``
so no Python overhead remains after the first (tracing) call.

GPU support
-----------
* **Apple Silicon (Metal)**: ``pip install jax-metal``
  Metal does not support float64, so the computation runs in complex64
  (float32 precision).  Accuracy is lower but throughput is higher.
* **NVIDIA (CUDA)**: install the appropriate ``jax[cuda]`` wheel.
  Full float64 / complex128 precision is available.

Usage
-----
from pygc._jax_backend import wilson_factorization_jax, JAX_AVAILABLE, JAX_FLOAT64
"""
from __future__ import annotations

import functools
import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)   # no-op on Metal, active on CPU/CUDA
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE: bool = True
    _JAX_VERSION: str = jax.__version__

    # Metal GPU supports only float32; CPU and CUDA support float64.
    # Detect by probing whether a float64 array keeps its dtype after a round-trip.
    _probe = jnp.ones(1, dtype=jnp.float64)
    JAX_FLOAT64: bool = bool(_probe.dtype == jnp.float64)
    del _probe

    # Choose working dtypes based on device capability.
    _NP_CDTYPE  = np.complex128  if JAX_FLOAT64 else np.complex64
    _JNP_CDTYPE = jnp.complex128 if JAX_FLOAT64 else jnp.complex64

except ImportError:           # pragma: no cover
    JAX_AVAILABLE = False
    JAX_FLOAT64   = False
    _JAX_VERSION  = ""
    _NP_CDTYPE    = None
    _JNP_CDTYPE   = None


# ---------------------------------------------------------------------------
# Internal helpers (only defined when JAX is present)
# ---------------------------------------------------------------------------

if JAX_AVAILABLE:

    def _plus_op(g: "jnp.ndarray", freq_len: int) -> "jnp.ndarray":
        """Plus-operator: batch IFFT → zero causal half → batch FFT."""
        gam  = jnp.fft.ifft(g, axis=2)
        gamp = (gam
                .at[:, :, 0].set(jnp.triu(0.5 * gam[:, :, 0]))
                .at[:, :, freq_len:].set(0.0 + 0.0j))
        return jnp.fft.fft(gamp, axis=2)

    @functools.partial(jax.jit, static_argnames=["freq_len", "Niterations"])
    def _wilson_loop(
        psi: "jnp.ndarray",
        Sarr: "jnp.ndarray",
        I: "jnp.ndarray",
        freq_len: int,
        Niterations: int,
        tol: float,
    ) -> "jnp.ndarray":
        """Full Wilson iteration loop compiled to XLA via lax.while_loop."""

        def body(carry: tuple) -> tuple:
            psi, _, itr = carry
            psi_T   = psi.transpose(2, 0, 1)                       # (2N, m, m)
            psi_inv = jnp.linalg.inv(psi_T)
            g_T     = psi_inv @ Sarr.transpose(2, 0, 1) @ jnp.conj(psi_inv).swapaxes(-1, -2) + I
            g       = g_T.transpose(1, 2, 0)

            gp      = _plus_op(g, freq_len)

            new_psi_T = psi_T @ gp.transpose(2, 0, 1)
            diff    = new_psi_T - psi_T
            psierr  = jnp.abs(diff).sum(axis=-2).max(axis=-1).mean()

            return new_psi_T.transpose(1, 2, 0), psierr, itr + 1

        def cond(carry: tuple) -> "jnp.ndarray":
            _, psierr, itr = carry
            return (psierr >= tol) & (itr < Niterations)

        init = (psi, jnp.array(jnp.inf), jnp.int32(0))
        psi_final, _, _ = lax.while_loop(cond, body, init)
        return psi_final

    def wilson_factorization_jax(
        S: np.ndarray,
        freq: np.ndarray,
        fs: float,
        Niterations: int = 100,
        tol: float = 1e-12,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wilson spectral factorization using a JAX JIT-compiled loop.

        The first call incurs XLA compilation time (~1–5 s); subsequent calls
        with the same array shapes reuse the compiled kernel.

        Precision is chosen automatically:
        - **complex128** on CPU and NVIDIA GPU (float64 available).
        - **complex64**  on Apple Metal GPU (float64 not supported).

        Parameters
        ----------
        S           : ndarray (m, m, N+1) — cross-spectral matrix.
        freq        : ndarray (N+1,)      — frequency axis.
        fs          : float               — sampling rate (unused; kept for API compat).
        Niterations : int.
        tol         : float.
        verbose     : bool                — print JAX device / precision info.

        Returns
        -------
        Snew, Hnew : ndarray (m, m, N+1)
        Znew       : ndarray (m, m)
        """
        m        = S.shape[0]
        N        = freq.shape[0] - 1
        freq_len = len(freq)

        if verbose:
            prec = "complex128 (float64)" if JAX_FLOAT64 else "complex64 (float32, Metal GPU)"
            print(f'[JAX] devices: {jax.devices()}  precision: {prec}')

        # Build Hermitian-extended Sarr (numpy; done once outside JIT)
        Sarr_np = np.zeros([m, m, 2 * N], dtype=_NP_CDTYPE)
        Sarr_np[:, :, :N + 1] = S
        for k in range(1, N):
            Sarr_np[:, :, 2 * N - k] = S[:, :, k].T

        # Initialise psi from Cholesky of gam0
        gam0   = np.fft.ifft(Sarr_np, axis=2).real[:, :, 0]
        h      = np.linalg.cholesky(gam0).T
        psi_np = np.tile(h[:, :, np.newaxis], (1, 1, 2 * N)).astype(_NP_CDTYPE)

        # Move to JAX arrays and run compiled loop
        psi   = jnp.array(psi_np)
        Sarr  = jnp.array(Sarr_np)
        I_jax = jnp.eye(m, dtype=_JNP_CDTYPE)[jnp.newaxis]  # (1, m, m)

        psi = _wilson_loop(psi, Sarr, I_jax, freq_len, Niterations, tol)
        psi = np.array(psi)  # back to NumPy for output

        # Compute outputs (NumPy; trivially fast)
        psi_T  = psi.transpose(2, 0, 1)
        Snew   = (psi_T[:N + 1] @ np.conj(psi_T[:N + 1]).swapaxes(-1, -2)).transpose(1, 2, 0)
        gamtmp = np.fft.ifft(psi, axis=2).real
        A0     = gamtmp[:, :, 0]
        A0inv  = np.linalg.inv(A0)
        Znew   = (A0 @ A0.T).real
        Hnew   = (psi_T[:N + 1] @ A0inv[np.newaxis]).transpose(1, 2, 0)

        return Snew, Hnew, Znew

else:  # JAX not available — stubs that raise a helpful error

    def wilson_factorization_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "JAX is not installed. Install it with: pip install jax"
        )
