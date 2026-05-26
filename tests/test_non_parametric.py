import numpy as np
import pytest
from pygc.non_parametric import wilson_factorization
from pygc.parametric import YuleWalker, compute_transfer_function


FS = 1.0
NVARS = 2
N_FREQ = 33  # odd so N = 32


def _identity_spectrum(nvars, n_freq):
    S = np.zeros([nvars, nvars, n_freq], dtype=complex)
    for k in range(n_freq):
        S[:, :, k] = np.eye(nvars)
    return S


class TestWilsonFactorization:
    def test_output_shapes(self):
        freq = np.linspace(0, 0.5, N_FREQ)
        S = _identity_spectrum(NVARS, N_FREQ)
        Snew, Hnew, Znew = wilson_factorization(S, freq, FS, Niterations=50, verbose=False)
        assert Snew.shape == (NVARS, NVARS, N_FREQ)
        assert Hnew.shape == (NVARS, NVARS, N_FREQ)
        assert Znew.shape == (NVARS, NVARS)

    def test_noise_covariance_symmetric(self):
        freq = np.linspace(0, 0.5, N_FREQ)
        S = _identity_spectrum(NVARS, N_FREQ)
        _, _, Znew = wilson_factorization(S, freq, FS, Niterations=50, verbose=False)
        np.testing.assert_allclose(Znew, Znew.T, atol=1e-6)

    def test_reconstructed_spectrum_close_to_input(self):
        """Snew ≈ S after factorization (up to numerical tolerance)."""
        rng = np.random.default_rng(0)
        freq = np.linspace(0, 0.5, N_FREQ)
        # Build a valid PSD from a random AR model
        data = rng.standard_normal((NVARS, 4000))
        AR, sigma = YuleWalker(data, 2)
        _, S = compute_transfer_function(AR, sigma, freq * FS * 2, FS)
        Snew, _, _ = wilson_factorization(S, freq, FS, Niterations=200, verbose=False)
        np.testing.assert_allclose(np.abs(Snew), np.abs(S), rtol=0.05)

    def test_identity_spectrum_gives_identity_noise(self):
        """Wilson factorization of the identity spectrum should give Znew ≈ I."""
        freq = np.linspace(0, 0.5, N_FREQ)
        S = _identity_spectrum(NVARS, N_FREQ)
        _, _, Znew = wilson_factorization(S, freq, FS, Niterations=200, verbose=False)
        np.testing.assert_allclose(Znew, np.eye(NVARS), atol=0.05)
