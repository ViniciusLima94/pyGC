import numpy as np
import pytest
from pygc import JAX_AVAILABLE
from pygc.non_parametric import wilson_factorization


def _simple_spectrum(nvars=2, n_freq=33):
    S = np.zeros([nvars, nvars, n_freq], dtype=complex)
    for k in range(n_freq):
        S[:, :, k] = np.eye(nvars)
    return S


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJaxBackend:
    def test_jax_output_shapes(self):
        from pygc import wilson_factorization_jax
        freq = np.linspace(0, 0.5, 33)
        S = _simple_spectrum(2, 33)
        Snew, Hnew, Znew = wilson_factorization_jax(S, freq, 1.0, Niterations=50)
        assert Snew.shape == (2, 2, 33)
        assert Hnew.shape == (2, 2, 33)
        assert Znew.shape == (2, 2)

    def test_jax_matches_numpy(self):
        """JAX and NumPy Wilson factorization agree to numerical precision."""
        from pygc import wilson_factorization_jax
        from pygc.parametric import YuleWalker, compute_transfer_function

        rng = np.random.default_rng(0)
        data = rng.standard_normal((2, 3000))
        AR, sigma = YuleWalker(data, 2)
        freq = np.linspace(0, 0.5, 33)
        _, S = compute_transfer_function(AR, sigma, freq * 200 * 2, 200)

        Snew_np, Hnew_np, Znew_np = wilson_factorization(S, freq, 1.0,
                                                          Niterations=200, verbose=False)
        Snew_jax, Hnew_jax, Znew_jax = wilson_factorization_jax(S, freq, 1.0,
                                                                  Niterations=200)

        np.testing.assert_allclose(np.abs(Snew_jax), np.abs(Snew_np), rtol=1e-5)
        np.testing.assert_allclose(Znew_jax, Znew_np, rtol=1e-5)

    def test_jax_noise_covariance_symmetric(self):
        from pygc import wilson_factorization_jax
        freq = np.linspace(0, 0.5, 33)
        S = _simple_spectrum(2, 33)
        _, _, Znew = wilson_factorization_jax(S, freq, 1.0, Niterations=100)
        np.testing.assert_allclose(Znew, Znew.T, atol=1e-5)
