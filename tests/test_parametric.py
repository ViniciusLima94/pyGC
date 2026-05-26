import numpy as np
import pytest
from pygc.parametric import YuleWalker, compute_transfer_function


FS = 200
NVARS = 2
ORDER = 2
NSAMPLES = 2000


@pytest.fixture
def simple_ar_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((NVARS, NSAMPLES))


class TestYuleWalker:
    def test_ar_coefficient_shape(self, simple_ar_data):
        AR, eps = YuleWalker(simple_ar_data, ORDER)
        assert AR.shape == (ORDER, NVARS, NVARS)

    def test_noise_covariance_shape(self, simple_ar_data):
        AR, eps = YuleWalker(simple_ar_data, ORDER)
        assert eps.shape == (NVARS, NVARS)

    def test_noise_covariance_symmetric(self, simple_ar_data):
        _, eps = YuleWalker(simple_ar_data, ORDER)
        np.testing.assert_allclose(eps, eps.T, atol=1e-4)

    def test_known_ar1_coefficients(self):
        """For a scalar AR(1) x[t] = a*x[t-1] + noise, Yule-Walker should recover a."""
        rng = np.random.default_rng(7)
        a = 0.7
        N = 5000
        x = np.zeros(N)
        noise = rng.standard_normal(N) * 0.1
        for t in range(1, N):
            x[t] = a * x[t - 1] + noise[t]
        X = x[np.newaxis, :]
        AR, _ = YuleWalker(X, 1)
        np.testing.assert_allclose(AR[0, 0, 0], a, atol=0.05)


class TestComputeTransferFunction:
    def test_output_shapes(self, simple_ar_data):
        AR, sigma = YuleWalker(simple_ar_data, ORDER)
        f = np.linspace(1, 99, 50)
        H, S = compute_transfer_function(AR, sigma, f, FS)
        assert H.shape == (NVARS, NVARS, len(f))
        assert S.shape == (NVARS, NVARS, len(f))

    def test_spectral_matrix_hermitian(self, simple_ar_data):
        AR, sigma = YuleWalker(simple_ar_data, ORDER)
        f = np.linspace(1, 99, 30)
        _, S = compute_transfer_function(AR, sigma, f, FS)
        for k in range(len(f)):
            np.testing.assert_allclose(S[:, :, k], np.conj(S[:, :, k].T), atol=1e-5)

    def test_diagonal_spectra_real_positive(self, simple_ar_data):
        AR, sigma = YuleWalker(simple_ar_data, ORDER)
        f = np.linspace(1, 99, 30)
        _, S = compute_transfer_function(AR, sigma, f, FS)
        for k in range(len(f)):
            for i in range(NVARS):
                assert S[i, i, k].real > 0
                assert abs(S[i, i, k].imag) < 1e-6
