import numpy as np
import pytest
from pygc.granger import granger_causality
from pygc.ar_model import ar_model_dhamala

FS = 200


def _make_data(nvars=2, N=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nvars, N))


class TestGrangerCausality:
    def test_output_shapes_welch(self):
        X = _make_data(N=1024)
        Ix2y, Iy2x, Ixy = granger_causality(X, FS, spectral_method='welch', verbose=False)
        assert Ix2y.ndim == 1
        assert Iy2x.shape == Ix2y.shape
        assert Ixy.shape == Ix2y.shape

    def test_output_shapes_fourier_multitrials(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2, 512))  # (trials, nvars, N)
        Ix2y, Iy2x, Ixy = granger_causality(X, FS, spectral_method='fourier', verbose=False)
        assert Ix2y.ndim == 1
        assert Iy2x.shape == Ix2y.shape
        assert Ixy.shape == Ix2y.shape

    def test_uncoupled_system_zero_gc(self):
        """Independent signals via Welch: GC measures should be ~0."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2, 5000))
        Ix2y, Iy2x, _ = granger_causality(X, FS, spectral_method='welch',
                                           Niterations=200, verbose=False)
        np.testing.assert_allclose(Ix2y, 0.0, atol=0.15)
        np.testing.assert_allclose(Iy2x, 0.0, atol=0.15)

    def test_known_coupling_direction(self):
        """Dhamala AR model: Y drives X (C > 0). GC(Y->X) > GC(X->Y)."""
        np.random.seed(0)
        cov = np.eye(2) * 0.5
        Z = ar_model_dhamala(N=3000, Trials=10, Fs=FS, C=0.25,
                              t_start=0, t_stop=None, cov=cov)
        # Z: (2, Trials, N) → welch expects (trials, nvars, N)
        X = Z.transpose(1, 0, 2)
        Ix2y, Iy2x, _ = granger_causality(X, FS, spectral_method='welch', verbose=False)
        assert np.mean(Iy2x) > np.mean(Ix2y), (
            f"Expected GC(Y->X)={np.mean(Iy2x):.4f} > GC(X->Y)={np.mean(Ix2y):.4f}"
        )

    def test_invalid_backend_raises(self):
        X = _make_data()
        with pytest.raises(ValueError, match="Unknown backend"):
            granger_causality(X, FS, spectral_method='welch', backend='invalid')

    def test_invalid_spectral_method_raises(self):
        X = _make_data()
        with pytest.raises(ValueError, match="Unknown spectral_method"):
            granger_causality(X, FS, spectral_method='invalid')

    def test_morlet_missing_freqs_raises(self):
        X = _make_data()
        with pytest.raises(ValueError, match="freqs"):
            granger_causality(X, FS, spectral_method='morlet')

    def test_morlet_multitrials(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((5, 2, 512))  # (trials, nvars, N)
        freqs = np.linspace(10, 80, 20)
        Ix2y, Iy2x, Ixy = granger_causality(
            X, FS, spectral_method='morlet',
            spectral_params={'freqs': freqs, 'n_cycles': 5.0},
            verbose=False,
        )
        assert Ix2y.shape == (len(freqs),)
