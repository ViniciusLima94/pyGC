import numpy as np
import pytest
from pygc.spectral_analysis.fourier import (
    compute_freq, csd_fourier, morlet_transform, morlet_csd,
)
from pygc.spectral_analysis.time_frequency import welch_spectrum, wavelet_transform


FS = 200.0
N = 512
FREQS = np.arange(10, 80, 5, dtype=float)


class TestComputeFreq:
    def test_length(self):
        f = compute_freq(N, FS)
        assert len(f) == N // 2 + 1

    def test_range(self):
        f = compute_freq(N, FS)
        assert f[0] > 0
        assert f[-1] < FS / 2

    def test_monotonic(self):
        f = compute_freq(N, FS)
        assert np.all(np.diff(f) > 0)


class TestCsdFourier:
    def test_auto_spectrum_real_positive(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N)
        f = compute_freq(N, FS)
        Pxx = csd_fourier(x, f=f, Fs=FS)
        assert np.all(Pxx.real >= 0)
        np.testing.assert_allclose(Pxx.imag, 0.0, atol=1e-12)

    def test_cross_spectrum_shape(self):
        rng = np.random.default_rng(1)
        x, y = rng.standard_normal(N), rng.standard_normal(N)
        f = compute_freq(N, FS)
        Pxy = csd_fourier(x, Y=y, f=f, Fs=FS)
        assert len(Pxy) == len(f)

    def test_cross_spectrum_conjugate_symmetry(self):
        """csd(x,y) == conj(csd(y,x))"""
        rng = np.random.default_rng(2)
        x, y = rng.standard_normal(N), rng.standard_normal(N)
        f = compute_freq(N, FS)
        Pxy = csd_fourier(x, Y=y, f=f, Fs=FS)
        Pyx = csd_fourier(y, Y=x, f=f, Fs=FS)
        np.testing.assert_allclose(Pxy, np.conj(Pyx), atol=1e-12)


class TestMorletTransform:
    def test_output_shape(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(N)
        W = morlet_transform(x, FREQS, Fs=FS, n_cycles=7.0)
        assert W.shape == (N, len(FREQS))

    def test_output_complex(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(N)
        W = morlet_transform(x, FREQS, Fs=FS)
        assert np.iscomplexobj(W)

    def test_power_peak_at_input_frequency(self):
        """Pure sine at 30 Hz → wavelet power should peak near 30 Hz."""
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 30 * t)
        W = morlet_transform(x, FREQS, Fs=FS, n_cycles=7.0)
        power = np.abs(W).mean(axis=0)
        peak_idx = np.argmax(power)
        np.testing.assert_allclose(FREQS[peak_idx], 30.0, atol=5.0)


class TestMorletCsd:
    def test_auto_spectrum_real_positive(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(N)
        Pxx = morlet_csd(x, freqs=FREQS, Fs=FS)
        assert Pxx.shape == (N, len(FREQS))
        assert np.all(Pxx.real >= 0)

    def test_cross_spectrum_shape(self):
        rng = np.random.default_rng(6)
        x, y = rng.standard_normal(N), rng.standard_normal(N)
        Pxy = morlet_csd(x, Y=y, freqs=FREQS, Fs=FS)
        assert Pxy.shape == (N, len(FREQS))


class TestWelchSpectrum:
    def test_output_shape(self):
        rng = np.random.default_rng(7)
        # shape: (trials, channels, timepoints)
        data = rng.standard_normal((3, 2, N))
        S = welch_spectrum(data=data, fs=FS)
        assert S.shape[0] == 2  # channels
        assert S.shape[1] == 2
        assert S.dtype == complex

    def test_diagonal_real_positive(self):
        rng = np.random.default_rng(8)
        data = rng.standard_normal((3, 2, N))
        S = welch_spectrum(data=data, fs=FS)
        for i in range(2):
            assert np.all(S[i, i, :].real > 0)

    def test_invalid_scaling_raises(self):
        data = np.zeros((1, 2, N))
        with pytest.raises(ValueError):
            welch_spectrum(data=data, fs=FS, scaling="bad")


class TestWaveletTransform:
    def test_output_shape_morlet(self):
        rng = np.random.default_rng(9)
        # MNE expects (epochs, channels, times)
        data = rng.standard_normal((2, 3, N))
        W = wavelet_transform(data=data, fs=FS, freqs=FREQS, n_cycles=7.0, method="morlet")
        assert W.shape == (2, 3, len(FREQS), N)

    def test_invalid_method_raises(self):
        data = np.zeros((1, 2, N))
        with pytest.raises(ValueError):
            wavelet_transform(data=data, fs=FS, freqs=FREQS, method="unknown")
