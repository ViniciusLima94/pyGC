import numpy as np
import mne.time_frequency


def compute_freq(N, Fs):
    """Frequency axis for a signal of length N sampled at Fs Hz."""
    T = N / Fs
    return np.linspace(1 / T, Fs / 2 - 1 / T, N // 2 + 1)


def csd_fourier(X, Y=None, f=None, Fs=1):
    """
    Cross-spectral density via FFT.

    Parameters
    ----------
    X : ndarray, shape (N,)
    Y : ndarray, shape (N,), optional. If None, returns auto-spectrum of X.
    f : ndarray — frequency axis returned by compute_freq; used to slice FFT output.
    Fs : float, sampling rate (Hz).

    Returns
    -------
    ndarray, shape (len(f),) — complex cross-spectrum.
    """
    N = X.shape[0]
    Xfft = np.fft.fft(X)[1 : len(f) + 1]
    if Y is not None:
        Yfft = np.fft.fft(Y)[1 : len(f) + 1]
        return Xfft * np.conj(Yfft) / N
    return Xfft * np.conj(Xfft) / N


def morlet_transform(X, freqs, Fs=1.0, n_cycles=7.0):
    """
    Morlet wavelet transform using MNE.

    Parameters
    ----------
    X : ndarray, shape (N,)
    freqs : ndarray — frequencies of interest (Hz).
    Fs : float, sampling rate (Hz).
    n_cycles : float or ndarray — number of cycles per frequency.

    Returns
    -------
    ndarray, shape (N, n_freqs) — complex analytic signal.
    """
    data = X[np.newaxis, np.newaxis, :]  # (1, 1, N)
    out = mne.time_frequency.tfr_array_morlet(
        data, Fs, freqs, n_cycles=n_cycles, zero_mean=False,
        output="complex", n_jobs=1,
    )
    # out shape: (1, 1, n_freqs, N) → (N, n_freqs)
    return out[0, 0, :, :].T


def morlet_csd(X, Y=None, freqs=None, Fs=1.0, n_cycles=7.0):
    """
    Cross-spectral density via Morlet wavelets using MNE.

    Parameters
    ----------
    X : ndarray, shape (N,)
    Y : ndarray, shape (N,), optional. If None, returns auto-spectrum of X.
    freqs : ndarray — frequencies of interest (Hz).
    Fs : float, sampling rate (Hz).
    n_cycles : float or ndarray.

    Returns
    -------
    ndarray, shape (N, n_freqs) — complex cross-spectrum (time × frequency).
    """
    N = X.shape[0]
    Wx = morlet_transform(X, freqs, Fs, n_cycles)
    if Y is not None:
        Wy = morlet_transform(Y, freqs, Fs, n_cycles)
        return Wx * np.conj(Wy) / N
    return Wx * np.conj(Wx) / N
