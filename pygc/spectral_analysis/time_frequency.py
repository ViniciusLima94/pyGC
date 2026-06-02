import numpy as np
import mne
from   scipy            import signal
from   joblib           import Parallel, delayed
from   ..misc           import smooth_spectra


def welch_spectrum(data=None, fs=20, window='hann', nperseg=None, nfft=None,
                   scaling='density', n_jobs=1):
    """Cross-spectral matrix via Welch's method, averaged over trials.

    Parameters
    ----------
    data     : ndarray (trials, channels, timepoints).
    fs       : float — sampling rate (Hz).
    window   : str or ndarray — window function.
    nperseg  : int or None — segment length (controls frequency resolution).
    nfft     : int or None — FFT length (zero-padding; must be >= nperseg).
    scaling  : 'density' or 'spectrum'.
    n_jobs   : int — joblib workers for parallel channel-pair computation.

    Returns
    -------
    S : ndarray (channels, channels, n_freq) complex — cross-spectral matrix.
    """
    if scaling not in ['density', 'spectrum']:
        raise ValueError('scaling should be either "density" or "spectrum"')
    T, C, _ = data.shape

    # Probe output size from a single csd call
    f_probe, _ = signal.csd(data[0, 0, :], data[0, 0, :],
                             fs, window=window, nperseg=nperseg, nfft=nfft,
                             scaling=scaling)
    n_freq = len(f_probe)

    def _csd_pair(i, j):
        acc = np.zeros(n_freq, dtype=complex)
        for trial in range(T):
            _, Saux = signal.csd(data[trial, i, :], data[trial, j, :],
                                 fs, window=window, nperseg=nperseg, nfft=nfft,
                                 scaling=scaling)
            acc += Saux
        return i, j, acc / T

    pairs = [(i, j) for i in range(C) for j in range(C)]
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_csd_pair)(i, j) for i, j in pairs
    )

    S = np.zeros([C, C, n_freq], dtype=complex)
    for i, j, val in results:
        S[i, j] = val
    return S


def multitaper_freq(N, fs, fmin=0, fmax=np.inf, n_fft=None):
    """Frequency axis matching csd_array_multitaper output.

    Parameters
    ----------
    N     : int — number of time samples.
    fs    : float — sampling rate (Hz).
    fmin  : float — minimum frequency (Hz).
    fmax  : float — maximum frequency (Hz).
    n_fft : int or None — FFT length (defaults to N).

    Returns
    -------
    freq : ndarray — frequency axis (Hz).
    """
    if n_fft is None:
        n_fft = N
    freq = np.fft.rfftfreq(n_fft, 1.0 / fs)[1:]   # drop DC
    return freq[(freq >= fmin) & (freq <= fmax)]


def multitaper_spectrum(data=None, fs=20, bandwidth=None, adaptive=False,
                        low_bias=True, fmin=0, fmax=np.inf, n_fft=None, n_jobs=1):
    """Cross-spectral matrix via multitaper (DPSS) method, averaged over trials.

    Parameters
    ----------
    data      : ndarray (trials, channels, timepoints).
    fs        : float — sampling rate (Hz).
    bandwidth : float or None — frequency bandwidth of the multitaper window (Hz).
    adaptive  : bool — use adaptive weights to combine tapered spectra.
    low_bias  : bool — only use tapers with >90% spectral concentration.
    fmin      : float — minimum frequency of interest (Hz).
    fmax      : float — maximum frequency of interest (Hz).
    n_fft     : int or None — FFT length.
    n_jobs    : int or None — parallel workers.

    Returns
    -------
    S    : ndarray (channels, channels, n_freq) complex — cross-spectral matrix.
    freq : ndarray (n_freq,) — frequency axis (Hz).
    """
    csd = mne.time_frequency.csd_array_multitaper(
        data, sfreq=fs, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
        adaptive=adaptive, low_bias=low_bias, n_fft=n_fft,
        n_jobs=n_jobs, verbose=False,
    )
    freq = csd.frequencies
    S = np.stack([np.asarray(csd.get_data(frequency=f)) for f in freq], axis=-1)
    return S, freq


def wavelet_transform(data = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0,
                      time_bandwidth = None, delta = 1, method = 'morlet', n_jobs = 1):
    if method not in ['morlet', 'multitaper']:
        raise ValueError('Method should be either "morlet" or "multitaper"')
    if method == 'morlet' and time_bandwidth is not None:
        print('For method equals "morlet" time_bandwidth is not used')
    if method == 'morlet':
        out = mne.time_frequency.tfr_array_morlet(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
                                                  output='complex', decim = delta, n_jobs=n_jobs)
    if method == 'multitaper':
        out = mne.time_frequency.tfr_array_multitaper(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
                                                      time_bandwidth = time_bandwidth, output='complex',
                                                      decim = delta, n_jobs=n_jobs)
    return out


def wavelet_transform(data = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0,
                      time_bandwidth = None, delta = 1, method = 'morlet', n_jobs = 1):
    if method not in ['morlet', 'multitaper']:
        raise ValueError('Method should be either "morlet" or "multitaper"')
    if method == 'morlet' and time_bandwidth is not None:
        print('For method equals "morlet" time_bandwidth is not used')
    if method == 'morlet':
        out = mne.time_frequency.tfr_array_morlet(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
                                                  output='complex', decim = delta, n_jobs=n_jobs)
    if method == 'multitaper':
        out = mne.time_frequency.tfr_array_multitaper(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
                                                      time_bandwidth = time_bandwidth, output='complex',
                                                      decim = delta, n_jobs=n_jobs)
    return out

def gabor_transform(signal = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0):
    n      = len(signal)
    sigma2 = 1
    if n%2 == 0:
        omega  = np.concatenate( (np.arange(0, n/2), np.arange(-np.ceil(n/2), 0) ) ) * fs/n
    else:
        omega  = np.concatenate( (np.arange(0, n/2), np.arange(-np.ceil(n/2)+1, 0) ) ) * fs/n

    fftx   = np.fft.fft(signal)

    tolerance = 0.5

    mincenterfreq = 2*tolerance*np.sqrt(sigma2)*fs*n_cycles/n
    maxcenterfreq = fs*n_cycles/(n_cycles+tolerance/np.sqrt(sigma2))

    s_array  = n_cycles/freqs
    minscale = n_cycles/maxcenterfreq
    maxscale = n_cycles/mincenterfreq

    nscale = len(freqs)
    wt     = np.zeros([n,nscale]) * (1+1j)
    scaleindices = np.arange(0,len(s_array))[(s_array>=minscale)*(s_array<=maxscale)]
    psi_array = np.zeros([n, nscale]) * (1+1j)

    for kscale in scaleindices:
        s    = s_array[kscale]
        freq = (s*omega - n_cycles)
        Psi  = (4*np.pi*sigma2)**(1/4) * np.sqrt(s) * np.exp(-sigma2/2*freq**2)
        wt[:,kscale] = np.fft.ifft(fftx*Psi)
        psi_array[:,kscale]=np.fft.ifft(Psi)

    return wt

def gabor_spectrum(signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),
                    win_time = 1, win_freq = 1, n_cycles = 7.0):
    if type(signal2) != np.ndarray:
        wt1    = gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
        Sxx    = wt1*np.conj(wt1)
        return smooth_spectra.smooth_spectra(Sxx.T, win_time, win_freq, fft=True, axes=(0,1))
    else:
        wt1 = gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
        wt2 = gabor_transform(signal=signal2,fs=fs,freqs=freqs,n_cycles=n_cycles)

        Sxy    = wt1*np.conj(wt2)
        Sxx    = wt1*np.conj(wt1)
        Syy    = wt2*np.conj(wt2)

        Sxx = smooth_spectra.smooth_spectra(Sxx.T, win_time, win_freq, fft=True, axes=(0,1))
        Syy = smooth_spectra.smooth_spectra(Syy.T, win_time, win_freq, fft=True, axes=(0,1))
        Sxy = smooth_spectra.smooth_spectra(Sxy.T, win_time, win_freq, fft=True, axes=(0,1))
        return Sxx, Syy, Sxy

def gabor_coherence(signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),
                     win_time = 1, win_freq = 1, n_cycles = 7.0):
    Sxx, Syy, Sxy = gabor_spectrum(signal1 = signal1, signal2 = signal2, fs = fs, freqs = freqs,
                    win_time = win_time, win_freq = win_freq, n_cycles = n_cycles)

    return Sxy * np.conj(Sxy) / (Sxx * Syy)
