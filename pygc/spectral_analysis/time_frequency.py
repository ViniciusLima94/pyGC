import numpy            as     np
import mne
import os
import h5py
import multiprocessing
from   joblib           import Parallel, delayed
from   ..misc            import smooth_spectra, downsample   

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

def wavelet_coherence(data = None, pairs = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
                      time_bandwidth = None, delta = 1, method = 'morlet', win_time = 1, win_freq = 1, 
                      dir_out = None, n_jobs = 1):

    # Data dimension
    T, C, L = data.shape
    # All possible pairs of channels

    # Computing wavelets
    W = wavelet_transform(data = data, fs = fs, freqs = freqs, n_cycles = n_cycles, 
                               time_bandwidth = time_bandwidth, delta = delta, 
                               method = method, n_jobs = -1)
    # Auto spectra
    S_auto = W * np.conj(W)

    def pairwise_coherence(index_pair, win_time, win_freq):
        channel1, channel2 = pairs[index_pair, 0], pairs[index_pair, 1]
        Sxy = W[:, channel1, :, :] * np.conj(W[:, channel2, :, :])
        Sxx = smooth_spectra.smooth_spectra(S_auto[:,channel1, :, :], win_time, win_freq, fft=True, axes = (1,2))
        Syy = smooth_spectra.smooth_spectra(S_auto[:,channel2, :, :], win_time, win_freq, fft=True, axes = (1,2))
        Sxy = smooth_spectra.smooth_spectra(Sxy, win_time, win_freq, fft=True, axes = (1,2))
        coh = Sxy * np.conj(Sxy) / (Sxx * Syy)

        file_name = os.path.join( dir_out, 'ch1_' + str(channel1) + '_ch2_' + str(channel2) +'.h5')
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('coherence', data=np.abs(coh).astype(np.float32))
            #  hf.create_dataset('frequency', data=freqs)
            #  hf.create_dataset('delta',	 data=delta) 

    #for trial_index in range(T):
    Parallel(n_jobs=n_jobs, backend='loky', timeout=1e6)(delayed(pairwise_coherence)(i, win_time, win_freq) for i in range(pairs.shape[0]) )

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
        #kernel = np.ones([win_time, win_freq])		
        return smooth_spectra.smooth_spectra(Sxx.T, win_time, win_freq, fft=True, axes=(0,1))#sig.convolve2d(Sxx, kernel, mode='same').T
    else:
        wt1 = gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
        wt2 = gabor_transform(signal=signal2,fs=fs,freqs=freqs,n_cycles=n_cycles)

        npts  = wt1.shape[0]
        nfreq = len(freqs)

        Sxy    = wt1*np.conj(wt2)
        Sxx    = wt1*np.conj(wt1)
        Syy    = wt2*np.conj(wt2)
        #print(len(Sxy.shape))

        # Smoothing spectra
        #kernel = np.ones([win_time, win_freq])		
        Sxx = smooth_spectra.smooth_spectra(Sxx.T, win_time, win_freq, fft=True, axes=(0,1))#sig.convolve2d(Sxx, kernel, mode='same')
        Syy = smooth_spectra.smooth_spectra(Syy.T, win_time, win_freq, fft=True, axes=(0,1))#sig.convolve2d(Syy, kernel, mode='same')
        Sxy = smooth_spectra.smooth_spectra(Sxy.T, win_time, win_freq, fft=True, axes=(0,1))#sig.convolve2d(Sxy, kernel, mode='same')
        return Sxx, Syy, Sxy

def gabor_coherence(signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),  
                     win_time = 1, win_freq = 1, n_cycles = 7.0):
    Sxx, Syy, Sxy = gabor_spectrum(signal1 = signal1, signal2 = signal2, fs = fs, freqs = freqs,  
                    win_time = win_time, win_freq = win_freq, n_cycles = n_cycles)

    return Sxy * np.conj(Sxy) / (Sxx * Syy)
