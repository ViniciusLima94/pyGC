import numpy            as     np
import mne.filter
import os
import h5py
import multiprocessing
from   joblib           import Parallel, delayed
from   .misc            import smooth_spectra, downsample   

class spectral_analysis():

	def __init__(self,):
		None

	def filter(self, data = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):

		signal_filtered = mne.filter.filter_data(data, fs, f_low, f_high,
		                                         method = 'iir', verbose=False, n_jobs=n_jobs)

		return signal_filtered

	def wavelet_transform(self, data = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
		                  time_bandwidth = None, delta = 1, method = 'morlet', n_jobs = 1):
		if method == 'morlet':
			out = mne.time_frequency.tfr_array_morlet(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
				                                      output='complex', decim = delta, n_jobs=n_jobs)
		if method == 'multitaper':
			out = mne.time_frequency.tfr_array_multitaper(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
													      time_bandwidth = time_bandwidth, output='complex', 
													      decim = delta, n_jobs=n_jobs)
		return out

	def wavelet_coherence(self, data = None, pairs = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
		                  time_bandwidth = None, delta = 1, method = 'morlet', win_time = 1, win_freq = 1, 
		                  dir_out = None, n_jobs = 1):

		# Data dimension
		T, C, L = data.shape
		# All possible pairs of channels

		# Computing wavelets
		W = self.wavelet_transform(data = data, fs = fs, freqs = freqs, n_cycles = n_cycles, 
		                           time_bandwidth = time_bandwidth, delta = delta, 
		                           method = method, n_jobs = -1)
		# Auto spectra
		S_auto = W * np.conj(W)

		def pairwise_coherence(index_pair, win_time, win_freq):
			channel1, channel2 = pairs[index_pair, 0], pairs[index_pair, 1]
			Sxy = W[:, channel1, :, :] * np.conj(W[:, channel2, :, :])
			#print(len(Sxy.shape))
			Sxx = smooth_spectra.smooth_spectra(S_auto[:,channel1, :, :], win_time, win_freq, fft=True, axes = (1,2))
			Syy = smooth_spectra.smooth_spectra(S_auto[:,channel2, :, :], win_time, win_freq, fft=True, axes = (1,2))
			Sxy = smooth_spectra.smooth_spectra(Sxy, win_time, win_freq, fft=True, axes = (1,2))
			coh = Sxy * np.conj(Sxy) / (Sxx * Syy)
			# Saving to file
			file_name = os.path.join( dir_out, 
				'ch1_' + str(channel1) + '_ch2_' + str(channel2) +'.h5')
			#print(file_name)
			#np.save(file_name, {'coherence' : np.abs(coh).astype(np.float32) })
			# Using HDF5 file format
			#dataset_name = 'ch1_' + str(channel1) + '_ch2_' + str(channel2)
			#hf.create_dataset(dataset_name, data=coh)
			#file_name = os.path.join( dir_out, 
			#	'ch1_' + str(channel1) + '_ch2_' + str(channel2) +'.h5')
			#hf = h5py.File(file_name, 'w')
			#hf.create_dataset('coherence', data=np.abs(coh).astype(np.float32))
			#hf.close()
			with h5py.File(file_name, 'w') as hf:
				hf.create_dataset('coherence', data=np.abs(coh).astype(np.float32))

		#for trial_index in range(T):
		Parallel(n_jobs=n_jobs, backend='loky', timeout=1e6)(
			delayed(pairwise_coherence)(i, win_time, win_freq) for i in range(pairs.shape[0]) )

