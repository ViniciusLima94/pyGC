import mne.filter

def bp_filter(data = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):

		signal_filtered = mne.filter.filter_data(data, fs, f_low, f_high,
		                                         method = 'iir', verbose=False, n_jobs=n_jobs)

		return signal_filtered
