import numpy as np 
import neo
import elephant
from quantities import s, Hz

def compute_freq(N, Fs):
	# Simulated time
	T = N / Fs
	# Frequency array
	f = np.linspace(1/T,Fs/2-1/T,N/2+1)

	return f

def cxy(X, Y=[], f=None, Fs=1):
	# Number of data points
	N = X.shape[0]

	if len(Y) > 0:
		Xfft = np.fft.fft(X)[1:len(f)+1]
		Yfft = np.fft.fft(Y)[1:len(f)+1]
		Pxy  = Xfft*np.conj(Yfft) / N
		return Pxy
	else:
		Xfft = np.fft.fft(X)[1:len(f)+1]
		Pxx  = Xfft*np.conj(Xfft) / N
		return Pxx

def morlet(X, f, Fs=1):
	N = X.shape[0]
	
	X = neo.AnalogSignal(X.T, t_start=0*s, sampling_rate=Fs*Hz, units='dimensionless')
	return elephant.signal_processing.wavelet_transform(X,f,fs=Fs).reshape((N,len(f)))

def morlet_power(X, Y=[], f=None, Fs=1):
	N = X.shape[0]

	if len(Y) > 0:
		Wx = morlet(X=X, f=f, Fs=Fs)
		Wy = morlet(Y=Y, f=f, Fs=Fs)
		return Wx*np.conj(Wy) / N
	else:
		Wx = morlet(X=X, f=f, Fs=Fs)
		return Wx*np.conj(Wx) / N
