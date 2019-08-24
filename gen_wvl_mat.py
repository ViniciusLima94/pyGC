import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from ar_model import *
from pyGC import *
from pySpec import *

N  = 900
Fs = 200
dt = 1.0 / Fs
C  = 0.25
Trials = 5000

cov = np.array([ [1.00, 0.00],[0.00, 1.00] ])

f = compute_freq(N, Fs)

S = np.zeros([2,2,N,N//2+1]) + 1j*np.zeros([2,2,N,N//2+1])

Z = ARmodel(N=N, Trials = Trials, C=C, Fs=Fs, t_start=0, t_stop=2.25, cov=cov)

for i in range(Trials):
	if i%50 == 0:
		print('Trial = ' + str(i))

	Wx = morlet(Z[0,i,:], f, Fs)
	Wy = morlet(Z[1,i,:], f, Fs)

	S[0,0] += Wx*np.conj(Wx) / Trials
	S[0,1] += Wx*np.conj(Wy) / Trials
	S[1,0] += Wy*np.conj(Wx) / Trials
	S[1,1] += Wy*np.conj(Wy) / Trials

np.save('data/wavelet_matrix.npy', {'f': f, 'W': S})
'''
Snew, Hnew, Znew = wilson_factorization(S, f, Fs)
Snew, Hnew, Znew = Parallel(n_jobs=40)(delayed(wilson_factorization)(S[:,:,i,:], f, Fs) for i in range(N))

Ix2y, Iy2x, Ixy  = granger_causality(S, Hnew, Znew) 

np.save('data/gc_fft.npy', {'f': f, 'S': S, 'H': Hnew, 'Z': Znew, 'Ix2y': Ix2y, 'Iy2x': Iy2x, 'Ixy': Ixy})
'''