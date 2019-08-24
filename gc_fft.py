import numpy as np
import matplotlib.pyplot as plt
from ar_model import *
from pyGC import *
from pySpec import *

N  = 5000
Fs = 200
dt = 1.0 / Fs
C  = 0.25
Trials = 5000

cov = np.array([ [1.00, 0.00],[0.00, 1.00] ])

f = compute_freq(N, Fs)

S = np.zeros([2,2,N//2+1]) + 1j*np.zeros([2,2,N//2+1])

Z = ARmodel(N=N, Trials = Trials, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov)

for i in range(Trials):
	if i%50 == 0:
		print('Trial = ' + str(i))

	S[0,0] += cxy(X=Z[0,i,:], Y=[], f=f, Fs=Fs) / Trials
	S[0,1] += cxy(X=Z[0,i,:], Y=Z[1,i,:], f=f, Fs=Fs) / Trials
	S[1,0] += cxy(X=Z[1,i,:], Y=Z[0,i,:], f=f, Fs=Fs) / Trials
	S[1,1] += cxy(X=Z[1,i,:], Y=[], f=f, Fs=Fs) / Trials

Snew, Hnew, Znew = wilson_factorization(S, f, Fs)
Ix2y, Iy2x, Ixy  = granger_causality(S, Hnew, Znew) 

np.save('data/gc_fft.npy', {'f': f, 'S': S, 'H': Hnew, 'Z': Znew, 'Ix2y': Ix2y, 'Iy2x': Iy2x, 'Ixy': Ixy})