import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import simps
from pySpec import *
from pyGC import *

'''
	AR model defined by Baccalá and Sameshima (2001).
	ref: Partial directed coherence: a new concept in neural struc- ture determination
'''
def ar_model(nvars, N, ntrials):

	Y = np.random.uniform(size=(nvars, N, ntrials))
	w = np.random.normal(0, 1, size=(nvars, N, ntrials))

	for i in range(ntrials):
		for t in range(5,N):
			Y[0,t,i] = 0.95*np.sqrt(2.0)*Y[0,t-1,i] - 0.9025*Y[0,t-2,i] + w[0,t,i]
			Y[1,t,i] = 0.5*Y[0,t-2,i]  + w[1,t,i]
			Y[2,t,i] = -0.4*Y[0,t-3,i] + w[2,t,i]
			Y[3,t,i] = -0.5*Y[0,t-2,i] + 0.25*np.sqrt(2.0)*Y[3,t-1,i] + 0.25*np.sqrt(2.0)*Y[4,t-1,i] + w[3,t,i]
			Y[4,t,i] = -0.25*np.sqrt(2.0)*Y[3,t-1,i] + 0.25*np.sqrt(2.0)*Y[4,t-1,i] + w[4,t,i]

	return Y

# AR model parameters
N       = 5000
ntrials = 1000
nvars   = 5

Fs = 2*np.pi
dt = 1.0 / Fs
f  = compute_freq(N, Fs)

# Simulating
Y = ar_model(nvars, N, ntrials)
# Plotting system's sugnal
plt.figure()
for i in range(nvars):
	plt.plot((Y[i,:,0]-Y[i,:,0].mean())/Y[i,:,0].std() + i + (i-1)*5)
plt.savefig('figures/baccala_signal.pdf', dpi = 300)
plt.close()

# Computing spectral matrix
S = np.zeros([nvars, nvars, N//2 + 1]) * (1 + 1j)

for trial in range(ntrials):
	if (trial % 100 == 0):
		print('Trial = ' + str(trial))
	for i in range(nvars):
		for j in range(nvars):
			S[i,j] += cxy(X=Y[i,:,trial], Y=Y[j,:,trial], f=f, Fs=Fs) / ntrials

Niterations = 15
tol         = 1e-12


GC = np.zeros([nvars, nvars])

for i in range(nvars):
	for j in range(nvars):
		if i == j:
			continue
		else:
			S_aux = np.array([[S[i,i], S[i,j]],[S[j,i], S[j,j]]])
			_, H, Z = wilson_factorization(S_aux, f, Fs, Niterations=10, tol=1e-12, verbose=False)
			Ix2y, Iy2x, _ = granger_causality(S_aux, H, Z)
			GC[i,j] = simps(Ix2y, f) / 2*np.pi
			GC[j,i] = simps(Iy2x, f) / 2*np.pi


F = conditional_granger_causality(S, f, Fs, verbose=False)



cGC = conditional_spec_granger_causality(S, f, Fs, Niterations=10, tol=1e-12, verbose=True)
#plt.imshow(F, aspect='auto', cmap='gist_yarg'); plt.colorbar()

count = 1
for i in range(nvars):
	for j in range(nvars):
		plt.subplot(nvars, nvars, count)
		plt.plot(f,cGC[j,i], 'k')
		plt.ylim(0,1.7)
		if j > 0:
			plt.yticks([])
		if i < 4:
			plt.xticks([])
		#plt.xlim(0,500)
		count += 1
plt.tight_layout()
plt.savefig('baccala_signal_spec.pdf', dpi = 600)
plt.close()