########################################################################################
# Run examples from our paper in RBEF
########################################################################################
import sys
import numpy             as     np 
import matplotlib.pyplot as     plt
import multiprocessing
import scipy
import scipy.signal
from   scipy.integrate   import simps
from   joblib            import Parallel, delayed
from   ar_model          import *
import pygc.pySpec
import pygc.parametric
import pygc.non_parametric
import pygc.granger
import plot_results 

p = int(sys.argv[-1])

if p == 0:
	# Generates figure 3 from the paper

	print('Generating Figure 3 from RBEF paper...')

	N  = 5000      # Number of observations
	Fs = 200       # Sampling frequency
	dt = 1.0 / Fs  # Time resolution
	C  = 0.25      # Coupling parameter
	Trials = 5000  # Number of trials
	# Covariance matrix
	cov = np.array([ [1.00, 0.00],
					 [0.00, 1.00] ])

	f = pygc.pySpec.compute_freq(N, Fs)

	S = np.zeros([2,2,N//2+1]) + 1j*np.zeros([2,2,N//2+1])

	print('Generating AR model time series...')
	Z = ar_model_dhamala(N=N, Trials = Trials, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov)

	print('Estimating spectral matrix from ' + str(Trials) + ' trials...')
	for i in range(Trials):
		if i%500 == 0:
			print('Trial = ' + str(i))

		S[0,0] += pygc.pySpec.cxy(X=Z[0,i,:], Y=[], f=f, Fs=Fs) / Trials
		S[0,1] += pygc.pySpec.cxy(X=Z[0,i,:], Y=Z[1,i,:], f=f, Fs=Fs) / Trials
		S[1,0] += pygc.pySpec.cxy(X=Z[1,i,:], Y=Z[0,i,:], f=f, Fs=Fs) / Trials
		S[1,1] += pygc.pySpec.cxy(X=Z[1,i,:], Y=[], f=f, Fs=Fs) / Trials

	print('Computing Granger Causality...')
	Snew, Hnew, Znew = pygc.non_parametric.wilson_factorization(S, f, Fs, Niterations=30)
	Ix2y, Iy2x, Ixy  = pygc.granger.granger_causality(S, Hnew, Znew)

	print('Saving data...')
	np.save('data/fig3.npy', {'f': f, 'S': S, 'H': Hnew, 'Z': Znew, 'Ix2y': Ix2y, 'Iy2x': Iy2x, 'Ixy': Ixy})

	print('Plotting results...')
	plot_results.fig3()

if p == 1:
	# Generates figure 4 from the paper

	N  = 900      # Number of observations
	Fs = 200      # Sampling frequency
	dt = 1.0 / Fs # Time resolution
	C  = 0.25     # Coupling parameter
	Trials = 5000 # Number of trials

	cov = np.array([ [1.00, 0.00],
					 [0.00, 1.00] ])

	f = pygc.pySpec.compute_freq(N, Fs)

	S = np.zeros([2,2,N,N//2+1]) + 1j*np.zeros([2,2,N,N//2+1])

	print('Generating AR model time series...')
	Z = ar_model_dhamala(N=N, Trials = Trials, C=C, Fs=Fs, t_start=0, t_stop=2.25, cov=cov)

	print('Estimating wavelet matrix from ' + str(Trials) + ' trials...')
	for i in range(Trials):
		if i%500 == 0:
			print('Trial = ' + str(i))

		Wx = pygc.pySpec.morlet(Z[0,i,:], f, Fs)
		Wy = pygc.pySpec.morlet(Z[1,i,:], f, Fs)

		S[0,0] += Wx*np.conj(Wx) / Trials
		S[0,1] += Wx*np.conj(Wy) / Trials
		S[1,0] += Wy*np.conj(Wx) / Trials
		S[1,1] += Wy*np.conj(Wy) / Trials


#	S   = S[:,:,idx,:]
	print('Computing Granger Causality...')
	def save_granger(S, idx):
		Snew, Hnew, Znew = pygc.non_parametric.wilson_factorization(S[:,:,idx,:], f, Fs, Niterations=30, verbose=False)
		Ix2y, Iy2x, Ixy  = pygc.granger.granger_causality(S[:,:,idx,:], Hnew, Znew) 
		np.save('data/fig4_'+str(idx)+'.npy', {'f': f, 'Ix2y': Ix2y, 'Iy2x': Iy2x, 'Ixy': Ixy})

	Parallel(n_jobs=40,	 backend='loky', max_nbytes=1e6)(delayed(save_granger)(S, idx) for idx in range(N))
	print('Plotting results...')
	plot_results.fig4()

if p == 2:
	# Generates figure 7 and 8 from the paper
	N       = 5000  # Number of observations
	Trials  = 1000  # Number of trials
	nvars   = 5     # Number of variables

	Fs = 2*np.pi
	dt = 1.0 / Fs
	f  = pygc.pySpec.compute_freq(N, Fs)

	print('Generating AR model time series...')
	Y = ar_model_baccala(nvars, N, Trials)

	print('Estimating spectral matrix from ' + str(Trials) + ' trials...')
	S = np.zeros([nvars, nvars, N//2 + 1]) * (1 + 1j)

	for trial in range(Trials):
		if (trial % 100 == 0):
			print('Trial = ' + str(trial))
		for i in range(nvars):
			for j in range(nvars):
				S[i,j] += pygc.pySpec.cxy(X=Y[i,:,trial], Y=Y[j,:,trial], f=f, Fs=Fs) / Trials

	print('Estimating pairwise Granger casalities')
	GC = np.zeros([nvars, nvars])
	for i in range(nvars):
		for j in range(nvars):
			if i == j:
				continue
			else:
				S_aux = np.array([[S[i,i], S[i,j]],[S[j,i], S[j,j]]])
				_, H, Z = pygc.non_parametric.wilson_factorization(S_aux, f, Fs, Niterations=10, tol=1e-12, verbose=False)
				Ix2y, Iy2x, _ = pygc.granger.granger_causality(S_aux, H, Z)
				GC[i,j] = simps(Ix2y, f) / 2*np.pi
				GC[j,i] = simps(Iy2x, f) / 2*np.pi

	print('Estimating conditional Granger casalities')
	F   = pygc.granger.conditional_granger_causality(S, f, Fs, Niterations = 10, verbose=False)
	cGC = pygc.granger.conditional_spec_granger_causality(S, f, Fs, Niterations=100, tol=1e-12, verbose=False)

	print('Saving data...')
	np.save('data/fig_7_8.npy', {'f':f,'GC': GC, 'F': F, 'cGC': cGC})

	print('Plotting results...')
	plot_results.fig7_8()

if p == 3:
	# Fits an AR model by solving YW equations as in appendix A of the paper.

	
	Trials = 1000  # Number of trials
	Fs     = 200   # Sampling frequency
	N      = 1000  # Number of data points
	X      = np.zeros([1,N, Trials]) # Data matrix
	tsim   = N/Fs # Simulation time

	# Coefficients of the ar model
	c = [0.7, 0.2, -0.1, -0.3]

	print('Generating AR model time series...')
	for T in range(Trials):
		X[0,:,T] = scipy.signal.lfilter([1], -np.array([-1]+c), np.random.randn(N))
	
	print('Estimating AR model coefficients for ' + str(Trials) + ' trials')

	for m in [2, 3, 4, 5, 6]:

		print()

		AR  = np.zeros([1,1,m])
		SIG = np.zeros([1,1])
		for T in range(Trials):
			aux1, aux2 = pygc.parametric.YuleWalker(X[:,:,T], m, maxlags=100)
			AR  += aux1.T/Trials
			SIG += aux2.T/Trials

		AR = np.round(AR, 2)
		SIG = np.round(SIG, 2)
		print('Using order = ' + str(m)+ '. Original coefficients: ' + str(c) + '. Estimated coefficients ' + str(AR[0][0]) + '. Noise variace: ' + str(SIG[0][0]))

if p == 4:
	# Generates figure 3C from the paper, but using a paramtreic method
	# Generates figure 3 from the paper

	print('Generating Figure 3 from RBEF paper...')

	N  = 5000      # Number of observations
	Fs = 200       # Sampling frequency
	dt = 1.0 / Fs  # Time resolution
	C  = 0.25      # Coupling parameter
	Trials = 5000  # Number of trials
	# Covariance matrix
	cov = np.array([ [1.00, 0.00],
					 [0.00, 1.00] ])

	print('Generating AR model time series...')
	X = ar_model_dhamala(N=N, Trials = Trials, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov)

	print('Estimating VAR coefficients using oreder m=2...')
	m   = 2
	AR  = np.zeros([m, 2,2])
	SIG = np.zeros([2,2])
	for T in range(Trials):
		aux1, aux2 = pygc.parametric.YuleWalker(X[:,T,:], m, maxlags=100)
		AR  += aux1/Trials
		SIG += aux2/Trials

	print('Computing Granger Causality...')
	f    = pygc.pySpec.compute_freq(N, Fs)
	H, S = pygc.parametric.compute_transfer_function(AR, SIG, f, Fs)
	Ix2y, Iy2x, _  = pygc.granger.granger_causality(S, H, SIG)

	plt.figure(figsize=(6,2))
	plt.plot(f, Ix2y)
	plt.plot(f, Iy2x)
	plt.xlim([0, 100])
	plt.ylim([-0.01, 1.2])
	plt.ylabel('GC')
	plt.xlabel('Frequency [Hz]')
	plt.legend([r'$X_{1}\rightarrow X_{2}$', r'$X_{2}\rightarrow X_{1}$'])
	plt.savefig('figures/fig9.pdf', dpi = 600)
	plt.close()



