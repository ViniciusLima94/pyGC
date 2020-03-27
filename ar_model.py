import numpy as np

def ar_model_dhamala(N=5000, Trials = 10, Fs = 200, C=0.2, t_start=0, t_stop=None, cov = None):
	'''
		AR model from Dhamala et. al.
	'''
	
	T = N / Fs

	time = np.linspace(0, T, N)

	X = np.random.random([Trials, N])
	Y = np.random.random([Trials, N])

	def interval(t, t_start, t_stop):
		if t_stop==None:
			return (t>=t_start)
		else:
			return (t>=t_start)*(t<=t_stop)

	for i in range(Trials):
		E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))
		for t in range(2, N):
			X[i,t] = 0.55*X[i,t-1] - 0.8*X[i,t-2] + interval(time[t],t_start,t_stop)*C*Y[i,t-1] + E[t,0]
			Y[i,t] = 0.55*Y[i,t-1] - 0.8*Y[i,t-2] +E[t,1]

	Z = np.zeros([2, Trials, N])

	Z[0] = X
	Z[1] = Y

	return Z

def ar_model_baccala(nvars, N, ntrials):
	'''
		AR model defined by Baccalá and Sameshima (2001).
	'''
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
