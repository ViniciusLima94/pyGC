########################################################################################
# Module with auxiliary functions
########################################################################################
import numpy as np 

def xcorr(x,y,maxlags):
	'''
		Estimate the auto (x=y) or cross (x!=y) correlation between two signals 
		Inputs:
		> x : Signal x of size [Number of variables, Number of observations].
		> y : Signal y of size [Number of variables, Number of observations].
		> maxlag : maximum number of lags for the correlations.
		Outputs:
		> lag : Lags of the correlations function (> 0).
		> Rxx : Correlation function.  
	'''

	Nvars = x.shape[0]
	N     = x.shape[1]
	lags = np.arange(0,maxlags)
	Rxx  = np.zeros([lags.shape[0],Nvars,Nvars])
	for k in lags:
		Rxx[k,:,:] = np.matmul(x[:,0:N-k],y[:,k:].T)/N
	return lags, Rxx

def PlusOperator(g,m,fs,freq):

	N = freq.shape[0]-1

	gam = np.zeros([m,m,2*N]) * (1+1j)

	for i in range(m):
		for j in range(m):
			gam[i,j,:] = np.fft.ifft(g[i,j,:])

	gamp = gam.copy()
	beta0 = 0.5*gam[:,:,0]
	gamp[:,:,0] = np.triu(beta0)
	gamp[:,:,len(freq):] = 0

	gp = np.zeros([m,m,2*N]) * (1+1j)

	for i in range(m):
		for j in range(m):
			gp[i,j,:] = np.fft.fft(gamp[i,j,:])

	return gp
