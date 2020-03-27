########################################################################################
# Module with functions for parametric estimation of GC
########################################################################################
import numpy as np
import scipy.linalg
from   .tools import *

def YuleWalker(X, m, maxlags=100):
	'''
		Estimate the VAR model coefficients by solving the YW equations.
		Inputs:
		> X : Data with size [Number of variables, Number of observations].
		> m : Model order
		Outputs:
		> AR_yw : Coefficient matrix
		> eps_yw: 
	'''

	Nvars = X.shape[0]
	N     = X.shape[1] 

	# Compute cross-correlations matrices for each lag
	lag, Rxx = xcorr(X,X,maxlags)

	#  Reorganizing data to compute crosscorrelation matrix
	b = X.T[m:]
	A = np.zeros([N-m,Nvars*m])

	count = 0
	for i in np.arange(0,m):
		for j in range(0,Nvars):
			A[:,count] = X.T[m-i-1:N-i-1,j]
			count      += 1

	r = np.reshape( Rxx[1:m+1], (Nvars*m,Nvars) )
	R = np.matmul(A.T, A)/N

	AR_yw  = np.matmul(scipy.linalg.inv(R).T,r).T
	AR_yw  = AR_yw.T.reshape((m,Nvars,Nvars))

	eps_yw = Rxx[0] 
	for i in range(m):
		eps_yw += np.matmul(-AR_yw[i].T,Rxx[i+1])

	return AR_yw, eps_yw

def compute_transfer_function(AR, sigma, f, Fs):

	m     = AR.shape[0]
	Nvars = AR.shape[1]

	H = np.zeros([Nvars,Nvars,f.shape[0]]) * (1 + 1j)
	S = np.zeros([Nvars,Nvars,f.shape[0]]) * (1 + 1j)

	for i in range(0,m+1):
		comp = np.exp(-1j * f * 2 * np.pi * i/Fs)
		if i == 0:
			for j in range(comp.shape[0]):
				H[:,:,j] += np.eye(Nvars) * comp[j]
		else:
			for j in range(comp.shape[0]):
				H[:,:,j] += -AR[i-1].T * comp[j]

	for i in range(f.shape[0]):
		H[:,:,i] = np.linalg.inv(H[:,:,i])

	for i in range(f.shape[0]):
		S[:,:,i] = np.matmul( np.matmul(H[:,:,i], sigma), np.conj(H[:,:,i]).T )

	return H, S
