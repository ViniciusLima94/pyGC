########################################################################################
# Module with functions for non-parametric estimation of GC
########################################################################################
import numpy             as     np 
import matplotlib.pyplot as     plt 
import scipy.io          as     scio
from   .tools             import *

def wilson_factorization(S, freq, fs, Niterations=100, tol=1e-12, verbose=True):
	'''
		Algorithm for the Wilson Factorization of the spectral matrix.
	'''

	m = S.shape[0]    
	N = freq.shape[0]-1

	Sarr  = np.zeros([m,m,2*N]) * (1+1j)

	f_ind = 0

	for f in freq:
		Sarr[:,:,f_ind] = S[:,:,f_ind]
		if(f_ind>0):
			Sarr[:,:,2*N-f_ind] = S[:,:,f_ind].T
		f_ind += 1
	
	#Sarr[:,:,0:N+1] = S[:,:,:].copy()
	#Sarr[:,:,N+2:]  = S[:,:,::-1]

	gam = np.zeros([m,m,2*N])

	for i in range(m):
		for j in range(m):
			gam[i,j,:] = (np.fft.ifft(Sarr[i,j,:])).real

	gam0 = gam[:,:,0]
	h    = np.linalg.cholesky(gam0).T

	psi = np.ones([m,m,2*N]) * (1+1j)

	for i in range(0,Sarr.shape[2]):
		psi[:,:,i] = h

	I = np.eye(m)

	g = np.zeros([m,m,2*N]) * (1+1j)
	for iteration in range(Niterations):

		for i in range(Sarr.shape[2]):
			# g(:,:,ind)=inv(psi(:,:,ind))*Sarr(:,:,ind)*inv(psi(:,:,ind))'+I;%'
			g[:,:,i] = np.matmul(np.matmul(np.linalg.inv(psi[:,:,i]),Sarr[:,:,i]),np.conj(np.linalg.inv(psi[:,:,i])).T)+I
			#g[:,:,i] = np.linalg.inv(psi[:,:,i])*Sarr[:,:,i]*np.conj(np.linalg.inv(psi[:,:,i]).T) + I

		gp = PlusOperator(g, m, fs, freq)
		psiold = psi.copy()
		psierr = 0
		for i in range(Sarr.shape[2]):
			psi[:,:,i] =np.matmul(psi[:,:,i], gp[:,:,i])# psi[:,:,i]*gp[:,:,i] #
			psierr    += np.linalg.norm(psi[:,:,i]-psiold[:,:,i],1) / Sarr.shape[2]

		if(psierr<tol):
			break

		if verbose == True:
			print('Err = ' + str(psierr))


	Snew = np.zeros([m,m,N+1]) * (1 + 1j)

	for i in range(N+1):
		Snew[:,:,i] = np.matmul(psi[:,:,i], np.conj(psi[:,:,i]).T)

	gamtmp = np.zeros([m,m,2*N]) * (1 + 1j)

	for i in range(m):
		for j in range(m):
			gamtmp[i,j,:] = np.fft.ifft(psi[i,j,:]).real

	A0    = gamtmp[:,:,0]
	A0inv = np.linalg.inv(A0)
	Znew  = np.matmul(A0, A0.T).real

	Hnew = np.zeros([m,m,N+1]) * (1 + 1j)

	for i in range(N+1):
		Hnew[:,:,i] = np.matmul(psi[:,:,i], A0inv)

	return Snew, Hnew, Znew