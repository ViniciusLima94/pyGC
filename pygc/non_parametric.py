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

import numpy as np


def rdet(A):
        if not np.shape(A):
                return A
        else:
                return np.linalg.det(A)


def covgc_time(X, dt, lag, t0):

    """
    [GC, pairs] = covGC_time(X, dt, lag, t0)
    Computes single-trials covariance-based Granger Causality for gaussian variables
    X   = data arranged as sources x timesamples
    dt  = duration of the time window for covariance correlation in samples
    lag = number of samples for the lag within each trial
    t0  = zero time in samples
    GC  = Granger Causality arranged as (number of pairs) x (3 directionalities (pair(:,1)->pair(:,2), pair(:,2)->pair(:,1), instantaneous))
    pairs = indices of sources arranged as number of pairs x 2
    -------------------- Total Granger interdependence ----------------------
    Total Granger interdependence:
    TGI = GC(x,y)
    TGI = sum(GC,2):
    TGI = GC(x->y) + GC(y->x) + GC(x.y)
    TGI = GC(x->y) + GC(y->x) + GC(x.y) = Hycy + Hxcx - Hxxcyy
    This quantity can be defined as the Increment of Total
    Interdependence and it can be calculated from the different of two
    mutual informations as follows
    ----- Relations between Mutual Informarion and conditional entropies ----
    % I(X_i+1,X_i|Y_i+1,Y_i) = H(X_i+1) + H(Y_i+1) - H(X_i+1,Y_i+1)
    Ixxyy   = log(det_xi1) + log(det_yi1) - log(det_xyi1);
    % I(X_i|Y_i) = H(X_i) + H(Y_i) - H(X_i, Y_i)
    Ixy     = log(det_xi) + log(det_yi) - log(det_yxi);
    ITI(np) = Ixxyy - Ixy;
    Reference
    Brovelli A, Chicharro D, Badier JM, Wang H, Jirsa V (2015)
    Copyright of Andrea Brovelli (Jan 2015) - Matlab version -
    Copyright of Andrea Brovelli & Michele Allegra (Jan 2020) - Python version -
    """

    X = np.array(X)

    # Data parameters. Size = sources x time points
    nSo, nTi = X.shape

    ind_t = (t0 - lag) * np.ones((dt, lag+1))

    for i in range(dt):
        for j in range(lag+1):
            ind_t[i, j] = ind_t[i, j] + i + (lag-j)

    ind_t = ind_t.astype(int)

    # Pairs between sources
    nPairs = nSo * (nSo-1)/2
    nPairs = np.int(nPairs)

    # Init
    GC = np.zeros((nPairs, 3))

    pairs = np.zeros((nPairs, 2))

    # Normalisation coefficient for gaussian entropy
    C = np.log(2*np.pi*np.exp(1))

    # Loop over number of pairs
    cc = 0

    for i in range(nSo):
        for j in range(i+1, nSo):

            # Define pairs of channels
            pairs[cc, 0] = i
            pairs[cc, 1] = j


            # Extract data for a given pair of sources
            x = X[i, ind_t]
            y = X[j, ind_t]

            # ---------------------------------------------------------------------
            # Conditional Entropies
            # ---------------------------------------------------------------------
            # Hycy: H(Y_i+1|Y_i) = H(Y_i+1) - H(Y_i)
            det_yi1 = rdet(np.cov(y.T))
            det_yi = rdet(np.cov(y[:, 1:].T))
            Hycy = np.log(det_yi1) - np.log(det_yi)
            # Hxcx: H(X_i+1|X_i) = H(X_i+1) - H(X_i)
            det_xi1 = rdet(np.cov(x.T))
            det_xi = rdet(np.cov(x[:, 1:].T))
            Hxcx = np.log(det_xi1) - np.log(det_xi)
            # Hycx: H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_yxi1 = rdet(np.cov(np.column_stack((y, x[:, 1:])).T))
            det_yxi = rdet(np.cov(np.column_stack((y[:, 1:], x[:, 1:])).T))
            Hycx = np.log(det_yxi1) - np.log(det_yxi)
            # Hxcy: H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_xyi1 = rdet(np.cov(np.column_stack((x, y[:, 1:])).T))
            Hxcy = np.log(det_xyi1) - np.log(det_yxi)
            # Hxxcyy: H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_xyi1 = rdet(np.cov(np.column_stack((x, y)).T))
            Hxxcyy = np.log(det_xyi1) - np.log(det_yxi)

            # ---------------------------------------------------------------------
            # Granger Causality measures
            # ---------------------------------------------------------------------
            # GC(pairs(:,1)->pairs(:,2))
            GC[cc, 0] = Hycy - Hycx
            # GC(pairs(:,2)->pairs(:,1))
            GC[cc, 1] = Hxcx - Hxcy
            # GC(x.y)
            GC[cc, 2] = Hycx + Hxcy - Hxxcyy

            cc = cc + 1


    return np.column_stack((pairs, GC))