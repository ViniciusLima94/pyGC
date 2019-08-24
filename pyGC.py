import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as scio

def wilson_factorization(S, freq, fs, Niterations=100, tol=1e-12, verbose=True):

	m = S.shape[0]    
	N = freq.shape[0]-1

	Sarr  = np.zeros([2,2,2*N]) * (1+1j)

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

def granger_causality(S, H, Z):

	N = S.shape[2]

	Hxx = H[0,0,:]
	Hxy = H[0,1,:]
	Hyx = H[1,0,:]
	Hyy = H[1,1,:]

	Hxx_tilda = Hxx + (Z[0,1]/Z[0,0]) * Hxy
	Hyx_tilda = Hyx + (Z[0,1]/Z[0,0]) * Hxx
	Hyy_circf = Hyy + (Z[1,0]/Z[1,1]) * Hyx

	Syy = Hyy_circf*Z[1,1]*np.conj(Hyy_circf) + Hyx*(Z[0,0]-Z[1,0]*Z[1,0]/Z[1,1]) * np.conj(Hyx)
	Sxx = Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda) + Hxy*(Z[1,1]-Z[0,1]*Z[0,1]/Z[0,0]) * np.conj(Hxy)

	Ix2y = np.log( Syy/(Hyy_circf*Z[1,1]*np.conj(Hyy_circf)) )
	Iy2x = np.log( Sxx/(Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda)) )

	Ixy  = np.zeros(N)

	for i in range(N):
		Ixy[i]  = np.log( (Hxx_tilda[i]*Z[0,0]*np.conj(Hxx_tilda[i]))*(Hyy_circf[i]*Z[1,1]*np.conj(Hyy_circf[i])/np.linalg.det(S[:,:,i])) ).real
	
	return Ix2y.real, Iy2x.real, Ixy.real

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



