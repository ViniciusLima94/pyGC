import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.signal import csd
import neo
import elephant
from pycwt import wavelet
from pyGC import *

def ARmodel(N=5000, Trials = 10, Fs = 200, C=0.2, t_start=0, t_stop=None, cov = None):
	
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



#cov = np.array([ [1.00, 0.40],[0.40, 0.70] ])




#for i in range(Trials):
	#if i%50 == 0:
	#	print('Trial = ' + str(i))
	#x, y = ARmodel(N=N, Trials = Trials, C=C, t_start=0, t_stop=2.25, cov=cov)

	#
	#Y = neo.AnalogSignal(y, t_start=0*s, sampling_rate=200*Hz, units='dimensionless')

	#Wx = e
	#Wy = elephant.signal_processing.wavelet_transform(Y,f,fs=Fs).reshape((N,N//2+1)) 


#	S[0,0] += Wx*np.conj(Wx) / Trials#cxy(X=x, Y=[], Fs=Fs) / Trials#
#	S[0,1] += Wx*np.conj(Wy) / Trials#cxy(X=x, Y=y,  Fs=Fs) / Trials
#	S[1,0] += Wy*np.conj(Wx) / Trials#cxy(X=y, Y=x,  Fs=Fs) / Trials
#	S[1,1] += Wy*np.conj(Wy) / Trials#cxy(X=y, Y=[], Fs=Fs) / Trials

#scio.savemat('spec_mat.mat', {'f':f, 'S': S})

#plt.figure()
#plt.plot(f, S[0,0,:].real)
#plt.plot(f, S[1,1,:].real)
#plt.plot(f, S[0,1,:].real)
#plt.legend([r'$S_{xx}$', r'$S_{yy}$', r'$S_{xy}$'])
#plt.show()
'''
Ix2y = np.zeros([N,N//2+1])
Iy2x = np.zeros([N,N//2+1])

for i in range(N):
	print('N = ' + str(i))
	Snew, Hnew, Znew = wilson_factorization(S[:,:,i,:], f, Fs)
	Ix2y[i,:], Iy2x[i,:], _ = granger_causality(S[:,:,i,:], Hnew, Znew) 

plt.plot(f, Ix2y.real)
plt.plot(f, Iy2x.real) 
plt.show()


'''