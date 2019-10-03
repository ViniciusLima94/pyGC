import numpy as np
import matplotlib.pyplot as plt 
from   scipy.ndimage import gaussian_filter
from   pySpec import * 

def gen_sig(Fs, T, seed):

	np.random.seed(seed)

	t = np.arange(0, T+1/Fs, 1/Fs)

	aux_x = 0
	aux_y = 0

	X = 10*np.sin(2*np.pi*10*t) + np.random.normal(0,15,len(t))
	Y = 10*np.sin(2*np.pi*10*t) + 10*np.sin(2*np.pi*20*t)  + np.random.normal(0,15,len(t))

	return X, Y

T  = 20
Fs = 80
dt = 1.0/Fs
Trials = 100

t = np.arange(0, T+1/Fs, 1/Fs)

f = compute_freq(len(t), Fs)

Z = np.zeros([2, len(t), Trials])

for i in range(Trials):
	Z[0,:,i], Z[1,:,i] = gen_sig(Fs, T, i)

S = np.zeros([2,2,len(t)//2+1]) * (1 + 1j)

for i in range(Trials):
	if i%50 == 0:
		print('Trial = ' + str(i))

	S[0,0] += cxy(X=Z[0,:,i], Y=[], f=f, Fs=Fs) / Trials
	S[0,1] += cxy(X=Z[0,:,i], Y=Z[1,:,i], f=f, Fs=Fs) / Trials
	S[1,0] += cxy(X=Z[1,:,i], Y=Z[0,:,i], f=f, Fs=Fs) / Trials
	S[1,1] += cxy(X=Z[1,:,i], Y=[], f=f, Fs=Fs) / Trials

C = np.abs(S[0,1])**2 / (S[0,0]*S[1,1])


plt.figure(figsize=(3,4))
plt.subplot(2,1,1)
plt.plot(f, gaussian_filter(S[0,0].real, 3))
plt.plot(f, gaussian_filter(S[1,1].real, 3), '--')
plt.legend(['X', 'Y'])
plt.subplot(2,1,2)
plt.plot(f, gaussian_filter(C.real,3), 'k')
plt.savefig('fig1_A.pdf', dpi=600)
plt.tight_layout()
plt.close()