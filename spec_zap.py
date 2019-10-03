import numpy as np
import matplotlib.pyplot as plt 
from   scipy.ndimage import gaussian_filter
from   pySpec import * 

def senao(t, t1, t2):
	if t < t1:
		return np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.random.randn()
	elif t1 <= t < t2:
		return np.sin(2*np.pi*15*t) + np.random.randn()
	else:
		return np.sin(2*np.pi*30*t) + np.random.randn()

Fs = 100
dt = 1 / Fs
T  = 1000
N  = T // dt
trials = 50

t = np.arange(0, T, dt)

f = compute_freq(N, Fs)

y = np.zeros([trials, len(t)])

for i in range(trials):
	print('Trial = ' + str(i))
	for j in range(len(t)):
		y[i,j] = senao(t[j], 340, 700)

S = np.zeros(int(N/2+2)) * (1+1j)

for i in range(trials):
	print('Trial = ' + str(i))
	S += cxy(X=y[i,:], Y=[], f=f, Fs=Fs) / trials

plt.plot(f,gaussian_filter(S.real, 5))
plt.show()

f_v = np.linspace(1/T,Fs/2-0.1,500)

W = np.zeros([len(t),len(f_v)]) + 1j*np.zeros([len(t),len(f_v)])

for i in range(trials):
	print('Trial = ' + str(i))
	W += morlet_power(X=y[i,:], Y=[], f=f_v, Fs=Fs) / trials

plt.figure(figsize=(3,4))
plt.subplot(2,1,1)
plt.plot(f, gaussian_filter(S.real, 3))
plt.legend(['X', 'Y'])
plt.subplot(2,1,2)
plt.imshow(((W.real-W.real.min())/(W.real.max()-W.real.min())).T, aspect='auto', cmap='jet', origin='lower', extent=[0, T, 0, Fs/2]); plt.colorbar()
plt.savefig('fig2_Ac.pdf', dpi=600)
plt.tight_layout()
plt.close()


plt.imshow(Wll.real.T, aspect='auto', cmap='jet', origin='lower', extent=[0, T, 0, Fs/2])
plt.show()