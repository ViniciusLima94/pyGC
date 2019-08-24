import numpy as np 
import matplotlib.pyplot as plt 

N = 900

Ix2y = np.zeros([N//2+1, N])
Iy2x = np.zeros([N//2+1, N])

for i in range(N):
	data = np.load('data/gc_wvl_'+str(i)+'.npy').item()
	f         = data['f']
	Ix2y[:,i] = data['Ix2y']
	Iy2x[:,i] = data['Iy2x']

plt.subplot(2,1,1)
plt.imshow(Iy2x, aspect='auto', cmap='jet', origin='lower', extent=[0, 4.5, f.min(), f.max()], vmin=0, vmax=np.round(Iy2x.max(),1))
plt.title(r'$Y\rightarrow X$')
plt.subplot(2,1,2)
plt.imshow(Ix2y, aspect='auto', cmap='jet', origin='lower', extent=[0, 4.5, f.min(), f.max()], vmin=0, vmax=np.round(Iy2x.max(),1))
plt.title(r'$X\rightarrow Y$')