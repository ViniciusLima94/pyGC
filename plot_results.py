import numpy as np 
import matplotlib.pyplot as plt 


######################################################################################################
# FIGURE 3
######################################################################################################
def fig3():
	N = 5000
	data = np.load('data/fig3.npy', allow_pickle=True).item()
	f    = data['f']
	Sxx  = data['S'][0,0]
	Sxy  = data['S'][0,1]
	Syy  = data['S'][1,1]
	Cxy  = Sxy*np.conj(Sxy) / (Sxx * Syy)
	Iy2x = data['Iy2x']
	Ix2y = data['Ix2y']

	plt.subplot2grid((2,2), (0,0))
	plt.plot(f, Sxx.real / 100)
	plt.plot(f, Syy.real / 100)
	#plt.plot(f, Sxy * np.conj(Sxy) / 100)
	plt.xlim([0, 100])
	plt.ylim([-0.15, 0.8])
	#plt.legend([r'$S_{11}(\omega)$', r'$S_{22}(\omega)$', r'$|S_{12}(\omega)|$'])
	plt.legend([r'$S_{11}(\omega)$', r'$S_{22}(\omega)$'])
	plt.xlabel('Frequency [Hz]')
	plt.subplot2grid((2,2), (0,1))
	plt.plot(f, Cxy.real)
	plt.xlim([0, 100])
	plt.ylim([-0.01, 0.67])
	plt.ylabel(r'$C_{12}(\omega)$')
	plt.xlabel('Frequency [Hz]')
	plt.subplot2grid((2,2), (1,0), colspan=2)
	plt.plot(f, Ix2y)
	plt.plot(f, Iy2x)
	plt.xlim([0, 100])
	plt.ylim([-0.01, 1.2])
	plt.ylabel('GC')
	plt.xlabel('Frequency [Hz]')
	plt.legend([r'$X_{1}\rightarrow X_{2}$', r'$X_{2}\rightarrow X_{1}$'])
	plt.tight_layout()
	plt.savefig('figures/fig1.pdf', dpi=600)
	plt.close()


######################################################################################################
# FIGURE 4
######################################################################################################
def fig4():
	N = 900

	Ix2y = np.zeros([N//2+1, N])
	Iy2x = np.zeros([N//2+1, N])

	for i in range(N):
		data = np.load('data/fig4_'+str(i)+'.npy').item()
		f         = data['f']
		Ix2y[:,i] = data['Ix2y']
		Iy2x[:,i] = data['Iy2x']

	plt.subplot(3,1,1)
	t = np.linspace(0, 4.5, N)
	Cyx = 0.25 * (t<=2.25)
	Cxy = np.zeros(N)
	plt.plot(t, Cxy, '--')
	plt.plot(t, Cyx)
	plt.ylim([-0.1, 0.3])
	plt.xlim(0, 4.5)
	plt.ylabel('Coupling')
	plt.legend([r'$X_{1}\rightarrow X_{2}$', r'$X_{2}\rightarrow X_{1}$'])
	plt.subplot(3,1,2)
	plt.imshow(Iy2x, aspect='auto', cmap='jet', origin='lower', extent=[0, 4.5, f.min(), f.max()], vmin=0, vmax=np.round(Iy2x.max(),1))
	plt.ylabel('frequency (Hz)')
	plt.title(r'Granger causality: $X_{2}\rightarrow X_{1}$')
	plt.subplot(3,1,3)
	plt.imshow(Ix2y, aspect='auto', cmap='jet', origin='lower', extent=[0, 4.5, f.min(), f.max()], vmin=0, vmax=np.round(Iy2x.max(),1))
	plt.ylabel('frequency (Hz)')
	plt.xlabel('time (sec)')
	plt.title(r'Granger causality: $X_{1}\rightarrow X_{2}$')
	plt.tight_layout()
	plt.savefig('figures/fig2.pdf', dpi=600)
	plt.close()

######################################################################################################
# FIGURE 7 and 8
######################################################################################################
def fig7_8():

	nvars = 5

	data = np.load('data/fig_7_8.npy').item()
	f    = data['f']
	F    = data['F']
	GC   = data['GC']
	cGC  = data['cGC']

	plt.figure(figsize=(4,6))
	plt.subplot(2,1,1)
	plt.imshow(F, aspect='auto', cmap='gist_yarg'); plt.colorbar()
	plt.ylabel('to')
	plt.title('GC')
	plt.xticks([])
	plt.subplot(2,1,2)
	plt.imshow(GC.T, aspect='auto', cmap='gist_yarg'); plt.colorbar()
	plt.ylabel('to')
	plt.xlabel('from')
	plt.title('cGC')
	plt.savefig('figures/fig7.pdf', dpi=600)
	plt.close()

	plt.figure()
	count = 1
	for i in range(nvars):
		for j in range(nvars):
			plt.subplot(nvars, nvars, count)
			plt.plot(f,cGC[j,i], 'k')
			plt.ylim(0,1.7)
			if j > 0:
				plt.yticks([])
			if i < 4:
				plt.xticks([])
			#plt.xlim(0,500)
			count += 1
	plt.tight_layout()
	plt.savefig('figures/fig8.pdf', dpi = 600)
	plt.close()
