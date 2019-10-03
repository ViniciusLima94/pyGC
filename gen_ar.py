import numpy as np 
import matplotlib.pyplot as plt 

a1, a2 = np.array([0.3, -0.5])
b1, b2, b3, b4 = np.array([-.2, .5, .6, -.2])

T = 300

X = np.random.random(size=T)
Y = np.random.random(size=T)

for t in range(2,T):
	X[t] = a1*X[t-1] + a2*X[t-2] + np.random.randn()

for t in range(5,T):
	Y[t] = b1*Y[t-1] + b2*Y[t-2] + b3*Y[t-3] + b4*Y[t-4] + np.random.randn()

plt.subplot(2,1,1)
plt.plot(X)
plt.xlim([100,300])
plt.subplot(2,1,2)
plt.plot(Y)
plt.xlim([100,300])
plt.savefig('ar_ex.pdf', dpi = 600)
plt.close()