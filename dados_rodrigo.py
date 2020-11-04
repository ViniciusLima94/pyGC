import numpy as np
from   pygc.covgc import covgc_time

X = np.loadtxt('dall.dat')

for i in range(3):
	X[i] = (X[i] - X[i].mean()) / X[0].std()

N  = X.shape[1] # Number of observations
dt = 0.020      # Time resolution
Fs = 200 / dt   # Sampling frequency
C  = 0.25       # Coupling parameter

##################################################################################################
# Covariance Based
##################################################################################################

lag = 1
dt  = N/2
t0  = N/2

for lag in range(1,10):
	print('lag = ' + str(lag))
	res = np.squeeze(covgc_time(X, dt, lag, t0))
	print(res)