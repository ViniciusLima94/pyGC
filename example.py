import numpy as np
from   pygc.covgc import covgc_time
from   ar_model          import *


N  = 5000      # Number of observations
Fs = 200       # Sampling frequency
dt = 1.0 / Fs  # Time resolution
C  = 0.25      # Coupling parameter

# Covariance matrix
cov = np.array([ [1.00, 0.00],
				 [0.00, 1.00] ])


X = np.squeeze( ar_model_dhamala(N=N, Trials = 1, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov) )

lag = 2
dt  = N/2
t0  = N/2


res = covgc_time(X, dt, lag, t0)

