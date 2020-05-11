import numpy as np
import pygc.parametric
import pygc.pySpec
from   pygc.covgc import covgc_time
from   ar_model   import *
from   scipy.integrate import simps
import pygc.granger

N  = 100000      # Number of observations
Fs = 200       # Sampling frequency
dt = 1.0 / Fs  # Time resolution
C  = 0.25      # Coupling parameter

# Covariance matrix
cov = np.array([ [1.00, 0.00],
				 [0.00, 1.00] ])


X = np.squeeze( ar_model_dhamala(N=N, Trials = 1, C=C, Fs=Fs, t_start=0, t_stop=None, cov=cov) )

##################################################################################################
# Covariance Based
##################################################################################################

lag = 2
dt  = N/2
t0  = N/2

res = np.squeeze(covgc_time(X, dt, lag, t0))
res = np.round(res, 6)
##################################################################################################
# Covariance Based
##################################################################################################

AR  = np.zeros([lag, 2,2])
SIG = np.zeros([2,2])
AR, SIG = pygc.parametric.YuleWalker(X[:,int(t0):], lag, maxlags=100)

print('Computing Granger Causality...')
f    = pygc.pySpec.compute_freq(N, Fs)
H, S = pygc.parametric.compute_transfer_function(AR, SIG, f, Fs)
Ix2y, Iy2x, Ixy  = pygc.granger.granger_causality(S, H, SIG)
fx2y = np.round( simps(Ix2y, f) / f.max(), 6)
fy2x = np.round( simps(Iy2x, f) / f.max(), 6)
fxy  = np.round( simps(Ixy, f)  / f.max(), 6)

##################################################################################################
# Printing results
##################################################################################################
print('-----------------+--------------------------------------')
print('                 |    1->2    |    2->1    |    1.2    ')
print('-----------------+--------------------------------------')
print('Cov GC           |  ' +str(res[0]) +'  |  ' +str(res[1])   +    '  |  ' + str(res[2])     )
print('-----------------+--------------------------------------')
print('Parametric GC    |  ' +str(fx2y) +'  |  ' +str(fy2x)   +    '  |  ' + str(fxy)     )
print('-----------------+--------------------------------------')