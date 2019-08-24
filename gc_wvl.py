import numpy as np
import matplotlib.pyplot as plt
import sys
from ar_model import *
from pyGC import *
from pySpec import *

N  = 900
Fs = 200
dt = 1.0 / Fs
C  = 0.25

idx = int(sys.argv[-1])

mat = np.load('data/wavelet_matrix.npy').item()

f   = mat['f']
S   = mat['W'][:,:,idx,:]

Snew, Hnew, Znew = wilson_factorization(S, f, Fs, Niterations=300, verbose=False)
Ix2y, Iy2x, Ixy  = granger_causality(S, Hnew, Znew) 

np.save('data/gc_wvl_'+str(idx)+'.npy', {'f': f, 'Ix2y': Ix2y, 'Iy2x': Iy2x, 'Ixy': Ixy})
