# Python Package for Granger Causality estimation (pyGC)

*Description:* This repository includes a python package to estimate Granger Causality (GC) from data, and it is structured as below:

```
pygc/
├── parametric.py
├── non_parametric.py
├── granger.py
├── tools.py
├── pySpec.py
```

Where,

- parametric.py: Contains funtions for parametric estimation of GC.
- non_parametric.py: Contains funtions for non-parametric estimation of GC.
- granger.py: Contains function to compute GC from the parameters estimated with parametric or non-parametric methods in the codes above.
- tools.py: Contains auxiliary functions.
- pySpec: Contains functions to compute power spectrum from data (with Fourier or Wavelet transforms).

## Examples of usage

### Computing GC via parametric estimation

```
import pygc.parametric
import pygc.granger

X = # Data which you want to estimate GC, dimension must be [Nvariables, Nobservations, Ntrials]
Nvars  = X.shape[0]  # For now I recommend Nvars to be equal two.
N      = X.shape[1]
Trials = X.shape[2]
Fs = # Sample frequency of the data

# Computing the frequency axis
f = pygc.pySpec.compute_freq(N, Fs)

m   = 2 # Model order you wish to fit
AR  = np.zeros([m,Nvars,Nvars])  # Store the auto-regressive coefficients
SIG = np.zeros([Nvars,Nvars])    # Store the noise covariance matrix
for T in range(Trials):
  aux1, aux2 = pygc.parametric.YuleWalker(X[:,:,T], m, maxlags=100)
	AR  += aux1/Trials
	SIG += aux2/Trials

# The below works for Nvars = 2, for more variables you should do this in a pairwise fashion
# Compute transfer matrix (H) and spectral matrix (S) both with size [Nvars, Nvars,len(f)]
H, S = pygc.parametric.compute_transfer_function(AR, SIG, f, Fs)
# Return the three components of GC
Ix2y, Iy2x, Ixy  = pygc.granger.granger_causality(S, H, SIG)
```

### Computing GC via Fourier transform (non-parametric)
```
import pygc.parametric
import pygc.granger

X = # Data which you want to estimate GC, dimension must be [Nvariables, Nobservations, Ntrials]
Nvars  = X.shape[0]  # For now I recommend Nvars to be equal two.
N      = X.shape[1]
Trials = X.shape[2]
Fs = # Sample frequency of the data

# Computing the frequency axis
f = pygc.pySpec.compute_freq(N, Fs)

# Declare spectral matrix
S = np.zeros([Nvars,Nvars,N//2+1]) + 1j*np.zeros([Nvars,Nvars,N//2+1])
# Compute the avarage spectral matrix for each Trial
# Here I use FFT to compute the spectrum you can use other methods you like
for i in range(Trials):
	S[0,0] += pygc.pySpec.cxy(X=X[0,:,T], Y=[], f=f, Fs=Fs) / Trials
	S[0,1] += pygc.pySpec.cxy(X=X[0,:,T], Y=Z[1,i,:], f=f, Fs=Fs) / Trials
	S[1,0] += pygc.pySpec.cxy(X=X[1,:,T], Y=Z[0,i,:], f=f, Fs=Fs) / Trials
	S[1,1] += pygc.pySpec.cxy(X=X[1,:,T], Y=[], f=f, Fs=Fs) / Trials

# The below works for Nvars = 2, for more variables you should do this in a pairwise fashion
# Uses Wilson factorization to estimate the transfer function and the noise covariance matrix
Snew, H, SIG = pygc.non_parametric.wilson_factorization(S, f, Fs, Niterations=30)
# Return the three components of GC
Ix2y, Iy2x, Ixy  = pygc.granger.granger_causality(S, H, SIG)
```

### Conditional GC

It is also possible to measure using the non-parametric estimation. If ```Nvars >= 3```, the procedure is similar to the example above:

```
# Now Nvars >= 3
S = np.zeros([Nvars, Nvars, N//2 + 1]) * (1 + 1j)

# Compute the spectral matrix
for trial in range(Trials):
	for i in range(Nvars):
		for j in range(nvars):
			S[i,j] += pygc.pySpec.cxy(X=Y[i,:,trial], Y=Y[j,:,trial], f=f, Fs=Fs) / Trials

# Estimate the conditional GC in time domain
F   = pygc.granger.conditional_granger_causality(S, f, Fs, Niterations = 30, verbose=False)
# Estimate the conditional GC in frequency domain
cGC = pygc.granger.conditional_spec_granger_causality(S, f, Fs, Niterations=30, tol=1e-12, verbose=True)
```

### Running an example 

In this repository you will find the code runRBEF.py, you can run it from the terminal using ```ipython runRBEF p```, ```p``` is a command line argumen and:

1. For ```p=0```, the code will generate Figure 1 from 10.1103/PhysRevLett.100.018701 (GC via Fourier transform)
2. For ```p=1```, the code will generate Figure 2 from 10.1103/PhysRevLett.100.018701 (GC via Wavelet transform)
3. For ```p=2```, the code will generate Figure 2 from doi:10.1016/j.jneumeth.2009.11.020 (Conditional GC in time domain, and the frequency domain)
4. For ```p=3```, simple example to illustrate the YuleWalker function in parametric.py to fit an AR model.
5. For ```p=4```, the same as ```p=0``` but using parametric estimation.
