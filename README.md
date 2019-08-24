# pyGC
 
> Non-parametric implementarion of Granger Causality in python based on Dhamala's paper "__Estimating Granger causality from Fourier and wavelet transforms of time series data__".

> This package contains the module pyGC contains the function wilson_factorization for factorizing spectral matrices based on the implementation by Nedungadi et. al., in "__Analyzing multiple spike trains with nonparametric granger causality__" (direct translation from their Matlab code), and the function granger_causality to compute the causal influences between two signal.

> pySpec contains functions to compute power spectrum and wavelet spectrum.

> With our code we reproduce Dhamala's example, to run the code first do:

* sh generate_enviroment.sh to create data and out folders
* run ipython gc_fft.py to generate data to reproduze Figure 1 from Dhamala's paper
* run ipython gen_wvl_mat.py to save the wavelet matrix for the data
* run sbatch run.sh to generate data with time-frequency Granger Causality.
* run ipython plot_gc.py to generate figures.
