---
title: 'pyGC: A Python Package for Parametric and Non-Parametric Frequency-Domain Granger Causality Estimation'
tags:
  - Python
  - Granger Causality
  - neuroscience
  - time series
  - spectral analysis
authors:
  - name: Vinicius Lima Cordeiro
    orcid: 0000-0001-7115-9041
    corresponding: true
    affiliation: 1

affiliations:
  - name: Institut de Neurosciences de La Timone, UMR 7289, CNRS, Aix-Marseille Universit´e, Marseille 13005, France
    index: 1

date: 29 May 2026
bibliography: paper.bib
---

# Summary

`pyGC` is an open-source Python library for estimating Granger Causality (GC) in the
frequency domain from multivariate time-series data. The package implements both the
*parametric* pathway — fitting a Vector Auto-Regressive (VAR) model via the Yule-Walker
equations and deriving the transfer function analytically — and the *non-parametric*
pathway based on Wilson spectral factorization of a directly estimated cross-spectral
matrix [@wilson1972factorization; @dhamala2008estimating]. Three spectral estimators are
integrated directly into the GC pipeline: a trial-averaged FFT periodogram, Welch's
overlapping-window method, and a Morlet wavelet CSD. An optional JAX back-end
JIT-compiles the entire Wilson iteration loop via XLA for CPU/GPU acceleration.
The library targets neuroscience applications (EEG, MEG, LFP) but is applicable to
any domain where directional information flow between signals needs to be quantified.

# Statement of Need

Granger Causality [@granger1969investigating] is a standard tool for inferring directed
connectivity from neural time series. Frequency-domain formulations
[@geweke1982measurement; @dhamala2008estimating] are particularly popular because they
reveal the frequency band at which influence operates, a detail lost in time-domain
summaries.

While individual routines exist in MATLAB toolboxes such as MVGC [@barnett2014mvgc]
and in scattered Python snippets, a cohesive, tested, and pip-installable Python library
that covers both estimation pathways, conditional GC, multiple spectral estimators, and
GPU-accelerated computation has been lacking. `pyGC` fills this gap with a clean
NumPy/SciPy API, a full pytest suite, and optional JAX acceleration.

# Background

## Granger Causality in the Frequency Domain

Given a stationary multivariate process $\mathbf{X}(t)$, its VAR representation of
order $m$ is

$$\mathbf{X}(t) = \sum_{k=1}^{m} \mathbf{A}_k \mathbf{X}(t-k) + \boldsymbol{\varepsilon}(t), \qquad \boldsymbol{\varepsilon}(t) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}),$$

where $\mathbf{A}_k$ are the coefficient matrices and $\boldsymbol{\Sigma}$ is the
noise covariance. The transfer function is
$\mathbf{H}(f) = \bigl[\mathbf{I} - \sum_k \mathbf{A}_k^{\top} e^{-2\pi i k f / F_s}\bigr]^{-1}$
and the cross-spectral matrix is
$\mathbf{S}(f) = \mathbf{H}(f)\,\boldsymbol{\Sigma}\,\mathbf{H}^*(f)$.

For a bivariate system $(X, Y)$ the pairwise spectral GC measures are
[@geweke1982measurement; @lima2020granger]:

$$I_{X \to Y}(f) = \ln \frac{S_{YY}(f)}{\tilde{H}_{YY}(f)\,\Sigma_{YY}\,\tilde{H}_{YY}^*(f)},$$

$$I_{Y \to X}(f) = \ln \frac{S_{XX}(f)}{H_{XX}^{\circ}(f)\,\Sigma_{XX}\,{H_{XX}^{\circ}}^*(f)},$$

where $\tilde{H}_{YY} = H_{YY} + (\Sigma_{YX}/\Sigma_{XX})\,H_{XY}$ and
$H_{XX}^{\circ} = H_{XX} + (\Sigma_{XY}/\Sigma_{YY})\,H_{YX}$ are the intrinsic
transfer functions after absorbing off-diagonal noise correlations.

## Wilson Spectral Factorization

When VAR model fitting is not desired, the transfer function and noise covariance can
be recovered directly from a non-parametric estimate of $\mathbf{S}(f)$ via Wilson
spectral factorization [@wilson1972factorization]. The algorithm iteratively finds an
analytic function $\boldsymbol{\Psi}(f)$ such that
$\mathbf{S}(f) = \boldsymbol{\Psi}(f)\,\boldsymbol{\Psi}^*(f)$, yielding
$\mathbf{H}(f)$ and $\boldsymbol{\Sigma}$ without assuming a finite-order VAR model.

## Conditional GC

For multivariate ($p > 2$) systems, `pyGC` provides conditional GC [@geweke1984measures],
which removes shared driving by conditioning on all remaining channels. Separate Wilson
factorizations are run on each $(p{-}1)$-dimensional reduced model; the results are
assembled into a $p \times p$ GC matrix. These reduced-model factorizations are
embarrassingly parallel and are executed with `joblib` when `n_jobs > 1`.

# Package Structure

`pyGC` is organised as a single installable package (`pygc`) with the following modules:

- `parametric` — Yule-Walker VAR fitting (`YuleWalker`) and transfer-function
  computation (`compute_transfer_function`).
- `non_parametric` — vectorised Wilson spectral factorization (`wilson_factorization`).
- `granger` — bivariate GC (`granger_causality`), conditional time-domain GC
  (`conditional_granger_causality`), and conditional spectral GC
  (`conditional_spec_granger_causality`). All three functions accept raw signal data
  and a `spectral_method` argument (`'fourier'`, `'welch'`, or `'morlet'`) so that
  spectral estimation is performed internally.
- `ar_model` — synthetic benchmark processes: the two-variable AR model of
  @dhamala2008estimating and the five-variable model of @baccala2001partial.
- `spectral_analysis` — spectral estimation helpers (Fourier CSD, Morlet wavelet CSD,
  Welch cross-spectrum, Gabor spectrum) built on MNE-Python [@gramfort2013mne] and
  SciPy.
- `_jax_backend` — optional JAX/XLA back-end with a JIT-compiled Wilson loop for
  CPU/GPU acceleration.

# Implementation Details


## Integrated Spectral Estimation

Rather than requiring users to pre-compute a cross-spectral matrix, `pyGC` integrates
three spectral estimators directly into the GC pipeline via the `spectral_method`
parameter:

- `'fourier'` — trial-averaged FFT periodogram; frequency resolution $= F_s / N$.
- `'welch'` — Welch overlapping-window average via `scipy.signal.csd`; controllable
  via `nperseg` and `window` in `spectral_params`.
- `'morlet'` — Morlet wavelet CSD time-averaged across trials and time, computed via
  MNE-Python; frequency grid specified in `spectral_params`.

## JAX Back-End

When JAX [@jax2018github] is installed, `wilson_factorization_jax` exposes a
JIT-compiled version of the entire Wilson loop via `jax.lax.while_loop`. The
convergence condition is encoded as a JAX boolean predicate so that the compiled
kernel exits as soon as the matrix 1-norm drops below `tol` without returning to
Python between iterations. CPU and GPU execution are both supported and selected
automatically by the JAX device back-end.

## Parallelism

Conditional (spectral) GC requires one Wilson factorization per channel (reduced
models). These are independent and are dispatched via `joblib.Parallel` with a thread
pool (`prefer='threads'`), which avoids serialisation overhead for NumPy-heavy
workloads.

# Usage Example

The following snippet demonstrates the bivariate non-parametric workflow using the
built-in Dhamala benchmark model, where channel $Y$ drives channel $X$ at 40 Hz:

```python
import numpy as np
from pygc.ar_model import ar_model_dhamala
from pygc import granger_causality

Fs   = 200
data = ar_model_dhamala(N=5000, Trials=50, Fs=Fs, C=0.25)
# data shape: (2, Trials, N); transpose to (Trials, 2, N)
X = data.transpose(1, 0, 2)

Ix2y, Iy2x, _ = granger_causality(X, Fs, spectral_method='welch')
# Iy2x peaks at ~40 Hz; Ix2y is near zero.
```

The parametric pathway (Yule-Walker + transfer function) and the JAX-accelerated
backend are selected by passing `spectral_method='fourier'` with pre-fitted AR
coefficients, or `backend='jax'`, respectively.

# Testing

`pyGC` ships with a pytest suite of 38 tests covering:

- Correctness of VAR fitting (Yule-Walker residuals and noise covariance symmetry).
- Spectral factorization convergence and reconstruction error.
- GC direction recovery on the Dhamala and Baccalá benchmark models
  ($I_{Y \to X} > I_{X \to Y}$).
- Conditional GC matrix sparsity on the Baccalá five-variable model.
- API consistency between the NumPy and JAX back-ends when JAX is available.
- Validation of all three spectral estimators (`fourier`, `welch`, `morlet`).

Tests are run with `pytest` and a coverage report is generated via `pytest-cov`.

# Related Software

**MVGC** [@barnett2014mvgc] is a comprehensive MATLAB toolbox for GC analysis but
does not provide a Python interface. **MNE-Connectivity** provides spectral
connectivity measures in Python but does not implement Wilson factorization or
conditional spectral GC. **nitime** offers VAR-based GC but has not been actively
maintained. `pyGC` complements these tools by providing a modern, tested,
pip-installable Python library that covers the full non-parametric pipeline with
multiple spectral estimators and GPU acceleration.

# Acknowledgements

The theoretical foundations of this package were developed alongside the tutorial paper
@lima2020granger. The Wilson factorization implementation is based on the algorithm
described in @dhamala2008estimating.

# References
