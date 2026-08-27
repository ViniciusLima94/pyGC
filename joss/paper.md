---
title: "pyGC: A Python Package for Parametric and Non-Parametric Frequency-Domain Granger Causality Estimation"
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
  - name: Renan Oliveira Shimoura
    orcid: 0000-0002-6580-5999
    corresponding: false
    affiliation: 2
  - name: Rodrigo Felipe de Oliveira Pena
    orcid: 0000-0002-2037-9746
    corresponding: false
    affiliation: 3

affiliations:
  - name: Institut de Neurosciences de La Timone, UMR 7289, CNRS, Aix-Marseille Universit´e, Marseille 13005, France
    index: 1
  - name: Institute for Advanced Simulation (IAS-6), Jülich Research Centre, Jülich, Germany
    index: 2
  - name: Department of Biological Sciences, Florida Atlantic University, Jupiter, FL 33458, USA
    index: 3

date: 29 May 2026
bibliography: paper.bib
---

# Summary

`pyGC` is an open-source Python library for estimating Granger Causality (GC) from
multivariate time-series data, both in the time domain (a scalar influence per channel
pair) and in the frequency domain (a frequency-resolved influence spectrum), for
pairwise as well as conditional (multivariate) analyses. Two estimation pathways are
provided and selected through a single `model` argument: the _parametric_ pathway fits
a Vector Auto-Regressive (VAR) model via the Yule-Walker equations and derives the
transfer function analytically, while the _non-parametric_ pathway applies Wilson
spectral factorization to a directly estimated cross-spectral matrix
[@wilson1972factorization; @dhamala2008estimating]. Four spectral estimators are
integrated directly into the GC pipeline: a trial-averaged FFT periodogram, Welch's
overlapping-window method, a Morlet wavelet CSD, and a multitaper (DPSS) CSD. The VAR
fit, transfer-function evaluation, and Wilson iteration are vectorised over frequency
bins (and, for the multi-trial VAR fit, over trials), and every GC function returns a
labelled `xarray.Dataset` so that directions, channel pairs, and frequency axes stay
attached to the numbers. An optional JAX back-end JIT-compiles the entire Wilson
iteration loop via XLA for CPU/GPU acceleration. The library targets neuroscience
applications (EEG, MEG, LFP) but is applicable to any domain where directional
information flow between signals needs to be quantified.

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
NumPy/SciPy API offering parametric and non-parametric estimation (a single `model`
switch), time- and frequency-domain measures, and pairwise and conditional GC through a
consistent set of functions; a full pytest suite; and optional JAX acceleration.
Results are returned as labelled `xarray.Dataset` objects, keeping channel pairs,
source/target directions, and frequency axes attached to the estimates.

# Background

## Granger Causality in the Frequency Domain

Given a stationary multivariate process $\mathbf{X}(t)$ recorded with sampling rate $F_s$, its VAR representation of
order $m$ is

$$\mathbf{X}(t) = \sum_{k=1}^{m} \mathbf{A}_k \mathbf{X}(t-k) + \boldsymbol{\varepsilon}(t) \qquad \boldsymbol{\varepsilon}(t) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\mathbf{A}_k$ are the coefficient matrices and $\boldsymbol{\Sigma}$ is the
noise covariance. The transfer function is
$\mathbf{H}(f) = \bigl[\mathbf{I} - \sum_k \mathbf{A}_k e^{-2\pi i k f / F_s}\bigr]^{-1}$
and the cross-spectral matrix is
$\mathbf{S}(f) = \mathbf{H}(f)\,\boldsymbol{\Sigma}\,\mathbf{H}^*(f)$.

For a bivariate system $(X, Y)$ the pairwise spectral GC measures are
[@geweke1982measurement; @lima2020granger]:

$$I_{X \to Y}(f) = \ln \frac{S_{YY}(f)}{\tilde{H}_{YY}(f)\ \Sigma_{YY}\ \tilde{H}_{YY}^*(f)},$$

$$I_{Y \to X}(f) = \ln \frac{S_{XX}(f)}{H_{XX}^{\circ}(f)\ \Sigma_{XX}\ {H_{XX}^{\circ}}^*(f)},$$

where $\tilde{H}_ {YY} = H_{YY} + (\Sigma_{YX}/\Sigma_{XX})\ H_{YX}$ and
$H_{XX}^{\circ} = H_{XX} + (\Sigma_{XY}/\Sigma_{YY})\ H_{XY}$ are the intrinsic
transfer functions after absorbing off-diagonal noise correlations.

## Wilson Spectral Factorization

When VAR model fitting is not desired, the transfer function and noise covariance can
be recovered directly from a non-parametric estimate of $\mathbf{S}(f)$ via Wilson
spectral factorization [@wilson1972factorization]. The algorithm iteratively finds an
analytic function $\boldsymbol{\Psi}(f)$ such that
$\mathbf{S}(f) = \boldsymbol{\Psi}(f)\\boldsymbol{\Psi}^*(f)$, yielding
$\mathbf{H}(f)$ and $\boldsymbol{\Sigma}$ without assuming a finite-order VAR model.

## Conditional GC

For multivariate ($p > 2$) systems, `pyGC` provides conditional GC [@geweke1984measures],
which removes shared driving by conditioning on all remaining channels. Separate
factorizations are run on each $(p{-}1)$-dimensional reduced model; the results are
assembled into a $p \times p$ GC matrix, available both as a time-domain scalar summary
(`conditional_granger_causality`) and as a frequency-resolved spectrum
(`spectral_conditional_granger_causality`). Each reduced model is refit from the raw
data — with `model='parametric'`, by refitting the reduced VAR rather than slicing the
full-model coefficients, which would not give the correct reduced model. These
reduced-model estimates are embarrassingly parallel and are executed with `joblib` when
`n_jobs > 1`. A pairwise time-domain GC (`granger_causality`) is obtained the same way,
fitting each channel pair on its own two-channel subset.

# Package Structure

`pyGC` is organised as a single installable package (`pygc`) with the following modules:

- `parametric` — Yule-Walker VAR fitting for single-trial (`YuleWalker`) and multi-trial
  (`YuleWalker_multitrial`, vectorised with `numpy.einsum`) data, plus analytic
  transfer-function and cross-spectrum computation (`compute_transfer_function`),
  vectorised over all frequency bins via a single `einsum` and a batched matrix inversion.
- `non_parametric` — vectorised Wilson spectral factorization (`wilson_factorization`),
  with every frequency-indexed loop replaced by batched `linalg.inv`, batched `matmul`,
  and axis-wise FFTs.
- `granger` — pairwise time-domain GC (`granger_causality`), pairwise spectral GC
  (`spectral_granger_causality`), conditional time-domain GC
  (`conditional_granger_causality`), and conditional spectral GC
  (`spectral_conditional_granger_causality`). Every function takes raw signal data, a
  `model` argument (`'nonparametric'` or `'parametric'`), and a `spectral_method`
  argument (`'fourier'`, `'welch'`, `'morlet'`, or `'multitaper'`); the pairwise
  functions additionally accept a `pairs` list and the conditional functions a `targets`
  list. Spectral estimation and — for `model='parametric'` — VAR fitting are performed
  internally.
- `output` — assembles raw result arrays into labelled `xarray.Dataset` objects
  (`build_granger_dataset`, `build_conditional_gc_dataset`,
  `build_conditional_spec_gc_dataset`) carrying direction, channel-pair, and frequency
  coordinates.
- `ar_model` — synthetic benchmark processes: the two-variable AR model of
  @dhamala2008estimating (`ar_model_dhamala`) and the five-variable model of
  @baccala2001partial (`ar_model_baccala`).
- `spectral_analysis` — sub-package providing the cross-spectral estimators exposed
  through `spectral_method`: a trial-averaged Fourier periodogram (NumPy) and Welch's
  method (SciPy), plus Morlet-wavelet and multitaper (DPSS) cross-spectra via MNE-Python
  [@gramfort2013mne].
- `misc` — internal helper for spectral smoothing (`smooth_spectra`).
- `_jax_backend` — optional JAX/XLA back-end with a JIT-compiled Wilson loop
  (`wilson_factorization_jax`) for CPU/GPU acceleration.

# Implementation Details

## Unified Parametric / Non-Parametric Interface

Every GC function accepts `model='nonparametric'` (default) or `model='parametric'`. In
the non-parametric case the cross-spectral matrix is estimated with the chosen
`spectral_method` and factorised with the Wilson algorithm. In the parametric case a VAR
model of the requested `order` is fitted by Yule-Walker — `YuleWalker` for single-trial
input `(nvars, N)` and the `einsum`-vectorised `YuleWalker_multitrial` for
`(trials, nvars, N)` input — and the transfer function, cross-spectrum, and innovations
covariance are obtained analytically from `compute_transfer_function`. The two pathways
share the downstream GC decomposition, so switching between them requires changing only
the `model` (and, for the parametric case, `order`) argument.

## Vectorised Estimation

The performance-critical kernels are written without Python-level frequency loops.
`compute_transfer_function` builds the lagged phase factors with a single `einsum` and
inverts $\mathbf{I} - \sum_k \mathbf{A}_k e^{-2\pi i k f / F_s}$ for all frequencies in
one batched `numpy.linalg.inv` call. `wilson_factorization` likewise replaces its
per-frequency loops with batched `linalg.inv`, batched `matmul`, and axis-wise FFTs,
for a substantial speedup over a per-frequency Python loop.

## Numerical Stability

`ensure_stability=True` (default) adds a microscopic term — scaled by machine epsilon
and the largest spectral magnitude — to the diagonal of the Hermitian-extended spectral
matrix before factorization. This prevents singular intermediate matrices across
different BLAS/LAPACK backends without measurably affecting the result.

## Labelled Output

All GC routines return an `xarray.Dataset` built by the `output` module:
`spectral_granger_causality` yields `x2y`, `y2x`, and `xy` over (`pairs`, `freq`); the
time-domain `granger_causality` and `conditional_granger_causality` yield an
(`source`, `target`) influence matrix `F`; and `spectral_conditional_granger_causality`
yields `GC` over (`target`, `source`, `freq`). Labels passed via `channel_names`
propagate to the coordinates, so downstream code can select results by name rather than
by positional index.

## Integrated Spectral Estimation

Rather than requiring users to pre-compute a cross-spectral matrix, `pyGC` integrates
four spectral estimators directly into the GC pipeline via the `spectral_method`
parameter:

- `'fourier'` — trial-averaged FFT periodogram; frequency resolution $= F_s / N$.
- `'welch'` — Welch overlapping-window average via `scipy.signal.csd`; controllable
  via `nperseg` and `window` in `spectral_params`.
- `'morlet'` — Morlet wavelet CSD time-averaged across trials and time, computed via
  MNE-Python; frequency grid specified in `spectral_params`.
- `'multitaper'` — multitaper CSD using discrete prolate spheroidal sequences (DPSS /
  Slepian tapers) via `mne.time_frequency.csd_array_multitaper`; the resolution
  bandwidth (in Hz) is set by `bandwidth` in `spectral_params`. Compared to the
  single-taper periodogram, multitaper estimates trade a modest increase in frequency
  smoothing for substantially reduced variance [@percival1993spectral], making them
  particularly robust for short or noisy epochs.

## JAX Back-End

When JAX [@jax2018github] is installed, `wilson_factorization_jax` exposes a
JIT-compiled version of the entire Wilson loop via `jax.lax.while_loop`. The
convergence condition is encoded as a JAX boolean predicate so that the compiled
kernel exits as soon as the matrix 1-norm drops below `tol` without returning to
Python between iterations. CPU and GPU execution are both supported and selected
automatically by the JAX device back-end.

## Parallelism

Independent sub-problems are dispatched with `joblib.Parallel`. Conditional (spectral)
GC requires one factorization per channel (reduced models), run on a thread pool
(`prefer='threads'`), which avoids serialisation overhead for NumPy-heavy workloads.
For pairwise GC the per-pair work is distributed through a selectable `joblib_backend`
(default `'loky'`): the time-domain routine refits each two-channel subset
independently, while the frequency-domain routine shares one factorization and
parallelises only the per-pair GC decomposition. The number of workers is controlled by
`n_jobs`.

# Usage Example

The following snippet demonstrates the bivariate non-parametric workflow using the
built-in Dhamala benchmark model, where channel $Y$ drives channel $X$ at 40 Hz:

```python
import numpy as np
from pygc.ar_model import ar_model_dhamala
from pygc import spectral_granger_causality

Fs   = 200
data = ar_model_dhamala(N=5000, Trials=50, Fs=Fs, C=0.25)
# data shape: (2, Trials, N); transpose to (Trials, 2, N)
X = data.transpose(1, 0, 2)

ds = spectral_granger_causality(X, Fs, spectral_method='welch', pairs=[(0, 1)])
# ds['y2x'] peaks at ~40 Hz; ds['x2y'] is near zero.
peak_freq = ds.freq.values[ds['y2x'].values[0].argmax()]
```

The parametric pathway is selected by passing `model='parametric'` with a VAR `order`,
and the JAX-accelerated non-parametric backend by passing `backend='jax'`; both keep the
same call signature and `xarray.Dataset` output:

```python
ds_par = spectral_granger_causality(X, Fs, model='parametric', order=20)
ds_jax = spectral_granger_causality(X, Fs, backend='jax')
```

# Testing

`pyGC` ships with a pytest suite of 38 tests covering:

- VAR fitting: Yule-Walker coefficient recovery on a known AR(1) process, and residual
  covariance shape and symmetry.
- Wilson spectral factorization: output shape, reconstruction error against the input
  spectrum, noise-covariance symmetry, and the identity-spectrum → identity
  noise-covariance limit.
- Transfer-function and cross-spectrum output: shape, Hermitian symmetry, and
  real-positive diagonal spectra.
- Standalone spectral estimators: frequency-axis length, range and monotonicity;
  real-positive auto-spectra and conjugate symmetry of the Fourier CSD; a power peak at
  a known input frequency for the Morlet transform; Welch output shape and
  invalid-scaling handling.
- GC direction recovery on the Dhamala benchmark ($I_{Y \to X} > I_{X \to Y}$) and
  near-zero GC for an uncoupled system.
- Output shapes for the `fourier`, `welch`, and `morlet` spectral methods, including
  multi-trial input, plus error handling for invalid `backend` / `spectral_method` and a
  missing Morlet frequency axis.
- Agreement between the NumPy and JAX back-ends (shapes, values, noise-covariance
  symmetry) when JAX is available.

Tests are run with `pytest` and a coverage report is generated via `pytest-cov`.

# Related Software

**MVGC** [@barnett2014mvgc] is a comprehensive MATLAB toolbox for GC analysis but
does not provide a Python interface. **MNE-Connectivity** provides spectral
connectivity measures in Python but does not implement Wilson factorization or
conditional spectral GC. **nitime** offers VAR-based GC but has not been actively
maintained. **Elephant** [@elephant18] offers time-domain parametric GC estimation,
but restricts its non-parametric approach to pairwise spectral GC. `pyGC` complements
these tools by providing a modern, tested, pip-installable Python library that covers
the full non-parametric pipeline with multiple spectral estimators and GPU acceleration.

# Acknowledgements

The theoretical foundations of this package were developed alongside the tutorial paper
@lima2020granger. The Wilson factorization implementation is based on the algorithm
described in @dhamala2008estimating.

# AI usage disclosure

Claude (Anthropic, Claude Sonnet 4.6) was used for code assistance during debugging, for proofreading
and editing the text of this paper; all AI-generated code and text were reviewed, tested, and verified by the authors, who take full responsibility for the correctness of the software and paper.

# References
