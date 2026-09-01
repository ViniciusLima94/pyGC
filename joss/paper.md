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
  - name: Institut de Neurosciences de la Timone, UMR 7289, CNRS, Aix-Marseille Université, Marseille 13005, France
    index: 1
  - name: Institute for Advanced Simulation (IAS-6), Jülich Research Centre, Jülich, Germany
    index: 2
  - name: Department of Biological Sciences, Florida Atlantic University, Jupiter, FL 33458, USA
    index: 3

date: 28 August 2026
bibliography: paper.bib
---

# Summary

`pyGC` is an open-source Python library for estimating Granger Causality (GC) in the
frequency domain from multivariate time-series data, for pairwise as well as conditional
(multivariate) analyses. Each estimate is a frequency-resolved influence spectrum for a
directed channel pair. A time-domain scalar summary per pair is also provided. Two
estimation pathways are available. The default _non-parametric_ pathway applies Wilson
spectral factorization to a directly estimated cross-spectral matrix
[@wilson1972factorization; @dhamala2008estimating] and needs no model-order selection.
The _parametric_ pathway instead fits a Vector Auto-Regressive (VAR) model via the
Yule-Walker equations and derives the transfer function analytically. Four cross-spectral estimators are
available for the non-parametric pathway: a trial-averaged FFT periodogram, Welch's
overlapping-window method, a Morlet wavelet CSD, and a multitaper (DPSS) CSD. Every GC
function returns a labelled `xarray.Dataset` so that directions, channel pairs, and
frequency axes stay attached to the numbers, and an optional JAX back-end provides
CPU/GPU acceleration. The library targets neuroscience applications (EEG, MEG, LFP) but
is applicable to any domain where directional information flow between signals needs to
be quantified.

# Statement of Need

Granger Causality [@granger1969investigating] is a standard tool for inferring directed
connectivity from neural time series. Frequency-domain formulations
[@geweke1982measurement; @dhamala2008estimating] are particularly popular because they
reveal the frequency band at which influence operates, a detail lost in time-domain
summaries.

While individual routines exist in MATLAB toolboxes such as MVGC [@barnett2014mvgc]
and in scattered Python snippets, Python users have lacked a single tested,
pip-installable library for non-parametric, frequency-resolved GC: Wilson factorization
of a directly estimated cross-spectrum, a choice of cross-spectral estimators, and
conditional spectral GC, alongside the parametric pathway and optional GPU acceleration
behind one consistent API. `pyGC` fills this gap, lowering the barrier to reproducible
frequency-domain connectivity analysis in the scientific Python ecosystem.

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

where $\tilde{H}_ {YY} = H_{YY} + (\Sigma_{YX}/\Sigma_{YY})\ H_{YX}$ and
$H_{XX}^{\circ} = H_{XX} + (\Sigma_{XY}/\Sigma_{XX})\ H_{XY}$ are the intrinsic
transfer functions after absorbing off-diagonal noise correlations.

## Wilson Spectral Factorization

When VAR model fitting is not desired, the transfer function and noise covariance can
be recovered directly from a non-parametric estimate of $\mathbf{S}(f)$ via Wilson
spectral factorization [@wilson1972factorization]. The algorithm iteratively finds an
analytic function $\boldsymbol{\Psi}(f)$ such that
$\mathbf{S}(f) = \boldsymbol{\Psi}(f)\ \boldsymbol{\Psi}^*(f)$, yielding
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
  (`YuleWalker_multitrial`) data, and analytic transfer-function computation
  (`compute_transfer_function`).
- `non_parametric` — vectorised Wilson spectral factorization (`wilson_factorization`).
- `granger` — pairwise GC (`granger_causality`), pairwise spectral GC
  (`spectral_granger_causality`), conditional time-domain GC
  (`conditional_granger_causality`), and conditional spectral GC
  (`spectral_conditional_granger_causality`). Every function accepts raw signal data, a
  `model` argument (`'nonparametric'` or `'parametric'`), and a `spectral_method`
  argument (`'fourier'`, `'welch'`, `'morlet'`, or `'multitaper'`), so estimation is
  performed internally.
- `output` — assembly of raw results into labelled `xarray.Dataset` objects.
- `ar_model` — synthetic benchmark processes: the two-variable AR model of
  @dhamala2008estimating and the five-variable model of @baccala2001partial.
- `spectral_analysis` — spectral estimation helpers (Fourier CSD, Morlet wavelet CSD,
  Welch cross-spectrum, multitaper DPSS CSD, Gabor spectrum) built on MNE-Python
  [@gramfort2013mne] and SciPy.
- `_jax_backend` — optional JAX/XLA back-end with a JIT-compiled Wilson loop for
  CPU/GPU acceleration.

# Implementation Details

## Unified Parametric / Non-Parametric Interface

Every GC function accepts `model='nonparametric'` (default) or `model='parametric'`. The
non-parametric pathway estimates the cross-spectral matrix with the chosen
`spectral_method` and factorises it with the Wilson algorithm. The parametric pathway
fits a VAR model of the requested `order` by Yule-Walker (single- and multi-trial) and
derives the transfer function, cross-spectrum, and innovations covariance analytically.
Both pathways share the downstream GC decomposition, so switching between them changes
only the `model` (and, for the parametric case, `order`) argument.

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
  Slepian tapers) via `mne.time_frequency.csd_array_multitaper`; the multitaper window
  bandwidth (in Hz) is set by `bandwidth` in `spectral_params`. Compared to the
  single-taper periodogram, multitaper estimates trade a modest increase in frequency
  smoothing for substantially reduced variance [@percival1993spectral], making them
  particularly robust for short or noisy epochs.

## JAX Back-End

When JAX [@jax2018github] is installed, `wilson_factorization_jax` exposes a
JIT-compiled version of the entire Wilson loop via `jax.lax.while_loop`. The convergence
condition is encoded as a JAX boolean predicate so that the compiled kernel exits as
soon as the matrix 1-norm drops below `tol` without returning to Python between
iterations. CPU and GPU execution are both supported and selected automatically by the
JAX device back-end.

## Performance and Parallelism

The transfer-function and Wilson-factorization kernels are vectorised over frequency
bins (and, for the multi-trial VAR fit, over trials) rather than looping in Python,
which is what makes whole-dataset estimation and the JAX back-end practical.
Independent sub-problems are then dispatched with `joblib.Parallel`: conditional
(spectral) GC runs one factorization per channel on a thread pool
(`prefer='threads'`), which avoids serialisation overhead for NumPy-heavy workloads.

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

The same call produces the parametric estimate with `model='parametric'` (plus a VAR
`order`) or the JAX-accelerated non-parametric backend with `backend='jax'`:

```python
ds_par = spectral_granger_causality(X, Fs, model='parametric', order=20)
ds_jax = spectral_granger_causality(X, Fs, backend='jax')
```

# Testing

`pyGC` ships with a pytest suite of 38 tests covering:

- Correctness of VAR fitting (Yule-Walker coefficient recovery and noise covariance
  symmetry).
- Spectral factorization convergence and reconstruction error.
- GC direction recovery on the Dhamala benchmark model ($I_{Y \to X} > I_{X \to Y}$),
  and near-zero GC for an uncoupled system.
- API consistency between the NumPy and JAX back-ends when JAX is available.
- Validation of the `fourier`, `welch`, `morlet`, and `multitaper` spectral estimators,
  plus error handling for invalid arguments.

Tests are run with `pytest` and a coverage report is generated via `pytest-cov`.

# Related Software

**MVGC** [@barnett2014mvgc] covers parametric time- and frequency-domain GC,
pairwise and conditional, but is MATLAB-only and has no non-parametric pathway.
In Python, **MNE-Connectivity** [@mne_connectivity] provides frequency-domain GC
via state-space models [@barnett2015statespace], without Wilson factorization or
conditioning on all remaining channels. **Elephant** [@elephant18] offers
parametric time-domain GC, pairwise and conditional, plus a non-parametric
spectral GC for the pairwise case. **nitime** [@rokem2009nitime] includes
parametric VAR-based spectral GC within a broader time-series analysis toolbox. `pyGC`
complements these tools, providing both estimation pathways and conditional spectral GC
through one consistent, pip-installable Python API, with optional JAX/GPU acceleration.

# Acknowledgements

The theoretical foundations of this package were developed alongside the tutorial paper @lima2020granger.

# AI usage disclosure

Claude (Anthropic, Claude Sonnet 4.6) was used for code assistance during debugging, for proofreading
and editing the text of this paper; all AI-generated code and text were reviewed, tested, and verified by the authors, who take full responsibility for the correctness of the software and paper.

# References
