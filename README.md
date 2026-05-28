# pyGC — Granger Causality in the Frequency Domain

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

A Python package for estimating Granger Causality (GC) in the frequency domain, supporting both parametric (VAR/Yule-Walker) and non-parametric (Wilson spectral factorization) pipelines.

If you use this package, please cite:

> Lima et al. (2020). *Granger causality in the frequency domain: derivation and applications.* Revista Brasileira de Ensino de Física, 42, e20190105. [https://doi.org/10.1590/1806-9126-rbef-2019-0105](https://doi.org/10.1590/1806-9126-rbef-2019-0105)

---

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"       # pytest + coverage
pip install -e ".[jax]"       # JAX/XLA GPU-accelerated backend
pip install -e ".[notebooks]" # Jupyter notebooks
```

---

## Package structure

```
pygc/
├── __init__.py               # Public API
├── parametric.py             # Yule-Walker VAR fitting + transfer function
├── non_parametric.py         # Wilson spectral factorization
├── granger.py                # Bivariate GC, conditional GC (time + spectral)
├── covgc.py                  # Covariance-based time-domain GC
├── tools.py                  # xcorr, demean, rdet
├── ar_model.py               # Synthetic AR benchmarks (Dhamala, Baccalá)
├── _jax_backend.py           # Optional JAX/XLA Wilson factorization
├── spectral_analysis/
│   ├── fourier.py            # compute_freq, CSD, Morlet transforms (MNE)
│   └── time_frequency.py     # Welch, wavelet coherence, Gabor
└── misc/
    └── __init__.py           # smooth_spectra, downsample utilities
```

---

## Usage

### Parametric estimation (Yule-Walker)

```python
import numpy as np
from pygc import YuleWalker, compute_transfer_function, granger_causality
from pygc.spectral_analysis import compute_freq
from pygc import ar_model

Fs = 200
# Dhamala benchmark: Y drives X at 40 Hz
data = ar_model.ar_model_dhamala(N=5000, Trials=20, Fs=Fs, C=0.25)
# data shape: (2, Trials, N)

f = compute_freq(data.shape[2], Fs)
m = 2                          # VAR model order
AR  = np.zeros((m, 2, 2))
SIG = np.zeros((2, 2))

for trial in range(data.shape[1]):
    a, s = YuleWalker(data[:, trial, :], m)
    AR  += a / data.shape[1]
    SIG += s / data.shape[1]

_, S = compute_transfer_function(AR, SIG, f, Fs)
Ix2y, Iy2x, Ixy = granger_causality(S, f, Fs)
# Iy2x peaks at ~40 Hz; Ix2y is near zero
```

### Non-parametric estimation (Wilson factorization)

```python
from pygc import granger_causality
from pygc.spectral_analysis import csd_fourier, compute_freq

f = compute_freq(N, Fs)
S = csd_fourier(data, f, Fs)           # shape (2, 2, N//2+1)

Ix2y, Iy2x, Ixy = granger_causality(S, f, Fs)
```

### Conditional GC (multivariate, p ≥ 3)

```python
from pygc import conditional_granger_causality, conditional_spec_granger_causality

# Time-domain conditional GC — returns (p, p) matrix
F = conditional_granger_causality(S, f, Fs, n_jobs=-1)

# Spectral conditional GC — returns (p, p, N) matrix
cGC = conditional_spec_granger_causality(S, f, Fs, n_jobs=-1)
```

### Covariance-based GC (time domain)

```python
from pygc import covgc_time

# X: (nSources, nTime); dt: window samples; lag: lag samples; t0: zero-time index
GC = covgc_time(X, dt=100, lag=5, t0=200, n_jobs=-1)
# GC shape: (nPairs, 3) — [GC(x->y), GC(y->x), GC(x.y)]
```

### JAX/XLA accelerated backend

```python
from pygc import granger_causality, JAX_AVAILABLE

if JAX_AVAILABLE:
    Ix2y, Iy2x, Ixy = granger_causality(S, f, Fs, method='jax')
```

---

## Testing

```bash
pytest              # run all 43 tests
pytest --cov=pygc   # with coverage report
```

---

## Example notebooks

See `notebooks/` for end-to-end worked examples:

- `01_basic_granger_causality.ipynb` — parametric and non-parametric GC on the Dhamala benchmark
- `02_spectral_analysis_mne.ipynb` — MNE-based spectral estimation helpers
- `03_conditional_granger_causality.ipynb` — time-domain and spectral conditional GC on the 5-variable Baccalá model
