# pyGC — Granger Causality in the Frequency Domain

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

A Python package for estimating Granger Causality (GC) in the frequency domain, supporting multiple spectral estimators (Fourier, Welch, Morlet, multitaper) and both NumPy and JAX/XLA backends.

If you use this package, please cite:

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
├── ar_model.py               # Synthetic AR benchmarks (Dhamala, Baccalá)
├── _jax_backend.py           # Optional JAX/XLA Wilson factorization
├── spectral_analysis/
│   ├── fourier.py            # compute_freq, CSD, Morlet transforms (MNE)
│   └── time_frequency.py     # Welch, multitaper, wavelet/Gabor transforms
└── misc/
    └── smooth_spectra.py     # smooth_spectra, downsample utilities
```

---

## Usage

All main functions accept raw signal data `X` and a sampling rate `fs`. The cross-spectral density is computed internally via the chosen `spectral_method`.

### Bivariate GC

```python
import numpy as np
from pygc import granger_causality
from pygc import ar_model

Fs = 200
# Dhamala benchmark: Y drives X at 40 Hz
data = ar_model.ar_model_dhamala(N=5000, Trials=20, Fs=Fs, C=0.25)
# data shape: (2, Trials, N)

# Non-parametric (Fourier CSD + Wilson factorization) — default
Ix2y, Iy2x, Ixy, f = granger_causality(data, fs=Fs, spectral_method='fourier')
# Iy2x peaks at ~40 Hz; Ix2y is near zero
```

Available spectral methods: `'fourier'` (default), `'welch'`, `'morlet'`, `'multitaper'`.

```python
# Welch CSD
Ix2y, Iy2x, Ixy, f = granger_causality(
    data, fs=Fs, spectral_method='welch',
    spectral_params={'nperseg': 512}
)

# Morlet CSD (requires explicit frequency axis)
freqs = np.linspace(1, 80, 80)
Ix2y, Iy2x, Ixy, f = granger_causality(
    data, fs=Fs, spectral_method='morlet',
    spectral_params={'freqs': freqs, 'n_cycles': 7.0}
)

# Multitaper CSD
Ix2y, Iy2x, Ixy, f = granger_causality(
    data, fs=Fs, spectral_method='multitaper',
    spectral_params={'bandwidth': 4.0}
)
```

### Parametric estimation (Yule-Walker)

Use `YuleWalker` and `compute_transfer_function` to obtain the VAR-based cross-spectral matrix:

```python
import numpy as np
from pygc import YuleWalker, compute_transfer_function
from pygc.spectral_analysis import compute_freq

Fs = 200
data = ar_model.ar_model_dhamala(N=5000, Trials=20, Fs=Fs, C=0.25)
# data shape: (2, Trials, N)

f  = compute_freq(data.shape[2], Fs)
m  = 2                          # VAR model order
AR  = np.zeros((m, 2, 2))
SIG = np.zeros((2, 2))

for trial in range(data.shape[1]):
    a, s = YuleWalker(data[:, trial, :], m)
    AR  += a / data.shape[1]
    SIG += s / data.shape[1]

H, S = compute_transfer_function(AR, SIG, f, Fs)
# H: (2, 2, n_freq) transfer function
# S: (2, 2, n_freq) cross-spectral matrix
```

### Conditional GC (multivariate, p ≥ 3)

```python
from pygc import conditional_granger_causality, conditional_spec_granger_causality

# Time-domain conditional GC — returns (p, p) matrix
F = conditional_granger_causality(data, fs=Fs, n_jobs=-1)

# Spectral conditional GC — returns ((p, p, n_freq) matrix, frequency axis)
cGC, f = conditional_spec_granger_causality(data, fs=Fs, n_jobs=-1)
```

Both functions support the same `spectral_method` and `spectral_params` arguments as `granger_causality`.

### JAX/XLA accelerated backend

```python
from pygc import granger_causality, JAX_AVAILABLE

if JAX_AVAILABLE:
    Ix2y, Iy2x, Ixy, f = granger_causality(data, fs=Fs, backend='jax')
```

### `ensure_stability` parameter

All three main functions accept `ensure_stability=True` (default), which clips near-zero or negative diagonal entries of the noise covariance after Wilson factorization to improve numerical stability.

---

## Testing

```bash
pytest              # run all 38 tests
pytest --cov=pygc   # with coverage report
```

---

## Example notebooks

See `notebooks/` for end-to-end worked examples:

- `01_basic_granger_causality.ipynb` — bivariate GC on the Dhamala benchmark
- `02_spectral_analysis_mne.ipynb` — MNE-based spectral estimation helpers
- `03_conditional_granger_causality.ipynb` — time-domain and spectral conditional GC on the 5-variable Baccalá model
- `04_benchmarks.ipynb` — performance benchmarks across backends and model sizes
- `05_spectral_methods_comparison.ipynb` — comparison of Fourier, Welch, Morlet, and multitaper estimators
- `06_example_with_eletrophysiological_data.ipynb` — GC applied to real electrophysiological recordings
