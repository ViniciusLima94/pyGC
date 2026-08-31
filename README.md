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
├── output.py                 # Packages results into labeled xarray.Dataset objects
├── ar_model.py                # Synthetic AR benchmarks (Dhamala, Baccalá)
├── _jax_backend.py           # Optional JAX/XLA Wilson factorization
├── spectral_analysis/
│   ├── fourier.py            # compute_freq, CSD, Morlet transforms (MNE)
│   └── time_frequency.py     # Welch, multitaper, wavelet/Gabor transforms
└── misc/
    └── smooth_spectra.py     # smooth_spectra, downsample utilities
```

---

## Usage

All main functions accept raw signal data `X` and a sampling rate `fs`. The cross-spectral density is computed internally via the chosen `spectral_method`. Every GC function returns an `xarray.Dataset` (see `output.py`) rather than raw arrays, so results come back labeled and ready to index by name.

> **Data shape convention:** the GC functions expect `X` as `(trials, nvars, N)` (or `(nvars, N)` for a single trial).

### Bivariate GC

```python
import numpy as np
from pygc import spectral_granger_causality
from pygc import ar_model

Fs = 200
# Dhamala benchmark: Y drives X at 40 Hz
data = ar_model.ar_model_dhamala(N=5000, Trials=20, Fs=Fs, C=0.25)
# data shape: (2, Trials, N) -> transpose to (Trials, 2, N) for the GC functions
data = np.transpose(data, (1, 0, 2))

# Non-parametric (Fourier CSD + Wilson factorization) — default
ds = spectral_granger_causality(data, fs=Fs, spectral_method='fourier', pairs=[(0, 1)])
# ds.y2x peaks at ~40 Hz; ds.x2y is near zero
peak_freq = ds.freq.values[ds.y2x.values[0].argmax()]
```

`spectral_granger_causality` returns a frequency-resolved `Dataset` with `x2y`, `y2x`, and `xy` variables (dims `pairs`, `freq`). For a single scalar per direction per pair instead (the broadband/time-domain summary), use `granger_causality`, which returns a `Dataset` with a single `F` variable (dims `source`, `target`):

```python
from pygc import granger_causality

ds_f = granger_causality(data, fs=Fs, spectral_method='fourier')
# ds_f.F[1, 0] -> broadband GC from channel 1 (Y) to channel 0 (X)
```

Available spectral methods: `'fourier'` (default), `'welch'`, `'morlet'`, `'multitaper'`.

```python
# Welch CSD
ds = spectral_granger_causality(
    data, fs=Fs, spectral_method='welch',
    spectral_params={'nperseg': 512}
)

# Morlet CSD (requires explicit frequency axis)
freqs = np.linspace(1, 80, 80)
ds = spectral_granger_causality(
    data, fs=Fs, spectral_method='morlet',
    spectral_params={'freqs': freqs, 'n_cycles': 7.0}
)

# Multitaper CSD
ds = spectral_granger_causality(
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
from pygc import ar_model

Fs = 200
data = ar_model.ar_model_dhamala(N=5000, Trials=20, Fs=Fs, C=0.25)
# data shape: (2, Trials, N) — YuleWalker takes (nvars, N) per trial, so no
# transpose is needed here (unlike the GC functions above).

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

`YuleWalker_multitrial` is also available if you'd rather not average the per-trial fits by hand.

### Conditional GC (multivariate, p ≥ 3)

```python
import numpy as np
from pygc import conditional_granger_causality, spectral_conditional_granger_causality
from pygc import ar_model

Fs = 200
data = ar_model.ar_model_baccala(nvars=5, N=3000, ntrials=20)
# data shape: (nvars, N, trials) -> transpose to (trials, nvars, N)
data = np.transpose(data, (2, 0, 1))

# Time-domain conditional GC — Dataset with an (nvars, nvars) 'F' matrix
ds_f = conditional_granger_causality(data, fs=Fs, n_jobs=-1)

# Spectral conditional GC — Dataset with an (nvars, nvars, n_freq) 'GC' array
ds_gc = spectral_conditional_granger_causality(data, fs=Fs, n_jobs=-1)
```

Both functions support the same `spectral_method` and `spectral_params` arguments as `spectral_granger_causality`.

### JAX/XLA accelerated backend

```python
from pygc import spectral_granger_causality, JAX_AVAILABLE

if JAX_AVAILABLE:
    ds = spectral_granger_causality(data, fs=Fs, backend='jax')
```

### `ensure_stability` parameter

The GC functions accept `ensure_stability=True` (default), which clips near-zero or negative diagonal entries of the noise covariance after Wilson factorization to improve numerical stability.

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
- `02_spectral_analysis.ipynb` — cross-spectral and time-frequency estimation (Fourier, Welch, multitaper, Morlet)
- `03_conditional_granger_causality.ipynb` — time-domain and spectral conditional GC on the 5-variable Baccalá model
- `04_benchmarks.ipynb` — performance benchmarks across backends and model sizes
- `05_spectral_methods_comparison.ipynb` — comparison of Fourier, Welch, Morlet, and multitaper estimators
- `06_example_with_eletrophysiological_data.ipynb` — GC applied to real electrophysiological recordings
