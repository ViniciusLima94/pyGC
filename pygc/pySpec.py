# Replaced by pygc.spectral_analysis.fourier (MNE-based, no neo/elephant dependency).
# This file is kept for reference only.
from .spectral_analysis.fourier import compute_freq, csd_fourier as cxy, morlet_transform as morlet, morlet_csd as morlet_power

__all__ = ["compute_freq", "cxy", "morlet", "morlet_power"]
