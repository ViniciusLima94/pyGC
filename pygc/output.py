import xarray as xr
import numpy as np


def build_granger_dataset(Ix2y, Iy2x, Ixy, pairs, f):
    """
    Package spectral Granger causality results into an xarray Dataset.

    Parameters
    ----------
    Ix2y, Iy2x, Ixy : array-like
        Directional (x->y, y->x) and instantaneous causality spectra.
        May arrive pre-squeezed (e.g. shape (n_freq,) when n_pairs == 1),
        so shape is normalized to (n_pairs, n_freq) here rather than assumed.
    pairs : array-like, shape (n_pairs, 2) or list of tuples
        Channel/variable index pairs.
    f : array-like, shape (n_freq,)
        Frequency values.
    """
    f = np.asarray(f)
    n_pairs = len(pairs)
    n_freq = len(f)

    def _to_pairs_freq(arr, name):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            # squeeze() collapsed the pairs dim (only valid when n_pairs == 1)
            if arr.shape[0] != n_freq:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected ({n_pairs}, {n_freq}) "
                    f"or ({n_freq},) when n_pairs == 1"
                )
            arr = arr.reshape(1, n_freq)
        if arr.shape != (n_pairs, n_freq):
            raise ValueError(
                f"{name} has shape {arr.shape}, expected ({n_pairs}, {n_freq})"
            )
        return arr

    Ix2y = _to_pairs_freq(Ix2y, "Ix2y")
    Iy2x = _to_pairs_freq(Iy2x, "Iy2x")
    Ixy  = _to_pairs_freq(Ixy, "Ixy")

    pair_labels = [f"{i}->{j}" for i, j in pairs]

    ds = xr.Dataset(
        {
            "x2y": (("pairs", "freq"), Ix2y),
            "y2x": (("pairs", "freq"), Iy2x),
            "xy":  (("pairs", "freq"), Ixy),
        },
        coords={
            "pairs": pair_labels,
            "freq": f,
        },
    )
    return ds

def build_conditional_gc_dataset(F, channel_names=None):
    """
    Package `conditional_granger_causality` output into an xarray Dataset.

    Parameters
    ----------
    F : ndarray, shape (nvars, nvars)
        Conditional GC matrix, F[i, j] = influence of i on j given all others.
    channel_names : list of str/int, optional
        Labels for the channels. Defaults to range(nvars).

    Returns
    -------
    ds : xarray.Dataset
        Variable 'F' with dims ('source', 'target').
    """
    F = np.asarray(F)
    nvars = F.shape[0]
    channel_names = list(channel_names) if channel_names is not None else list(range(nvars))

    ds = xr.Dataset(
        {"F": (("source", "target"), F)},
        coords={"source": channel_names, "target": channel_names},
    )
    return ds


def build_conditional_spec_gc_dataset(GC, f, channel_names=None):
    """
    Package `conditional_spec_granger_causality` output into an xarray Dataset.

    Parameters
    ----------
    GC : ndarray, shape (nvars, nvars, n_freq)
        Spectral conditional GC, GC[j, i, :] = influence of i on j given all
        others, as a function of frequency.
    f : array-like, shape (n_freq,)
        Frequency values.
    channel_names : list of str/int, optional
        Labels for the channels. Defaults to range(nvars).

    Returns
    -------
    ds : xarray.Dataset
        Variable 'GC' with dims ('target', 'source', 'freq'), matching the
        GC[j, i, :] indexing convention from the function itself.
    """
    GC = np.asarray(GC)
    f = np.asarray(f)
    nvars = GC.shape[0]
    channel_names = list(channel_names) if channel_names is not None else list(range(nvars))

    ds = xr.Dataset(
        {"GC": (("target", "source", "freq"), GC)},
        coords={"target": channel_names, "source": channel_names, "freq": f},
    )
    return ds