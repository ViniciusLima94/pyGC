import numpy as np
import pytest
from pygc.granger import granger_causality, spectral_granger_causality
from pygc.ar_model import ar_model_dhamala

FS = 200


def _make_data(nvars=2, N=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nvars, N))


def _dhamala_trials(N=3000, Trials=10, C=0.25):
    """Dhamala AR model as (trials, nvars, N). Channel 1 (Y) drives channel 0 (X)."""
    np.random.seed(0)
    Z = ar_model_dhamala(N=N, Trials=Trials, Fs=FS, C=C,
                         t_start=0, t_stop=None, cov=np.eye(2) * 0.5)
    return Z.transpose(1, 0, 2)  # (2, Trials, N) -> (Trials, 2, N)


class TestSpectralGrangerCausality:
    def test_output_shapes_welch(self):
        X = _make_data(N=1024)
        ds = spectral_granger_causality(X, FS, spectral_method='welch')
        assert set(ds.data_vars) == {"x2y", "y2x", "xy"}
        assert ds.x2y.dims == ("pairs", "freq")
        assert ds.sizes["pairs"] == 1
        assert ds.y2x.shape == ds.x2y.shape == ds.xy.shape

    def test_output_shapes_fourier_multitrials(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2, 512))  # (trials, nvars, N)
        ds = spectral_granger_causality(X, FS, spectral_method='fourier')
        assert ds.x2y.dims == ("pairs", "freq")
        assert ds.y2x.shape == ds.x2y.shape == ds.xy.shape

    def test_uncoupled_system_zero_gc(self):
        """Independent signals via Welch: GC measures should be ~0."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2, 5000))
        ds = spectral_granger_causality(X, FS, spectral_method='welch', Niterations=200)
        np.testing.assert_allclose(ds.x2y.values, 0.0, atol=0.15)
        np.testing.assert_allclose(ds.y2x.values, 0.0, atol=0.15)

    def test_known_coupling_direction(self):
        """Dhamala AR model: Y drives X (C > 0). GC(Y->X) > GC(X->Y)."""
        ds = spectral_granger_causality(_dhamala_trials(), FS, spectral_method='welch')
        x2y, y2x = float(ds.x2y.mean()), float(ds.y2x.mean())
        assert y2x > x2y, f"Expected GC(Y->X)={y2x:.4f} > GC(X->Y)={x2y:.4f}"

    def test_multiple_pairs(self):
        X = _make_data(nvars=3, N=1024)
        ds = spectral_granger_causality(X, FS, spectral_method='welch', n_jobs=1)
        assert list(ds.pairs.values) == ["0->1", "0->2", "1->2"]

    def test_morlet_multitrials(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((5, 2, 512))  # (trials, nvars, N)
        freqs = np.linspace(10, 80, 20)
        ds = spectral_granger_causality(
            X, FS, spectral_method='morlet',
            spectral_params={'freqs': freqs, 'n_cycles': 5.0},
        )
        assert ds.x2y.shape == (1, len(freqs))


class TestGrangerCausality:
    """Time-domain (broadband) pairwise GC."""

    def test_output_matrix(self):
        X = _make_data(nvars=3, N=1024)
        ds = granger_causality(X, FS, spectral_method='welch', n_jobs=1)
        F = ds.F.values
        assert ds.F.dims == ("source", "target")
        assert F.shape == (3, 3)
        assert np.all(np.isnan(np.diag(F)))
        off_diag = F[~np.eye(3, dtype=bool)]
        assert np.all(np.isfinite(off_diag))

    def test_known_coupling_direction(self):
        """F[i, j] = GC(i -> j); channel 1 drives channel 0."""
        F = granger_causality(_dhamala_trials(), FS, spectral_method='welch', n_jobs=1).F.values
        assert F[1, 0] > F[0, 1], f"Expected GC(1->0)={F[1, 0]:.4f} > GC(0->1)={F[0, 1]:.4f}"

    def test_invalid_backend_raises(self):
        X = _make_data()
        with pytest.raises(ValueError, match="Unknown backend"):
            granger_causality(X, FS, spectral_method='welch', backend='invalid')

    def test_invalid_spectral_method_raises(self):
        X = _make_data()
        with pytest.raises(ValueError, match="Unknown spectral_method"):
            granger_causality(X, FS, spectral_method='invalid')

    def test_morlet_missing_freqs_raises(self):
        X = _make_data()
        with pytest.raises(ValueError, match="freqs"):
            granger_causality(X, FS, spectral_method='morlet')
