import numpy as np
import pytest
from pygc.granger import granger_causality
from pygc.parametric import YuleWalker, compute_transfer_function
from pygc.ar_model import ar_model_dhamala


FS = 200
N_FREQ = 65


def _identity_spectrum(n_freq):
    S = np.zeros([2, 2, n_freq], dtype=complex)
    for k in range(n_freq):
        S[:, :, k] = np.eye(2)
    return S


class TestGrangerCausality:
    def test_output_shapes(self):
        S = _identity_spectrum(N_FREQ)
        f = np.linspace(0, FS / 2, N_FREQ)
        Ix2y, Iy2x, Ixy = granger_causality(S, f, FS, verbose=False)
        assert Ix2y.shape == (N_FREQ,)
        assert Iy2x.shape == (N_FREQ,)
        assert Ixy.shape == (N_FREQ,)

    def test_uncoupled_system_zero_gc(self):
        """Independent signals: GC measures should be ~0."""
        S = _identity_spectrum(N_FREQ)
        f = np.linspace(0, FS / 2, N_FREQ)
        Ix2y, Iy2x, _ = granger_causality(S, f, FS, Niterations=200, verbose=False)
        np.testing.assert_allclose(Ix2y, 0.0, atol=1e-3)
        np.testing.assert_allclose(Iy2x, 0.0, atol=1e-3)

    def test_known_coupling_direction(self):
        """
        Dhamala AR model: Y drives X (C > 0).
        Spectral GC on the parametric cross-spectrum should show GC(Y->X) > GC(X->Y).
        """
        np.random.seed(0)

        cov = np.eye(2) * 0.5
        Z = ar_model_dhamala(N=3000, Trials=1, Fs=FS, C=0.25,
                              t_start=0, t_stop=None, cov=cov)
        X = Z[:, 0, :]  # (2, N)

        order = 5
        f = np.linspace(1, 90, N_FREQ)
        AR, sigma = YuleWalker(X, order)
        _, S = compute_transfer_function(AR, sigma, f, FS)

        Ix2y, Iy2x, _ = granger_causality(S, f, FS, verbose=False)

        # Dhamala model has Y->X coupling, so Iy2x should dominate.
        assert np.mean(Iy2x) > np.mean(Ix2y), (
            f"Expected GC(Y->X)={np.mean(Iy2x):.4f} > GC(X->Y)={np.mean(Ix2y):.4f}"
        )

    def test_invalid_method_raises(self):
        S = _identity_spectrum(N_FREQ)
        f = np.linspace(0, FS / 2, N_FREQ)
        with pytest.raises(ValueError, match="Unknown method"):
            granger_causality(S, f, FS, method='invalid')
