import numpy as np
import pytest
from pygc.granger import granger_causality
from pygc.parametric import YuleWalker, compute_transfer_function
from pygc.ar_model import ar_model_dhamala


FS = 200
N_FREQ = 65


def _identity_system(n_freq):
    H = np.zeros([2, 2, n_freq], dtype=complex)
    S = np.zeros([2, 2, n_freq], dtype=complex)
    Z = np.eye(2)
    for k in range(n_freq):
        H[:, :, k] = np.eye(2)
        S[:, :, k] = np.eye(2)
    return S, H, Z


class TestGrangerCausality:
    def test_output_shapes(self):
        S, H, Z = _identity_system(N_FREQ)
        Ix2y, Iy2x, Ixy = granger_causality(S, H, Z)
        assert Ix2y.shape == (N_FREQ,)
        assert Iy2x.shape == (N_FREQ,)
        assert Ixy.shape == (N_FREQ,)

    def test_uncoupled_system_zero_gc(self):
        """Independent signals: all three GC measures should be ~0."""
        S, H, Z = _identity_system(N_FREQ)
        Ix2y, Iy2x, Ixy = granger_causality(S, H, Z)
        np.testing.assert_allclose(Ix2y, 0.0, atol=1e-10)
        np.testing.assert_allclose(Iy2x, 0.0, atol=1e-10)

    def test_known_coupling_direction(self):
        """
        Dhamala AR model: Y drives X (C > 0).
        After fitting, GC(Y->X) should be substantially larger than GC(X->Y).
        """
        rng_state = np.random.RandomState(0)
        np.random.seed(0)

        cov = np.eye(2) * 0.5
        Z = ar_model_dhamala(N=3000, Trials=1, Fs=FS, C=0.25,
                              t_start=0, t_stop=None, cov=cov)
        # Z shape: (2, Trials, N) — use first trial
        X = Z[:, 0, :]  # (2, N)

        order = 5
        f = np.linspace(1, 90, N_FREQ)
        AR, sigma = YuleWalker(X, order)
        H, S = compute_transfer_function(AR, sigma, f, FS)
        Ix2y, Iy2x, _ = granger_causality(S, H, sigma)

        # In granger_causality convention:
        #   Ix2y = GC(X->Y): influence of channel 0 (X) on channel 1 (Y)
        #   Iy2x = GC(Y->X): influence of channel 1 (Y) on channel 0 (X)
        # Dhamala model has Y->X coupling, so Iy2x should dominate.
        assert np.mean(Iy2x) > np.mean(Ix2y), (
            f"Expected GC(Y->X)={np.mean(Iy2x):.4f} > GC(X->Y)={np.mean(Ix2y):.4f}"
        )
