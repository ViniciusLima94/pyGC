import numpy as np
import pytest
from pygc.covgc import covgc_time


def _make_data(nvars=2, nsamples=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nvars, nsamples))


class TestCovgcTime:
    def test_output_shape_two_vars(self):
        X = _make_data(nvars=2, nsamples=300)
        dt, lag, t0 = 50, 5, 100
        GC = covgc_time(X, dt, lag, t0)
        # One pair, three directionality measures
        assert GC.shape == (1, 3)

    def test_output_shape_three_vars(self):
        X = _make_data(nvars=3, nsamples=500)
        dt, lag, t0 = 50, 5, 200
        GC = covgc_time(X, dt, lag, t0)
        # Three pairs for 3 variables
        assert GC.shape == (3, 3)

    def test_known_coupling(self):
        """
        X[1] depends on X[0] at lag 1. GC(X[0]->X[1]) should exceed GC(X[1]->X[0]).
        """
        rng = np.random.default_rng(42)
        N = 600
        X = np.zeros((2, N))
        noise = rng.standard_normal((2, N)) * 0.2
        X[0] = np.cumsum(noise[0])  # random walk
        # X[1] driven by X[0] at previous step
        for t in range(1, N):
            X[1, t] = 0.6 * X[0, t - 1] + noise[1, t]

        dt, lag, t0 = 80, 3, 200
        GC = covgc_time(X, dt, lag, t0)
        # GC[:,0] = X[0]->X[1], GC[:,1] = X[1]->X[0]
        assert GC[0, 0] > GC[0, 1], (
            f"Expected GC(X0->X1)={GC[0,0]:.4f} > GC(X1->X0)={GC[0,1]:.4f}"
        )

    def test_gc_can_be_negative(self):
        """covgc_time returns log-ratios, which are not strictly bounded."""
        X = _make_data()
        GC = covgc_time(X, dt=50, lag=5, t0=100)
        # Just check it runs and returns finite values
        assert np.all(np.isfinite(GC))
