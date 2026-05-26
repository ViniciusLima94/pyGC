import numpy as np
import pytest
from pygc.tools import xcorr, demean, rdet, PlusOperator


class TestXcorr:
    def test_output_shapes(self):
        x = np.random.randn(2, 100)
        lags, Rxx = xcorr(x, x, 10)
        assert lags.shape == (10,)
        assert Rxx.shape == (10, 2, 2)

    def test_lag_zero_matches_variance(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 2000))
        _, Rxx = xcorr(x, x, 1)
        np.testing.assert_allclose(Rxx[0, 0, 0], np.var(x), rtol=0.1)

    def test_cross_correlation_asymmetry(self):
        rng = np.random.default_rng(1)
        x = np.zeros((2, 200))
        x[0] = rng.standard_normal(200)
        x[1, 5:] = x[0, :-5]  # x[1] is x[0] delayed by 5 samples
        _, Rxx = xcorr(x, x, 20)
        # Cross-correlation at lag 5 should be high
        assert abs(Rxx[5, 0, 1]) > abs(Rxx[0, 0, 1])


class TestDemean:
    def test_removes_mean(self):
        # demean computes the mean over the flattened (trials × time) axis per source
        rng = np.random.default_rng(2)
        n, m, N = 3, 5, 100
        x = rng.standard_normal((n, m, N)) + 10.0
        y = demean(x)
        # After demeaning, mean over the flattened axis should be 0 per source
        y_flat = np.swapaxes(y, 1, 2).reshape((n, m * N))
        np.testing.assert_allclose(y_flat.mean(axis=1), 0.0, atol=1e-10)

    def test_normalise_sets_unit_std(self):
        rng = np.random.default_rng(3)
        n, m, N = 2, 4, 500
        x = rng.standard_normal((n, m, N)) * 5.0 + 3.0
        y = demean(x, norm=True)
        y_flat = np.swapaxes(y, 1, 2).reshape((n, m * N))
        np.testing.assert_allclose(y_flat.std(axis=1), 1.0, atol=0.05)

    def test_shape_preserved(self):
        x = np.ones((3, 5, 100))
        assert demean(x).shape == x.shape


class TestRdet:
    def test_scalar_returns_self(self):
        assert rdet(np.float64(3.0)) == 3.0

    def test_identity_matrix(self):
        assert rdet(np.eye(3)) == pytest.approx(1.0)

    def test_singular_matrix(self):
        A = np.zeros((2, 2))
        assert rdet(A) == pytest.approx(0.0)
