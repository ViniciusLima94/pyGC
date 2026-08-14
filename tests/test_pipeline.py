import numpy as np
import pytest
import xarray as xr

from pygc import JAX_AVAILABLE, Pipeline
from pygc.ar_model import ar_model_baccala

FS = 200.0


def _random_data(trials=10, nvars=4, N=512, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((trials, nvars, N))


def _baccala_data(ntrials=150, N=1024, seed=0):
    np.random.seed(seed)
    Y = ar_model_baccala(nvars=5, N=N, ntrials=ntrials)  # (nvars, N, ntrials)
    return Y.transpose(2, 0, 1)  # (trials, nvars, N)


class TestPipelineValidation:
    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            Pipeline(metric='invalid', fs=FS)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Pipeline(metric='gc', fs=FS, backend='invalid')

    def test_missing_fs_raises(self):
        with pytest.raises(ValueError, match="fs"):
            Pipeline(metric='gc')


class TestBivariateMetrics:
    @pytest.mark.parametrize('metric', ['gc', 'spectral_gc'])
    def test_output_shape_and_dims(self, metric):
        nvars = 4
        data = _random_data(trials=10, nvars=nvars, N=256)
        result = Pipeline(metric=metric, fs=FS, spectral_method='welch').fit(data)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('edge', 'freq')
        assert result.sizes['edge'] == nvars * (nvars - 1)

    def test_edge_labels(self):
        data = _random_data(trials=10, nvars=3, N=256)
        result = Pipeline(metric='gc', fs=FS, spectral_method='welch').fit(data)
        edges = set(result.coords['edge'].values.tolist())
        expected = {f"Node_{i}->Node_{j}" for i in range(3) for j in range(3) if i != j}
        assert edges == expected

    def test_result_attribute_matches_return_value(self):
        data = _random_data(trials=5, nvars=3, N=256)
        pipe = Pipeline(metric='gc', fs=FS, spectral_method='welch')
        result = pipe.fit(data)
        assert pipe.result_ is result

    def test_accepts_single_trial_2d_data(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 512))  # (nodes, time), fourier default
        result = Pipeline(metric='gc', fs=FS).fit(data)
        assert result.sizes['edge'] == 3 * 2

    def test_known_coupling_direction(self):
        """Baccalá model: node 0 drives node 1 directly; node 1 does not drive node 0."""
        data = _baccala_data(ntrials=150, N=1024)
        result = Pipeline(metric='gc', fs=FS, spectral_method='fourier').fit(data)
        e01 = result.sel(edge='Node_0->Node_1').values.mean()
        e10 = result.sel(edge='Node_1->Node_0').values.mean()
        assert e01 > e10


class TestConditionalMetric:
    def test_output_shape_and_dims(self):
        nvars = 4
        data = _random_data(trials=10, nvars=nvars, N=256)
        result = Pipeline(metric='conditional', fs=FS, spectral_method='welch').fit(data)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('edge',)
        assert result.sizes['edge'] == nvars * (nvars - 1)

    def test_known_coupling_direction(self):
        data = _baccala_data(ntrials=150, N=1024)
        result = Pipeline(metric='conditional', fs=FS, spectral_method='fourier').fit(data)
        e01 = float(result.sel(edge='Node_0->Node_1'))
        e10 = float(result.sel(edge='Node_1->Node_0'))
        assert e01 > e10
        assert e10 == pytest.approx(0.0, abs=0.05)


class TestConditionalSpectralMetric:
    def test_output_shape_and_dims(self):
        nvars = 4
        data = _random_data(trials=10, nvars=nvars, N=256)
        result = Pipeline(metric='conditional_spectral', fs=FS,
                           spectral_method='welch').fit(data)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('edge', 'freq')
        assert result.sizes['edge'] == nvars * (nvars - 1)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJaxBackend:
    def test_jax_matches_numpy(self):
        data = _random_data(trials=20, nvars=4, N=512)
        result_np = Pipeline(metric='gc', fs=FS, spectral_method='welch',
                              backend='numpy').fit(data)
        result_jax = Pipeline(metric='gc', fs=FS, spectral_method='welch',
                               backend='jax').fit(data)
        np.testing.assert_allclose(result_np.values, result_jax.values,
                                    rtol=1e-4, atol=1e-6)
