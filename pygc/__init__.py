from .parametric import YuleWalker, compute_transfer_function
from .non_parametric import wilson_factorization
from .granger import granger_causality, conditional_granger_causality, conditional_spec_granger_causality
from .covgc import covgc_time
from .tools import xcorr, demean, rdet
from ._jax_backend import JAX_AVAILABLE, wilson_factorization_jax

__version__ = "0.1.0"

__all__ = [
    "YuleWalker",
    "compute_transfer_function",
    "wilson_factorization",
    "wilson_factorization_jax",
    "JAX_AVAILABLE",
    "granger_causality",
    "conditional_granger_causality",
    "conditional_spec_granger_causality",
    "covgc_time",
    "xcorr",
    "demean",
    "rdet",
]
