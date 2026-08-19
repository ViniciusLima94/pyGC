from .parametric import YuleWalker, YuleWalker_multitrial, compute_transfer_function
from .non_parametric import wilson_factorization
from .granger import granger_causality, conditional_granger_causality, conditional_spec_granger_causality
from ._jax_backend import JAX_AVAILABLE, JAX_FLOAT64, wilson_factorization_jax
from .output import build_granger_dataset, build_conditional_gc_dataset, build_conditional_spec_gc_dataset

__version__ = "2.0.0"

__all__ = [
    "YuleWalker",
    "YuleWalker_multitrial",
    "compute_transfer_function",
    "wilson_factorization",
    "wilson_factorization_jax",
    "JAX_AVAILABLE",
    "JAX_FLOAT64",
    "granger_causality",
    "conditional_granger_causality",
    "conditional_spec_granger_causality",
    "build_granger_dataset",
    "build_conditional_gc_dataset",
    "build_conditional_spec_gc_dataset"
]
