import jax.numpy as jnp
import linsdex as lsde
import jax
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Annotated, List
from jaxtyping import Array, PRNGKeyArray, Float, Scalar
from linsdex import TimeSeries, AbstractLinearSDE
import jax.random as random
import equinox as eqx
# from experiment.config_base import AbstractModelConfig, ExperimentConfig
from dataclasses import dataclass, field, asdict
import jax.tree_util as jtu
import abc
from functools import partial
from linsdex.nn.generative_models.prob_model_abstract import AbstractGenerativeModel

def get_probabilistic_model(model_config: Any, dataset_config: Any, random_seed: int) -> AbstractGenerativeModel:
  from linsdex.nn.generative_models.autoregressive import AutoregressiveModel
  from linsdex.nn.generative_models.diffusion import DiffusionTimeSeriesModel, DiffusionModel
  # from linsdex.nn.generative_models.non_probabilistic import NonProbabilisticModel
  # from linsdex.nn.generative_models.structured_diffusion import StructuredDiffusionTimeSeriesModel
  from linsdex.nn.generative_models.diffusion import ImprovedDiffusionTimeSeriesModel

  if model_config.name == "autoregressive":
    raise NotImplementedError(f"Have not tested this yet.")
  elif model_config.name == "diffusion_time_series":
    return DiffusionTimeSeriesModel(model_config, dataset_config, random_seed)
  elif model_config.name == "diffusion":
    return DiffusionModel(model_config, dataset_config, random_seed)
  elif model_config.name == "non_probabilistic_latent":
    raise NotImplementedError(f"Have not tested this yet.")
  elif model_config.name == "structured_diffusion_time_series":
    # return StructuredDiffusionTimeSeriesModel(model_config, dataset_config, random_seed)
    raise NotImplementedError("StructuredDiffusionTimeSeriesModel is currently missing.")
  elif model_config.name == "improved_diffusion_time_series":
    return ImprovedDiffusionTimeSeriesModel(model_config, dataset_config, random_seed)
  else:
    raise ValueError(f"Unknown model: {model_config.name}")
