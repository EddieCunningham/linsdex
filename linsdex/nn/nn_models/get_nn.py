from functools import partial
from typing import Literal, Optional, Union, Tuple, Callable, List, Any
import einops
import equinox as eqx
import jax.random as random
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from linsdex.nn.nn_models.nn_abstract import AbstractSeq2SeqModel, AbstractNeuralNet
# from experiment.config_base import ExperimentConfig
from linsdex.nn.nn_models.rnn import GRUSeq2SeqModel, GRUSeq2SeqHypers, TimeDependentGRUSeq2SeqModel
from linsdex.nn.nn_models.resnet import ResNet, ResNetHypers, TimeDependentResNet, TimeDependentResNetHypers
from linsdex.nn.nn_models.s5 import (
  S5Seq2SeqModel,
  S5SeqHypers,
  TimeDependentS5Seq2SeqModel,
  TimeDependentS5SeqHypers,
)
import jax.nn as jnn

################################################################################################################

def get_nn(model_config: Any, dataset_config: Any, random_seed: int) -> Union[AbstractSeq2SeqModel, AbstractNeuralNet]:
  key = random.PRNGKey(random_seed)

  # Determine if this is 1D or 3D (image) data
  is_image_data = hasattr(dataset_config, 'height') and hasattr(dataset_config, 'width')

  # Check for conditioning information in dataset config
  cond_shape = None
  if hasattr(dataset_config, 'cond_shape'):
    cond_shape = dataset_config.cond_shape
  elif hasattr(dataset_config, 'conditioning_shape'):
    cond_shape = dataset_config.conditioning_shape

  # Handle nn models for time series
  if model_config.nn_type == "gru_rnn":

    assert hasattr(model_config, "hidden_size")
    assert hasattr(model_config, "n_layers")
    assert hasattr(model_config, "intermediate_channels")

    hypers = GRUSeq2SeqHypers(
      hidden_size=model_config.hidden_size,
      n_layers=model_config.n_layers,
      intermediate_channels=model_config.intermediate_channels
    )

    return GRUSeq2SeqModel(
      cond_in_channels=dataset_config.n_features,
      in_channels=dataset_config.n_features,
      out_channels=dataset_config.n_features,
      hypers=hypers,
      key=key
    )
  elif model_config.nn_type == "time_dependent_gru_rnn":

    assert hasattr(model_config, "hidden_size")
    assert hasattr(model_config, "n_layers")
    assert hasattr(model_config, "intermediate_channels")

    hypers = GRUSeq2SeqHypers(
      hidden_size=model_config.hidden_size,
      n_layers=model_config.n_layers,
      intermediate_channels=model_config.intermediate_channels
    )

    return TimeDependentGRUSeq2SeqModel(
      cond_in_channels=dataset_config.n_features,
      in_channels=dataset_config.n_features,
      out_channels=dataset_config.n_features,
      hypers=hypers,
      key=key
    )
  elif model_config.nn_type == "transformer":
    raise NotImplementedError
  elif model_config.nn_type == "time_dependent_transformer":
    raise NotImplementedError
  elif model_config.nn_type == "wavenet":
    raise NotImplementedError
  elif model_config.nn_type == "time_dependent_wavenet":
    raise NotImplementedError
  elif model_config.nn_type == "ssm":
    # Expose SSM params with defaults
    d_model = getattr(model_config, 'd_model', None) or dataset_config.n_features
    ssm_size = getattr(model_config, 'ssm_size', None) or 256
    blocks = getattr(model_config, 'blocks', None) or 4
    n_layers = getattr(model_config, 'num_layers', None) or getattr(model_config, 'n_layers', None) or 4
    cond_size = getattr(model_config, 'cond_size', None)
    if cond_size is None and cond_shape is not None:
      cond_size = cond_shape if isinstance(cond_shape, int) else cond_shape[-1]
    s5_hypers = S5SeqHypers(
      d_model=d_model,
      ssm_size=ssm_size,
      blocks=blocks,
      num_layers=n_layers,
      cond_size=cond_size,
    )

    return S5Seq2SeqModel(
      input_size=dataset_config.n_features,
      output_size=dataset_config.n_features,
      hypers=s5_hypers,
      key=key,
    )
  elif model_config.nn_type == "time_dependent_ssm":
    d_model = getattr(model_config, 'd_model', None) or dataset_config.n_features
    ssm_size = getattr(model_config, 'ssm_size', None) or 256
    blocks = getattr(model_config, 'blocks', None) or 4
    n_layers = getattr(model_config, 'num_layers', None) or getattr(model_config, 'n_layers', None) or 4
    cond_size = getattr(model_config, 'cond_size', None)
    if cond_size is None and cond_shape is not None:
      cond_size = cond_shape if isinstance(cond_shape, int) else cond_shape[-1]

    s5_hypers = TimeDependentS5SeqHypers(
      d_model=d_model,
      ssm_size=ssm_size,
      blocks=blocks,
      num_layers=n_layers,
      cond_size=cond_size,
    )

    return TimeDependentS5Seq2SeqModel(
      input_size=dataset_config.n_features,
      output_size=dataset_config.n_features,
      hypers=s5_hypers,
      key=key,
    )

  elif model_config.nn_type == "resnet":

    if is_image_data:
      # For 3D image data
      input_shape = (dataset_config.height, dataset_config.width, dataset_config.n_features)
      filter_shape = getattr(model_config, 'filter_shape', (3, 3))  # Default to 3x3 filters
      groups = getattr(model_config, 'n_groups', 8)  # Default groups for images
      out_size = dataset_config.n_features  # Keep same number of channels
    else:
      # For 1D data
      input_shape = (dataset_config.n_features,)
      filter_shape = None
      groups = None
      out_size = dataset_config.n_features

    hypers = ResNetHypers(
      working_size=model_config.working_size,
      hidden_size=model_config.hidden_size,
      n_blocks=model_config.n_blocks,
      filter_shape=filter_shape,
      groups=groups,
      activation=jnn.gelu
    )

    return ResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

  elif model_config.nn_type == "time_dependent_resnet":

    if is_image_data:
      # For 3D image data
      input_shape = (dataset_config.height, dataset_config.width, dataset_config.n_features)
      filter_shape = getattr(model_config, 'filter_shape', (3, 3))  # Default to 3x3 filters
      groups = getattr(model_config, 'n_groups', 8)  # Default groups for images
      out_size = dataset_config.n_features  # Keep same number of channels
    else:
      # For 1D data
      input_shape = (dataset_config.n_features,)
      filter_shape = None
      groups = None
      out_size = dataset_config.n_features

    hypers = TimeDependentResNetHypers(
      working_size=model_config.working_size,
      hidden_size=model_config.hidden_size,
      n_blocks=model_config.n_blocks,
      filter_shape=filter_shape,
      groups=groups,
      activation=jnn.gelu,
      embedding_size=model_config.embedding_size,
      out_features=model_config.out_features
    )

    return TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

  else:
    raise ValueError(f"Invalid neural network type: {model_config.nn_type}")
