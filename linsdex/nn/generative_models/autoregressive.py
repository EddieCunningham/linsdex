import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Dict, overload, Literal
import einops
import equinox as eqx
import abc
import diffrax
from jaxtyping import Array, PRNGKeyArray
import jax.tree_util as jtu
import os
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool
from jax._src.util import curry
from linsdex import AbstractSDE
from linsdex import AbstractBatchableObject
from linsdex import TimeSeries
from linsdex import InterleavedTimes
from linsdex.ssm.simple_encoder import AbstractEncoder, PaddingLatentVariableEncoderWithPrior
from linsdex import AbstractLinearSDE, MixedGaussian, NaturalGaussian
from linsdex import AbstractSquareMatrix
from linsdex import AbstractTransition, AbstractPotential
from linsdex import GaussianTransition
import linsdex.util as util
from linsdex import Messages, CRF
from linsdex import StandardGaussian, MixedGaussian, NaturalGaussian
from linsdex.nn.generative_models.prob_model_abstract import AbstractGenerativeModel
from linsdex.nn.nn_models.nn_abstract import AbstractSeq2SeqModel
from linsdex.nn.nn_layers.util import apply_mask_to_series
# from experiment.config_base import ExperimentConfig
from linsdex import BrownianMotion, WienerVelocityModel, TimeScaledLinearTimeInvariantSDE
from dataclasses import dataclass
# from experiment.config_base import AbstractModelConfig, ExperimentConfig

################################################################################################################

def predict_next_state_distribution(
  nn: AbstractSeq2SeqModel,
  obs_series: TimeSeries,
) -> AbstractPotential:

  # Predict the mean and diagonal covariance matrix
  mean_and_cov = nn(obs_series).values

  dim = mean_and_cov.shape[-1]//2
  means, covs = mean_and_cov[...,:dim], mean_and_cov[...,dim:]

  def create_gaussian(mean: Float[Array, 'D'], cov: Float[Array, 'D']) -> AbstractPotential:
    return StandardGaussian(mean, cov)

  next_distributions = jax.vmap(create_gaussian)(means[:-1], covs[:-1])
  return next_distributions

def gaussian_autoregressive_loss(
  model_config: Any,
  dataset_config: Any,
  nn: AbstractSeq2SeqModel,
  obs_series: TimeSeries,
  key: PRNGKeyArray,
  debug: Optional[bool] = False
) -> Dict[str, Scalar]:

  masked_obs_series = apply_mask_to_series(obs_series, dataset_config.condition_length)
  next_distributions = predict_next_state_distribution(nn, masked_obs_series)
  log_probs = next_distributions.log_prob(obs_series.values[1:])

  ml_loss = -log_probs.mean()

  if debug:
    import pdb; pdb.set_trace()

  losses = dict(ml=ml_loss)
  return losses

def gaussian_autoregressive_model_sample(
  model_config: Any,
  dataset_config: Any,
  nn: AbstractSeq2SeqModel,
  obs_series: TimeSeries,
  key: PRNGKeyArray,
  debug: Optional[bool] = False
) -> TimeSeries:
  """Sample from p(y_{k+1:N} | Y_{1:k}) autoregressively."""

  # Keep a buffer of the observed series.  There is a deprecation variable in TimSeries (mask_value)
  # that causes an error if we just set yts_buffer to yts.  So we need to do this manually.
  yts_buffer = obs_series

  # Encode the encoder series context for our transformer
  masked_obs_series = apply_mask_to_series(obs_series, dataset_config.condition_length)
  context = nn.create_context(masked_obs_series)
  filled_seq2seq = partial(nn, context=context)

  #########################################################
  # Autoregressive sampling
  #########################################################
  def scan_body(carry, inputs, debug=False):
    yts_buffer = carry
    key, k = inputs

    # Predict all of the transitions.  The model is autoregressive so we don't need to worry about
    # masking yts_buffer or anything like that.  Also, we cut off the unobserved parts of yts inside
    # the predict_next_state_distribution function.
    transitions = predict_next_state_distribution(filled_seq2seq, yts_buffer)

    # Get p(y_{i+1} | y_k)
    transition = transitions[k]

    # Sample y_{i+1} ~ p(y_{i+1} | y_k)
    ykp1 = transition.sample(key)

    # Update the buffer with the new prediction
    new_values = util.fill_array(yts_buffer.values, k+1, ykp1)

    # Update the observation mask to reflect the new prediction
    new_mask = util.fill_array(yts_buffer.mask, k+1, True)

    # Create the new time series
    new_yts = TimeSeries(yts_buffer.times, new_values, new_mask)
    if debug:
      import pdb; pdb.set_trace()
    return new_yts, new_yts

  carry = yts_buffer
  keys = random.split(key, len(yts_buffer.times)-1)
  inputs = (keys, jnp.arange(len(yts_buffer.times)-1))

  if debug:
    for i, item in enumerate(zip(*inputs)):
      carry, _ = scan_body(carry, item, debug=True)
    import pdb; pdb.set_trace()
  else:
    carry, all_yts_buffer = jax.lax.scan(scan_body, carry, inputs)

  return carry

################################################################################################################

class AutoregressiveModelConfig(eqx.Module):
  name: str

  cond_len: int

  hidden_size: int
  n_layers: int
  intermediate_channels: int

class AutoregressiveModel(AbstractGenerativeModel):

  model: 'AutoregressiveRNNModel'

  def __init__(self, model_config: Any, dataset_config: Any, random_seed: int):

    key = random.PRNGKey(random_seed)

    if model_config.sde_type == 'brownian':
      sde = BrownianMotion(sigma=dataset_config.process_noise, dim=dataset_config.n_features)
    elif model_config.sde_type == 'wiener_velocity':
      sde = WienerVelocityModel(sigma=dataset_config.process_noise, position_dim=dataset_config.n_features, order=2)
    else:
      raise ValueError(f"Unknown SDE type: {model_config.sde_type}")

    sde = TimeScaledLinearTimeInvariantSDE(sde, time_scale=dataset_config.time_scale_mult)

    encoder = PaddingLatentVariableEncoderWithPrior(
      y_dim=dataset_config.n_features,
      x_dim=sde.dim,
      sigma=dataset_config.observation_noise,
      use_prior=True
    )

    self.model = AutoregressiveRNNModel(sde,
                                                  encoder,
                                                  hidden_size=model_config.hidden_size,
                                                  potential_cov_type=model_config.potential_cov_type,
                                                  parametrization=model_config.parametrization,
                                                  interpolation_freq=model_config.interpolation_freq,
                                                  seq_len=dataset_config.seq_length,
                                                  cond_len=model_config.cond_len,
                                                  key=key,
                                                  n_layers=model_config.n_layers,
                                                  intermediate_channels=model_config.intermediate_channels)

  def loss_fn(self, key: PRNGKeyArray, yts: TimeSeries, debug: Optional[bool] = False) -> Scalar:
    losses = self.model.loss_fn(yts, key, debug)
    return losses['ml'], losses

  def sample(self, key: PRNGKeyArray, yts: TimeSeries, debug: Optional[bool] = False) -> TimeSeries:
    return self.model.sample(key, yts, debug=debug)

  def basic_interpolation(self, key: PRNGKeyArray, yts: TimeSeries) -> TimeSeries:
    return self.model.basic_interpolation(key, yts)

################################################################################################################
