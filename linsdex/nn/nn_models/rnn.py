from functools import partial
from typing import Literal, Optional, Union, Tuple, Callable, List, Any
import einops
import equinox as eqx
import jax.random as random
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Bool
from linsdex import AbstractBatchableObject, auto_vmap
from linsdex.nn.nn_models.nn_abstract import AbstractSeq2SeqModel, AbstractTimeDependentSeq2SeqModel
from linsdex import TimeSeries
from linsdex.nn.nn_models.nn_abstract import AbstractHyperParams
from linsdex.nn.nn_layers.time_condition import TimeFeatures
from linsdex.nn.nn_layers.rnn_layers import GRURNN

################################################################################################################

class StackedGRUSequenceHypers(AbstractHyperParams):
  hidden_size: int
  intermediate_channels: int
  num_layers: int

class StackedGRURNN(AbstractBatchableObject):

  gru_blocks: GRURNN
  initial_state: Float[Array, 'N hidden_size']
  in_proj: eqx.nn.Linear
  out_proj: eqx.nn.Linear
  hypers: StackedGRUSequenceHypers

  def __init__(self,
               in_channels: int,
               out_channels: int,
               hypers: StackedGRUSequenceHypers,
               key: PRNGKeyArray):
    k1, k2, k3, k4 = random.split(key, 4)

    self.in_proj = eqx.nn.Linear(in_features=in_channels,
                            out_features=hypers.intermediate_channels,
                            key=k1)

    self.out_proj = eqx.nn.Linear(in_features=hypers.intermediate_channels,
                             out_features=out_channels,
                             key=k2)

    def make_block(key: PRNGKeyArray) -> GRURNN:
      return GRURNN(in_channels=hypers.intermediate_channels,
                    out_channels=hypers.intermediate_channels,
                    hidden_size=hypers.hidden_size,
                    key=key)
    keys = random.split(k1, hypers.num_layers)
    self.gru_blocks = jax.vmap(make_block)(keys)

    self.initial_state = jnp.zeros((hypers.num_layers, hypers.hidden_size))
    self.hypers = hypers

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.gru_blocks[0].batch_size

  @auto_vmap
  def __call__(
    self,
    xts: Float[Array, 'T C'],
    global_context: Optional[Float[Array, 'N hidden_size']] = None
  ) -> Float[Array, 'T C']:

    hidden_state = self.initial_state
    if global_context is not None:
      hidden_state = hidden_state + global_context

    params, static = eqx.partition(self.gru_blocks, eqx.is_array)

    # Scan over the depth of the GRU
    def scan_body(carry, inputs):
      hts = carry
      params, starting_hidden_state = inputs
      gru = eqx.combine(params, static)
      hts = gru(hts, starting_hidden_state)
      return hts, ()

    hts = jax.vmap(self.in_proj)(xts)
    hts, _ = jax.lax.scan(scan_body, hts, (params, hidden_state))
    out = jax.vmap(self.out_proj)(hts)
    return out

################################################################################################################

class GRUSeq2SeqHypers(AbstractHyperParams):
  hidden_size: int # Hidden size of recurrent part of the GRU
  n_layers: int # Number of layers in the GRU
  intermediate_channels: int # Dimensionality of the intermediate sequence

class GRUSeq2SeqModel(AbstractSeq2SeqModel):
  """Naive GRU-based encoder-decoder model.  The encoder creates inputs for each of
  the decoder layers.
  """

  encoder: StackedGRURNN
  decoder: StackedGRURNN
  time_features: TimeFeatures

  hypers: GRUSeq2SeqHypers = eqx.field(static=True)

  def __init__(self,
               cond_in_channels: int,
               in_channels: int,
               out_channels: int,
               hypers: GRUSeq2SeqHypers,
               key: PRNGKeyArray):
    """
    The encoder is a GRU-based encoder that takes in the condition information and the input series
    and outputs a hidden state.  The decoder is a GRU-based decoder that takes in the hidden state
    and the input series and outputs a hidden state.  The hidden state is then passed through a
    linear layer to get the output series.

    **Arguments:**
      cond_in_channels - Dimensionality of the condition information
      in_channels - Dimensionality of the input series
      out_channels - Dimensionality of the output series
      hypers - Hyperparameters for the GRU-based encoder-decoder model
      key - Random key for initialization
    """
    k1, k2, k3 = random.split(key, 3)

    # Create the time embedding
    time_feature_size = hypers.hidden_size
    self.time_features = TimeFeatures(embedding_size=2*time_feature_size,
                                       out_features=time_feature_size,
                                       key=k3)

    # Create the encoder and decoder
    gru_hypers = StackedGRUSequenceHypers(hidden_size=hypers.hidden_size,
                                      intermediate_channels=hypers.intermediate_channels,
                                      num_layers=hypers.n_layers)
    encoder = StackedGRURNN(in_channels=cond_in_channels + time_feature_size,
                    out_channels=hypers.hidden_size*hypers.n_layers,
                    hypers=gru_hypers,
                    key=k1)
    decoder = StackedGRURNN(in_channels=in_channels + time_feature_size,
                    out_channels=out_channels,
                    hypers=gru_hypers,
                    key=k2)

    self.encoder = encoder
    self.decoder = decoder

    self.hypers = hypers

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.encoder.batch_size

  def series_to_array(self, series: TimeSeries) -> Float[Array, 'T C']:
    time_features = jax.vmap(self.time_features)(series.times)
    return jnp.concatenate([series.values, time_features], axis=-1)

  def create_context(self, condition_info: TimeSeries) -> Float[Array, 'S C']:
    xts_enc = self.series_to_array(condition_info)
    out = self.encoder(xts_enc)[-1]
    return out.reshape((self.hypers.n_layers, self.hypers.hidden_size))

  def __call__(self,
               series: TimeSeries,
               condition_info: TimeSeries,
               context: Optional[Float[Array, 'S C']] = None) -> TimeSeries:

    if context is None:
      context = self.create_context(condition_info)

    xts_dec = self.series_to_array(series)

    out = self.decoder(xts_dec, context)
    return out

################################################################################################################

class TimeDependentGRUSeq2SeqHypers(AbstractHyperParams):
  hidden_size: int # Hidden size of recurrent part of the GRU
  n_layers: int # Number of layers in the GRU
  intermediate_channels: int # Dimensionality of the intermediate sequence

class TimeDependentGRUSeq2SeqModel(AbstractTimeDependentSeq2SeqModel):
  """Naive GRU-based encoder-decoder model.  The encoder creates inputs for each of
  the decoder layers.
  """

  encoder: StackedGRURNN
  decoder: StackedGRURNN
  time_features: TimeFeatures
  simulation_time_features: TimeFeatures

  hypers: TimeDependentGRUSeq2SeqHypers = eqx.field(static=True)

  def __init__(self,
               cond_in_channels: int,
               in_channels: int,
               out_channels: int,
               hypers: TimeDependentGRUSeq2SeqHypers,
               key: PRNGKeyArray):
    k1, k2, k3 = random.split(key, 3)

    # Create the time embedding
    time_feature_size = hypers.hidden_size
    self.time_features = TimeFeatures(embedding_size=2*time_feature_size,
                                       out_features=time_feature_size,
                                       key=k3)

    self.simulation_time_features = TimeFeatures(embedding_size=2*time_feature_size,
                                       out_features=time_feature_size,
                                       key=k2)

    # Create the encoder and decoder
    gru_hypers = StackedGRUSequenceHypers(hidden_size=hypers.hidden_size,
                                      intermediate_channels=hypers.intermediate_channels,
                                      num_layers=hypers.n_layers)
    encoder = StackedGRURNN(in_channels=cond_in_channels + time_feature_size,
                    out_channels=hypers.hidden_size*hypers.n_layers,
                    hypers=gru_hypers,
                    key=k1)
    decoder = StackedGRURNN(in_channels=in_channels + 2*time_feature_size,
                    out_channels=out_channels,
                    hypers=gru_hypers,
                    key=k2)

    self.encoder = encoder
    self.decoder = decoder

    self.hypers = hypers

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.encoder.batch_size

  def series_to_array(self, series: Union[TimeSeries, None]) -> Float[Array, 'T C']:
    time_features = jax.vmap(self.time_features)(series.times)
    return jnp.concatenate([series.values, time_features], axis=-1)

  def create_context(self, condition_info: TimeSeries) -> Float[Array, 'S C']:
    xts_enc = self.series_to_array(condition_info)
    out = self.encoder(xts_enc)[-1]
    return out.reshape((self.hypers.n_layers, self.hypers.hidden_size))

  def __call__(self,
               s: Scalar,
               series: TimeSeries,
               condition_info: Optional[TimeSeries] = None, # Only optional if context is provided
               context: Optional[Float[Array, 'S C']] = None) -> TimeSeries:

    if condition_info is None:
      condition_info = TimeSeries(times=series.times, values=jnp.zeros(series.values.shape))

    if context is None:
      context = self.create_context(condition_info)

    xts_dec = self.series_to_array(series)

    sim_time_features = self.simulation_time_features(s)
    sim_time_features = jnp.broadcast_to(sim_time_features[None], (xts_dec.shape[0], sim_time_features.shape[-1]))

    xts_with_sim_time_features = jnp.concatenate([xts_dec, sim_time_features], axis=-1)

    out = self.decoder(xts_with_sim_time_features, context)
    return out
