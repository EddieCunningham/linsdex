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
from linsdex.nn.nn_layers.attention import MultiheadAttention
from linsdex import AbstractBatchableObject, auto_vmap
import abc
from linsdex.nn.nn_models.wavenet import *
from linsdex import TimeSeries
from linsdex.nn.nn_models.nn_abstract import AbstractEncoderDecoderModel

class TransformerWaveNetResBlockHypers(AbstractHyperParams):
  wavenet_kernel_width: int = 8
  num_transformer_heads: int = 4

class TransformerWaveNetResBlock(AbstractBatchableObject):

  layernorm1: eqx.nn.LayerNorm
  layernorm2: eqx.nn.LayerNorm
  multiheaded_attention: MultiheadAttention
  wavenet_block: WaveNetResBlock

  cross_attention: Optional[MultiheadAttention]
  layernorm3: Optional[eqx.nn.LayerNorm]

  def __init__(self,
               in_channels: int,
               cond_channels: Optional[int] = None,
               hypers: Optional[TransformerWaveNetResBlockHypers] = TransformerWaveNetResBlockHypers(),
               *,
               causal: bool = True,
               key: PRNGKeyArray):
    k1, k2, k3, k4 = random.split(key, 4)

    # Create all of the layer norms
    self.layernorm1 = eqx.nn.LayerNorm(shape=(in_channels,))
    self.layernorm2 = eqx.nn.LayerNorm(shape=(in_channels,))

    # Create the multiheaded attention
    self.multiheaded_attention = MultiheadAttention(num_heads=hypers.num_transformer_heads,
                                        query_size=in_channels,
                                        key_value_size=in_channels,
                                        output_size=in_channels,
                                        causal=causal,
                                        key=k2)

    # Instead of using a feedforward, we'll use a single wavenet block
    wavenet_hypers = WaveNetResBlockHypers(kernel_width=hypers.wavenet_kernel_width,
                                   dilation=1,
                                   hidden_channels=2*in_channels)
    self.wavenet_block = WaveNetResBlock(in_channels=in_channels,
                                         hypers=wavenet_hypers,
                                         key=k1)

    # Optional cross attention
    if cond_channels is not None:
      self.cross_attention = MultiheadAttention(num_heads=hypers.num_transformer_heads,
                                                query_size=in_channels,
                                                key_value_size=cond_channels,
                                                output_size=in_channels,
                                                causal=False,
                                                key=k2)
      self.layernorm3 = eqx.nn.LayerNorm(shape=(in_channels,))
    else:
      self.cross_attention = None
      self.layernorm3 = None

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.wavenet_block.batch_size

  def __call__(self, xts: Float[Array, 'T C'], hts: Optional[Float[Array, 'S C']] = None) -> Tuple[Float[Array, 'T C'], Float[Array, 'T C']]:

    # Layer norm
    yts = jax.vmap(self.layernorm1)(xts)

    # Self attention
    yts = self.multiheaded_attention(query=yts, key_and_value=yts)
    xts = yts + xts

    if hts is not None:
      assert self.cross_attention is not None
      yts = self.cross_attention(query=yts, key_and_value=hts)
      xts = yts + xts

    # Layer norm
    yts = jax.vmap(self.layernorm2)(xts)

    # Wavenet includes final skip connection
    out, _ = self.wavenet_block(yts)
    return out

################################################################################################################

class TransformerWavenetModelHypers(AbstractHyperParams):

  n_blocks: int
  hidden_channel_size: int = 32

  wavenet_kernel_width: int = 8
  num_transformer_heads: int = 4

class TransformerWavenetModel(AbstractTimeDependentEncoderDecoderModel):

  encoder: TransformerWaveNet
  decoder: TransformerWaveNet
  time_features_encoder: TimeFeatures
  time_features_decoder: TimeFeatures

  hypers: TransformerWavenetModelHypers = eqx.field(static=True)

  def __init__(self,
               cond_in_channels: int,
               in_channels: int,
               out_channels: int,
               hypers: TransformerWavenetModelHypers,
               key: PRNGKeyArray,
               strict_autoregressive: bool = False,
               causal_decoder: bool = True):
    k1, k2, k3, k4 = random.split(key, 4)

    # Create the time embedding
    time_feature_size = hypers.hidden_channel_size
    self.time_features_encoder = TimeFeatures(embedding_size=2*time_feature_size,
                                       out_features=time_feature_size,
                                       key=k3)
    self.time_features_decoder = TimeFeatures(embedding_size=2*time_feature_size,
                                       out_features=time_feature_size,
                                       key=k4)

    encoder = TransformerWaveNet(in_channels=cond_in_channels + time_feature_size,
                                 out_channels=hypers.hidden_channel_size,
                                 hypers=hypers,
                                 key=k1,
                                 causal=False, # Don't need autoregression for the encoder
                                 strict_autoregressive=strict_autoregressive)

    decoder = TransformerWaveNet(in_channels=in_channels + time_feature_size,
                                 out_channels=out_channels,
                                 hypers=hypers,
                                 key=k2,
                                 cond_channels=hypers.hidden_channel_size,
                                 causal=causal_decoder,
                                 strict_autoregressive=strict_autoregressive)

    self.encoder = encoder
    self.decoder = decoder

    self.hypers = hypers

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.encoder.batch_size

  def create_context(self, condition_series: TimeSeries) -> Float[Array, 'S C']:
    time_features_enc = jax.vmap(self.time_features_encoder)(condition_series.times)
    xts_enc = jnp.concatenate([condition_series.values, time_features_enc], axis=-1)
    return self.encoder(xts_enc)

  def __call__(self,
               condition_series: TimeSeries,
               latent_series: TimeSeries,
               context: Optional[Float[Array, 'S C']] = None) -> Float[Array, 'T C']:

    if context is None:
      context = self.create_context(condition_series)

    # Create the time embedding
    time_features_dec = jax.vmap(self.time_features_decoder)(latent_series.times)

    # Concatentate the time features with the series
    xts_dec = jnp.concatenate([latent_series.values, time_features_dec], axis=-1)

    out = self.decoder(xts_dec, context)

    return out

################################################################################################################
