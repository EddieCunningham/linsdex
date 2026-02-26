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
from linsdex import StandardGaussian
from linsdex import AbstractBatchableObject, auto_vmap
import abc
from linsdex.nn.nn_models.nn_abstract import AbstractHyperParams

################################################################################################################

class CausalConv1dHypers(AbstractHyperParams):
  kernel_width: int = 3
  stride: int = 1
  dilation: int = 1
  use_bias: bool = True

  @property
  def padding(self) -> int:
    return self.kernel_width - 1

class CausalConv1d(AbstractBatchableObject):
  conv1d: eqx.nn.Conv1d
  hypers: CausalConv1dHypers

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    hypers: Optional[CausalConv1dHypers] = CausalConv1dHypers(),
    *,
    key: PRNGKeyArray
  ):
    self.conv1d = eqx.nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=hypers.kernel_width,
                                stride=hypers.stride,
                                padding=hypers.padding,
                                use_bias=hypers.use_bias,
                                key=key)
    self.hypers = hypers



  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.conv1d.weight.ndim == 3:
      return None
    elif self.conv1d.weight.ndim == 4:
      return self.conv1d.weight.shape[0]
    elif self.conv1d.weight.ndim > 4:
      return self.conv1d.weight.shape[:-3]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.conv1d.weight.ndim}")

  @auto_vmap
  def __call__(self, x: Float[Array, 'T Din']) -> Float[Array, 'T Dout']:
    return self.conv1d(x.T)[:,:x.shape[0]].T

################################################################################################################

class WaveNetResBlockHypers(AbstractHyperParams):
  kernel_width: int = 2
  dilation: int = 1
  hidden_channels: int = 32

class WaveNetResBlock(AbstractBatchableObject):
  gating_conv: CausalConv1d
  filter_conv: CausalConv1d
  out_conv: CausalConv1d
  skip_conv: CausalConv1d

  def __init__(self,
               in_channels: int,
               hypers: Optional[WaveNetResBlockHypers] = WaveNetResBlockHypers(),
               *,
               key: PRNGKeyArray):
    k1, k2, k3, k4 = random.split(key, 4)

    dilation_conv_hypers = CausalConv1dHypers(kernel_width=hypers.kernel_width,
                                     stride=1,
                                     dilation=hypers.dilation,
                                     use_bias=True)

    self.gating_conv = CausalConv1d(in_channels=in_channels,
                                    out_channels=hypers.hidden_channels,
                                    hypers=dilation_conv_hypers,
                                    key=k1)
    self.filter_conv = CausalConv1d(in_channels=in_channels,
                                    out_channels=hypers.hidden_channels,
                                    hypers=dilation_conv_hypers,
                                    key=k2)


    conv1x1_hypers = CausalConv1dHypers(kernel_width=1,
                                        stride=1,
                                        dilation=1,
                                        use_bias=True)

    self.out_conv = CausalConv1d(in_channels=hypers.hidden_channels,
                                 out_channels=in_channels,
                                 hypers=conv1x1_hypers,
                                 key=k3)
    self.skip_conv = CausalConv1d(in_channels=hypers.hidden_channels,
                                  out_channels=in_channels,
                                  hypers=conv1x1_hypers,
                                  key=k4)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.gating_conv.batch_size

  def __call__(self, x: Float[Array, 'T C']) -> Tuple[Float[Array, 'T C'], Float[Array, 'T C']]:
    # sigmoid = gx.square_sigmoid
    # tanh = lambda x: 2*sigmoid(x) - 1

    sigmoid = jax.nn.sigmoid
    tanh = jax.nn.tanh

    gate_out = sigmoid(self.gating_conv(x))
    filter_out = tanh(self.filter_conv(x))


    p = gate_out * filter_out
    out = self.out_conv(p)

    new_hidden = out + x
    skip = self.skip_conv(p)
    return new_hidden, skip

################################################################################################################

class WaveNetHypers(AbstractHyperParams):
  dilations: Int[Array, 'N']
  initial_filter_width: int = 4
  filter_width: int = 2
  residual_channels: int = 32
  dilation_channels: int = 32
  skip_channels: int = 32

class WaveNet(AbstractBatchableObject):
  blocks: List[WaveNetResBlock]

  in_projection_conv: CausalConv1d
  skip_conv: CausalConv1d
  out_projection_conv: CausalConv1d

  strict_autoregressive: bool = eqx.field(static=True)

  def __init__(self,
               in_channels: int,
               out_channels: int,
               hypers: WaveNetHypers,
               key: PRNGKeyArray,
               strict_autoregressive: bool = False):
    k1, k2, k3, k4 = random.split(key, 4)

    # Create the first projection
    initial_hypers = CausalConv1dHypers(kernel_width=hypers.initial_filter_width,
                                        stride=1,
                                        dilation=1,
                                        use_bias=True)
    self.in_projection_conv = CausalConv1d(in_channels=in_channels,
                                           out_channels=hypers.residual_channels,
                                           hypers=initial_hypers,
                                           key=k1)

    # Create the intermediate blocks
    def make_block(dilation: int, key: PRNGKeyArray) -> WaveNetResBlock:
      block_hypers = WaveNetResBlockHypers(kernel_width=hypers.filter_width,
                                     dilation=dilation,
                                     hidden_channels=hypers.dilation_channels)
      return WaveNetResBlock(in_channels=hypers.residual_channels,
                             hypers=block_hypers,
                             key=key)

    keys = random.split(k2, len(hypers.dilations))
    self.blocks = jax.vmap(make_block)(hypers.dilations, keys)

    # Create the final projections
    conv1x1_hypers = CausalConv1dHypers(kernel_width=1,
                                        stride=1,
                                        dilation=1,
                                        use_bias=True)
    self.skip_conv = CausalConv1d(in_channels=hypers.residual_channels,
                                   out_channels=hypers.skip_channels,
                                   hypers=conv1x1_hypers,
                                   key=k3)
    self.out_projection_conv = CausalConv1d(in_channels=hypers.skip_channels,
                                            out_channels=out_channels,
                                            hypers=conv1x1_hypers,
                                            key=k4)

    self.strict_autoregressive = strict_autoregressive

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.blocks[0].batch_size

  @auto_vmap
  def __call__(self, x: Float[Array, 'T C']) -> Float[Array, 'T C']:
    if self.strict_autoregressive:
      x = jnp.pad(x, ((1, 0), (0, 0)))

    # Initial projection
    hidden = self.in_projection_conv(x)

    # Residual blocks
    params, static = eqx.partition(self.blocks, eqx.is_array)
    def f(hidden, params):
      block = eqx.combine(params, static)
      hidden, out_partial = block(hidden)
      return hidden, (hidden, out_partial)

    # last_hidden, (all_hiddens, outs) = scan(f, hidden, params)
    last_hidden, (all_hiddens, outs) = jax.lax.scan(f, hidden, params)
    out_pre_swish = outs.sum(axis=0)

    # Output projection
    out = jax.nn.swish(out_pre_swish)
    out = self.skip_conv(out)
    out = jax.nn.swish(out)# + out
    out = self.out_projection_conv(out)

    if self.strict_autoregressive:
      out = out[:-1]

    return out
