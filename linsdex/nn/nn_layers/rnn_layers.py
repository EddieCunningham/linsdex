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

################################################################################################################

class CausalConv1d(AbstractBatchableObject):
  conv1d: eqx.nn.Conv1d
  kernel_width: int = eqx.field(static=True)
  stride: int = eqx.field(static=True)
  dilation: int = eqx.field(static=True)
  use_bias: bool = eqx.field(static=True)

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_width: int = 3,
    stride: int = 1,
    dilation: int = 1,
    use_bias: bool = True,
    *,
    key: PRNGKeyArray
  ):
    padding = kernel_width - 1

    self.conv1d = eqx.nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_width,
                                stride=stride,
                                padding=padding,
                                use_bias=use_bias,
                                key=key)
    self.kernel_width = kernel_width
    self.stride = stride
    self.dilation = dilation
    self.use_bias = use_bias

  @property
  def padding(self) -> int:
    return self.kernel_width - 1

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

class WaveNetResBlock(AbstractBatchableObject):
  """https://arxiv.org/pdf/1609.03499"""
  gating_conv: CausalConv1d
  filter_conv: CausalConv1d
  out_conv: CausalConv1d
  skip_conv: CausalConv1d

  cond_filter_conv: Optional[CausalConv1d] = None
  cond_gating_conv: Optional[CausalConv1d] = None

  kernel_width: int = eqx.field(static=True)
  dilation: int = eqx.field(static=True)
  hidden_channels: int = eqx.field(static=True)

  def __init__(self,
               in_channels: int,
               kernel_width: int = 2,
               dilation: int = 1,
               hidden_channels: int = 32,
               cond_channels: Optional[int] = None,
               *,
               key: PRNGKeyArray):
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)

    self.gating_conv = CausalConv1d(in_channels=in_channels,
                                    out_channels=hidden_channels,
                                    kernel_width=kernel_width,
                                    stride=1,
                                    dilation=dilation,
                                    use_bias=True,
                                    key=k1)
    self.filter_conv = CausalConv1d(in_channels=in_channels,
                                    out_channels=hidden_channels,
                                    kernel_width=kernel_width,
                                    stride=1,
                                    dilation=dilation,
                                    use_bias=True,
                                    key=k2)

    self.out_conv = CausalConv1d(in_channels=hidden_channels,
                                 out_channels=in_channels,
                                 kernel_width=1,
                                 stride=1,
                                 dilation=1,
                                 use_bias=True,
                                 key=k3)
    self.skip_conv = CausalConv1d(in_channels=hidden_channels,
                                  out_channels=in_channels,
                                  kernel_width=1,
                                  stride=1,
                                  dilation=1,
                                  use_bias=True,
                                  key=k4)

    self.kernel_width = kernel_width
    self.dilation = dilation
    self.hidden_channels = hidden_channels

    if cond_channels is not None:
      self.cond_filter_conv = CausalConv1d(in_channels=cond_channels,
                                    out_channels=in_channels,
                                    kernel_width=1,
                                    stride=1,
                                    dilation=1,
                                    use_bias=True,
                                    key=k5)
      self.cond_gating_conv = CausalConv1d(in_channels=cond_channels,
                                    out_channels=in_channels,
                                    kernel_width=1,
                                    stride=1,
                                    dilation=1,
                                    use_bias=True,
                                    key=k5)
    else:
      self.cond_filter_conv = None
      self.cond_gating_conv = None

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.gating_conv.batch_size

  def __call__(
    self,
    x: Float[Array, 'T C'],
    y: Optional[Float[Array, 'T C']] = None,
    h: Optional[Float[Array, 'T C']] = None,
  ) -> Tuple[Float[Array, 'D'], Float[Array, 'T C']]:
    if h is not None:
      raise NotImplementedError("Conditioning on global context is not implemented yet.")

    sigmoid = jax.nn.sigmoid
    tanh = jax.nn.tanh

    Wfx = self.filter_conv(x)
    Wgx = self.gating_conv(x)

    if self.cond_filter_conv is not None:
      assert y is not None
      Vfy = self.cond_filter_conv(y)
      Vgy = self.cond_gating_conv(y)

      Wfx = Wfx + Vfy
      Wgx = Wgx + Vgy

    gate_out = sigmoid(Wgx)
    filter_out = tanh(Wfx)

    p = gate_out * filter_out
    out = self.out_conv(p)

    new_hidden = out + x
    skip = self.skip_conv(p)
    return new_hidden, skip

################################################################################################################

class GRURNN(AbstractBatchableObject):

  initial_state: Float[Array, 'hidden_size'] # Input to left of the GRU cell
  gru: eqx.nn.GRUCell # Recurrent part from left to right
  out_proj: eqx.nn.Linear # Output to the top of the GRU cell

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    hidden_size: int,
    *,
    key: PRNGKeyArray
  ):
    k1, k2 = random.split(key, 2)

    self.gru = eqx.nn.GRUCell(input_size=in_channels,
                              hidden_size=hidden_size,
                              use_bias=True,
                              key=k1)

    self.initial_state = jnp.zeros(hidden_size)
    self.out_proj = eqx.nn.Linear(in_features=hidden_size,
                                  out_features=out_channels,
                                  use_bias=True,
                                  key=k2)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.gru.weight_ih.ndim == 2:
      return None
    elif self.gru.weight_ih.ndim == 3:
      return self.gru.weight_ih.shape[0]
    elif self.gru.weight_ih.ndim > 3:
      return self.gru.weight_ih.shape[:-2]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.gru.weight_ih.ndim}")

  @auto_vmap
  def __call__(
    self,
    xts: Float[Array, 'T C'],
    global_context: Optional[Float[Array, 'hidden_size']] = None
  ) -> Float[Array, 'T C']:

    hidden_state = self.initial_state
    if global_context is not None:
      hidden_state = hidden_state + global_context

    # Scan over the time axis
    def scan_body(carry, inputs):
      hidden_state_in = carry
      x = inputs
      hidden_state_out = self.gru(x, hidden_state_in)
      return hidden_state_out, hidden_state_out

    last_hidden_state, hidden_states = jax.lax.scan(scan_body, hidden_state, xts)
    out = jax.vmap(self.out_proj)(hidden_states)
    return out

################################################################################################################
