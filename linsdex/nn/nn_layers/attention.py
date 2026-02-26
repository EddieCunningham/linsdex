import functools as ft
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import cast, Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Bool, Float, PRNGKeyArray
import equinox as eqx
from linsdex import AbstractBatchableObject, auto_vmap
from typing import Literal, Optional, Union, Tuple, Callable, List, Any

################################################################################################################

class MultiheadAttention(AbstractBatchableObject):
  """Multihead attention from equinox but using JAX's attention algorithm"""

  query_proj: eqx.nn.Linear
  key_proj: eqx.nn.Linear
  value_proj: eqx.nn.Linear
  output_proj: eqx.nn.Linear

  num_heads: int = eqx.field(static=True)
  query_size: int = eqx.field(static=True)
  key_value_size: int = eqx.field(static=True)
  output_size: int = eqx.field(static=True)
  qk_size: int = eqx.field(static=True)
  vo_size: int = eqx.field(static=True)
  causal: bool = eqx.field(static=True)

  query_layer_norm: eqx.nn.LayerNorm
  key_layer_norm: eqx.nn.LayerNorm

  def __init__(
    self,
    num_heads: int,
    query_size: int,
    key_value_size: Optional[int] = None,
    output_size: Optional[int] = None,
    causal: bool = True,
    *,
    key: PRNGKeyArray,
  ):
    qkey, kkey, vkey, okey = jrandom.split(key, 4)

    qk_size = query_size // num_heads
    vo_size = query_size // num_heads
    if output_size is None:
      output_size = query_size

    self.query_proj = eqx.nn.Linear(
      query_size,
      num_heads * qk_size,
      use_bias=True,
      key=qkey,
    )
    self.query_layer_norm = eqx.nn.LayerNorm(shape=(num_heads * qk_size,))
    self.key_proj = eqx.nn.Linear(
      key_value_size,
      num_heads * qk_size,
      use_bias=True,
      key=kkey
    )
    self.key_layer_norm = eqx.nn.LayerNorm(shape=(num_heads * qk_size,))
    self.value_proj = eqx.nn.Linear(
      key_value_size,
      num_heads * vo_size,
      use_bias=True,
      key=vkey,
    )
    self.output_proj = eqx.nn.Linear(
      num_heads * vo_size,
      output_size,
      use_bias=True,
      key=okey,
    )

    self.num_heads = num_heads
    self.query_size = query_size
    self.key_value_size = key_value_size
    self.output_size = output_size
    self.qk_size = qk_size
    self.vo_size = vo_size
    self.causal = causal

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    ndim = self.query_layer_norm.bias.ndim
    if ndim == 1:
      return None
    elif ndim == 2:
      return self.query_layer_norm.bias.shape[0]
    elif ndim > 2:
      return self.query_layer_norm.bias.shape[:-1]
    else:
      raise ValueError(f"Invalid batch size: {ndim}")

  def __call__(
    self,
    query: Float[Array, "q_seq q_size"],
    key_and_value: Float[Array, "kv_seq k_size"]
  ) -> Float[Array, "q_seq o_size"]:

    query_seq_length, _ = query.shape
    kv_seq_length, _ = key_and_value.shape

    query_heads = jax.vmap(self.query_proj)(query)
    key_heads = jax.vmap(self.key_proj)(key_and_value)
    value_heads = jax.vmap(self.value_proj)(key_and_value)

    # Normalize the query and key https://arxiv.org/pdf/2309.14322
    query_heads = eqx.filter_vmap(self.query_layer_norm)(query_heads)
    key_heads = eqx.filter_vmap(self.key_layer_norm)(key_heads)

    # Reshape the query and key
    query_heads = query_heads.reshape(query_seq_length, self.num_heads, -1)
    key_heads = key_heads.reshape(kv_seq_length, self.num_heads, -1)
    value_heads = value_heads.reshape(kv_seq_length, self.num_heads, -1)

    # JAX has fast a implementation!
    attn = jax.nn.dot_product_attention(query_heads,
                                        key_heads,
                                        value_heads,
                                        is_causal=self.causal)

    attn = attn.reshape(query_seq_length, -1)
    return jax.vmap(self.output_proj)(attn)
