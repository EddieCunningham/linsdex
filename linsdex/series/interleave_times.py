import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterable, Literal, List
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import jax.tree_util as jtu
import linsdex.util as util
from linsdex.series.batchable_object import AbstractBatchableObject

__all__ = ['InterleavedTimes']

class InterleavedTimes(AbstractBatchableObject):
  new_indices: Float[Array, 'T_new']
  base_indices: Float[Array, 'T_old']
  times: Float[Array, 'T_new + T_old']

  def __init__(self, new_times: Union[Float[Array, 'T2'], None], base_times: Float[Array, 'T1']):
    if new_times is None or new_times.size == 0:
      self.new_indices = None
      self.base_indices = jnp.arange(len(base_times))
      self.times = base_times
    elif base_times is None or base_times.size == 0:
      self.new_indices = jnp.arange(len(new_times))
      self.base_indices = None
      self.times = new_times
    else:

      # Find the position of the old potentials in the expanded node potentials
      indices_for_base_times = jnp.arange(len(base_times))
      def get_new_index(old_index):
        # Count the number of new times that are less than the old time
        return old_index + (new_times <= base_times[old_index]).sum()
      new_indices_for_base_times = jax.vmap(get_new_index)(indices_for_base_times)

      # Find the positions of the new times in the expanded node potentials
      indices_for_new_times = jnp.arange(len(new_times))
      def get_new_index(old_index):
        return old_index + (base_times < new_times[old_index]).sum()
      new_indices_for_new_times = jax.vmap(get_new_index)(indices_for_new_times)

      # Get the combined times
      combined_ts = jnp.concatenate([base_times, new_times])
      sorted_indices = jnp.argsort(combined_ts)
      ts = combined_ts[sorted_indices]

      self.new_indices = new_indices_for_new_times
      self.base_indices = new_indices_for_base_times
      self.times = ts

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.times.ndim == 1:
      return None
    elif self.times.ndim == 2:
      return self.times.shape[0]
    elif self.times.ndim > 2:
      return self.times.shape[:-1]
    else:
      raise ValueError(f"ts has {self.times.ndim} dimensions, which is not supported")

  @property
  def new_indices_mask(self):
    new_mask = jnp.ones(self.new_indices.shape, dtype=bool)
    base_mask = jnp.zeros(self.base_indices.shape, dtype=bool)
    return self.interleave(new_mask, base_mask)

  @property
  def base_indices_mask(self):
    new_mask = jnp.zeros(self.new_indices.shape, dtype=bool)
    base_mask = jnp.ones(self.base_indices.shape, dtype=bool)
    return self.interleave(new_mask, base_mask)

  @property
  def new_times(self):
    return self.times[self.new_indices]

  @property
  def base_times(self):
    return self.times[self.base_indices]

  def transpose(self):
    return InterleavedTimes(new_times=self.base_indices, base_times=self.new_indices)

  def interleave(self, new_xts: Float[Array, 'T_new D'], base_xts: Float[Array, 'T_old D']) -> Float[Array, 'T_new + T_old D']:
    """Interleave the new times and positions with the base times and positions"""

    if self.new_indices is None:
      return base_xts

    new_params, new_static = eqx.partition(new_xts, eqx.is_array)
    base_params, base_static = eqx.partition(base_xts, eqx.is_array)

    # Allocate memory for the combined times and positions
    combined_ts = self.times
    T = combined_ts.shape[0]
    def zeros_like(x):
      return jnp.zeros((T, *x.shape[1:]), dtype=x.dtype)
    combined_params = jtu.tree_map(zeros_like, base_params)

    # Fill the buffer with the base times
    filled_params = util.fill_array(combined_params, self.base_indices, base_params)

    # Fill the buffer with the new times
    combined_params = util.fill_array(filled_params, self.new_indices, new_params)

    combined = eqx.combine(combined_params, base_static)
    return combined

  def filter_base_times(self, xts):
    return jtu.tree_map(lambda x: x[self.base_indices], xts)

  def filter_new_times(self, xts):
    return jtu.tree_map(lambda x: x[self.new_indices], xts)

################################################################################################################
