import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Dict, overload
import einops
import equinox as eqx
import abc
import diffrax
from jaxtyping import Array, PRNGKeyArray
import jax.tree_util as jtu
import os
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool
from jax._src.util import curry
from linsdex.series.series import TimeSeries
from linsdex.series.interleave_times import InterleavedTimes

################################################################################################################

class AbstractDecoder(eqx.Module, abc.ABC):
  """An emission model that takes as input the latent variable and returns the observed data.
  In a Bayesian setting, this should ideally be a distribution but we're not doing that
  in this project."""

  def __call__(
    self,
    series: TimeSeries
  ) -> TimeSeries:
    """Return the observed data

    **Arguments**:
      - `series`: The latent variable

    **Returns**:
      - `yts`: The observed data
    """
    pass

################################################################################################################

class IdentityDecoder(AbstractDecoder):

  def __call__(
    self,
    series: TimeSeries
  ) -> TimeSeries:
    """Return the observed data

    **Arguments**:
      - `series`: The latent variable

    **Returns**:
      - `series`: The observed data
    """
    return series

################################################################################################################

class PaddingLatentVariableDecoder(AbstractDecoder):

  y_dim: int
  x_dim: int

  def __init__(
    self,
    y_dim: int,
    x_dim: int,
  ):
    """
    **Arguments**:
      - `y_dim`: The dimension of the observed data
      - `x_dim`: The dimension of the latent variable
    """
    self.y_dim = y_dim
    self.x_dim = x_dim

  def __call__(
    self,
    series: TimeSeries
  ) -> TimeSeries:
    """Return the observed data

    **Arguments**:
      - `series`: The latent variable

    **Returns**:
      - `series`: The observed data
    """
    xts = series.yts
    yts = xts[...,:self.y_dim]
    return TimeSeries(series.ts, yts, series.observation_mask[...,:self.y_dim])

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *

  key = random.PRNGKey(0)
  batch_size = 8
  y_dim = 3
  x_dim = 5
  sigma = 1.0