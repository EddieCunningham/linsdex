import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Annotated, Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterable, Literal, List
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import jax.tree_util as jtu
from linsdex.potential.gaussian.dist import MixedGaussian, NaturalGaussian, StandardGaussian
import linsdex.util as util
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.tags import TAGS
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from plum import dispatch

__all__ = ['GaussianPotentialSeries']

def _process_potentials(
    _,
    ts: Float[Array, 'N'],
    xts: Float[Array, 'N D'],
    standard_deviation: Optional[Float[Array, 'N D']] = None,
    certainty: Union[Float[Array, 'N D'], None] = None,
    parameterization: Optional[Literal['natural', 'mixed', 'standard']] = 'natural'
  ) -> Annotated[AbstractPotential, 'N']:
    """A probabilistic time series is a time series of Gaussian potential functions for the latent variable

    **Arguments**:
      - `ts`: The times at which the potential functions are evaluated
      - `xts`: The potential functions evaluated at each time
      - `standard_deviation`: The standard deviation of the potential functions.
                              If None, then we assume that the potential functions are fully certain.
      - `certainty`: The certainty (inverse of standard deviation) of the potential functions.
                     Positive values indicate certainty.  0 for fully uncertain and None if fully certain.
      - `parameterization`: The parameterization of the potential functions.  Defaults to 'natural'.

    **Returns**:
      - Batched AbstractPotential
    """
    if certainty is not None:
      if xts.shape != certainty.shape:
        raise ValueError("xts and certainty must have the same shape")
    if ts.shape != xts.shape[:-1]:
      raise ValueError("ts must have the same shape as xts except for the last dimension")

    # Determine which potentials are fully certain and which are fully uncertain
    if standard_deviation is None and certainty is None:
      # We will be fully certain for all observations
      standard_deviation = jnp.zeros_like(xts)
      certainty = jnp.ones_like(xts)*jnp.inf
      is_fully_certain = jnp.ones_like(ts, dtype=bool)
      is_fully_uncertain = jnp.zeros_like(ts, dtype=bool)
      parameterization = 'mixed' # Only used mixed parameterization if we're doing a bridge

    elif standard_deviation is not None and certainty is not None:
      raise ValueError("Both standard_deviation and certainty cannot be provided")

    elif standard_deviation is None:
      # Use certainty to determine which potentials are fully certain
      is_fully_certain = jnp.isinf(certainty)
      is_fully_uncertain = jnp.where(jnp.abs(certainty) < 1e-10, True, False)

      is_fully_certain = jnp.all(is_fully_certain, axis=-1)
      is_fully_uncertain = jnp.all(is_fully_uncertain, axis=-1)

    elif certainty is None:
      # Use standard deviation to determine which potentials are fully certain
      is_fully_certain = jnp.where(jnp.abs(standard_deviation) < 1e-10, True, False)
      is_fully_uncertain = jnp.isinf(standard_deviation)

      certainty = 1/standard_deviation
      certainty = jnp.where(is_fully_uncertain, 0.0, certainty)
      certainty = jnp.where(is_fully_certain, jnp.inf, certainty)

      is_fully_certain = jnp.all(is_fully_certain, axis=-1)
      is_fully_uncertain = jnp.all(is_fully_uncertain, axis=-1)

    else:
      raise ValueError("Either standard_deviation or certainty must be provided")

    assert is_fully_certain.shape == ts.shape
    assert is_fully_uncertain.shape == ts.shape

    # Turn the unceratinty into a Matrix type
    Jinv = jax.vmap(util.to_matrix)(certainty)

    # Certainties of inf correspond to fully certain potentials
    def set_total_certainty(Jinv, mask):
      def set_totally_certain(Jinv):
        return eqx.tree_at(lambda x: x.tags, Jinv, TAGS.inf_tags)
      return util.where(mask, set_totally_certain(Jinv), Jinv)
    Jinv = jax.vmap(set_total_certainty)(Jinv, is_fully_certain)

    # Certainties of 0 correspond to fully uncertain potentials
    def set_total_uncertainty(Jinv, mask):
      def set_totally_uncertain(Jinv):
        return eqx.tree_at(lambda x: x.tags, Jinv, TAGS.zero_tags)
      return util.where(mask, set_totally_uncertain(Jinv), Jinv)
    Jinv = jax.vmap(set_total_uncertainty)(Jinv, is_fully_uncertain)

    # Create the potentials
    def process_potential(x: Float[Array, 'D'],
                          Jinv: AbstractSquareMatrix):

      if parameterization == 'natural':
        h = Jinv@x
        potential = NaturalGaussian(Jinv, h)
      elif parameterization == 'mixed':
        potential = MixedGaussian(x, Jinv)
      elif parameterization == 'standard':
        potential = StandardGaussian(x, Jinv.get_inverse())
      else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
      return potential

    # Construct the node potentials
    node_potentials = jax.vmap(process_potential)(xts, Jinv)
    return node_potentials

################################################################################################################

class GaussianPotentialSeries(AbstractBatchableObject):
  """This class represents potential functions at different times.  It is what we use to condition
  SDEs."""

  times: Float[Array, 'N']
  node_potentials: Annotated[AbstractPotential, 'N']

  def __init__(self,
               ts: Float[Array, 'N'],
               xts: Float[Array, 'N D'],
               standard_deviation: Optional[Float[Array, 'N D']] = None,
               certainty: Union[Float[Array, 'N D'], None] = None,
               parameterization: Optional[Literal['natural', 'mixed', 'standard']] = 'natural'):
    """A probabilistic time series is a time series of Gaussian potential functions for the latent variable.
    This initializer will work if the inputs are batched.

    **Arguments**:
      - `ts`: The times at which the potential functions are evaluated
      - `xts`: The potential functions evaluated at each time
      - `standard_deviation`: The standard deviation of the potential functions.
                              If None, then we assume that the potential functions are fully certain.
      - `certainty`: The certainty (inverse of standard deviation) of the potential functions.
                     Positive values indicate certainty.  0 for fully uncertain and None if fully certain.
      - `parameterization`: The parameterization of the potential functions.  Defaults to 'natural' if
                            there is any (un)certainty provided.  Otherwise defaults to 'mixed'.
    """
    if ts.ndim == 0:
      assert xts.ndim == 1
      ts = ts[None]
      xts = xts[None]

    if isinstance(xts, AbstractPotential):
      # Hack to get easy initialization
      self.times = ts
      self.node_potentials = xts
      return

    if certainty is not None:
      if xts.shape != certainty.shape:
        raise ValueError("xts and certainty must have the same shape")
    if ts.shape != xts.shape[:-1]:
      raise ValueError("ts must have the same shape as xts except for the last dimension")

    self.times = ts
    self.node_potentials = auto_vmap(_process_potentials)(self,
                                                          ts,
                                                          xts,
                                                          standard_deviation,
                                                          certainty,
                                                          parameterization=parameterization)

  @classmethod
  def from_potentials(cls, ts: Float[Array, 'N'], node_potentials: AbstractPotential):
    """Alternative initializer that directly takes node potentials"""
    return GaussianPotentialSeries(ts, node_potentials)

  @property
  def batch_size(self):
    if self.times.ndim == 1:
      return None
    elif self.times.ndim == 2:
      return self.times.shape[0]
    else:
      return self.times.shape[:-1]

  def __len__(self):
    return self.times.shape[-1]

  @property
  def dim(self):
    return self.node_potentials.dim

  @property
  def is_fully_uncertain(self):
    return self.node_potentials.J.tags.is_zero

  @property
  def is_fully_certain(self):
    return self.node_potentials.J.tags.is_inf

  def to_mixed(self):
    return eqx.tree_at(lambda x: x.node_potentials, self, self.node_potentials.to_mixed())

  def to_nat(self):
    return eqx.tree_at(lambda x: x.node_potentials, self, self.node_potentials.to_nat())

  def to_std(self):
    return eqx.tree_at(lambda x: x.node_potentials, self, self.node_potentials.to_std())

  def make_windowed_batches(self, window_size: int):
    """Turn a single TimeSeries into a batch of TimeSeries from windows of size window_size"""
    from linsdex.series.series import _make_windowed_batches
    return _make_windowed_batches(self, window_size)
