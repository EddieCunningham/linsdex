import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type, Iterable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import lineax as lx
import abc
import warnings
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.crf.crf import CRF, Messages
from plum import dispatch
import linsdex.util as util
from jax._src.util import curry
from linsdex.series.series import TimeSeries
from linsdex.series.interleave_times import InterleavedTimes
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries

__all__ = ['DiscretizeResult',
           'AbstractContinuousCRF']

################################################################################################################

class DiscretizeResult(eqx.Module):
  """This represents the dicretization of a stochastic process at a given set of times.
  We assume that this stochastic process is already conditioned on some evidence at
  some other times.  The `info` attribute represents the mapping between the new times and the
  old times.  The `crf` attribute represents the discretization of the process at the new times."""
  crf: CRF
  info: InterleavedTimes

class AbstractContinuousCRF(AbstractBatchableObject, abc.ABC):
  """Represents a stochastic process that is conditioned on some evidence (potentials)
  at some times.  `parallel` attribute represents whether CRF message passing should be parallelized."""

  evidence: eqx.AbstractVar[GaussianPotentialSeries]
  parallel: bool = eqx.field(static=True)

  def __init__(
    self,
    evidence: GaussianPotentialSeries,
    parallel: bool = False,
  ):
    assert isinstance(evidence, GaussianPotentialSeries)
    self.evidence = evidence
    self.parallel = parallel

  @property
  def times(self) -> Float[Array, 'T']:
    return self.evidence.times

  @property
  def node_potentials(self) -> AbstractPotential:
    return self.evidence.node_potentials

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    node_batch_size = self.node_potentials.batch_size
    if isinstance(node_batch_size, Iterable):
      return node_batch_size[-2]
    elif isinstance(node_batch_size, int):
      return None
    else:
      raise ValueError(f"Invalid batch size: {node_batch_size}")

  @abc.abstractmethod
  def get_base_transition_distribution(
    self,
    s: Float[Array, 'D'],
    t: Float[Array, 'D']
  ) -> AbstractTransition:
    """Get the transition distribution between two times.  If this is tractable and
    compatible with the evidence, then we inference in the conditioned process is tractable.
    """
    pass

  def condition_on(self, new_evidence: GaussianPotentialSeries) -> 'AbstractContinuousCRF':
    """Condition this process on new evidence.  This will return a new process that is conditioned
    on the new evidence and the old evidence.

    Args:
      new_evidence: The new evidence to condition on.  This should be a PotentialSeries with the same
        times as the old evidence.

    Returns:
      A new process that is conditioned on the new evidence and the old evidence.
    """
    new_times, base_times = self.evidence.times, new_evidence.times
    info = InterleavedTimes(new_times, base_times)
    combined_evidence = info.interleave(self.evidence, new_evidence)
    return type(self).__init__(combined_evidence, parallel=self.parallel)

  def log_prob(self, series: TimeSeries) -> Scalar:
    """Evaluate the log of the probability of x under the distribution"""
    result = self.discretize(series.times)
    crf = result.crf
    info = result.info
    return crf.marginalize(info.new_indices).log_prob(series.values)

  def sample(self, key: PRNGKeyArray, ts: Float[Array, 'T']) -> TimeSeries:
    """Sample from this process at the given times by discretizing it at the given times.
    This will return a TimeSeries object with the samples at the new times.

    Args:
      key: The PRNG key to use for sampling.
      ts: The times at which to sample the process.

    Returns:
      A TimeSeries object with the samples at the new times.
    """
    crf_result = self.discretize(ts)
    crf = crf_result.crf
    info = crf_result.info
    samples = crf.sample(key)
    xts = info.filter_new_times(samples)
    return TimeSeries(ts, xts)

  def discretize(
    self,
    ts: Union[Float[Array, 'T'], None] = None,
    info: Optional[InterleavedTimes] = None
  ) -> Union[DiscretizeResult, CRF]:
    """Discretize this process at the given times.  If `ts` is None, then we will use the times
    of the evidence.  If `info` is provided, then we will use it to map the new times to the old times.

    Args:
      ts: The times at which to discretize the process.  If None, then we will use the times of the evidence.
      info: The mapping between the new times and the old times.  If None, then we will construct it from `ts` and `self.times`.

    Returns:
      A DiscretizeResult object containing the discretized CRF and the mapping between the new times and the old times.
    """
    if ts is not None:
      assert ts.ndim == 1

    # Interleave these with the new times
    if info is None:
      info = InterleavedTimes(ts, self.times)
    all_ts = info.times
    assert all_ts.shape[-1] > 1, 'There must be at least 2 times (including the times of this continuous CRF and the new times) when discretizing!'

    # Make a set of empty node potentials
    zero = self.node_potentials[0].total_uncertainty_like(self.node_potentials[0])
    def make_zero(i):
      return zero
    node_potentials = eqx.filter_vmap(make_zero)(jnp.arange(all_ts.shape[-1]))

    # Place our priors in the node potentials
    node_potentials = util.fill_array(node_potentials, info.base_indices, self.node_potentials)

    # Make the transitions for the new times
    s, t = all_ts[:-1], all_ts[1:]
    def make_transition_potential(s, t):
      return self.get_base_transition_distribution(s, t)

    transitions = eqx.filter_vmap(make_transition_potential)(s, t)

    crf = CRF(node_potentials, transitions, parallel=self.parallel)
    if ts is None:
      return crf
    return DiscretizeResult(crf, info)
