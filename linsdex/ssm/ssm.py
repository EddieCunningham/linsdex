import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterable
import einops
import abc
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree, Int
import jax.tree_util as jtu
from linsdex.util import misc as util
from linsdex.series.series import TimeSeries
from linsdex.series.interleave_times import InterleavedTimes
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.crf.crf import CRF

class AbstractStateSpaceModel(AbstractPotential, abc.ABC):
  """Abstract class for state space models.  These are chains with emissions and a prior."""

  prior: eqx.AbstractVar[AbstractPotential]
  transitions: eqx.AbstractVar[AbstractTransition]
  emissions: eqx.AbstractVar[AbstractTransition]
  parallel: eqx.AbstractVar[Optional[bool]]

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    node_batch_size = self.emissions.batch_size
    if isinstance(node_batch_size, Iterable):
      return node_batch_size[-2]
    elif isinstance(node_batch_size, int):
      return None
    else:
      raise ValueError(f"Invalid batch size: {node_batch_size}")

  def __len__(self):
    node_batch_size = self.emissions.batch_size
    if isinstance(node_batch_size, int):
      return node_batch_size
    elif isinstance(node_batch_size, Iterable):
      return node_batch_size[-1]
    else:
      raise ValueError(f"Invalid batch size: {node_batch_size}")

  def __call__(self, xs: Float[Array, 'N Dx'], ys: Float[Array, 'N Dy']) -> Scalar:
    prior_energy = self.prior(xs[0])
    transition_energy = jax.vmap(self.transitions)(xs[:-1], xs[1:]).sum()
    emission_energy = jax.vmap(self.emissions)(xs, ys).sum()
    return prior_energy + transition_energy + emission_energy

  def normalizing_constant(self) -> Scalar:
    return jnp.array(1.0)

  def log_prob(self, xs: Float[Array, 'N Dx'], ys: Float[Array, 'N Dy']) -> Scalar:

    def log_likelihood(xt, xtp1, transition):
      pxtp1 = transition.condition_on_x(xt)
      return pxtp1.log_prob(xtp1)

    log_p0 = self.prior.log_prob(xs[0])
    log_xkp1_xk = jax.vmap(log_likelihood)(xs[:-1], xs[1:], self.transitions)
    log_yt_xk = jax.vmap(self.emissions.log_prob)(xs, ys)
    return log_p0 + log_xkp1_xk.sum() + log_yt_xk.sum()

  def sample(self, key: PRNGKeyArray) -> Tuple[Float[Array, 'N Dx'], Float[Array, 'N Dy']]:
    keys = random.split(key, len(self))
    x0 = self.prior.sample(key)

    from linsdex.potential.gaussian.transition import GaussianTransition
    if isinstance(self.transitions, GaussianTransition) and self.parallel:
      # If Gaussian chain, then we can do parallel sampling
      from linsdex.potential.gaussian.transition import gaussian_chain_parallel_sample
      xts = gaussian_chain_parallel_sample(self.transitions, x0, keys[1:])

    else:

      def forward_sampling(carry, inputs):
        xt = carry
        transition, key = inputs

        xtp1 = transition.condition_on_x(xt).sample(key)
        return xtp1, xtp1

      # Otherwise, do sequential sampling
      xT, xts = jax.lax.scan(forward_sampling, x0, (self.transitions, keys[1:]))

      # Concatenate the first point
      xts = jtu.tree_map(lambda x0, x: jnp.concatenate([x0[None], x]), x0, xts)

    # Sample the emissions
    def sample_emissions(xt, emission, key):
      return emission.condition_on_x(xt).sample(key)

    keys = random.split(keys[0], len(self))
    yts = eqx.filter_vmap(sample_emissions)(xts, self.emissions, keys)
    return xts, yts

  def get_posterior(self, ys: Float[Array, 'N Dy']) -> CRF:
    assert ys.ndim == 2
    def make_emission_potential(emission, y):
      return emission.condition_on_y(y)
    emission_potentials = eqx.filter_vmap(make_emission_potential)(self.emissions, ys)
    first_potential = self.prior + emission_potentials[0]
    potentials = util.fill_array(emission_potentials, 0, first_potential)
    return CRF(potentials, self.transitions, parallel=self.parallel)