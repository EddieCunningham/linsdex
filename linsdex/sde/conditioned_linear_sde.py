import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type, Dict, Literal
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
from linsdex.crf.continuous_crf import DiscretizeResult, AbstractContinuousCRF
from linsdex.potential.gaussian.dist import MixedGaussian, NaturalGaussian, StandardGaussian, NaturalJointGaussian, GaussianStatistics
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.matrix.matrix_with_inverse import MatrixWithInverse
from linsdex.sde.sde_base import AbstractLinearSDE, AbstractLinearTimeInvariantSDE
from plum import dispatch
import linsdex.util as util
from linsdex.series.series import TimeSeries
from linsdex.series.interleave_times import InterleavedTimes
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from linsdex.series.series import TimeSeries
from linsdex.sde.ode_sde_simulation import ode_solve, ODESolverParams, SDESolverParams, sde_sample, DiffraxSolverState

__all__ = ['ConditionedLinearSDE']

class ConditionedLinearSDE(AbstractLinearSDE, AbstractContinuousCRF):

  sde: AbstractLinearSDE
  evidence: GaussianPotentialSeries

  parallel: bool = eqx.field(static=True)

  def __init__(
    self,
    sde: AbstractLinearSDE,
    evidence: GaussianPotentialSeries,
    parallel: Optional[bool] = None
  ):
    assert isinstance(sde, AbstractLinearSDE)
    assert isinstance(evidence, GaussianPotentialSeries)
    if isinstance(sde, ConditionedLinearSDE):
      # Then combine the two SDEs
      self.sde = sde.sde
      new_times, base_times = sde.evidence.times, evidence.times
      info = InterleavedTimes(new_times, base_times)
      self.evidence = info.interleave(sde.evidence, evidence)
    else:
      self.sde = sde
      self.evidence = evidence
    if parallel is None:
      parallel = jax.devices()[0].platform == 'gpu'
    self.parallel = parallel

  def get_base_transition_distribution(self, s: Float[Array, 'D'], t: Float[Array, 'D']) -> AbstractTransition:
    return self.sde.get_transition_distribution(s, t)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.evidence.batch_size

  @property
  def dim(self) -> int:
    return self.sde.dim

  def get_params(
    self,
    t: Scalar,
    xt: Optional[Float[Array, 'D']] = None,
    *,
    messages: Optional[Messages] = None
  ) -> Tuple[Float[Array, 'D D'],
             Float[Array, 'D'],
             Float[Array, 'D D']]:
    t = jnp.array(t)
    F, u, L = self.sde.get_params(t, xt)
    # F, u, L = self.sde.F, self.sde.u, self.sde.L
    LLT = L@L.T

    # Discretize and then get the backward messages
    crf_result = self.discretize(t[None])
    crf, info = crf_result.crf, crf_result.info

    # Get the index of the new time
    assert info.new_indices.size == 1
    new_index = info.new_indices[0]

    # Get the backward messages
    messages = Messages.from_messages(messages, crf, need_bwd=True)
    bwd = messages.bwd

    # Convert to natural parameters
    bwdt = bwd[new_index]
    if isinstance(bwdt, NaturalGaussian) == False:
      bwdt = bwdt.to_nat()

    F_cond = F - LLT@bwdt.J
    u_cond = u + LLT@bwdt.h

    return F_cond, u_cond, L

  def get_drift(
    self,
    t: Scalar,
    xt: Float[Array, 'D'],
    *,
    messages: Optional[Messages] = None
  ) -> Float[Array, 'D']:
    F, u, L = self.get_params(t, messages=messages)
    return F@xt + u

  def get_diffusion_coefficient(
    self,
    t: Scalar,
    xt: Float[Array, 'D'],
    *,
    messages: Optional[Messages] = None
  ) -> AbstractSquareMatrix:
    _, _, L = self.get_params(t, messages=messages)
    return L

  def get_transition_distribution(
    self,
    s: Scalar,
    t: Scalar,
    *,
    messages: Optional[Messages] = None
  ) -> GaussianTransition:
    st = jnp.array([s, t])

    # Discretize and then get the backward messages
    crf_result: DiscretizeResult = self.discretize(st)
    crf: CRF = crf_result.crf
    info: InterleavedTimes = crf_result.info
    transitions = crf.get_transitions(messages=messages)

    # Get the mask for the parallel scan that says when to reset the scan
    segment_ends = info.new_indices
    reset_mask = jnp.arange(len(crf))[:,None] == segment_ends[None,:]
    reset_mask = reset_mask.any(axis=1)[:-1]

    # Perform the parallel scan
    def operator(right: AbstractTransition, left: AbstractTransition) -> AbstractTransition:
      return left.chain(right) # The transitions are in reverse order

    transitions = util.parallel_segmented_scan(operator, transitions[::-1], reset_mask[::-1])[::-1]
    return transitions[segment_ends[1]]

  def get_marginal(
    self,
    t: Scalar,
    return_messages: bool = False,
    *,
    messages: Optional[Messages] = None
  ) -> Union[AbstractPotential, Tuple[AbstractPotential, AbstractPotential, AbstractPotential]]:
    t = jnp.array(t)

    # Discretize and then get the backward messages
    crf_result = self.discretize(t[None])
    crf, info = crf_result.crf, crf_result.info

    # This is the index of the node at time t
    index = info.new_indices[0]

    # Get the marginals.  Compute the messages out here to avoid recomputing them
    messages = Messages.from_messages(messages, crf, need_fwd=True, need_bwd=True)
    marginals = crf.get_marginals(messages=messages)
    pxt = marginals[index]

    if return_messages:
      return pxt, messages[index]
    else:
      return pxt

  def get_idx_before_t(self, t: Scalar) -> int:
    return jnp.searchsorted(self.times, t, side='right') - 1

  def get_local_sde_at_t(
    self,
    t: Scalar,
    *,
    messages: Optional[Messages] = None
  ) -> 'ConditionedLinearSDE':
    """
    Get the local SDE at time t.  This SDE will be conditioned on only 2 node potentials
    and will have the same distribution as this SDE at the times in between the node potentials.

    This saves having to do message passing again when computing the probability flow ODE.
    """
    t = jnp.array(t)

    # Discretize and then get the messages
    crf = self.discretize()
    messages = Messages.from_messages(messages, crf, need_fwd=True, need_bwd=True)

    # Find the index of the node at time t
    idx = self.get_idx_before_t(t)

    # Get the potentials for the ends of the interval that t is in
    fwd_prior = messages.fwd[idx] + self.node_potentials[idx]
    bwd_prior = messages.bwd[idx+1] + self.node_potentials[idx+1]

    # Concatenate them and make a new SDE
    new_potentials = jtu.tree_map(lambda f, b: jnp.concatenate([f[None],b[None]], axis=0), fwd_prior, bwd_prior)
    new_ts = jnp.array([self.times[idx], self.times[idx+1]])

    pts = GaussianPotentialSeries(new_ts, new_potentials)
    local_sde = ConditionedLinearSDE(self.sde, pts)
    return local_sde

  def get_flow(
    self,
    t: Scalar,
    xt: Float[Array, 'D'],
    method: Literal['jvp', 'score'] = 'score',
    *,
    messages: Optional[Messages] = None
  ) -> Float[Array, 'D']:
    t = jnp.array(t)

    if self.times.shape[-1] > 2:
      # This avoids needing to do message passing when we do a jvp
      local_sde = self.get_local_sde_at_t(t, messages=messages)
      return local_sde.get_flow(t, xt, method=method)

    # Get the base drift and diffusion coefficient
    vt = self.sde.get_drift(t, xt)
    L = self.sde.get_diffusion_coefficient(t, xt)
    LLT = L@L.T

    if method == 'jvp':
      # Get the noise that generated xt
      pxt = self.get_marginal(t, messages=messages)
      noise = pxt.get_noise(xt)

      def sample(t):
        pxt = self.get_marginal(t)
        return pxt._sample(noise)

      xt2, dxtdt = jax.jvp(sample, (t,), (jnp.ones_like(t),))
      return dxtdt

    elif method == 'score':
      crf_result = self.discretize(t[None])
      crf, info = crf_result.crf, crf_result.info
      index = info.new_indices[0]

      # Get the forward and backward messages
      messages = Messages.from_messages(messages, crf, need_fwd=True, need_bwd=True)
      fwd = messages.fwd
      bwd = messages.bwd
      fwdt, bwdt = fwd[index], bwd[index]

      # Get the transition distribution
      return vt + 0.5*LLT@(bwdt.score(xt) - fwdt.score(xt))

  def sample_matching_items(
    self,
    ts: Float[Array, 'T'],
    key: PRNGKeyArray,
    *,
    messages: Optional[Messages] = None
  ) -> Dict[str, Float[Array, 'T D']]:

    # Discretize the SDE at the self.times and ts
    crf_result = self.discretize(ts)
    crf, info = crf_result.crf, crf_result.info

    # Get the forward and backward messages
    fwd = crf.get_forward_messages()
    bwd = crf.get_backward_messages()
    messages = Messages(fwd, bwd)

    # Sample a trajectory and get the marginals
    xts = crf.sample(key, messages=messages)
    marginals = crf.get_marginals(messages=messages)

    # Compute the flow at each point
    def get_items(t, xt, fwd, bwd, pxt):
      vt = self.sde.get_drift(t, xt)
      L = self.sde.get_diffusion_coefficient(t, xt)
      LLT = L@L.T
      fwd_score, bwd_score = fwd.score(xt), bwd.score(xt)
      flow = vt + 0.5*LLT@(bwd_score - fwd_score)
      drift = vt + LLT@bwd_score
      return FlowItems(t=t,
                       xt=xt,
                       flow=flow,
                       score=pxt.score(xt),
                       drift=drift,
                       fwd=fwd,
                       bwd=bwd)

    items = jax.vmap(get_items)(info.times, xts, fwd, bwd, marginals)
    return items

  def simulate_probability_flow(
    self,
    x0: Float[Array, 'D'],
    save_times: Float[Array, 'T'],
    params: Optional[ODESolverParams] = ODESolverParams(),
  ) -> TimeSeries:
    """Simulates the probability flow ODE

    Args:
        x0: Initial state with shape [D] where D is the state dimension
        save_times: Array of times at which to save the trajectory
        key: PRNG key for stochastic simulation

    Returns:
        TimeSeries containing the simulated trajectory at the save_times
    """

    # Simulate the probability flow ODE of the neural SDE
    simulated_trajectory = ode_solve(self,
                               x0=x0,
                               save_times=save_times,
                               params=params)

    return simulated_trajectory

class FlowItems(AbstractBatchableObject):
  t: Scalar
  xt: Float[Array, 'D']
  flow: Float[Array, 'D']
  score: Float[Array, 'D']
  drift: Float[Array, 'D']
  fwd: AbstractPotential
  bwd: AbstractPotential

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.xt.ndim == 1:
      return None
    elif self.xt.ndim == 2:
      return self.xt.shape[0]
    elif self.xt.ndim > 2:
      return self.xt.shape[:-1]
    else:
      raise ValueError(f'xt has {self.xt.ndim} dimensions')
