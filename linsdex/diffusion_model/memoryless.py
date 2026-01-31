"""
Memoryless SDE utilities.

This module implements the memoryless forward SDE used by Reciprocal Adjoint
Matching (RAM). The core idea is to replace the original base process with a
linear, Gaussian, reverse-time process whose joint distribution between the
initial and terminal times factorizes ("memoryless"). This enables sampling
of pairs (X_t, X_1) without simulating full controlled trajectories.

Key components:
- memoryless_noise_schedule: returns the diffusion operator L_hat(t) which
  makes the process memoryless when combined with the appropriate drift
  reparameterization.
- MemorylessForwardSDE: reverse-time (s = t1 - t) linear SDE whose parameters
  are derived from the model components at forward time t. Conditioning this
  SDE on X_1 at s = 0 yields a closed-form posterior for X_t | X_1.
- sample_memoryless_trajectory: given X_1 and a forward time grid, samples the
  intermediate states either by closed-form discretization (preferred) or by
  direct SDE simulation in reverse time.

All routines here operate consistently in reverse time s where s = t1 - t.
Callers that operate in forward time t should use the provided helpers which
internally convert between t and s.
"""
import jax.numpy as jnp
from typing import Tuple, Callable, Literal, Any, Annotated
import diffrax
from jaxtyping import Array, PRNGKeyArray, Float, Scalar
from linsdex import AbstractLinearSDE, AbstractSquareMatrix, ConditionedLinearSDE, TimeSeries, DiagonalMatrix
from linsdex.diffusion_model.probability_path import (
  DiffusionModelComponents,
  BwdMeanToMarginalMean,
  ProbabilityPathSlice,
)
from linsdex.series.batchable_object import AbstractBatchableObject
from .adjoints import SimulationState, sde_simulation_with_internals
import jax
from linsdex.util.parallel_scan import parallel_scan, _tree_concatenate
from linsdex.potential.gaussian.transition import GaussianTransition
import equinox as eqx
import jax.random as random
from linsdex.crf.crf import CRF
from linsdex.linear_functional.functional_ops import resolve_functional
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.potential.gaussian.dist import StandardGaussian
from linsdex.potential.gaussian.transition import functional_potential_to_transition
from linsdex.util import misc as util

def memoryless_noise_schedule(
  components: DiffusionModelComponents,
  t: Scalar,
  xt: Float[Array, 'D'],
) -> Callable[[Scalar], AbstractSquareMatrix]:
  """Get the memoryless noise schedule for a given time and state.
  For diffusion models, the drift bt(xt) can be written in terms
  of the score function as bt(xt) = Kt@xt + lt + Qt@score(xt).
  The memoryless noise schedule is Lt_hat = (2Qt - Lt@Lt.T)^0.5 so that
  the SDE dxt = (bt(xt) + (Lt@Lt.T - Lt_hat@Lt_hat.T)@score(xt))dt + Lt_hat@dWt,
  is memoryless.

  This helper accepts forward time t and converts to reverse time s = t1 - t
  before delegating to MemorylessForwardSDE.get_params.
  """
  s = components.t1 - t
  return MemorylessForwardSDE(components).get_params(s)[-1]

class MemorylessForwardSDE(AbstractLinearSDE):
  """Reverse-time linear SDE that induces a memoryless path distribution.

  This SDE is parameterized in reverse time s = t1 - t. Its parameters are
  derived from the model components at the corresponding forward time t and
  ensure that conditioning on X_1 at s = 0 yields Gaussian posteriors for all
  intermediate states. This enables efficient sampling of X_t | X_1.
  """
  components: DiffusionModelComponents
  dim: int

  def __init__(self, components: DiffusionModelComponents):
    self.components = components
    self.dim = self.components.linear_sde.dim

  @property
  def batch_size(self):
    return self.components.batch_size

  def get_params(self, s: Scalar) -> Tuple[AbstractSquareMatrix, Float[Array, 'D'], AbstractSquareMatrix]:
    """Return reverse-time (A_s, b_s, L_s) at reverse time s.

    Internally converts to forward time t = t1 - s and computes the matrices
    following the derivations (Qt, Kt, lt) with the memoryless diffusion
    L_hat derived from (2Qt - L L^T)^{1/2}.
    """
    t = self.components.t1 - s

    quantities = ProbabilityPathSlice(self.components, t)
    Jb = quantities.beta_precision
    Jp = quantities.marginal_precision

    marginal_mean_to_bwd_mean = BwdMeanToMarginalMean(self.components, t, _quantities=quantities).get_inverse()
    A, b = marginal_mean_to_bwd_mean.A, marginal_mean_to_bwd_mean.b
    I = A.set_eye()

    Ft, ut, Lt = self.components.linear_sde.get_params(t)
    LLT = Lt @ Lt.T
    LLTJb = LLT @ Jb

    # Qt = Lt @ Lt.T @ Jb @ A @ Jp^{-1}
    # Kt = Lt @ Lt.T @ Jb @ (A - I) + Ft
    # lt = Lt @ Lt.T @ Jb @ b + ut

    Qt = LLTJb @ A @ Jp.get_inverse()
    Kt = LLTJb @ (A - I) + Ft
    lt = LLTJb @ b + ut

    # Ensure LLT_hat is symmetric and positive definite
    LLT_hat = 2 * Qt - LLT
    LLT_hat = 0.5 * (LLT_hat + LLT_hat.T)

    # Add a small amount of jitter to the diagonal to ensure positivity
    if isinstance(LLT_hat, DiagonalMatrix):
      LLT_hat = DiagonalMatrix(jnp.maximum(LLT_hat.elements, 1e-10))

    Lt_hat = LLT_hat.get_cholesky()

    return -Kt, -lt, Lt_hat

################################################################################################################

class MemorylessFullPath(AbstractBatchableObject):
  forward_sde: MemorylessForwardSDE
  times: Float[Array, "T"]
  functional_reversed_crf: CRF # CRF whose evidence at the y1 is not filled in yet

  # Precomputed for convenience
  p_y1_given_xt: Annotated[GaussianTransition, "T"]
  p_xt_given_y1: Annotated[GaussianTransition, "T"]
  base_drifts: Annotated[LinearFunctional, "T"]
  diffusion_coefficients: Annotated[AbstractSquareMatrix, "T"]

  @property
  def batch_size(self):
    return self.forward_sde.batch_size

  def sample(self, key: PRNGKeyArray, y1: Float[Array, "D"]) -> TimeSeries:
    """
    Samples a trajectory from the memoryless full path.
    """
    crf: CRF = resolve_functional(self.functional_reversed_crf, y1)
    reversed_xts = crf.sample(key)
    return TimeSeries(times=self.times, values=reversed_xts[::-1])

def get_memoryless_projection_adjoint_path(
  components: DiffusionModelComponents,
  times: Float[Array, "T"],
) -> MemorylessFullPath:
  """
  Computes the memoryless projected probability path at a set of times for a given diffusion model.
  """
  times = jnp.concatenate([times, jnp.array([components.t1])])
  reversed_times = (components.t1 - times)[::-1]

  # Compute the memoryless projection
  forward_memoryless_sde = MemorylessForwardSDE(components)

  # Get the transitions in reverse time
  def get_transition(t_start: Scalar, t_end: Scalar) -> GaussianTransition:
    return forward_memoryless_sde.get_transition_distribution(t_start, t_end)
  p_xtm1_given_xt = jax.vmap(get_transition)(reversed_times[:-1], reversed_times[1:])

  # Construct the functional evidence
  functional_y1 = LinearFunctional.identity(components.linear_sde.dim)
  functional_evidence = StandardGaussian(functional_y1, components.evidence_cov)
  def make_uncertain_potential(_) -> StandardGaussian:
    return StandardGaussian.total_uncertainty_like(functional_evidence)
  node_potentials = jax.vmap(make_uncertain_potential)(reversed_times)
  node_potentials = util.fill_array(node_potentials, 0, functional_evidence)

  # Construct the functional reversed CRF.  This is a CRF whose evidence at the y1 is not filled in yet.
  # We can perform all of the usual CRF optations on this CRF, and then when we need to sample from it,
  # we can fill in the evidence at the y1.
  functional_reversed_crf = CRF(node_potentials, p_xtm1_given_xt)

  ##################################
  # Precompute p(x_t | y_1) and p(y_1 | x_t)
  ##################################
  functional_transitions: Annotated[GaussianTransition, LinearFunctional] = functional_reversed_crf.get_transitions()
  def op(left: GaussianTransition, right: GaussianTransition) -> GaussianTransition:
    return left.chain(right)
  chained_reversed_transitions = parallel_scan(op, functional_transitions, reverse=False)
  functional_p_xt_given_y1: StandardGaussian = chained_reversed_transitions[::-1].condition_on_y(functional_y1)
  functional_p_y1_given_xt: StandardGaussian = chained_reversed_transitions.swap_variables()[::-1].condition_on_x(functional_y1)

  p_xt_given_y1: GaussianTransition = functional_potential_to_transition(functional_p_xt_given_y1).swap_variables()
  p_y1_given_xt: GaussianTransition = functional_potential_to_transition(functional_p_y1_given_xt)

  diffusion_coefficients: Annotated[AbstractSquareMatrix, "T"] = jax.vmap(forward_memoryless_sde.get_diffusion_coefficient, in_axes=(0, None))(reversed_times, jnp.zeros((components.linear_sde.dim,)))

  identity_functional: LinearFunctional = LinearFunctional.identity(components.linear_sde.dim)
  base_drifts: Annotated[LinearFunctional, "T"] = -jax.vmap(forward_memoryless_sde.get_drift, in_axes=(0, None))(reversed_times, identity_functional)


  # Bundle up the transitions into a single object
  return MemorylessFullPath(
    forward_sde=forward_memoryless_sde,
    times=times,
    functional_reversed_crf=functional_reversed_crf,
    p_y1_given_xt=p_y1_given_xt,
    p_xt_given_y1=p_xt_given_y1,
    base_drifts=base_drifts,
    diffusion_coefficients=diffusion_coefficients,
  )

################################################################################################################

def sample_memoryless_trajectory(
  components: DiffusionModelComponents,
  x1: Float[Array, "D"],
  ts: Float[Array, "T"],
  key: PRNGKeyArray,
  method: Literal["simulation", "discretization"] = "simulation",
  solver_name: Literal["shark", "euler"] = "shark",
) -> TimeSeries:
  """
  Sample a trajectory from the memoryless forward SDE on a provided forward-time grid `ts`,
  conditioned on terminal state x1 at t = t1.

  Internally simulates in reverse-time s = t1 - t from s=0 with initial state x1,
  then maps samples back to the forward time ordering of `ts`.

  Args:
    components: Diffusion model components providing linear SDE and schedules.
    x1: Terminal state at t = t1 used for conditioning.
    ts: 1D forward-time grid at which to return samples.
    key: PRNG key for randomness.
    method: "discretization" uses closed-form linear SDE conditioning; "simulation"
      integrates the reverse-time SDE numerically (usually slower, used for checks).

  Returns:
    TimeSeries(times=ts, values=X_t) where values align with the provided grid.
  """
  t0, t1 = components.t0, components.t1
  assert ts.ndim == 1, "ts must be a 1D array of forward times"

  # Build reverse-time evaluation grid: s = t1 - t. Need ascending s for samplers.
  s_eval_points_desc = t1 - ts
  s_eval_points_asc = s_eval_points_desc[::-1]

  sde_key = key
  forward_linear_sde = MemorylessForwardSDE(components)

  if method == "discretization":
    cond_sde: ConditionedLinearSDE = forward_linear_sde.condition_on_starting_point(0.0, x1)
    s_series_asc: TimeSeries = cond_sde.sample(sde_key, s_eval_points_asc)
    xts_values = s_series_asc.values[::-1]

  elif method == "simulation":
    if solver_name == "shark":
      solver = diffrax.ShARK()
    elif solver_name == "euler":
      solver = diffrax.Euler()
    else:
      raise ValueError(f"Invalid solver name: {solver_name}")

    @diffrax.ODETerm
    def wrapped_drift(t: Scalar, xt: Float[Array, 'D'], args: Any) -> Float[Array, 'D']:
      return forward_linear_sde.get_drift(t, xt)

    def diffusion_fn(t: Scalar, xt: Float[Array, 'D'], args: Any) -> Float[Array, 'D']:
      return forward_linear_sde.get_diffusion_coefficient(t, xt).as_matrix()

    simulation_state: SimulationState = sde_simulation_with_internals(
      solver,
      x1,
      wrapped_drift,
      diffusion_fn,
      0.0,
      s_eval_points_asc[-1],
      key=sde_key,
      args=None,
      ts=s_eval_points_asc,
    )
    xts_values = simulation_state.xts[::-1]

  else:
    raise ValueError(f"Invalid method: {method}")

  # Always append terminal pair (t1, x1); callers ensure t1 is not in ts
  times_aug = jnp.concatenate([ts, jnp.array([t1])])
  values_aug = jnp.concatenate([xts_values, x1[None]], axis=0)
  return TimeSeries(times=times_aug, values=values_aug)
