r"""
This module provides tools for converting between different parameterizations
of flow based generative models constructed from linear stochastic differential
equations with Gaussian priors. The framework assumes a base SDE of the form
$dx_t = (F_t x_t + u_t)dt + L_t dW_t$ and a Gaussian prior $p(x_0)$. Under these
conditions the stochastic bridge $p(x_t | x_0, y_1)$ is Gaussian and all common
model outputs including the score function, SDE drift and probability flow ODE
velocity are related by closed form affine transformations.

The following notation and objects are used throughout this module:
  x_t: The state of the stochastic process at time t.
  y_1: The terminal evidence or observation at time t = 1.
  \beta_t(x_t) = p(y_1 | x_t): The backward message representing the
    likelihood of the terminal evidence given the current state.
  p_t(x_t | y_1): The marginal distribution of the stochastic bridge
    at time t conditioned on the terminal evidence.
  s_t(x_t) = \nabla \log p_t(x_t | y_1): The score function of the
    marginal distribution.
  b_t(x_t): The drift of the Markovian projection SDE that preserves
    the marginals of the stochastic bridge.
  v_t(x_t): The velocity field of the probability flow ODE.
  \epsilon: Standard normal noise used for reparameterizing the
    stochastic bridge state.

The implementations follow the theoretical derivations for Markovian projection
and memoryless SDEs as detailed in the accompanying documentation. These
conversions enable practitioners to train a model in one parameterization
while sampling or performing inference in another.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Callable, Dict
from jaxtyping import Array, Float, Scalar, PRNGKeyArray, PyTree
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian, NaturalGaussian
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.sde.sde_base import AbstractLinearSDE
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.linear_functional.functional_ops import resolve_functional
from linsdex.potential.gaussian.dist import AbstractGaussianPotential

from linsdex.sde.conditioned_linear_sde import FlowItems
from linsdex.util.parallel_scan import parallel_scan, _tree_concatenate

class DiffusionModelComponents(AbstractBatchableObject):
  """
  Container for the fundamental components of a diffusion model. This includes
  the base linear stochastic differential equation, the Gaussian prior at the
  initial time and the evidence covariance at the terminal time. These
  components define the stochastic bridge that is used for model training and
  conversion.
  """
  linear_sde: AbstractLinearSDE
  t0: Scalar
  x_t0_prior: StandardGaussian
  t1: Scalar
  evidence_cov: AbstractSquareMatrix

  @property
  def batch_size(self):
    return self.linear_sde.batch_size

class ProbabilityPathSlice(AbstractGaussianPotential):
  r"""
  Computes and stores the intermediate probabilistic quantities for a diffusion
  model at a specific time. This class encapsulates the backward message
  $\beta_t(x_t) = p(y_1 | x_t)$ and the marginal distribution $p_t(x_t | y_1)$
  of the stochastic bridge.

  The backward message is computed by integrating the terminal evidence
  $p(y_1 | x_1)$ against the transition distribution $p(x_1 | x_t)$, which
  yields the expression
  $\beta_t(x_t) = \int p(y_1 | x_1) p(x_1 | x_t) dx_1$.

  The bridge path marginal is then obtained by incorporating this message into
  the forward transition $p(x_t | x_0)$ to form the joint $p(x_t, x_0 | y_1)$
  and marginalizing over the prior $p(x_0)$. This gives the closed form
  expression
  $p_t(x_t | y_1) = \int p(x_t | x_0, y_1) p(x_0) dx_0$.

  By precomputing these Gaussian distributions, we avoid redundant calculations
  when converting between different model parameterizations.
  """
  functional_beta_t: MixedGaussian
  functional_pt_given_y1: StandardGaussian
  components: DiffusionModelComponents
  t: Scalar
  fwd: StandardGaussian

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    phi_t1gt: Optional[GaussianTransition] = None,
    phi_tgt0: Optional[GaussianTransition] = None
  ):
    self.components = components
    self.t = t
    evidence_precision = components.evidence_cov.get_inverse()
    I, zero = DiagonalMatrix.eye(components.linear_sde.dim), jnp.zeros(components.linear_sde.dim)
    no_op_lf = LinearFunctional(I, zero)

    # --- Backward message quantities (from Y1ToBwdMean) ---
    functional_phi_t1 = MixedGaussian(no_op_lf, evidence_precision)
    if phi_t1gt is None:
      phi_t1gt = components.linear_sde.get_transition_distribution(t, components.t1)
    self.functional_beta_t = phi_t1gt.update_and_marginalize_out_y(functional_phi_t1)

    # --- Marginal distribution quantities (from Y1ToMarginalMean) ---
    if phi_tgt0 is None:
      phi_tgt0 = components.linear_sde.get_transition_distribution(components.t0, t)
    self.fwd = phi_tgt0.update_and_marginalize_out_x(components.x_t0_prior).to_std()

    # We want to compute the "bridge path" marginal:
    # p_t(x_t; y_1) = \int p(x_t | x_0, y_1) p(x_0) dx_0
    # This path has the property that p_0(x_0) = p(x_0) (the prior)
    # and p_1(x_1; y_1) = p(x_1 | y_1) (the observation).

    # 1. Incorporate the future evidence y_1 into the transition p(x_t | x_0)
    # This gives the joint p(x_t, x_0 | y_1) = p(x_t | x_0, y_1) p(y_1 | x_0)
    joint_xt_x0 = phi_tgt0.update_y(self.functional_beta_t)

    # 2. Extract the bridge transition p(x_t | x_0, y_1)
    bridge_transition = joint_xt_x0.transition

    # 3. Marginalize over the prior p(x_0)
    # This computes \int p(x_t | x_0, y_1) p(x_0) dx_0
    p_xt_bridge = bridge_transition.update_and_marginalize_out_x(components.x_t0_prior)

    self.functional_pt_given_y1 = p_xt_bridge.to_std()

  @property
  def y1_to_marginal_mean(self) -> LinearFunctional:
    return self.functional_pt_given_y1.to_mixed().mu

  @property
  def marginal_precision(self) -> AbstractSquareMatrix:
    return self.functional_pt_given_y1.Sigma.get_inverse()

  @property
  def beta_precision(self) -> AbstractSquareMatrix:
    return self.functional_beta_t.J

  @property
  def batch_size(self):
    return self.functional_beta_t.batch_size

  def sample(self, key: PRNGKeyArray) -> LinearFunctional:
    """
    Samples a functional from p(x_t | y_1)
    """
    return self.functional_pt_given_y1.sample(key)

  def _sample(self, epsilon: Float[Array, 'D']) -> LinearFunctional:
    """
    Samples a functional from p(x_t | y_1) using the reparameterization trick.
    """
    return self.functional_pt_given_y1._sample(epsilon)

  def normalizing_constant(self) -> Scalar:
    return self.functional_pt_given_y1()

  def log_prob(self, x: PyTree) -> Scalar:
    return self.functional_pt_given_y1.log_prob(x)

  @classmethod
  def total_certainty_like(cls, x: Float[Array, 'D'], other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    return cls.functional_pt_given_y1.total_certainty_like(x, other)

  @classmethod
  def total_uncertainty_like(cls, other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    return cls.functional_pt_given_y1.total_uncertainty_like(other)

  def sufficient_statistics(self, x: Float[Array, 'B D']) -> 'AbstractGaussianPotential':
    return self.functional_pt_given_y1.sufficient_statistics(x)

  @auto_vmap
  def integrate(self):
    r"""Compute the value of \int exp{-0.5*x^T J x + x^T h - logZ} dx"""
    return self.normalizing_constant() - self.logZ

  def score(self, x: Float[Array, 'D']) -> Float[Array, 'D']:
    return self.functional_pt_given_y1.score(x)

  def get_noise(self, x: Float[Array, 'D']) -> Float[Array, 'D']:
    return self.functional_pt_given_y1.get_noise(x)

  def __call__(self, x: PyTree) -> Scalar:
    return self.functional_pt_given_y1(x)

  def to_transition(self, epsilon: Float[Array, 'D']) -> GaussianTransition:
    """
    Converts the probability path to a transition distribution using the reparameterization trick.
    """
    x_given_y: LinearFunctional = self.functional_pt_given_y1._sample(epsilon)
    return GaussianTransition(x_given_y.A, x_given_y.b, self.functional_pt_given_y1.Sigma, self.functional_pt_given_y1.logZ)

  def _sample_matching_items(self, epsilon: Float[Array, 'D']) -> FlowItems:
    """
    Samples the matching items from p(x_t | y_1) using the reparameterization trick.

    Args:
      epsilon: Standard normal noise used for reparameterizing the stochastic bridge state.

    Returns:
      FlowItems: A container for the functional probabilistic quantities at time t:
        t: The current time.
        xt: The state of the stochastic process sampled from the marginal distribution
          p_t(x_t | y_1) as a mapping from terminal evidence y_1.
        flow: The velocity field of the probability flow ODE v_t(x_t), which
          preserves the marginal distributions of the stochastic bridge.
        score: The score function of the marginal distribution ∇ log p_t(x_t | y_1).
        noise: The standard normal noise ε used for reparameterization.
        drift: The drift of the Markovian projection SDE b_t(x_t) that preserves
          the marginals of the stochastic bridge.
    """
    # 1. State xt as a LinearFunctional (mapping from y1 to xt)
    xt = self.functional_pt_given_y1._sample(epsilon)

    # 2. Extract SDE parameters
    F, u, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T

    # 3. Compute quantities as LinearFunctionals
    score = self.functional_pt_given_y1.score(xt)
    # noise is just the constant epsilon
    noise = LinearFunctional(DiagonalMatrix.zeros(self.components.linear_sde.dim), epsilon)

    bwd_score = self.functional_beta_t.score(xt)

    vt = F @ xt + u
    drift = vt + LLT @ bwd_score
    flow = drift - 0.5 * LLT @ score

    return FlowItems(
      t=self.t,
      xt=xt,
      flow=flow,
      score=score,
      noise=noise,
      drift=drift
    )

def get_probability_path(
  components: DiffusionModelComponents,
  times: Float[Array, "T"],
) -> ProbabilityPathSlice:
  """
  Computes the probability path at a set of times for a given diffusion model.

  Args:
    components: The fundamental components of the diffusion model.
    times: A set of times at which to compute the probability path slices.

  Returns:
    ProbabilityPathSlice: A batched object containing the probabilistic quantities
      at each time in the input array.
  """

  def get_transition(t_start, t_end):
    return components.linear_sde.get_transition_distribution(t_start, t_end)

  transitions = jax.vmap(get_transition)(times[:-1], times[1:])

  def op(left, right):
    return left.chain(right)

  phi_tgt0_scan = parallel_scan(op, transitions, reverse=False)
  phi_t1gt_scan = parallel_scan(op, transitions, reverse=True)

  # Prepend identity to phi_tgt0 and append to phi_t1gt
  # This ensures both have length T matching the times array
  identity = GaussianTransition.no_op_like(transitions[0])
  identity = jax.tree_util.tree_map(lambda x: x[None] if eqx.is_array(x) else x, identity)

  phi_tgt0 = _tree_concatenate(identity, phi_tgt0_scan, axis=0)
  phi_t1gt = _tree_concatenate(phi_t1gt_scan, identity, axis=0)

  return jax.vmap(lambda t, p1, p0: ProbabilityPathSlice(components, t, phi_t1gt=p1, phi_tgt0=p0))(times, phi_t1gt, phi_tgt0)

class Y1ToBwdMean(LinearFunctional):
  """
  Represents the affine mapping from terminal evidence to the mean of the
  backward message at the current time.
  """
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  precision: AbstractSquareMatrix

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    _quantities: Optional[ProbabilityPathSlice] = None,
  ):
    if _quantities is None:
      _quantities = ProbabilityPathSlice(components, t)

    self.A = _quantities.functional_beta_t.mu.A
    self.b = _quantities.functional_beta_t.mu.b
    self.precision = _quantities.functional_beta_t.J

class Y1ToMarginalMean(LinearFunctional):
  """
  Represents the affine mapping from terminal evidence to the mean of the
  marginal distribution at the current time.
  """
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  precision: AbstractSquareMatrix

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    _quantities: Optional[ProbabilityPathSlice] = None,
  ):
    if _quantities is None:
      _quantities = ProbabilityPathSlice(components, t)

    self.A = _quantities.y1_to_marginal_mean.A
    self.b = _quantities.y1_to_marginal_mean.b
    self.precision = _quantities.marginal_precision

class BwdMeanToMarginalMean(LinearFunctional):
  """
  Represents the affine mapping from the backward message mean to the marginal
  distribution mean at the current time.
  """
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  bwd_precision: AbstractSquareMatrix
  marginal_precision: AbstractSquareMatrix

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    _quantities: Optional[ProbabilityPathSlice] = None,
  ):
    if _quantities is None:
      _quantities = ProbabilityPathSlice(components, t)

    y1_to_marginal_mean = Y1ToMarginalMean(components, t, _quantities=_quantities)
    y1_to_bwd_mean = Y1ToBwdMean(components, t, _quantities=_quantities)
    bwd_mean_to_y1 = y1_to_bwd_mean.get_inverse()
    bwd_mean_to_marginal_mean = bwd_mean_to_y1(y1_to_marginal_mean)
    self.A = bwd_mean_to_marginal_mean.A
    self.b = bwd_mean_to_marginal_mean.b
    self.bwd_precision = y1_to_bwd_mean.precision
    self.marginal_precision = y1_to_marginal_mean.precision


class DiffusionModelConversions(AbstractBatchableObject):
  """
  Provides methods to transform between different representations of the
  generative process. This class implements the affine mappings between the
  terminal evidence, the backward message, the marginal distribution, the score
  function, the SDE drift and the probability flow ODE velocity. These
  transformations are derived from the Gaussian structure of the linear SDE and
  the initial prior.
  """
  quantities: ProbabilityPathSlice
  components: DiffusionModelComponents
  t: Scalar

  def __init__(self, components: DiffusionModelComponents, t: Scalar):
    self.quantities = ProbabilityPathSlice(components, t)
    self.components = components
    self.t = t

  @property
  def batch_size(self):
    return self.quantities.batch_size

  def y1_to_bwd_message(self, y1: Float[Array, 'D']) -> MixedGaussian:
    """
    Maps the terminal evidence to the backward message at the current time.
    """
    y1_to_bwd_mean_lf = LinearFunctional(self.quantities.functional_beta_t.mu.A, self.quantities.functional_beta_t.mu.b)
    bwd_mean = y1_to_bwd_mean_lf(y1)
    return MixedGaussian(bwd_mean, self.quantities.functional_beta_t.J)

  def bwd_message_to_y1(self, bwd_message: MixedGaussian) -> Float[Array, 'D']:
    """
    Maps the backward message at the current time to the terminal evidence.
    """
    y1_to_bwd_mean_lf = LinearFunctional(self.quantities.functional_beta_t.mu.A, self.quantities.functional_beta_t.mu.b)
    bwd_mean_to_y1 = y1_to_bwd_mean_lf.get_inverse()
    return bwd_mean_to_y1(bwd_message.mu)

  def y1_to_marginal(self, y1: Float[Array, 'D']) -> StandardGaussian:
    """
    Maps the terminal evidence to the marginal distribution at the current time.
    """
    marginal_mean = self.quantities.y1_to_marginal_mean(y1)
    return StandardGaussian(marginal_mean, self.quantities.marginal_precision.get_inverse())

  def marginal_to_y1(self, marginal: StandardGaussian) -> Float[Array, 'D']:
    """
    Maps the marginal distribution at the current time to the terminal evidence.
    """
    marginal_mean_to_y1 = self.quantities.y1_to_marginal_mean.get_inverse()
    return marginal_mean_to_y1(marginal.mu)

  def bwd_message_to_marginal(self, bwd_message: MixedGaussian) -> StandardGaussian:
    """
    Transforms the backward message into the marginal distribution at the current time.
    """
    bwd_mean_to_marginal_mean = BwdMeanToMarginalMean(self.components, self.t)
    marginal_mean = bwd_mean_to_marginal_mean(bwd_message.mu)
    return StandardGaussian(marginal_mean, bwd_mean_to_marginal_mean.marginal_precision.get_inverse())

  def marginal_to_bwd_message(self, marginal: StandardGaussian) -> MixedGaussian:
    """
    Transforms the marginal distribution into the backward message at the current time.
    """
    bwd_mean_to_marginal_mean = BwdMeanToMarginalMean(self.components, self.t)
    bwd_mean = bwd_mean_to_marginal_mean.get_inverse()(marginal.mu)
    return MixedGaussian(bwd_mean, bwd_mean_to_marginal_mean.bwd_precision)

  def bwd_message_to_drift(self, bwd_message: MixedGaussian, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    r"""
    Computes the SDE drift $b_t(x_t)$ from the backward message $\beta_t(x_t)$ at
    the current state. The drift is given by the expression
    $b_t(x_t) = F_t x_t + u_t + L_t L_t^\top \nabla \log \beta_t(x_t)$.
    This transformation allows the model to steer the process toward the
    terminal evidence.
    """
    F, u, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    return F @ xt + u + LLT @ bwd_message.score(xt)

  def drift_to_bwd_message(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> MixedGaussian:
    r"""
    Recovers the backward message $\beta_t(x_t)$ from the SDE drift $b_t(x_t)$ at
    the current state. This involves solving the relation
    $\nabla \log \beta_t(x_t) = (L_t L_t^\top)^{-1} (b_t(x_t) - F_t x_t - u_t)$
    for the message parameters.
    """
    F, u, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    bwd_score = LLT.solve(drift - F @ xt - u)
    bwd_precision = self.quantities.functional_beta_t.J
    bwd_mean = bwd_precision.solve(bwd_score) + xt
    return MixedGaussian(bwd_mean, bwd_precision)

  def epsilon_to_bwd_message(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> MixedGaussian:
    """
    Maps the standard normal noise to the backward message at the current state.
    """
    bwd_precision = self.quantities.functional_beta_t.J
    L_chol = bwd_precision.get_cholesky()
    bwd_mean = xt - bwd_precision.solve(L_chol@epsilon)
    return MixedGaussian(bwd_mean, bwd_precision)

  def bwd_message_to_epsilon(self, xt: Float[Array, 'D'], bwd_message: MixedGaussian) -> Float[Array, 'D']:
    """
    Maps the backward message at the current state to the standard normal noise.
    """
    bwd_precision = self.quantities.functional_beta_t.J
    L_chol = bwd_precision.get_cholesky()
    return L_chol.T @ (xt - bwd_message.mu)

  def epsilon_to_drift(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Computes the SDE drift from the standard normal noise at the current state.
    """
    bwd_message = self.epsilon_to_bwd_message(xt, epsilon)
    return self.bwd_message_to_drift(bwd_message, xt)

  def drift_to_epsilon(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Recovers the standard normal noise from the SDE drift at the current state.
    """
    bwd_message = self.drift_to_bwd_message(xt, drift)
    return self.bwd_message_to_epsilon(xt, bwd_message)

  def epsilon_to_score(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Computes the score function from the standard normal noise at the current state.
    """
    bwd_message = self.epsilon_to_bwd_message(xt, epsilon)
    marginal = self.bwd_message_to_marginal(bwd_message)
    return self.marginal_to_score(marginal, xt)

  def score_to_epsilon(self, xt: Float[Array, 'D'], score: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Recovers the standard normal noise from the score function at the current state.
    """
    marginal = self.score_to_marginal(xt, score)
    bwd_message = self.marginal_to_bwd_message(marginal)
    return self.bwd_message_to_epsilon(xt, bwd_message)

  def epsilon_to_flow(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Computes the probability flow ODE velocity from the standard normal noise at the current state.
    """
    drift = self.epsilon_to_drift(xt, epsilon)
    return self.drift_to_flow(xt, drift)

  def flow_to_epsilon(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Recovers the standard normal noise from the probability flow ODE velocity at the current state.
    """
    drift = self.flow_to_drift(xt, flow)
    return self.drift_to_epsilon(xt, drift)

  def marginal_to_score(self, marginal: StandardGaussian, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Computes the score function from the marginal distribution at the current state.
    """
    return marginal.score(xt)

  def score_to_marginal(self, xt: Float[Array, 'D'], score: Float[Array, 'D']) -> StandardGaussian:
    """
    Recovers the marginal distribution from the score function at the current state.
    """
    marginal_precision = self.quantities.marginal_precision
    marginal_mean = marginal_precision.solve(score) + xt
    return StandardGaussian(marginal_mean, marginal_precision.get_inverse())

  def score_to_flow(self, xt: Float[Array, 'D'], score: Float[Array, 'D']) -> Float[Array, 'D']:
    r"""
    Computes the probability flow ODE velocity $v_t(x_t)$ from the score
    function $\nabla \log p_t(x_t)$ at the current state. The velocity is
    defined as $v_t(x_t) = b_t(x_t) - 0.5 L_t L_t^\top \nabla \log p_t(x_t)$,
    which ensures that the ODE preserves the marginal distributions of the SDE.
    """
    marginal = self.score_to_marginal(xt, score)
    bwd_message = self.marginal_to_bwd_message(marginal)
    drift = self.bwd_message_to_drift(bwd_message, xt)
    _, _, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    return drift - 0.5 * LLT @ score

  def y1_to_drift(self, y1: Float[Array, 'D'], xt: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the terminal evidence to the SDE drift at the current state.
    """
    bwd_message = self.y1_to_bwd_message(y1)
    return self.bwd_message_to_drift(bwd_message, xt)

  def drift_to_y1(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the SDE drift at the current state to the terminal evidence.
    """
    bwd_message = self.drift_to_bwd_message(xt, drift)
    return self.bwd_message_to_y1(bwd_message)

  def drift_to_score(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the SDE drift to the score function at the current state.
    """
    bwd_message = self.drift_to_bwd_message(xt, drift)
    marginal = self.bwd_message_to_marginal(bwd_message)
    return self.marginal_to_score(marginal, xt)

  def drift_to_flow(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the SDE drift to the probability flow ODE velocity at the current state.
    """
    score = self.drift_to_score(xt, drift)
    _, _, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    return drift - 0.5 * LLT @ score

  def y1_to_flow(self, y1: Float[Array, 'D'], xt: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the terminal evidence to the probability flow ODE velocity at the current state.
    """
    drift = self.y1_to_drift(y1, xt)
    return self.drift_to_flow(xt, drift)

  def flow_to_drift(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the probability flow ODE velocity $v_t(x_t)$ to the SDE drift $b_t(x_t)$
    at the current state. This requires inverting the relationship between the
    flow, drift and score while accounting for the Gaussian prior on the
    initial state. The calculation utilizes the precomputed precision matrices
    and affine mappings from the ProbabilityPathSlice.
    """
    F, u, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    bwd_to_marginal_mean = BwdMeanToMarginalMean(self.components, self.t)
    Jbeta = bwd_to_marginal_mean.bwd_precision
    J = bwd_to_marginal_mean.marginal_precision
    A = bwd_to_marginal_mean.A
    b = bwd_to_marginal_mean.b

    mat1 = F - LLT @ Jbeta + 0.5 * LLT @ J
    offset = mat1 @ xt + u - 0.5 * LLT @ J @ b
    mat2 = LLT @ Jbeta - 0.5 * LLT @ J @ A
    bwd_mean = mat2.solve(flow - offset)
    bwd_message = MixedGaussian(bwd_mean, Jbeta)
    drift = self.bwd_message_to_drift(bwd_message, xt)
    return drift

  def flow_to_y1(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the probability flow ODE velocity at the current state to the terminal evidence.
    """
    drift = self.flow_to_drift(xt, flow)
    return self.drift_to_y1(xt, drift)

  def flow_to_score(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the probability flow ODE velocity to the score function at the current state.
    """
    drift = self.flow_to_drift(xt, flow)
    return self.drift_to_score(xt, drift)

  def y1_to_score(self, xt: Float[Array, 'D'], y1: Float[Array, 'D']) -> Float[Array, 'D']:
    """
    Maps the terminal evidence to the score function at the current state.
    """
    marginal = self.y1_to_marginal(y1)
    return self.marginal_to_score(marginal, xt)

  def get_flow_covariance(self, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    """
    Computes the covariance of the probability flow ODE velocity induced by the standard normal noise.
    """
    def epsilon_to_flow(epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
      return self.epsilon_to_flow(xt, epsilon)
    flow_jac = jax.jacfwd(epsilon_to_flow)(jnp.zeros(self.components.linear_sde.dim))
    return flow_jac @ flow_jac.T

  def get_score_covariance(self, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    """
    Computes the covariance of the score function induced by the standard normal noise.
    """
    def epsilon_to_score(epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
      return self.epsilon_to_score(xt, epsilon)
    score_jac = jax.jacfwd(epsilon_to_score)(jnp.zeros(self.components.linear_sde.dim))
    return score_jac @ score_jac.T

  def get_drift_covariance(self, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    """
    Computes the covariance of the SDE drift induced by the standard normal noise.
    """
    def epsilon_to_drift(epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
      return self.epsilon_to_drift(xt, epsilon)
    drift_jac = jax.jacfwd(epsilon_to_drift)(jnp.zeros(self.components.linear_sde.dim))
    return drift_jac @ drift_jac.T

def noise_schedule_drift_correction(
  components: DiffusionModelComponents,
  t: Scalar,
  xt: Float[Array, 'D'],
  drift: Float[Array, 'D'],
  noise_schedule: Optional[Callable[[Scalar, Float[Array, 'D']], AbstractSquareMatrix]] = None,
  _conversions: Optional[DiffusionModelConversions] = None,
) -> Float[Array, 'D']:
  r"""
  Corrects the SDE drift when changing the noise schedule to ensure that the
  marginal distributions are preserved. The transformation is given by the
  expression
  $b_t'(x_t) = b_t(x_t) + 0.5(L_t' L_t'^\top - L_t L_t^\top) \nabla \log p_t(x_t)$
  where $L_t'$ is the new diffusion coefficient. This ensures that infinitely
  many different stochastic processes can share the same marginal distributions
  as established in the theoretical framework.
  """
  if noise_schedule is None:
    return drift
  conversions = _conversions or DiffusionModelConversions(components, t)
  score = conversions.drift_to_score(xt, drift)
  L = components.linear_sde.get_diffusion_coefficient(t, xt)
  Lhat = noise_schedule(t, xt)
  correction = 0.5*(Lhat@Lhat.T - L@L.T)@score
  return drift + correction


def probability_path_transition(
  components_t: DiffusionModelComponents,
  components_s: DiffusionModelComponents,
  t: Scalar,
  s: Scalar,
) -> GaussianTransition:
  r"""
  Computes the Gaussian transition distribution $p(x_t | x_s, y_1)$ between two
  marginal distributions of a stochastic bridge at different times. The
  resulting distribution is $N(x_t | A_{t|s} x_s + u_{t|s}, \Sigma_{t|s})$ where
  the parameters satisfy the composition of the forward transition and backward
  messages.
  """
  y1_to_marginal_mean_t = Y1ToMarginalMean(components_t, t)
  y1_to_marginal_mean_s = Y1ToMarginalMean(components_s, s)
  At, bt = y1_to_marginal_mean_t.A, y1_to_marginal_mean_t.b
  As, bs = y1_to_marginal_mean_s.A, y1_to_marginal_mean_s.b
  Sigmat = y1_to_marginal_mean_t.precision.get_inverse()
  Sigmas = y1_to_marginal_mean_s.precision.get_inverse()

  AtAsinv = At@As.get_inverse()

  A = AtAsinv
  u = -AtAsinv@bs + bt
  Sigma = Sigmat - AtAsinv@Sigmas@AtAsinv.T
  return GaussianTransition(A, u, Sigma)
