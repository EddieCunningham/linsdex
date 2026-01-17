import jax
import jax.numpy as jnp
from typing import Optional, Callable
from jaxtyping import Array, Float, Scalar
from linsdex.series.batchable_object import AbstractBatchableObject
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.sde.sde_base import AbstractLinearSDE
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.linear_functional.functional_ops import resolve_functional

class DiffusionModelComponents(AbstractBatchableObject):
  linear_sde: AbstractLinearSDE
  t0: Scalar
  x_t0_prior: StandardGaussian
  t1: Scalar
  evidence_cov: AbstractSquareMatrix

  @property
  def batch_size(self):
    return self.linear_sde.batch_size

class DiffusionPathQuantities(AbstractBatchableObject):
  """
  Helper class to compute and store intermediate quantities for a diffusion model at a specific time t.
  This avoids recomputing these quantities across different conversion functions.
  """
  virtual_beta_t: MixedGaussian
  y1_to_marginal_mean: LinearFunctional
  marginal_precision: AbstractSquareMatrix
  beta_precision: AbstractSquareMatrix

  def __init__(self, components: DiffusionModelComponents, t: Scalar):
    evidence_precision = components.evidence_cov.get_inverse()
    I, zero = DiagonalMatrix.eye(components.linear_sde.dim), jnp.zeros(components.linear_sde.dim)
    no_op_lf = LinearFunctional(I, zero)

    # --- Backward message quantities (from Y1ToBwdMean) ---
    virtual_phi_t1 = MixedGaussian(no_op_lf, evidence_precision)
    phi_t1gt = components.linear_sde.get_transition_distribution(t, components.t1)
    self.virtual_beta_t = phi_t1gt.update_and_marginalize_out_y(virtual_phi_t1)
    self.beta_precision = self.virtual_beta_t.J

    # --- Marginal distribution quantities (from Y1ToMarginalMean) ---
    x_t0_prior_std: StandardGaussian = components.x_t0_prior.to_std()
    m, P = x_t0_prior_std.mu, x_t0_prior_std.Sigma

    virtual_phi_t0 = MixedGaussian(no_op_lf, evidence_precision.set_inf())
    phi_tgt0 = components.linear_sde.get_transition_distribution(components.t0, t)
    virtual_alpha_t = phi_tgt0.update_and_marginalize_out_x(virtual_phi_t0)

    alpha_t = resolve_functional(virtual_alpha_t, m)
    p_xt_given_x0_and_y1: StandardGaussian = (alpha_t + self.virtual_beta_t).to_std()

    At_alpha = virtual_alpha_t.mu.A
    Jt_alpha = virtual_alpha_t.J
    At_alpha_beta = p_xt_given_x0_and_y1.Sigma @ Jt_alpha @ At_alpha
    corrected_covariance = p_xt_given_x0_and_y1.Sigma + At_alpha_beta @ P @ At_alpha_beta.T

    virtual_pt_given_y1 = StandardGaussian(p_xt_given_x0_and_y1.mu, corrected_covariance)

    self.y1_to_marginal_mean = virtual_pt_given_y1.mu
    self.marginal_precision = virtual_pt_given_y1.Sigma.get_inverse()

  @property
  def batch_size(self):
    return self.virtual_beta_t.batch_size


class Y1ToBwdMean(LinearFunctional):
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  precision: AbstractSquareMatrix

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    _quantities: Optional[DiffusionPathQuantities] = None,
  ):
    if _quantities is None:
      _quantities = DiffusionPathQuantities(components, t)

    self.A = _quantities.virtual_beta_t.mu.A
    self.b = _quantities.virtual_beta_t.mu.b
    self.precision = _quantities.virtual_beta_t.J

class Y1ToMarginalMean(LinearFunctional):
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  precision: AbstractSquareMatrix

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    _quantities: Optional[DiffusionPathQuantities] = None,
  ):
    if _quantities is None:
      _quantities = DiffusionPathQuantities(components, t)

    self.A = _quantities.y1_to_marginal_mean.A
    self.b = _quantities.y1_to_marginal_mean.b
    self.precision = _quantities.marginal_precision

class BwdMeanToMarginalMean(LinearFunctional):
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  bwd_precision: AbstractSquareMatrix
  marginal_precision: AbstractSquareMatrix

  def __init__(
    self,
    components: DiffusionModelComponents,
    t: Scalar,
    _quantities: Optional[DiffusionPathQuantities] = None,
  ):
    if _quantities is None:
      _quantities = DiffusionPathQuantities(components, t)

    y1_to_marginal_mean = Y1ToMarginalMean(components, t, _quantities=_quantities)
    y1_to_bwd_mean = Y1ToBwdMean(components, t, _quantities=_quantities)
    bwd_mean_to_y1 = y1_to_bwd_mean.get_inverse()
    bwd_mean_to_marginal_mean = bwd_mean_to_y1(y1_to_marginal_mean)
    self.A = bwd_mean_to_marginal_mean.A
    self.b = bwd_mean_to_marginal_mean.b
    self.bwd_precision = y1_to_bwd_mean.precision
    self.marginal_precision = y1_to_marginal_mean.precision


class DiffusionModelConversions(AbstractBatchableObject):
  quantities: DiffusionPathQuantities
  components: DiffusionModelComponents
  t: Scalar

  def __init__(self, components: DiffusionModelComponents, t: Scalar):
    self.quantities = DiffusionPathQuantities(components, t)
    self.components = components
    self.t = t

  @property
  def batch_size(self):
    return self.quantities.batch_size

  def y1_to_bwd_message(self, y1: Float[Array, 'D']) -> MixedGaussian:
    y1_to_bwd_mean_lf = LinearFunctional(self.quantities.virtual_beta_t.mu.A, self.quantities.virtual_beta_t.mu.b)
    bwd_mean = y1_to_bwd_mean_lf(y1)
    return MixedGaussian(bwd_mean, self.quantities.virtual_beta_t.J)

  def bwd_message_to_y1(self, bwd_message: MixedGaussian) -> Float[Array, 'D']:
    y1_to_bwd_mean_lf = LinearFunctional(self.quantities.virtual_beta_t.mu.A, self.quantities.virtual_beta_t.mu.b)
    bwd_mean_to_y1 = y1_to_bwd_mean_lf.get_inverse()
    return bwd_mean_to_y1(bwd_message.mu)

  def y1_to_marginal(self, y1: Float[Array, 'D']) -> StandardGaussian:
    marginal_mean = self.quantities.y1_to_marginal_mean(y1)
    return StandardGaussian(marginal_mean, self.quantities.marginal_precision.get_inverse())

  def marginal_to_y1(self, marginal: StandardGaussian) -> Float[Array, 'D']:
    marginal_mean_to_y1 = self.quantities.y1_to_marginal_mean.get_inverse()
    return marginal_mean_to_y1(marginal.mu)

  def bwd_message_to_marginal(self, bwd_message: MixedGaussian) -> StandardGaussian:
    bwd_mean_to_marginal_mean = BwdMeanToMarginalMean(self.components, self.t)
    marginal_mean = bwd_mean_to_marginal_mean(bwd_message.mu)
    return StandardGaussian(marginal_mean, bwd_mean_to_marginal_mean.marginal_precision.get_inverse())

  def marginal_to_bwd_message(self, marginal: StandardGaussian) -> MixedGaussian:
    bwd_mean_to_marginal_mean = BwdMeanToMarginalMean(self.components, self.t)
    bwd_mean = bwd_mean_to_marginal_mean.get_inverse()(marginal.mu)
    return MixedGaussian(bwd_mean, bwd_mean_to_marginal_mean.bwd_precision)

  def bwd_message_to_drift(self, bwd_message: MixedGaussian, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    F, u, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    return F @ xt + u + LLT @ bwd_message.score(xt)

  def drift_to_bwd_message(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> MixedGaussian:
    F, u, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    bwd_score = LLT.solve(drift - F @ xt - u)
    bwd_precision = self.quantities.virtual_beta_t.J
    bwd_mean = bwd_precision.solve(bwd_score) + xt
    return MixedGaussian(bwd_mean, bwd_precision)

  def epsilon_to_bwd_message(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> MixedGaussian:
    r"""
    x_{t|1} = \mu_{t|1}^beta + {\Sigma_{t|1}^beta}^{1/2} \epsilon, where \epsilon ~ N(0, I)
    """
    bwd_precision = self.quantities.virtual_beta_t.J
    L_chol = bwd_precision.get_cholesky()
    bwd_mean = xt - bwd_precision.solve(L_chol@epsilon)
    return MixedGaussian(bwd_mean, bwd_precision)

  def bwd_message_to_epsilon(self, xt: Float[Array, 'D'], bwd_message: MixedGaussian) -> Float[Array, 'D']:
    """
    Inverse of epsilon_to_bwd_message.
    """
    bwd_precision = self.quantities.virtual_beta_t.J
    L_chol = bwd_precision.get_cholesky()
    return L_chol.T @ (xt - bwd_message.mu)

  def epsilon_to_drift(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
    bwd_message = self.epsilon_to_bwd_message(xt, epsilon)
    return self.bwd_message_to_drift(bwd_message, xt)

  def drift_to_epsilon(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    bwd_message = self.drift_to_bwd_message(xt, drift)
    return self.bwd_message_to_epsilon(xt, bwd_message)

  def epsilon_to_score(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
    bwd_message = self.epsilon_to_bwd_message(xt, epsilon)
    marginal = self.bwd_message_to_marginal(bwd_message)
    return self.marginal_to_score(marginal, xt)

  def score_to_epsilon(self, xt: Float[Array, 'D'], score: Float[Array, 'D']) -> Float[Array, 'D']:
    marginal = self.score_to_marginal(xt, score)
    bwd_message = self.marginal_to_bwd_message(marginal)
    return self.bwd_message_to_epsilon(xt, bwd_message)

  def epsilon_to_flow(self, xt: Float[Array, 'D'], epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
    drift = self.epsilon_to_drift(xt, epsilon)
    return self.drift_to_flow(xt, drift)

  def flow_to_epsilon(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
    drift = self.flow_to_drift(xt, flow)
    return self.drift_to_epsilon(xt, drift)

  def marginal_to_score(self, marginal: StandardGaussian, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    return marginal.score(xt)

  def score_to_marginal(self, xt: Float[Array, 'D'], score: Float[Array, 'D']) -> StandardGaussian:
    marginal_precision = self.quantities.marginal_precision
    marginal_mean = marginal_precision.solve(score) + xt
    return StandardGaussian(marginal_mean, marginal_precision.get_inverse())

  def score_to_flow(self, xt: Float[Array, 'D'], score: Float[Array, 'D']) -> Float[Array, 'D']:
    marginal = self.score_to_marginal(xt, score)
    bwd_message = self.marginal_to_bwd_message(marginal)
    drift = self.bwd_message_to_drift(bwd_message, xt)
    _, _, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    return drift - 0.5 * LLT @ score

  def y1_to_drift(self, y1: Float[Array, 'D'], xt: Float[Array, 'D']) -> Float[Array, 'D']:
    bwd_message = self.y1_to_bwd_message(y1)
    return self.bwd_message_to_drift(bwd_message, xt)

  def drift_to_y1(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    bwd_message = self.drift_to_bwd_message(xt, drift)
    return self.bwd_message_to_y1(bwd_message)

  def drift_to_score(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    bwd_message = self.drift_to_bwd_message(xt, drift)
    marginal = self.bwd_message_to_marginal(bwd_message)
    return self.marginal_to_score(marginal, xt)

  def drift_to_flow(self, xt: Float[Array, 'D'], drift: Float[Array, 'D']) -> Float[Array, 'D']:
    score = self.drift_to_score(xt, drift)
    _, _, L = self.components.linear_sde.get_params(self.t)
    LLT = L @ L.T
    return drift - 0.5 * LLT @ score

  def y1_to_flow(self, y1: Float[Array, 'D'], xt: Float[Array, 'D']) -> Float[Array, 'D']:
    drift = self.y1_to_drift(y1, xt)
    return self.drift_to_flow(xt, drift)

  def flow_to_drift(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
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
    drift = self.flow_to_drift(xt, flow)
    return self.drift_to_y1(xt, drift)

  def flow_to_score(self, xt: Float[Array, 'D'], flow: Float[Array, 'D']) -> Float[Array, 'D']:
    drift = self.flow_to_drift(xt, flow)
    return self.drift_to_score(xt, drift)

  def y1_to_score(self, xt: Float[Array, 'D'], y1: Float[Array, 'D']) -> Float[Array, 'D']:
    marginal = self.y1_to_marginal(y1)
    return self.marginal_to_score(marginal, xt)

  def get_flow_covariance(self, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    def epsilon_to_flow(epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
      return self.epsilon_to_flow(xt, epsilon)
    flow_jac = jax.jacfwd(epsilon_to_flow)(jnp.zeros(self.components.linear_sde.dim))
    return flow_jac @ flow_jac.T

  def get_score_covariance(self, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    def epsilon_to_score(epsilon: Float[Array, 'D']) -> Float[Array, 'D']:
      return self.epsilon_to_score(xt, epsilon)
    score_jac = jax.jacfwd(epsilon_to_score)(jnp.zeros(self.components.linear_sde.dim))
    return score_jac @ score_jac.T

  def get_drift_covariance(self, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
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
  """
  If components_t is p(x_t | y_1) and components_s is p(x_s | y_1), then this function returns a GaussianTransition
  that is p(x_t | x_s).
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
