
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type, Dict, Literal, Annotated
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
from linsdex.potential.gaussian.dist import MixedGaussian, NaturalGaussian, StandardGaussian, AbstractGaussianPotential
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
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.sde.sde_base import AbstractSDE
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.linear_functional.functional_ops import resolve_functional

def get_integrated_marginal_given_y1(
  base_sde: AbstractLinearSDE,
  t0: Scalar,
  x_t0_prior: AbstractGaussianPotential,
  t1: Scalar,
  y1: Float[Array, 'D'],
  evidence_cov: AbstractSquareMatrix,
  t: Scalar
) -> StandardGaussian:
  """
  Computes p(x_t | y_1) = \int p(x_t | x_0, y_1) p(x_0) dx_0

  Args:
    base_sde: The base SDE to use
    t0: The start time
    x_t0_prior: The prior over y_t0
    t1: The end time
    y1: The evidence at time t1
    evidence_cov: The covariance of the evidence
    t: The time to compute the marginal at
  """
  evidence_precision = evidence_cov.get_inverse()

  # Create the virtual potential over x_{t0}.
  I, zero = DiagonalMatrix.eye(base_sde.dim), jnp.zeros(base_sde.dim)
  inf_precision = evidence_precision.set_inf() # This is to ensure that the distribution at t0 is exactly the prior
  virtual_phi_t0 = MixedGaussian(LinearFunctional(I, zero), inf_precision)

  # Create the potential over x_{t1}
  phi_t1 = MixedGaussian(y1, evidence_precision)

  # Get the transition distributions between t0 and t, and t and t1
  phi_tgt0 = base_sde.get_transition_distribution(t0, t)
  phi_t1gt = base_sde.get_transition_distribution(t, t1)

  # Compute p(x_t | y_{t0}; y_{t1})
  alpha_t: MixedGaussian = phi_tgt0.update_and_marginalize_out_x(virtual_phi_t0) # Mean is a linear functional
  beta_t: MixedGaussian = phi_t1gt.update_and_marginalize_out_y(phi_t1)
  p_xt_given_y0_and_y1: StandardGaussian = (alpha_t + beta_t).to_std()

  # The mean of the marginal distribution is a linear function dependent on y_{t0}
  mu_t: LinearFunctional = p_xt_given_y0_and_y1.mu # = A@y_{t0} + b
  Sigma_t: AbstractSquareMatrix = p_xt_given_y0_and_y1.Sigma

  transition_from_y0_to_xt_given_y1 = GaussianTransition(mu_t.A, mu_t.b, Sigma_t)

  # Integrate out y_{t0}
  p_xt_given_y1 = transition_from_y0_to_xt_given_y1.update_and_marginalize_out_x(x_t0_prior)
  return p_xt_given_y1

class MatchingConversion(AbstractBatchableObject):
  score_to_flow: LinearFunctional
  bwd_to_drift: LinearFunctional
  score_to_bwd: LinearFunctional
  y1_to_bwd: Callable[[Float[Array, 'D']], MixedGaussian]

  @property
  def batch_size(self):
    return self.score_to_flow.batch_size


class DiffusionModelComponents(AbstractBatchableObject):
  base_sde: AbstractLinearSDE
  t0: Scalar
  x_t0_prior: AbstractGaussianPotential
  t1: Scalar
  evidence_cov: AbstractSquareMatrix

  @property
  def batch_size(self):
    return self.base_sde.batch_size

class Y1ToBwdMean(LinearFunctional):
  """
  This class represents the mapping from y1 to the mean of the backward message at t.

  Example usage:
    y1_to_bwd_mean = Y1ToBwdMean(base_sde, t1, evidence_cov, t)
    bwd_mean = y1_to_bwd_mean(y1)
  """

  A: AbstractSquareMatrix
  b: Float[Array, 'D']

  precision: AbstractSquareMatrix

  def __init__(
    self,
    base_sde: AbstractLinearSDE,
    t1: Scalar,
    evidence_cov: AbstractSquareMatrix,
    t: Scalar
  ):
    evidence_precision = evidence_cov.get_inverse()

    # Create the virtual potential over x_{t0}.
    I, zero = DiagonalMatrix.eye(base_sde.dim), jnp.zeros(base_sde.dim)
    no_op_lf = LinearFunctional(I, zero)

    # Create the (virtual) potential over x_{t1}.  Don't fill with the evidence yet because we will want to return the mapping from y1 to bwd
    virtual_phi_t1 = MixedGaussian(no_op_lf, evidence_precision)

    # Get the transition distributions from t to t1
    phi_t1gt = base_sde.get_transition_distribution(t, t1)

    # Compute p(x_t | y_{t0}; y_{t1})
    virtual_beta_t: MixedGaussian = phi_t1gt.update_and_marginalize_out_y(virtual_phi_t1)

    self.A = virtual_beta_t.mu.A
    self.b = virtual_beta_t.mu.b
    self.precision = virtual_beta_t.J

class Y1ToMarginalMean(LinearFunctional):
  """
  This class represents the mapping from y1 to the mean of the marginal distribution at t.
  """

  A: AbstractSquareMatrix
  b: Float[Array, 'D']

  precision: AbstractSquareMatrix

  def __init__(
    self,
    base_sde: AbstractLinearSDE,
    t0: Scalar,
    x_t0_prior: AbstractGaussianPotential,
    t1: Scalar,
    evidence_cov: AbstractSquareMatrix,
    t: Scalar
  ):
    evidence_precision = evidence_cov.get_inverse()
    I, zero = DiagonalMatrix.eye(base_sde.dim), jnp.zeros(base_sde.dim)
    no_op_lf = LinearFunctional(I, zero)

    # Extract the mean and covariance of the prior
    x_t0_prior_std: StandardGaussian = x_t0_prior.to_std()
    m, P = x_t0_prior_std.mu, x_t0_prior_std.Sigma

    # Create the virtual potentials.  We are mostly just interested in their covariances
    virtual_phi_t0 = MixedGaussian(no_op_lf, evidence_precision.set_inf())
    virtual_phi_t1 = MixedGaussian(no_op_lf, evidence_precision)

    # Get the transition distributions between t0 and t, and t and t1
    phi_tgt0 = base_sde.get_transition_distribution(t0, t)
    phi_t1gt = base_sde.get_transition_distribution(t, t1)

    # Compute p(x_t | x_0, y_1)
    virtual_alpha_t: MixedGaussian = phi_tgt0.update_and_marginalize_out_x(virtual_phi_t0) # Mean is a linear functional
    virtual_beta_t:  MixedGaussian = phi_t1gt.update_and_marginalize_out_y(virtual_phi_t1) # Mean is a linear functional

    # This is a hack to marginalize out x0.  The math ends up working out if we plug the prior mean into
    # the forward message.  The only thing that we will need to do is compute the correction term for the covariance.
    p_xt_given_x0_and_y1: Callable[[Float[Array, 'D']], StandardGaussian] = (virtual_alpha_t(m) + virtual_beta_t).to_std()

    # Unpack the parameters of the forward message
    At_alpha = virtual_alpha_t.mu.A
    Jt_alpha = virtual_alpha_t.J

    # Construct the correction matrix for the covariance
    At_alpha_beta = p_xt_given_x0_and_y1.Sigma@Jt_alpha@At_alpha
    corrected_covariance = p_xt_given_x0_and_y1.Sigma + At_alpha_beta@P@At_alpha.T

    # Construct the correct marginal distribution
    virtual_pt_given_y1 = StandardGaussian(p_xt_given_x0_and_y1.mu, corrected_covariance)

    self.A = virtual_pt_given_y1.mu.A
    self.b = virtual_pt_given_y1.mu.b
    self.precision = virtual_pt_given_y1.Sigma.get_inverse()

def y1_to_drift(
  base_sde: AbstractLinearSDE,
  t1: Scalar,
  evidence_cov: AbstractSquareMatrix,
  t: Scalar,
  xt: Float[Array, 'D'],
  y1: Float[Array, 'D']
) -> LinearFunctional:
  """
  Compute the mapping from y1 to the drift at t.
  """
  # Get the backward message at t
  y1_to_bwd_mean = Y1ToBwdMean(base_sde, t1, evidence_cov, t)
  beta_t_mean = y1_to_bwd_mean(y1)
  beta_t = MixedGaussian(beta_t_mean, y1_to_bwd_mean.precision)

  # Get the parameters of the SDE at time t
  Ft, ut, Lt = base_sde.get_params(t)
  LtLtT = Lt@Lt.T

  # Compute the drift
  drift = Ft@xt + ut + LtLtT@beta_t.score(xt)
  return drift

def drift_to_y1(
  base_sde: AbstractLinearSDE,
  t1: Scalar,
  evidence_cov: AbstractSquareMatrix,
  t: Scalar,
  xt: Float[Array, 'D'],
  drift: Float[Array, 'D']
) -> LinearFunctional:
  """
  Compute the mapping from y1 to the drift at t.
  """
  # Get the backward message at t
  y1_to_bwd_mean = Y1ToBwdMean(base_sde, t1, evidence_cov, t)

  # Get the parameters of the SDE at time t
  Ft, ut, Lt = base_sde.get_params(t)
  LtLtT = Lt@Lt.T

  # Compute the score of the backward message
  betat_score = LtLtT.solve(drift - Ft@xt - ut)

  # Compute the mean of the backward message
  betat_mean = y1_to_bwd_mean.precision.solve(betat_score) + xt

  return y1_to_bwd_mean.get_inverse()(betat_mean)

def y1_to_score(
  components: DiffusionModelComponents,
  t: Scalar,
  xt: Float[Array, 'D'],
  y1: Float[Array, 'D']
) -> LinearFunctional:
  """
  Compute the mapping from y1 to the score at t.
  """
  # Get the backward message at t
  y1_to_marginal_mean = Y1ToMarginalMean(components.base_sde,
                                         components.t0,
                                         components.x_t0_prior,
                                         components.t1,
                                         components.evidence_cov,
                                         t)
  pt_mean = y1_to_marginal_mean(y1)
  pt = MixedGaussian(pt_mean, y1_to_marginal_mean.precision)
  return pt.score(xt)






















def score_to_bwd_linear_functional(
  base_sde: AbstractLinearSDE,
  t0: Scalar,
  x_t0_prior: AbstractGaussianPotential,
  t1: Scalar,
  y1: Float[Array, 'D'],
  evidence_cov: AbstractSquareMatrix,
  t: Scalar
) -> MatchingConversion:
  evidence_precision = evidence_cov.get_inverse()

  # Create the virtual potential over x_{t0}.
  I, zero = DiagonalMatrix.eye(base_sde.dim), jnp.zeros(base_sde.dim)
  inf_precision = evidence_precision.set_inf() # This is to ensure that the distribution at t0 is exactly the prior
  no_op_lf = LinearFunctional(I, zero)
  virtual_phi_t0 = MixedGaussian(no_op_lf, inf_precision)

  # Create the (virtual) potential over x_{t1}.  Don't fill with the evidence yet because we will want to return the mapping from y1 to bwd
  virtual_phi_t1 = MixedGaussian(no_op_lf, evidence_precision)
  # phi_t1 = MixedGaussian(y1, evidence_precision)

  # Get the transition distributions between t0 and t, and t and t1
  phi_tgt0 = base_sde.get_transition_distribution(t0, t)
  phi_t1gt = base_sde.get_transition_distribution(t, t1)

  # Compute p(x_t | y_{t0}; y_{t1})
  virtual_alpha_t: MixedGaussian = phi_tgt0.update_and_marginalize_out_x(virtual_phi_t0) # Mean is a linear functional
  virtual_beta_t: MixedGaussian = phi_t1gt.update_and_marginalize_out_y(virtual_phi_t1)
  beta_t: MixedGaussian = resolve_functional(virtual_beta_t, y1)
  p_xt_given_y0_and_y1: StandardGaussian = (virtual_alpha_t + beta_t).to_std()

  # The mean of the marginal distribution is a linear function dependent on y_{t0}
  mu_t: LinearFunctional = p_xt_given_y0_and_y1.mu # = A@y_{t0} + b
  Sigma_t: AbstractSquareMatrix = p_xt_given_y0_and_y1.Sigma
  transition_from_y0_to_xt_given_y1 = GaussianTransition(mu_t.A, mu_t.b, Sigma_t) # N(xt | A@y_{t0} + b, Sigma_t)

  # Integrate out y_{t0}
  p_xt_given_y1 = transition_from_y0_to_xt_given_y1.update_and_marginalize_out_x(x_t0_prior)

  #########################################################
  #########################################################
  #########################################################
  # Get the mapping between the score of the marginal and the backward message
  pxt_nat = p_xt_given_y1.to_nat()
  bwd_nat = beta_t.to_nat()

  Jp, hp = pxt_nat.J, pxt_nat.h
  Jb, hb = bwd_nat.J, bwd_nat.h

  A_score_to_bwd = Jb@Jp.get_inverse()
  b_score_to_bwd = hb - A_score_to_bwd@hp
  score_to_bwd = LinearFunctional(A_score_to_bwd, b_score_to_bwd) # score_to_bwd(score_xt) = bwd_xt

  #########################################################
  # Compute the mapping between the backward message and the drift
  # drift = Ft@xt + ut + LtLt^T bwd
  Ft, ut, Lt = base_sde.get_params(t)
  LtLtT = Lt@Lt.T

  A_bwd_to_drift = LtLtT - Ft@Jb.get_inverse()
  b_bwd_to_drift = ut + Ft@beta_t.mu
  bwd_to_drift = LinearFunctional(A_bwd_to_drift, b_bwd_to_drift) # bwd_to_drift(bwd_xt) = drift_xt

  #########################################################
  # Compute the mapping between the score and the probability flow ODE
  score_to_drift = score_to_bwd@bwd_to_drift
  score_to_flow = no_op_lf - 0.5*LtLtT@score_to_drift

  #########################################################
  # y1 to bwd message
  y1_to_bwd: Callable[[Float[Array, 'D']], MixedGaussian] = virtual_beta_t # y1_to_bwd(y1) = bwd

  #########################################################
  # Get the conversions that we'd use in practice
  drift_to_score = score_to_drift.get_inverse()
  drift_to_flow = drift_to_score@score_to_flow


  return MatchingConversion(score_to_flow, bwd_to_drift, score_to_bwd, y1_to_bwd, drift_to_flow)

################################################################################################################

class ForwardSDE(AbstractLinearSDE):
  r"""This class represents an SDE that transforms a sample from an unknown distribution into a Gaussian distribution
  as is the case for the forward SDE in diffusion models.  This is an SDE whose marginal distribution is equal to
  p(x_t | y_{t0}) = \int p(x_t | y_{t1}, y_{t0}) p(y_{t1} | y_{t0}) dy_{t1} where p(y_{t1} | y_{t0}) is the Gaussian
  prior and p(x_t | y_{t1}, y_{t0}) is marginal distribution of the conditioned base SDE.
  """

  base_sde: AbstractLinearSDE
  evidence_precision: AbstractSquareMatrix

  t0: Scalar
  phi_t0: MixedGaussian

  t1: Scalar
  virtual_phi_t1: MixedGaussian

  y_t1_prior: AbstractGaussianPotential

  def __init__(
    self,
    base_sde: AbstractLinearSDE,
    t0: Scalar,
    y0: Float[Array, 'D'],
    t1: Scalar,
    y_t1_prior: AbstractGaussianPotential,
    evidence_cov: Optional[AbstractSquareMatrix] = None,
  ):
    xdim = base_sde.dim
    if evidence_cov is None:
      # evidence_cov = DiagonalMatrix.eye(xdim)*1e-6 # For numerical stability
      evidence_cov = DiagonalMatrix.zeros(xdim)
    else:
      if isinstance(evidence_cov, AbstractSquareMatrix) == False:
        raise ValueError("evidence_cov must be an AbstractSquareMatrix")
    self.evidence_precision = evidence_cov.get_inverse()

    self.base_sde = base_sde

    # Construct the virtual evidence.  We have real evidence at t0 (y0), but at t1, we only have a prior over
    # an variable.  To be general, we can represent y_t1 by a trivial linear functional, lf(y_t1) = I@y_t1 + 0.
    self.t0 = t0
    self.phi_t0 = MixedGaussian(y0, self.evidence_precision)

    self.t1 = t1
    lfT = LinearFunctional(DiagonalMatrix.eye(xdim), jnp.zeros(xdim))
    self.virtual_phi_t1 = MixedGaussian(lfT, self.evidence_precision)

    self.y_t1_prior = y_t1_prior

  @property
  def batch_size(self):
    """Get the batch size from the base SDE"""
    return self.base_sde.batch_size

  @property
  def dim(self):
    """Get the dimension from the base SDE"""
    return self.base_sde.dim

  def get_params(self, t: Scalar) -> Tuple[AbstractSquareMatrix, Float[Array, 'D'], AbstractSquareMatrix]:
    """Get the parameters of the SDE at time t.  This has a few steps:
      1: Compute p(x_t | y_{t0}, y_{t1})
      2: Compute the Bayes estimate of the mean of p(y_{t1} | x_t, y_{t0})
      3: Get the Bayes estimate of the backward message at t
      4: Get the virtual parameters of the conditioned SDE
      5: Unpack the linear functional to apply a final correction

    All of these require message passing, which we will do explicitly instead of using the helpers in this library
    because we only have 2 times, t0 and t1, and because we have evidence of mixed types (float and linear functional).
    """

    phi_t1gt = self.base_sde.get_transition_distribution(t, self.t1)
    phi_tgt0 = self.base_sde.get_transition_distribution(self.t0, t)

    ##########################################
    # 1: Compute p(x_t | y_{t0}, y_{t1})
    ##########################################
    beta_t: MixedGaussian = phi_t1gt.update_and_marginalize_out_y(self.virtual_phi_t1) # Mean is a linear functional
    alpha_t: MixedGaussian = phi_tgt0.update_and_marginalize_out_x(self.phi_t0)
    p_t: StandardGaussian = (alpha_t + beta_t).to_std()

    # The mean of the marginal distribution is a linear function dependent on y_{t1}
    mu_t: LinearFunctional = p_t.mu # = A@y_{t1} + b
    Sigma_t: AbstractSquareMatrix = p_t.Sigma

    ##########################################
    # 2: Compute the Bayes estimate of the mean of p(y_{t1} | x_t, y_{t0})
    ##########################################
    # Turn the marginal into p(x_t | y_{t1}; y_{t0}) by unpacking the linear functional
    p_xt_given_yt1_and_yt0 = GaussianTransition(mu_t.A, mu_t.b, Sigma_t)

    # Swap the order to get a potential from x_t to y_{t1}
    phi_yt1_given_xt_and_yt0: GaussianTransition = p_xt_given_yt1_and_yt0.swap_variables()

    # Incorporate the prior evidence
    p_yt1_given_xt_and_yt0: GaussianTransition = phi_yt1_given_xt_and_yt0.update_y(self.y_t1_prior, only_return_transition=True)

    # The Bayes estimate of y_{t1} is the mean of the transition
    A, b = p_yt1_given_xt_and_yt0.A, p_yt1_given_xt_and_yt0.u

    y_t1_bayes_estimate: LinearFunctional = LinearFunctional(A, b) # Ax_t + b = E(y_{t1} | x_t, y_{t0})
    phi_t1_bayes_estimate: MixedGaussian = MixedGaussian(y_t1_bayes_estimate, self.evidence_precision)

    ##########################################
    # 3: Get the Bayes estimate of the backward message at t
    ##########################################
    beta_t_bayes_estimate: NaturalGaussian = phi_t1gt.update_and_marginalize_out_y(phi_t1_bayes_estimate).to_nat()
    J_beta_t = beta_t_bayes_estimate.J
    h_beta_t_bayes_estimate: LinearFunctional = beta_t_bayes_estimate.h

    ##########################################
    # 4: Get the virtual parameters of the conditioned SDE
    ##########################################
    F, u, L = self.base_sde.get_params(t)
    LLT = L@L.T

    F_cond: AbstractSquareMatrix = F - LLT@J_beta_t
    u_cond: LinearFunctional = u + LLT@h_beta_t_bayes_estimate

    ##########################################
    # 5: Unpack the linear functional to apply a final correction
    ##########################################
    F_corrected = F_cond + u_cond.A
    u_corrected = u_cond.b

    return F_corrected, u_corrected, L

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  from linsdex.sde.sde_examples import BrownianMotion, LinearTimeInvariantSDE
  from linsdex.matrix.diagonal import DiagonalMatrix
  from linsdex.matrix.dense import DenseMatrix
  from linsdex.sde.ode_sde_simulation import ode_solve, ODESolverParams
  from linsdex.ssm.simple_encoder import IdentityEncoder

  # turn on x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  k1, k2, k3 = random.split(key, 3)

  base_sde = BrownianMotion(sigma=1.0, dim=2)
  t0 = jnp.array(0.0)

  # prior_mean = jnp.array([0.0, 2.0])
  prior_mean = random.normal(k1, (2,))
  prior_covariance = DiagonalMatrix.eye(2)*0.5
  prior_gaussian = StandardGaussian(prior_mean, prior_covariance)

  t1 = jnp.array(1.0)
  # y1 = jnp.array([0.0, 0.0])
  y1 = random.normal(k2, (2,))

  evidence_cov = DiagonalMatrix.eye(2)*0.001

  t = jnp.array(0.1)
  p_xt_given_y1: StandardGaussian = get_integrated_marginal_given_y1(base_sde, t0, prior_gaussian, t1, y1, evidence_cov, t)

  xt = random.normal(key, (2,))
  score_to_bwd: LinearFunctional = score_to_bwd_linear_functional(base_sde, t0, prior_gaussian, t1, y1, evidence_cov, t)
  bwd = score_to_bwd(xt)
  import pdb; pdb.set_trace()


  def get_marginal(t: Scalar) -> StandardGaussian:
    pxt: StandardGaussian = get_integrated_marginal_given_y1(base_sde, t0, prior_gaussian, t1, y1, evidence_cov, t)
    xt = pxt.sample(key)
    return xt, pxt

  (xt, pxt), (flow, dpxt) = eqx.filter_jvp(get_marginal, (t,), (jnp.ones_like(t),))
  import pdb; pdb.set_trace()


  def get_matching_items(
    t: Scalar,
    y1: Float[Array, 'D']
  ) -> Dict[str, Float[Array, 'D']]:

    def get_marginal(t: Scalar) -> StandardGaussian:
      return get_integrated_marginal_given_y1(base_sde, t0, prior_gaussian, t1, y1, evidence_cov, t)

    pxt, dpxt = jax.jvp(get_marginal, (t,), (jnp.ones_like(t),))


  import pdb; pdb.set_trace()




  class FlowMatchingForwardSDE(AbstractLinearSDE):
    def batch_size(self):
      return None

    def get_params(self, s: Scalar) -> Tuple[AbstractSquareMatrix, Float[Array, 'D'], AbstractSquareMatrix]:
      t = 1-s
      kappa_t = 1/t
      eta_t = (1-t)/t
      F = DiagonalMatrix.eye(2)*-kappa_t
      u = jnp.zeros(2)
      L = DiagonalMatrix.eye(2)*jnp.sqrt(2*eta_t)
      return F, u, L

    @property
    def dim(self):
      return 2

  base_sde = FlowMatchingForwardSDE()
  transition = base_sde.get_transition_distribution(0.001, 1.0-0.001)
  import pdb; pdb.set_trace()






  # turn on x64
  # jax.config.update("jax_enable_x64", True)

  # Construct the endpoints of the bridge
  class GMM(eqx.Module):
    mus: Float[Array, 'n_components D']
    Sigmas: Float[Array, 'n_components D']
    weights: Float[Array, 'n_components']

    def __init__(self, mus, Sigmas, weights):
      self.mus = mus
      self.Sigmas = Sigmas
      self.weights = weights

    @property
    def n_components(self) -> int:
      return self.mus.shape[0]

    def sample(self, key: PRNGKeyArray) -> Float[Array, 'D']:
      key_cat, key_gauss = random.split(key)
      component_idx = random.categorical(key_cat, jnp.log(self.weights))
      mu = self.mus[component_idx]
      Sigma_diag = self.Sigmas[component_idx]
      return random.multivariate_normal(key_gauss, mu, jnp.diag(Sigma_diag))


  n_components = 10
  mus = jnp.array([[0.0, 0.0], [4.0, 1.0], [2.0, 5.0]])
  Sigmas = jnp.array([[1.0, 0.1], [0.1, 1.0], [1.0, 1.0]])
  weights = jnp.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
  gmm = GMM(mus, Sigmas, weights)
  keys = random.split(key, 1000)
  p0_samples = jax.vmap(gmm.sample)(keys)
  # plt.scatter(*p0_samples.T)
  # plt.show()

  # For the other endpoint, we will use a regular standard Gaussian
  prior_mean = jnp.array([0.0, 2.0])
  prior_covariance = DiagonalMatrix.eye(2)*0.5
  prior_gaussian = StandardGaussian(prior_mean, prior_covariance)
  p1_samples = jax.vmap(prior_gaussian.sample)(keys)
  # plt.scatter(*p1_samples.T)
  # plt.show()


  # Construct the forward SDE
  k1, k2 = random.split(key)
  F = DenseMatrix(random.normal(k1, (2, 2)))
  L = DenseMatrix(random.normal(k2, (2, 2)))*0.1
  # base_sde = LinearTimeInvariantSDE(F, L)
  # base_sde = BrownianMotion(sigma=1.0, dim=2)
  t0 = 0.0
  y0 = p0_samples[0]
  T = 1.0
  yT_prior = prior_gaussian
  forward_sde = ForwardSDE(base_sde, t0, y0, T, yT_prior)


  params = forward_sde.get_params(0.1)


  transition = forward_sde.get_transition_distribution(1e-3, 1.0-1e-3)
  # transition = forward_sde.get_transition_distribution(0.0, 1.0)
  endpoint_marginal = transition.condition_on_x(y0)
  import pdb; pdb.set_trace()