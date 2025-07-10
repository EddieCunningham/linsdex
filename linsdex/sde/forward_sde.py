
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
      evidence_cov = DiagonalMatrix.eye(xdim)*1e-6 # For numerical stability
      # evidence_cov = DiagonalMatrix.zeros(xdim)
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
      1. Get the Bayes estimate of yT given xt, E(y_T | x_t)
      2. Use the Bayes estimate of yT as virtual evidence to condition the base SDE, p(x_t | y_{t0}, y_T)
      3. Get the parameters of the conditioned SDE, F_t, u_t(x_t), L_t
      4. Unpack the linear functional u_t(x_t) into A_t, b_t to get the final parameters

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
    p_xt_given_yt1_and_yt0 = GaussianTransition(mu_t.A, mu_t.b, Sigma_t) # Stable

    # Swap the order to get a potential from x_t to y_{t1}
    phi_yt1_given_xt_and_yt0: GaussianTransition = p_xt_given_yt1_and_yt0.swap_variables()

    # Incorporate the prior evidence
    p_yt1_given_xt_and_yt0: GaussianTransition = phi_yt1_given_xt_and_yt0.update_y(self.y_t1_prior, only_return_transition=True)

    # The Bayes estimate of y_{t1} is the mean of the transition
    A, b = p_yt1_given_xt_and_yt0.A, p_yt1_given_xt_and_yt0.u

    y_t1_bayes_estimate: LinearFunctional = LinearFunctional(A, b)
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

  key = random.PRNGKey(0)

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
  L = DenseMatrix(random.normal(k2, (2, 2)))
  base_sde = LinearTimeInvariantSDE(F, L)
  # base_sde = BrownianMotion(sigma=1.0, dim=2)
  t0 = 0.0
  y0 = p0_samples[0]
  T = 1.0
  yT_prior = prior_gaussian
  forward_sde = ForwardSDE(base_sde, t0, y0, T, yT_prior)


  params = forward_sde.get_params(0.0)


  transition = forward_sde.get_transition_distribution(0.0, 1.0)
  endpoint_marginal = transition.condition_on_x(y0)
  import pdb; pdb.set_trace()