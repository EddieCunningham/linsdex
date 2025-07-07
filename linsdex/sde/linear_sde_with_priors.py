
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
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.sde.sde_base import AbstractSDE

def get_bayes_estimate_of_mean_of_evidence(
  transition_to_evidence: GaussianTransition,
  evidence_mean_prior: StandardGaussian,
  evidence_covariance: AbstractSquareMatrix,
  xt: Float[Array, 'D']
) -> Float[Array, 'D']:
  r"""Get the Bayes estimate of the mean of the evidence at time t and xt.

  Suppose that we have a potential function of a stochastic process at time t
  of the form N(xT | \mu, Sigma) where \mu ~ N(m, P).  This function computes
  E_{p(\mu|xt)}[\mu] where p(\mu|xt) is the posterior distribution of the mean of the
  evidence at time t given the evidence at time t.

  Another way to write the Bayes estimate is
  E_{p(\mu|xt)}[\mu] = \argmin{f(x_t)}E_{p(\mu,xt)}[\|f(x_t) - \mu|^2]

  This function is pretty much just to test get_linear_parameters_of_bayes_estimate_of_mean_of_evidence

  Args:
    transition_to_evidence: The transition distribution from time t to T.
    evidence_mean_prior: The prior distribution of the mean of the evidence.
    evidence_covariance: The covariance matrix of the evidence.
    xt: The current value of the stochastic process.
  """
  ATgt, uTgt, SigmaTgt = transition_to_evidence.A, transition_to_evidence.u, transition_to_evidence.Sigma
  SigmaT = evidence_covariance
  mu_new = ATgt@xt + uTgt
  Sigma_new = SigmaTgt + SigmaT
  combined = evidence_mean_prior + StandardGaussian(mu_new, Sigma_new)
  muT_bayes_estimate = combined.to_std().mu
  return muT_bayes_estimate

def get_linear_parameters_of_bayes_estimate_of_mean_of_evidence(
  transition_to_evidence: GaussianTransition,
  evidence_mean_prior: StandardGaussian,
  evidence_covariance: AbstractSquareMatrix
) -> Tuple[AbstractSquareMatrix, Float[Array, 'D']]:
  r"""Get a matrix A and vector b such that A@xt + b = E_{p(\mu|xt)}[\mu]

  Args:
    transition_to_evidence: The transition distribution from time t to T.
    evidence_mean_prior: The prior distribution of the mean of the evidence.
    evidence_covariance: The covariance matrix of the evidence.
  """
  ATgt, uTgt, SigmaTgt = transition_to_evidence.A, transition_to_evidence.u, transition_to_evidence.Sigma
  SigmaT = evidence_covariance

  m_prior = evidence_mean_prior.mu
  P_prior = evidence_mean_prior.Sigma

  Sigma_new = SigmaTgt + SigmaT
  P_plus_Sigma_new = P_prior + Sigma_new

  # The Bayes estimate is a weighted average of the prior mean and the data-driven mean.
  # mu_bayes = (I-S) * mu_data + S * mu_prior
  # where S = Sigma_new * (P_prior + Sigma_new)^-1
  # and mu_data = ATgt@xt + uTgt.
  # This can be rewritten as:
  # mu_bayes = (I-S)@ATgt@xt + (I-S)@uTgt + S@m_prior
  # So, A = (I-S)@ATgt and b = (I-S)@uTgt + S@m_prior
  #
  # We can more efficiently compute (I-S) and S.
  # (I-S) = P_prior * (P_prior + Sigma_new)^-1
  # S = Sigma_new * (P_prior + Sigma_new)^-1

  # S = Sigma_new @ inv(P_plus_Sigma_new)
  S = P_plus_Sigma_new.solve(Sigma_new)

  # I_minus_S = P_prior @ inv(P_plus_Sigma_new)
  I_minus_S = P_plus_Sigma_new.solve(P_prior)

  A = I_minus_S @ ATgt
  b = I_minus_S @ uTgt + S @ m_prior

  return A, b

class LinearSDEWithPriors(AbstractSDE):
  r"""This class represents a linear SDE with priors on the parameters.  We assume that there
  are priors over the trajectories produced by the input linear SDE of the form
  N(x_T | \mu_T, \Sigma_T) where \Sigma_T is what `evidence_covariances` represents,
  and the prior distribution on \mu_T, N(\mu_T | m_T, P_T), is what `evidence_mean_prior`
  represents.  This class represents the linear SDE that results from integrating out the
  means of the evidence.

  This is useful for constructing forward SDEs in diffusion models that are equal to the
  prior distribution at the times of the evidence.
  """

  linear_sde: AbstractLinearSDE # Contains the evidence covariances.
  evidence_mean_priors: GaussianPotentialSeries # N(\mu_T | m_T, P_T) for each T
  evidence_covariances: Annotated[AbstractSquareMatrix, 'n_times'] # \Sigma_T for each T

  def __init__(
    self,
    linear_sde: AbstractLinearSDE,
    evidence_mean_priors: GaussianPotentialSeries,
    evidence_covariances: Annotated[AbstractSquareMatrix, 'n_times']
  ):
    self.linear_sde = linear_sde
    self.evidence_mean_priors = evidence_mean_priors
    self.evidence_covariances = evidence_covariances

  @property
  def batch_size(self):
    """Get the batch size from the base SDE"""
    return self.linear_sde.batch_size

  @property
  def dim(self):
    """Get the dimension from the base SDE"""
    return self.linear_sde.dim

  def get_params(self, t: Scalar, xt: Float[Array, 'D']) -> Tuple[AbstractSquareMatrix, Float[Array, 'D'], AbstractSquareMatrix]:
    """We first need to compute the Bayes estimates of the prior means
    and then we will be able to simply use a conditioned SDE in order
    to get all of the parameters."""

    # Estimate the means of all of the evidence at time t
    def get_evidence_bayes_estimates(
      T: Scalar,
      evidence_mean_prior: StandardGaussian,
      evidence_covariance: AbstractSquareMatrix
    ) -> StandardGaussian:
      transition_to_evidence = self.linear_sde.get_transition_distribution(t, T)
      muT = get_bayes_estimate_of_mean_of_evidence(
        transition_to_evidence, evidence_mean_prior, evidence_covariance, xt
      )
      return StandardGaussian(muT, evidence_covariance)

    ts: Float[Array, 'n_times'] = self.evidence_mean_priors.times
    evidence_mean_priors: GaussianPotentialSeries = self.evidence_mean_priors.node_potentials
    evidence_covariances: Annotated[AbstractSquareMatrix, 'n_times'] = self.evidence_covariances

    evidence_potentials: Annotated[StandardGaussian, 'n_times'] = jax.vmap(get_evidence_bayes_estimates)(
      ts, evidence_mean_priors, evidence_covariances
    )

    evidence_series = GaussianPotentialSeries.from_potentials(ts, evidence_potentials)

    # Condition the linear SDE on this evidence
    conditioned_sde = self.linear_sde.condition_on(evidence_series)
    return conditioned_sde.get_params(t)

  def get_drift(self, t: Scalar, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    F, u, _ = self.get_params(t, xt)
    return F@xt + u

  def get_diffusion_coefficient(self, t: Scalar, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    _, _, L = self.get_params(t, xt)
    return L