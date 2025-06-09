import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Dict, overload
import einops
import equinox as eqx
import abc
import diffrax
from jaxtyping import Array, PRNGKeyArray
import jax.tree_util as jtu
import os
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool
from jax._src.util import curry
from linsdex.series.series import TimeSeries
from linsdex.series.interleave_times import InterleavedTimes
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from linsdex.potential.gaussian.dist import StandardGaussian, MixedGaussian
from linsdex.matrix import DiagonalMatrix, TAGS
from typing import Literal, Optional, Union, Tuple, Callable, List, Any

################################################################################################################

class AbstractEncoder(eqx.Module, abc.ABC):
  """A recognition model that predicts the h natural parameter of the CRF.
  Returns mean and inverse covariance."""

  potential_cov_type: eqx.AbstractVar[Literal['diagonal', 'dense', 'block2x2', 'block3x3']]

  def __call__(
    self,
    series: TimeSeries
  ) -> GaussianPotentialSeries:
    """Return the mean and inverse covariance of the potential function for the latent variable model

    **Arguments**:
      - `yts`: The observed data

    **Returns**:
      - `xts`: The probabilistic time series.  This indicates Gaussian potential functions
        for the latent variable at each time.
    """
    pass

################################################################################################################

class IdentityEncoder(AbstractEncoder, abc.ABC):

  y_dim: int
  x_dim: int
  potential_cov_type: Literal['diagonal', 'dense', 'block2x2', 'block3x3'] = 'diagonal'

  def __init__(self, dim: int):
    self.x_dim = dim
    self.y_dim = dim
    self.potential_cov_type = 'diagonal'

  def encode(self, series: TimeSeries) -> Float[Array, 'T D']:
    mask = jnp.broadcast_to(series.mask[:,None], series.values.shape)
    return jnp.where(mask, series.values, 0.0)

  def __call__(
    self,
    series: TimeSeries
  ) -> GaussianPotentialSeries:
    """Return the mean of the potential function for the latent variable model.
    We set the inverse covariance to None to indicate that we are placing
    100% certainty in this observation.  This can be used to build a stochastic bridge.

    **Arguments**:
      - `series`: The observed data

    **Returns**:
      - `series`: The probabilistic time series
    """
    xts = self.encode(series)
    inverse_covs = None
    prob_series = GaussianPotentialSeries(series.times, xts, inverse_covs)
    return prob_series

################################################################################################################

class PaddingLatentVariableEncoderWithPrior(AbstractEncoder):
  """Pad the observed data with zeros and add a standard Gaussian prior in the first position"""

  y_dim: int
  x_dim: int
  sigma: Scalar
  potential_cov_type: Literal['diagonal', 'dense', 'block2x2', 'block3x3'] = 'diagonal'
  use_prior: bool = True

  def __init__(
    self,
    y_dim: int,
    x_dim: int,
    sigma: Scalar,
    use_prior: bool = True
  ):
    """
    **Arguments**:
      - `y_dim`: The dimension of the observed data
      - `x_dim`: The dimension of the latent variable
      - `sigma`: The standard deviation of the observation noise
    """
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.sigma = sigma
    self.use_prior = use_prior

  def __call__(self,
    series: TimeSeries,
    parameterization: Optional[str] = None
  ) -> GaussianPotentialSeries:
    """Return the mean and inverse covariance of the potential function for the latent variable model.

    **Arguments**:
      - `series`: The observed data
      - `parameterization`: The parameterization of the potential function

    **Returns**:
      - `mean`: The mean of the potential function
      - `inverse_cov`: The inverse covariance of the potential function
    """
    def process_single_elt(y, mask):
      y = jnp.where(mask, y, 0.0)
      mean = jnp.pad(y, (0, self.x_dim - self.y_dim))

      # We place 0 certainty in unobserved elements
      mask = jnp.pad(mask, (0, self.x_dim - self.y_dim))
      inverse_cov = jnp.pad(jnp.ones(self.y_dim)/self.sigma**2, (0, self.x_dim - self.y_dim))
      inverse_cov = jnp.where(mask, inverse_cov, 0.0)
      return mean, inverse_cov

    full_mask = jnp.broadcast_to(series.mask[:,None], series.values.shape)
    means, inverse_covs = jax.vmap(process_single_elt)(series.values, full_mask)

    # Add a standard Gaussian prior in the first position
    if self.use_prior:
      inverse_covs = inverse_covs + 0.01 # For stability
      # inverse_covs = inverse_covs.at[0].add(1.0)

    if parameterization is None:
      parameterization = 'natural'
    prob_series = GaussianPotentialSeries(series.times, means, certainty=inverse_covs, parameterization=parameterization)
    return prob_series

  def from_observation_space_prob_series(self, prob_series: GaussianPotentialSeries) -> GaussianPotentialSeries:
    """If we have a potential in the observation space, we need to pad it with zeros
    so that it is a valid potential in the latent space"""

    potentials = prob_series.node_potentials

    def process_single_potential(potential: MixedGaussian) -> MixedGaussian:
      assert potential.J.batch_size is None
      assert isinstance(potential, MixedGaussian)
      assert isinstance(potential.J, DiagonalMatrix)

      J_elements = potential.J.elements
      J_elements = jnp.pad(J_elements, (0, self.x_dim - self.y_dim))
      J = DiagonalMatrix(J_elements, tags=TAGS.symmetric_tags)

      mu = potential.mu
      mu = jnp.pad(mu, (0, self.x_dim - self.y_dim))

      new_potential = MixedGaussian(mu, J).to_nat()
      return new_potential

    new_potentials = jax.vmap(process_single_potential)(potentials)
    prob_series = GaussianPotentialSeries(prob_series.times, new_potentials)
    return prob_series

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *

  key = random.PRNGKey(0)
  batch_size = 8
  y_dim = 3
  x_dim = 5
  sigma = 1.0
  encoder = SimpleLatentVariableEncoder(y_dim, x_dim, sigma)
  y = random.normal(key, (batch_size, y_dim))
  observation_mask = random.bernoulli(key, 0.5, (batch_size, y_dim)).astype(bool)
  means, covs = jax.vmap(encoder)(y, observation_mask)
  print(means)
  print(covs)
  import pdb; pdb.set_trace()


