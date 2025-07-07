import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Annotated
import einops
import equinox as eqx
from abc import ABC, abstractmethod
import diffrax
from jaxtyping import Array, PRNGKeyArray, Float, Scalar
from jax._src.util import curry
import abc
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix, TAGS
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
import linsdex.util as util
import warnings
import lineax as lx
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.potential.gaussian.dist import StandardGaussian, NaturalGaussian
from linsdex.matrix.matrix_with_inverse import MatrixWithInverse
from linsdex.sde.sde_base import AbstractLinearTimeInvariantSDE, AbstractLinearSDE
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.block.block_3x3 import Block3x3Matrix

__all__ = [
  "LinearTimeInvariantSDE",
  "BrownianMotion",
  "OrnsteinUhlenbeck",
  "VariancePreserving",
  "WienerVelocityModel",
  "WienerAccelerationModel",
  "CriticallyDampedLangevinDynamics",
  "TOLD",
  "StochasticHarmonicOscillator",]

class LinearTimeInvariantSDE(AbstractLinearTimeInvariantSDE):
  F: AbstractSquareMatrix
  L: AbstractSquareMatrix

class BrownianMotion(AbstractLinearTimeInvariantSDE):

  F: DiagonalMatrix
  L: DiagonalMatrix

  def __init__(
    self,
    sigma: Scalar,
    dim: int
  ):
    self.F = DiagonalMatrix.zeros(dim)
    self.L = DiagonalMatrix.eye(dim)*sigma

class OrnsteinUhlenbeck(AbstractLinearTimeInvariantSDE):

  F: DiagonalMatrix
  L: DiagonalMatrix

  def __init__(
    self,
    sigma: Scalar,
    lambda_: Scalar,
    dim: int
  ):
    self.F = DiagonalMatrix.eye(dim)*-lambda_
    self.L = DiagonalMatrix.eye(dim)*sigma

class VariancePreserving(AbstractLinearSDE, abc.ABC):

  beta_min: Scalar
  beta_max: Scalar
  dim: int

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.beta_min.ndim == 0:
      return None
    elif self.beta_min.ndim == 1:
      return self.beta_min.shape[0]
    else:
      return self.beta_min.shape[:-1]

  def beta(self, t: Scalar) -> Scalar:
    return self.beta_min + t*(self.beta_max - self.beta_min)

  def T(self, t: Scalar) -> Scalar:
    return t*self.beta_min + 0.5*t**2*(self.beta_max - self.beta_min)

  def get_params(self, t: Scalar) -> Tuple[AbstractSquareMatrix,
                                           Float[Array, 'D'],
                                           AbstractSquareMatrix]:
    beta = self.beta(t)
    I = DiagonalMatrix.eye(self.dim)
    F = -0.5*beta*I
    u = jnp.zeros((self.dim,))
    L = jnp.sqrt(beta)*I
    return F, u, L

  def get_transition_distribution(self,
                                  s: Scalar,
                                  t: Scalar) -> GaussianTransition:
    Tt = self.T(t)
    Ts = self.T(s)
    dT = Tt - Ts
    alpha = jnp.exp(-0.5*dT)
    I = DiagonalMatrix.eye(self.dim)

    A = alpha*I
    u = jnp.zeros((self.dim,))
    Sigma = (1 - jnp.exp(-dT))*I

    return GaussianTransition(A, u, Sigma)

################################################################################################################

class WienerVelocityModel(AbstractLinearTimeInvariantSDE):
  """A higher order tracking model.  This has latent variables corresponding
  to different derivatives of position.  For example, if order=3, then the
  latent variables correspond to position, velocity and acceleration.  There
  will be no noise added to position and less noise added to the lower derivatives
  than the higher derivatives.
  """

  F: Union[Block2x2Matrix,Block3x3Matrix]
  L: DiagonalMatrix
  position_dim: int = eqx.field(static=True)
  order: int = eqx.field(static=True)

  def __init__(self,
               sigma: Scalar,
               position_dim: int,
               order: int):
    assert order > 1
    self.position_dim = position_dim
    self.order = order

    # Build a block matrix with order x order blocks and to start
    # make each block the identitiy matrix
    I = jnp.eye(self.position_dim)
    F = jnp.block([[I]*self.order]*self.order)

    # Zero out everything that isn't the upper elements
    F = jnp.triu(F, k=self.position_dim)
    if order == 2:
      elements = jnp.zeros((2, 2, self.position_dim))
      elements = elements.at[0,1,:].set(1.0)
      def make_block(elements):
        return DiagonalMatrix(elements, tags=TAGS.no_tags)
      F = jax.vmap(jax.vmap(make_block))(elements)
      self.F = Block2x2Matrix(F, tags=TAGS.no_tags)
    elif order == 3:
      elements = jnp.zeros((3, 3, self.position_dim))
      elements = elements.at[0,1,:].set(1.0)
      elements = elements.at[1,2,:].set(1.0)
      def make_block(elements):
        return DiagonalMatrix(elements, tags=TAGS.no_tags)
      F = jax.vmap(jax.vmap(make_block))(elements)
      self.F = Block3x3Matrix(F, tags=TAGS.no_tags)
    else:
      self.F = DenseMatrix(jnp.tril(F, k=self.position_dim), tags=TAGS.no_tags)

    # Construct the diffusion matrix.  Place less noise on lower order terms
    sigma = jnp.array(sigma)
    if sigma.ndim == 0:
      factor = 3
      order_noise = (1/sigma**(factor - 1))*jnp.linspace(0.0, sigma, self.order)**factor
      order_noise = jnp.repeat(order_noise, self.position_dim)
      self.L = DiagonalMatrix(order_noise, tags=TAGS.no_tags)
    else:
      assert sigma.shape == ((self.order - 1),)
      sigma = jnp.pad(sigma, (1, 0))
      sigma = jnp.repeat(sigma, self.position_dim)
      self.L = DiagonalMatrix(sigma, tags=TAGS.no_tags)

class WienerAccelerationModel(AbstractLinearTimeInvariantSDE):

  F: Block3x3Matrix
  L: DiagonalMatrix
  position_dim: int = eqx.field(static=True)

  def __init__(self,
               sigma: Scalar,
               position_dim: int):
    self.position_dim = position_dim

    # Build a block matrix with order x order blocks and to start
    # make each block the identitiy matrix
    I = jnp.eye(self.position_dim)
    F = jnp.block([[I]*3]*3)

    # Zero out everything that isn't the upper elements
    F = jnp.triu(F, k=self.position_dim)
    elements = jnp.zeros((3, 3, self.position_dim))
    elements = elements.at[0,1,:].set(1.0)
    elements = elements.at[1,2,:].set(1.0)
    def make_block(elements):
      return DiagonalMatrix(elements, tags=TAGS.no_tags)
    F = jax.vmap(jax.vmap(make_block))(elements)
    self.F = Block3x3Matrix(F, tags=TAGS.no_tags)

    # Construct the diffusion matrix.  Place less noise on lower order terms
    sigma_diag = jnp.zeros(self.F.shape[0])
    sigma_diag = sigma_diag.at[-position_dim:].set(sigma)
    self.L = DiagonalMatrix(sigma_diag, tags=TAGS.no_tags)

class CriticallyDampedLangevinDynamics(AbstractLinearTimeInvariantSDE):
  """https://arxiv.org/pdf/2112.07068"""
  F: Annotated[Block2x2Matrix, DiagonalMatrix]
  L: DiagonalMatrix

  def __init__(self,
               mass: Union[Float[Array, 'dim'], Scalar],
               beta: Union[Float[Array, 'dim'], Scalar],
               dim: Optional[int] = None):
    mass = jnp.array(mass)
    beta = jnp.array(beta)
    if mass.ndim == 0:
      assert dim is not None
      mass = jnp.ones(dim)*mass
    else:
      dim = mass.shape[-1]

    if beta.ndim == 0:
      beta = jnp.ones(dim)*beta
    else:
      assert beta.shape[-1] == dim

    assert mass.shape ==  beta.shape

    gamma = jnp.sqrt(4*mass) # critical damping
    zero = jnp.zeros_like(gamma)
    top_left = DiagonalMatrix(zero, tags=TAGS.no_tags)
    top_right = DiagonalMatrix(beta/mass, tags=TAGS.no_tags)
    bottom_left = DiagonalMatrix(-beta, tags=TAGS.no_tags)
    bottom_right = DiagonalMatrix(-gamma*beta/mass, tags=TAGS.no_tags)

    self.F = Block2x2Matrix.from_blocks(top_left, top_right, bottom_left, bottom_right)

    elements = jnp.sqrt(2*gamma*beta)
    elements = jnp.pad(elements, (elements.shape[-1], 0))
    self.L = DiagonalMatrix(elements, tags=TAGS.no_tags)

class TOLD(AbstractLinearTimeInvariantSDE):
  """https://arxiv.org/pdf/2409.07697"""

  F: Block3x3Matrix
  L: DiagonalMatrix

  def __init__(self,
               L: Scalar = 1,
               *,
               dim: int):
    one = jnp.ones(dim)
    zero = jnp.zeros(dim)

    F_elements = jnp.array([[zero, one, zero],
                             [-one, zero, 2*jnp.sqrt(2)*one],
                             [zero, -2*jnp.sqrt(2)*one, -3*jnp.sqrt(3)*one]])
    def make_block(elements):
      return DiagonalMatrix(elements, tags=TAGS.no_tags)
    F = jax.vmap(jax.vmap(make_block))(F_elements)
    self.F = Block3x3Matrix(F, tags=TAGS.no_tags)

    L_elements = 3**(0.25)*jnp.sqrt(6/L)*one
    L_elements = jnp.pad(L_elements, (2*L_elements.shape[-1], 0))
    self.L = DiagonalMatrix(L_elements, tags=TAGS.no_tags)

class StochasticHarmonicOscillator(AbstractLinearTimeInvariantSDE):
  F: Annotated[Block2x2Matrix, DiagonalMatrix]
  L: DiagonalMatrix

  def __init__(self,
               freq: Union[Float[Array, 'dim'], Scalar], # Frequency of the oscillator
               coeff: Union[Float[Array, 'dim'], Scalar], # Damping coefficient
               sigma: Union[Float[Array, 'dim'], Scalar], # Noise level
               observation_dim: Optional[int] = None):
    freq = jnp.array(freq)
    coeff = jnp.array(coeff)
    sigma = jnp.array(sigma)
    if freq.ndim == 0:
      assert observation_dim is not None
      freq = jnp.ones(observation_dim)*freq
    else:
      observation_dim = freq.shape[-1]

    if coeff.ndim == 0:
      coeff = jnp.ones(observation_dim)*coeff
    else:
      assert coeff.shape[-1] == observation_dim

    if sigma.ndim == 0:
      sigma = jnp.ones(observation_dim)*sigma
    else:
      assert sigma.shape[-1] == observation_dim

    assert freq.shape ==  coeff.shape == sigma.shape

    zero = jnp.zeros_like(freq)
    one = jnp.ones_like(freq)

    F_elements = jnp.array([[zero, one],
                            [-freq**2, -coeff]])
    def make_block(elements):
      return DiagonalMatrix(elements, tags=TAGS.no_tags)
    F = jax.vmap(jax.vmap(make_block))(F_elements)
    self.F = Block2x2Matrix(F, tags=TAGS.no_tags)

    elements = sigma
    elements = jnp.pad(elements, (elements.shape[-1], 0))
    self.L = DiagonalMatrix(elements, tags=TAGS.no_tags)


if __name__ == '__main__':
  from debug import *
  from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
  from linsdex.series.series import TimeSeries

  key = random.PRNGKey(0)
  dim = 1
  freq = random.normal(key, (dim, dim))
  freq = freq.T@freq
  U, s, _ = jnp.linalg.svd(freq)
  freq_matrix = U@jnp.diag(s)@U.T

  zero = jnp.zeros_like(freq)
  one = jnp.ones_like(freq)

  F_elements = jnp.array([[zero, one],
                          [-freq_matrix, zero]])
  def make_block(elements):
    return DenseMatrix(elements, tags=TAGS.no_tags)
  F_mats = jax.vmap(jax.vmap(make_block))(F_elements)
  F = Block2x2Matrix(F_mats, tags=TAGS.no_tags)

  sigma = 0.1
  elements = sigma*jnp.ones(dim)
  elements = jnp.pad(elements, (elements.shape[-1], 0))
  L = DiagonalMatrix(elements, tags=TAGS.no_tags)
  sho = LinearTimeInvariantSDE(F, L)


  ts = jnp.linspace(0, 4, 4)
  xts = jnp.cos(ts)[:,None]
  series = TimeSeries(ts, xts)
  x_dim = series.values.shape[-1]*2
  y_dim = series.values.shape[-1]
  sigma = 0.1

  def process_single_elt(y, mask):
    y = jnp.where(mask, y, 0.0)
    mean = jnp.pad(y, (0, x_dim - y_dim))

    # We place 0 certainty in unobserved elements
    mask = jnp.pad(mask, (0, x_dim - y_dim))
    inverse_cov = jnp.pad(jnp.ones(y_dim)/sigma**2, (0, x_dim - y_dim))
    inverse_cov = jnp.where(mask, inverse_cov, 0.0)
    return mean, inverse_cov

  means, inverse_covs = jax.vmap(process_single_elt)(series.values, series.mask)
  latent_potentials = GaussianPotentialSeries(series.times, means, certainty=inverse_covs)
  cond_sde = sho.condition_on(latent_potentials)

  save_times = jnp.linspace(ts[0], ts[-1], 1000)
  samples = cond_sde.sample(key, save_times)
  import pdb; pdb.set_trace()