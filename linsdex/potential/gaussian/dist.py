import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, PyTree, Scalar, Bool
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
import warnings
import abc
import linsdex.util as util
from linsdex.matrix.tags import Tags, TAGS
from plum import dispatch
import jax.tree_util as jtu
from linsdex.potential.gaussian.config import USE_CHOLESKY_SAMPLING
from linsdex.linear_functional.functional_ops import vdot, zeros_like
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.quadratic_form import QuadraticForm

__all__ = [
    'NaturalGaussian',
    'NaturalJointGaussian',
    'StandardGaussian',
    'MixedGaussian',
    'GaussianStatistics',
    'gaussian_e_step',
    'gaussian_m_step'
]

################################################################################################################

class AbstractGaussianPotential(AbstractPotential):
  """Abstract base class for Gaussian potentials.

  This class provides a common interface for Gaussian potentials, which are
  used to represent Gaussian distributions in various forms.
  """

  @abc.abstractmethod
  def __call__(self, x: PyTree) -> Scalar:
    pass

  def __add__(self, other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    pass

  @abc.abstractmethod
  def normalizing_constant(self) -> Scalar:
    pass

  @abc.abstractmethod
  def log_prob(self, x: PyTree) -> Scalar:
    pass

  @abc.abstractmethod
  def sample(self, key: PRNGKeyArray) -> PyTree:
    pass

  @classmethod
  @abc.abstractmethod
  def total_certainty_like(cls, x: Float[Array, 'D'], other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    pass

  @classmethod
  @abc.abstractmethod
  def total_uncertainty_like(cls, other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    pass

  @abc.abstractmethod
  def sufficient_statistics(self, x: Float[Array, 'B D']) -> 'AbstractGaussianPotential':
    pass

  @auto_vmap
  def integrate(self):
    """Compute the value of \int exp{-0.5*x^T J x + x^T h - logZ} dx"""
    return self.normalizing_constant() - self.logZ

  @abc.abstractmethod
  def score(self, x: Float[Array, 'D']) -> Float[Array, 'D']:
    pass

  @abc.abstractmethod
  def get_noise(self, x: Float[Array, 'D']) -> Float[Array, 'D']:
    pass

################################################################################################################

class NaturalGaussian(AbstractGaussianPotential):
  """Gaussian distribution in natural parameter (information) form.

  Represents a Gaussian potential as exp{-0.5*x^T J x + x^T h - logZ}, where:
  - J is the precision matrix (inverse covariance)
  - h is the precision-adjusted mean (J*μ)
  - logZ is the log normalizing constant

  This parametrization is particularly useful for combining multiple Gaussian
  distributions, as the natural parameters add directly. It's also the form
  used in exponential families and variational inference.

  Attributes:
    J: Precision matrix (inverse covariance)
    h: Precision-adjusted mean vector
    logZ: Log normalizing constant
  """

  J: AbstractSquareMatrix
  h: Union[Float[Array, 'D'], LinearFunctional]
  logZ: Union[Scalar, QuadraticForm]

  def __init__(
    self,
    J: AbstractSquareMatrix,
    h: Union[Float[Array, 'D'], LinearFunctional],
    logZ: Optional[Union[Scalar, QuadraticForm]] = None
  ):
    """Initialize a Gaussian in natural parameter form.

    Args:
      J: Precision matrix (inverse covariance)
      h: Precision-adjusted mean vector
      logZ: Log normalizing constant (if None, computed automatically)

    Raises:
      AssertionError: If J is not a matrix or has incorrect dimensions
    """
    assert isinstance(J, AbstractSquareMatrix)
    assert J.ndim == 2
    # Check that J is positive definite and symmetric
    J = util.psd_check(J)
    # Ensure symmetry and add jitter for numerical stability
    J = 0.5*(J + J.T)
    self.J = J
    self.h = h

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  @property
  def dim(self) -> int:
    """Get the dimensionality of the Gaussian distribution.

    Returns:
      The number of dimensions of the distribution
    """
    return self.h.shape[-1]

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    """Get the batch dimensions of this Gaussian.

    Returns:
      Batch dimensions from the precision matrix J
    """
    return self.J.batch_size

  @classmethod
  def total_certainty_like(cls, x: Float[Array, 'D'], other: 'NaturalGaussian') -> 'NaturalGaussian':
    """Create a deterministic (Dirac delta) distribution at point x.

    Args:
      x: The point at which to place the Dirac delta
      other: Template Gaussian to match structure with

    Raises:
      AssertionError: Natural parametrization cannot represent Dirac deltas
    """
    assert 0, 'Cannot express a dirac distribution as a natural Gaussian'

  @classmethod
  def total_uncertainty_like(cls, other: 'NaturalGaussian') -> 'NaturalGaussian':
    """Create a completely uninformative (uniform) distribution.

    Creates a Gaussian with zero precision matrix, representing
    a uniform distribution over the space.

    Args:
      other: Template Gaussian to match structure with

    Returns:
      An uninformative Gaussian with the same structure as other
    """
    if other.batch_size:
      return eqx.filter_vmap(lambda o: cls.total_uncertainty_like(o))(other)

    out = super().zeros_like(other)
    return eqx.tree_at(lambda x: x.J.tags, out, TAGS.zero_tags)

  @classmethod
  def zeros_like(cls, other: 'NaturalGaussian') -> 'NaturalGaussian':
    return cls.total_uncertainty_like(other)

  @auto_vmap
  def sufficient_statistics(self, x: Float[Array, 'B D']):
    assert x.ndim == 2
    B, D = x.shape
    T1 = jnp.einsum('...i,...j->ij', x, x)/B
    t2 = jnp.mean(x, axis=0)
    T1 = DenseMatrix(T1, tags=TAGS.no_tags)
    return NaturalGaussian(T1, t2)

  def to_nat(self):
    return self

  @auto_vmap
  def to_std(self):
    Sigma = self.J.get_inverse()
    Sigma = Sigma._force_fix_tags()
    mu = Sigma@self.h
    return StandardGaussian(mu, Sigma, self.logZ)

  @auto_vmap
  def to_mixed(self):
    return MixedGaussian(self.J.solve(self.h), self.J, self.logZ)

  @auto_vmap
  def to_ess(self):
    return self.to_std().to_ess()

  @auto_vmap
  @dispatch
  def __add__(self, other: 'NaturalGaussian') -> 'NaturalGaussian':
    J_sum = self.J + other.J
    J_sum = J_sum.set_symmetric()
    h_sum = self.h + other.h
    logZ_sum = self.logZ + other.logZ
    return NaturalGaussian(J_sum, h_sum, logZ_sum)

  @auto_vmap
  @dispatch
  def __add__(self, other: 'StandardGaussian') -> 'NaturalGaussian':
    return self + other.to_nat()

  @auto_vmap
  @dispatch
  def __add__(self, other: 'MixedGaussian') -> 'NaturalGaussian':
    return self + other.to_nat()

  @auto_vmap
  def normalizing_constant(self):
    """Compute the log normalizing constant of the Gaussian potential.

    Calculates log(∫ exp{-0.5*x^T J x + x^T h} dx), which is the log partition
    function needed to normalize the potential into a proper probability distribution.

    This is different from the logZ attribute, which can be an arbitrary scalar
    used for bookkeeping or unnormalized representations.

    Returns:
      The log normalizing constant as a scalar
    """
    Jinv_h = self.J.solve(self.h)
    dim = self.h.shape[0]

    # Formula: logZ = 0.5*h^T J^{-1} h - 0.5*log|J| + 0.5*d*log(2π)
    logZ = 0.5*vdot(Jinv_h, self.h)
    logZ -= 0.5*self.J.get_log_det()
    logZ += 0.5*dim*jnp.log(2*jnp.pi)

    # Handle special case when J is zero
    return util.where(self.J.is_zero,
                    logZ*0.0,
                    logZ)

  @auto_vmap
  def __call__(self, x: Array):
    """Evaluate the unnormalized log density at point x.

    Computes -0.5*x^T J x + x^T h - logZ, which is the log of the
    unnormalized probability density at x.

    Args:
      x: Point at which to evaluate the log density

    Returns:
      Unnormalized log probability density at x
    """
    return -0.5*vdot(x, self.J@x) + vdot(self.h, x) - self.logZ

  @auto_vmap
  def log_prob(self, x: Array):
    """Evaluate the normalized log probability density at point x.

    Computes the log probability density of the normalized distribution at x.

    Args:
      x: Point at which to evaluate the log probability

    Returns:
      Log probability density at x
    """
    nc = self.normalizing_constant()
    return -0.5*vdot(x, self.J@x) + vdot(self.h, x) - nc

  @auto_vmap
  def score(self, x: Array) -> Array:
    """Score function of the Gaussian potential"""
    return self.h - self.J@x

  @auto_vmap
  def sample(self, key: PRNGKeyArray):
    """Sample from the Gaussian distribution.

    Generates a random sample from the Gaussian distribution using
    the reparameterization trick. First generates standard normal noise,
    then transforms it to match the target distribution's parameters.

    Args:
      key: JAX PRNG key for random number generation

    Returns:
      A random sample from the distribution with shape matching the dimensionality
    """
    eps = random.normal(key, self.h.shape)
    return self._sample(eps)

  if USE_CHOLESKY_SAMPLING:

    @auto_vmap
    def _sample(self, eps: Float[Array, 'D']) -> Float[Array, 'D']:
      J = self.J# + self.J.eye(self.J.shape[0])*1e-8
      L_chol = J.get_cholesky()
      out = J.solve(L_chol@eps + self.h)
      return out

    @auto_vmap
    def get_noise(self, x: Float[Array, 'D']):
      L_chol = self.J.get_cholesky()
      return L_chol.solve(self.J@x - self.h)

  else:

    @auto_vmap
    def _sample(self, eps: Float[Array, 'D']):
      U, S, V = self.J.get_svd()
      S_chol = S.get_cholesky()
      S_chol_inv = S_chol.get_inverse()
      out = U@S_chol_inv@(eps + S_chol_inv@U.T@self.h)
      return out

    @auto_vmap
    def get_noise(self, x: Float[Array, 'D']):
      U, S, V = self.J.get_svd()
      S_chol = S.get_cholesky()
      return S_chol@U.T@x - S_chol.solve(U.T@self.h)

  @auto_vmap
  def to_joint(self, *, dim: int) -> 'NaturalJointGaussian':
    J = self.J.as_matrix()
    J11 = DenseMatrix(J[:dim, :dim], tags=TAGS.no_tags)
    J12 = DenseMatrix(J[:dim, dim:], tags=TAGS.no_tags)
    J22 = DenseMatrix(J[dim:, dim:], tags=TAGS.no_tags)
    if isinstance(self.h, LinearFunctional):
      raise NotImplementedError("Cannot convert to joint form when h is a LinearFunctional.")
    h1 = self.h[:dim]
    h2 = self.h[dim:]
    return NaturalJointGaussian(J11, J12, J22, h1, h2, self.logZ)

  def make_deterministic(self) -> 'MixedGaussian':
    new_J = self.J.set_inf()
    return NaturalGaussian(new_J, self.h, self.logZ)

################################################################################################################

class NaturalJointGaussian(NaturalGaussian):
  """Represents a joint Gaussian distribution over x and y.
  exp{-0.5*y^T J11 y + y^T J12 x + 0.5*x^T J22 x - h1^T y - h2^T x - logZ}
  This is here as a ground truth for testing transition"""

  J11: AbstractSquareMatrix
  J12: AbstractSquareMatrix
  J22: AbstractSquareMatrix
  h1: Float[Array, 'D']
  h2: Float[Array, 'D']
  logZ: Scalar

  def __init__(
    self,
    J11: AbstractSquareMatrix,
    J12: AbstractSquareMatrix,
    J22: AbstractSquareMatrix,
    h1: Float[Array, 'D'],
    h2: Float[Array, 'D'],
    logZ: Optional[Scalar] = None
  ):
    assert isinstance(J11, AbstractSquareMatrix)
    assert isinstance(J12, AbstractSquareMatrix)
    assert isinstance(J22, AbstractSquareMatrix)
    self.J11 = J11
    self.J12 = J12
    self.J22 = J22
    self.h1 = h1
    self.h2 = h2

    if logZ is None:
      logZ = NaturalGaussian(self.J, self.h).normalizing_constant()
    self.logZ = logZ

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.J11.batch_size

  @property
  @auto_vmap
  def J21(self) -> AbstractSquareMatrix:
    return self.J12.T

  @property
  @auto_vmap
  def J(self) -> AbstractSquareMatrix:
    matrix = jnp.block([[self.J11.as_matrix(), self.J12.as_matrix()],
                        [self.J12.as_matrix().T, self.J22.as_matrix()]])
    return DenseMatrix(matrix, tags=TAGS.no_tags)

  @property
  @auto_vmap
  def h(self) -> Float[Array, 'D']:
    return jnp.concatenate([self.h1, self.h2])

  @auto_vmap
  def to_std(self):
    Sigma = self.J.get_inverse()
    mu = Sigma@self.h
    return StandardGaussian(mu, Sigma, self.logZ)

  @auto_vmap
  def __add__(self, other: 'NaturalGaussian') -> 'NaturalGaussian':
    J11 = self.J11 + other.J11
    J12 = self.J12 + other.J12
    J22 = self.J22 + other.J22
    J11 = J11.set_symmetric()
    J22 = J22.set_symmetric()
    h1 = self.h1 + other.h1
    h2 = self.h2 + other.h2
    logZ = self.logZ + other.logZ
    return NaturalJointGaussian(J11, J12, J22, h1, h2, logZ)

  @classmethod
  def from_joint_potential(cls, joint: 'JointPotential') -> 'NaturalJointGaussian':
    if joint.batch_size:
      return eqx.filter_vmap(lambda j: cls.from_joint_potential(j))(joint)

    transition = joint.transition
    transition_nat = transition.to_nat()
    J11, J12, J22 = transition_nat.J11, transition_nat.J12, transition_nat.J22
    h1, h2 = transition_nat.h1, transition_nat.h2
    logZ1 = transition.logZ

    prior_nat = joint.prior.to_nat()
    J, h, logZ2 = prior_nat.J, prior_nat.h, prior_nat.logZ
    return NaturalJointGaussian(J11, J12, J22 + J, h1, h2 + h, logZ1 + logZ2)

  @auto_vmap
  def to_block(self) -> 'NaturalGaussian':
    J_mat = jnp.block([[self.J11.as_matrix(), self.J12.as_matrix()],
                       [self.J12.as_matrix().T, self.J22.as_matrix()]])
    J = DenseMatrix(J_mat, tags=TAGS.no_tags)
    h = jnp.concatenate([self.h1, self.h2])
    logZ = self.logZ
    return NaturalGaussian(J, h, logZ)

  @auto_vmap
  def swap_variables(self) -> 'NaturalJointGaussian':
    """Swap the order of the variables"""
    return NaturalJointGaussian(self.J22,
                                self.J21,
                                self.J11,
                                self.h2,
                                self.h1,
                                self.logZ)

  @auto_vmap
  def update_y(self, potential: NaturalGaussian) -> 'NaturalJointGaussian':
    J11 = self.J11 + potential.J
    J12 = self.J12
    J22 = self.J22
    J11 = J11.set_symmetric()
    h1 = self.h1 + potential.h
    h2 = self.h2
    logZ = self.logZ + potential.logZ
    return NaturalJointGaussian(J11, J12, J22, h1, h2, logZ)

  @auto_vmap
  def marginalize_out_y(self) -> 'NaturalGaussian':
    """p(y, x) -> p(x)"""
    J21_J11inv = self.J11.solve(self.J12).T

    Jy = self.J22 - J21_J11inv@self.J12
    Jy = Jy.set_symmetric()
    hy = self.h2 - J21_J11inv@self.h1

    # Compute the new logZ
    logZ_offset = NaturalGaussian(self.J11, self.h1).normalizing_constant()
    return NaturalGaussian(Jy, hy, self.logZ - logZ_offset)

  @auto_vmap
  def condition_on_y(self, y: Float[Array, 'Dy']) -> 'NaturalJointGaussian':
    """p(y, x) -> p(x | y)"""
    Jy = self.J22
    hy = self.h2 - self.J21@y
    logZy = self.logZ + 0.5*vdot(y, self.J11@y) - vdot(y, self.h1)
    return NaturalGaussian(Jy, hy, logZy)

  @auto_vmap
  def update_x(self, potential: NaturalGaussian) -> 'NaturalJointGaussian':
    return self.swap_variables().update_y(potential).swap_variables()

  @auto_vmap
  def marginalize_out_x(self) -> 'NaturalGaussian':
    return self.swap_variables().marginalize_out_y()

  @auto_vmap
  def condition_on_x(self, y: Float[Array, 'Dy']) -> 'NaturalJointGaussian':
    return self.swap_variables().condition_on_y(y)

  @auto_vmap
  def update_and_marginalize_out_x(self, potential: NaturalGaussian) -> 'NaturalGaussian':
    return self.update_x(potential).marginalize_out_x()

  @auto_vmap
  def update_and_marginalize_out_y(self, potential: NaturalGaussian) -> 'NaturalGaussian':
    return self.update_y(potential).marginalize_out_y()

  @auto_vmap
  def chain(self, other: 'AbstractPotential') -> 'AbstractPotential':
    """Combine two transitions into a single transition.

    If self is p(x|y) and other is p(y|z) then this returns p(x|z) = \int p(x|y)p(y|z) dy

    # We want to integrate out the middle row and column of the joint
    joint_J = [[self.J11,      self.J12       ,      zero],
               [self.J21, self.J22 + other.J11, other.J12],
               [zero    ,      other.J21      , other.J22]]
    joint_h = [self.h1, self.h2 + other.h1, other.h2]

    # Do this by rearranging the rows and columns to use regular marginalization
    joint_J = [[self.J22 + other.J11,      self.J21, other.J12],
               [self.J12            ,      self.J11,      zero],
               [other.J21           ,          zero, other.J22]]
    joint_h = [self.h2 + other.h1, self.h1, other.h2]
    """
    zero = jnp.zeros((self.J11.shape[0], other.J22.shape[1]))
    joint_J11 = self.J22 + other.J11
    joint_J11 = joint_J11.set_symmetric()
    joint_J12 = jnp.block([self.J21.as_matrix(), other.J12.as_matrix()])
    joint_J22 = jnp.block([[self.J11.as_matrix(), zero],
                          [zero.T, other.J22.as_matrix()]])
    joint_J12 = DenseMatrix(joint_J12, tags=TAGS.no_tags)
    joint_J22 = DenseMatrix(joint_J22, tags=TAGS.no_tags)
    joint_J22 = joint_J22.set_symmetric()
    joint_h1 = self.h2 + other.h1
    joint_h2 = jnp.concatenate([self.h1, other.h2])

    joint_logZ = self.logZ + other.logZ
    joint = NaturalJointGaussian(joint_J11, joint_J12, joint_J22, joint_h1, joint_h2, joint_logZ)
    return joint.marginalize_out_y().to_joint(dim=self.J11.shape[0])

  @auto_vmap
  def sample(self, key: PRNGKeyArray):
    return NaturalGaussian(self.J, self.h).sample(key)

  @classmethod
  def total_certainty_like(cls, x: Float[Array, 'D'], other: 'AbstractPotential') -> 'AbstractPotential':
    raise NotImplementedError

  @classmethod
  def total_uncertainty_like(cls, other: 'AbstractPotential') -> 'AbstractPotential':
    raise NotImplementedError

################################################################################################################

class StandardGaussian(AbstractGaussianPotential):
  """Gaussian distribution in standard (mean and covariance) form.

  Represents a Gaussian distribution N(μ, Σ) with mean vector μ and
  covariance matrix Σ. The density function is:
  exp{-0.5*(x-μ)^T Σ^{-1} (x-μ) - logZ}

  This is the most common and intuitive parametrization of Gaussian distributions,
  particularly useful for sampling and when working with observed data.

  Attributes:
    mu: Mean vector
    Sigma: Covariance matrix
    logZ: Log normalizing constant
  """

  mu: Union[Float[Array, 'D'], LinearFunctional]
  Sigma: AbstractSquareMatrix
  logZ: Union[Scalar, QuadraticForm]

  def __init__(
    self,
    mu: Union[Float[Array, 'D'], LinearFunctional],
    Sigma: AbstractSquareMatrix,
    logZ: Optional[Union[Scalar, QuadraticForm]] = None
  ):
    """Initialize a Gaussian in standard (mean and covariance) form.

    Args:
      mu: Mean vector
      Sigma: Covariance matrix
      logZ: Log normalizing constant (if None, computed automatically)

    Raises:
      AssertionError: If Sigma is not a matrix or has incorrect dimensions
    """
    assert isinstance(Sigma, AbstractSquareMatrix)
    assert Sigma.ndim == 2
    Sigma = util.psd_check(Sigma)
    # Ensure symmetry and add jitter for numerical stability
    self.Sigma = 0.5*(Sigma + Sigma.T)
    self.mu = mu

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  @property
  def dim(self) -> int:
    return self.mu.shape[-1]

  @classmethod
  def total_certainty_like(self, x: Float[Array, 'D'], other: 'StandardGaussian') -> 'StandardGaussian':
    if other.batch_size:
      return eqx.filter_vmap(lambda _x, o: self.total_certainty_like(_x, o))(x, other)

    out = super().zeros_like(other)
    out = eqx.tree_at(lambda x: x.Sigma.tags, out, TAGS.zero_tags)
    out = eqx.tree_at(lambda x: x.mu, out, x)
    return out

  @classmethod
  def total_uncertainty_like(cls, other: 'StandardGaussian') -> 'StandardGaussian':
    if other.batch_size:
      return eqx.filter_vmap(lambda o: cls.total_uncertainty_like(o))(other)

    out = super().zeros_like(other)
    return eqx.tree_at(lambda x: x.Sigma.tags, out, TAGS.inf_tags)

  @classmethod
  def zeros_like(cls, other: 'StandardGaussian') -> 'StandardGaussian':
    return cls.total_uncertainty_like(other)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.Sigma.batch_size

  @auto_vmap
  def sufficient_statistics(self, x: Float[Array, 'B D']):
    return self.to_nat().sufficient_statistics(x)

  @auto_vmap
  def to_nat(self):
    """Convert to natural parameter form (precision form).

    Transforms from standard parameterization N(μ, Σ) to natural parameterization
    with precision matrix J = Σ^{-1} and h = J*μ.

    Returns:
      Equivalent Gaussian in natural parameter form
    """
    J = self.Sigma.get_inverse()
    J = J._force_fix_tags()
    h = J@self.mu
    return NaturalGaussian(J, h, self.logZ)

  def to_std(self):
    """Convert to standard form (identity operation).

    Returns:
      Self, as this is already in standard form
    """
    return self

  @auto_vmap
  def to_mixed(self):
    """Convert to mixed parameter form.

    Transforms from standard parameterization N(μ, Σ) to mixed parameterization
    with mean μ and precision matrix J = Σ^{-1}.

    Returns:
      Equivalent Gaussian in mixed parameter form
    """
    J = self.Sigma.get_inverse()
    return MixedGaussian(self.mu, J, self.logZ)

  @auto_vmap
  def to_ess(self):
    mumuT = jnp.einsum('...i,...j->ij', self.mu, self.mu)
    Xi = self.Sigma.as_matrix() + mumuT
    return GaussianStatistics(self.mu, Xi)
    # return ExpectedSufficientStatisticsGaussian(self.mu, Xi, self.logZ)

  def cast(self, other: 'StandardGaussian'):
    mu = self.mu + zeros_like(other.mu) # In case either is a linear functional
    cov = self.Sigma + other.Sigma.zeros_like(other.Sigma) # Correct type for covariance
    logZ = self.logZ + zeros_like(other.logZ) # In case either is a quadratic form
    return StandardGaussian(mu, cov, logZ)

  @auto_vmap
  @dispatch
  def __add__(self, other: 'StandardGaussian') -> 'StandardGaussian':
    """Combine two Gaussian distributions.

    Implements a numerically stable product of Gaussians in standard form.
    This is equivalent to a Kalman filter update step and produces a new
    Gaussian that represents the product of the two input distributions.

    Args:
      other: Another Gaussian in standard form to combine with this one

    Returns:
      A new Gaussian representing the product of the distributions
    """
    mu, Sigma = self.mu, self.Sigma
    mux, Sigmax = other.mu, other.Sigma

    # This determines the output type of the covariance
    Sigma_plus_Sigmax = Sigma + Sigmax
    # Compute the Kalman gain: Sigmax@(Sigma + Sigmax)^{-1}
    S = Sigma_plus_Sigmax.T.solve(Sigmax.T).T
    # Compute the new covariance
    P = S@Sigma
    P = P.set_symmetric()
    # Compute the new mean
    m = S@mu + Sigma@(Sigma_plus_Sigmax.solve(mux))
    logZ = self.logZ + other.logZ
    out = StandardGaussian(m, P, logZ)

    # Handle special cases: infinite covariance (uninformative distribution)
    out = util.where(Sigma.is_inf, other.cast(out), out)
    out = util.where(Sigmax.is_inf, self.cast(out), out)

    # Need to add the logZs together
    out = eqx.tree_at(lambda x: x.logZ, out, self.logZ + other.logZ)
    return out

  @auto_vmap
  @dispatch
  def __add__(self, other: NaturalGaussian) -> 'StandardGaussian':
    """Combine with a Gaussian in natural form.

    Args:
      other: A Gaussian in natural form to combine with this one

    Returns:
      A new Gaussian representing the product of the distributions
    """
    return self + other.to_std()

  @auto_vmap
  @dispatch
  def __add__(self, other: 'MixedGaussian') -> 'StandardGaussian':
    """Combine with a Gaussian in mixed form.

    Args:
      other: A Gaussian in mixed form to combine with this one

    Returns:
      A new Gaussian representing the product of the distributions
    """
    return self + other.to_std()

  @auto_vmap
  def normalizing_constant(self):
    """Compute the normalizing constant, which is
    \int exp{-0.5*x^T Sigma^{-1} x + x^T Sigma^{-1}mu} dx. This is different
    than logZ which can be an arbitrary scalar."""
    covinv_mu = self.Sigma.solve(self.mu)
    dim = self.mu.shape[-1]
    logZ = 0.5*vdot(covinv_mu, self.mu)
    logZ += 0.5*self.Sigma.get_log_det()
    logZ += 0.5*dim*jnp.log(2*jnp.pi)

    return util.where(self.Sigma.is_inf|self.Sigma.is_zero, logZ * 0.0, logZ)

  @auto_vmap
  def __call__(self, x: Array):
    Sigma_inv_x = self.Sigma.solve(x)
    return -0.5*vdot(x, Sigma_inv_x) + vdot(self.mu, Sigma_inv_x) - self.logZ

  @auto_vmap
  def log_prob(self, x: Float[Array, 'D']):
    """Calculate the log probability density of a point.

    Computes the log probability density of x under the N(μ, Σ) distribution.

    Args:
      x: Point to evaluate

    Returns:
      Log probability density at x
    """
    nc = self.normalizing_constant()
    Sigma_inv_x = self.Sigma.solve(x)
    return -0.5*vdot(x, Sigma_inv_x) + vdot(self.mu, Sigma_inv_x) - nc

  @auto_vmap
  def score(self, x: Array) -> Array:
    """Compute the score function (gradient of log density).

    The score function is ∇_x log p(x) = Σ^{-1}(μ - x)

    Args:
      x: Point at which to evaluate the score

    Returns:
      Gradient of the log density at x
    """
    return self.Sigma.solve(self.mu - x)

  @auto_vmap
  def sample(self, key: PRNGKeyArray):
    """Generate a random sample from the Gaussian distribution.

    Uses the reparameterization trick to generate samples efficiently.
    First draws a standard normal sample, then transforms it to have
    the correct mean and covariance.

    Args:
      key: JAX PRNG key for random number generation

    Returns:
      A random sample from the N(μ, Σ) distribution
    """
    eps = random.normal(key, self.mu.shape)
    return self._sample(eps)


  if USE_CHOLESKY_SAMPLING:
    @auto_vmap
    def _sample(self, eps: Float[Array, 'D']):
      L = self.Sigma.get_cholesky()
      out = self.mu + L@eps
      out = util.where(self.Sigma.is_zero, self.mu, out)
      return out

    @auto_vmap
    def get_noise(self, x: Float[Array, 'D']):
      L = self.Sigma.get_cholesky()
      eps = L.solve(x - self.mu)
      return eps

  else:

    @auto_vmap
    def _sample(self, eps: Float[Array, 'D']):
      U, Sinv, V = self.Sigma.get_svd()
      out = U@Sinv.get_cholesky()@eps + self.mu
      out = util.where(self.Sigma.is_zero, self.mu, out)
      return out

    @auto_vmap
    def get_noise(self, x: Float[Array, 'D']):
      U, Sinv, V = self.Sigma.get_svd()
      S_chol = Sinv.get_cholesky().get_inverse()
      return S_chol@U.T@(x - self.mu)

  def make_deterministic(self) -> 'StandardGaussian':
    new_Sigma = self.Sigma.zeros_like(self.Sigma)
    return StandardGaussian(self.mu, new_Sigma, self.logZ)

################################################################################################################

class MixedGaussian(AbstractGaussianPotential):
  """Gaussian distribution in mixed parameter form.

  Represents a Gaussian distribution with mean vector μ and precision matrix J
  (inverse covariance). The density function is:
  exp{-0.5*(x-μ)^T J (x-μ) - logZ} = exp{-0.5*x^T J x + x^T Jμ - 0.5*μ^T J μ - logZ}

  This parameterization combines aspects of both standard and natural forms,
  maintaining the intuitive mean parameter while using the precision matrix
  for certain computations that are more efficient in that form.

  Attributes:
    mu: Mean vector
    J: Precision matrix (inverse covariance)
    logZ: Log normalizing constant
  """

  mu: Union[Float[Array, 'D'], LinearFunctional]
  J: AbstractSquareMatrix
  logZ: Union[Scalar, QuadraticForm]

  def __init__(
    self,
    mu: Union[Float[Array, 'D'], LinearFunctional],
    J: AbstractSquareMatrix,
    logZ: Optional[Union[Scalar, QuadraticForm]] = None
  ):
    """Initialize a Gaussian in mixed parameter form.

    Args:
      mu: Mean vector
      J: Precision matrix (inverse covariance)
      logZ: Log normalizing constant (if None, computed automatically)

    Raises:
      AssertionError: If J is not a matrix or has incorrect dimensions
    """
    assert isinstance(J, AbstractSquareMatrix)
    assert J.ndim == 2
    J = util.psd_check(J)
    # Ensure symmetry and add jitter for numerical stability
    self.J = 0.5*(J + J.T)
    self.mu = mu

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  @property
  def dim(self) -> int:
    return self.mu.shape[-1]

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.J.batch_size

  @classmethod
  def construct_deterministic_potential(self, x: Float[Array, 'D']) -> 'MixedGaussian':
    J = DiagonalMatrix(jnp.zeros_like(x), tags=TAGS.inf_tags)
    return MixedGaussian(x, J)

  @classmethod
  def total_certainty_like(cls, x: Float[Array, 'D'], other: 'MixedGaussian') -> 'MixedGaussian':
    if other.batch_size:
      return eqx.filter_vmap(lambda _x, o: cls.total_certainty_like(_x, o))(x, other)

    out = super().zeros_like(other)
    out = eqx.tree_at(lambda x: x.J.tags, out, TAGS.inf_tags)
    out = eqx.tree_at(lambda x: x.mu, out, x)
    return out

  @classmethod
  def total_uncertainty_like(cls, other: 'MixedGaussian') -> 'MixedGaussian':
    if other.batch_size:
      return eqx.filter_vmap(lambda o: cls.total_uncertainty_like(o))(other)

    out = super().zeros_like(other)
    return eqx.tree_at(lambda x: x.J.tags, out, TAGS.zero_tags)

  @classmethod
  def zeros_like(cls, other: 'MixedGaussian') -> 'MixedGaussian':
    return cls.total_uncertainty_like(other)

  @auto_vmap
  def sufficient_statistics(self, x: Float[Array, 'B D']):
    return self.to_nat().sufficient_statistics(x)

  @auto_vmap
  def to_nat(self):
    """Convert to natural parameter form.

    Transforms from mixed parameterization (μ, J) to natural parameterization
    with precision matrix J and h = J*μ.

    Returns:
      Equivalent Gaussian in natural parameter form
    """
    h = self.J@self.mu
    return NaturalGaussian(self.J, h, self.logZ)

  @auto_vmap
  def to_std(self):
    """Convert to standard parameter form.

    Transforms from mixed parameterization (μ, J) to standard parameterization
    with mean μ and covariance matrix Σ = J^{-1}.

    Returns:
      Equivalent Gaussian in standard parameter form
    """
    Sigma = self.J.get_inverse()
    Sigma = Sigma._force_fix_tags()
    return StandardGaussian(self.mu, Sigma, self.logZ)

  def to_mixed(self):
    """Convert to mixed form (identity operation).

    Returns:
      Self, as this is already in mixed form
    """
    return self

  def cast(self, other: 'MixedGaussian'):
    mu = self.mu + zeros_like(other.mu) # In case either is a linear functional
    J = self.J + other.J.zeros_like(other.J) # Correct type for covariance
    logZ = self.logZ + zeros_like(other.logZ) # In case either is a quadratic form
    return MixedGaussian(mu, J, logZ)

  @auto_vmap
  @dispatch
  def __add__(self, other: 'MixedGaussian') -> 'MixedGaussian':
    """Numerically stable update for standard gaussians.  This is what is used
    in a Kalman filter update."""
    mu, J = self.mu, self.J
    mux, Jx = other.mu, other.J

    Jbar = J + Jx
    mubar = Jbar.solve(J@mu) + Jbar.solve(Jx@mux)
    logZ = self.logZ + other.logZ
    out = MixedGaussian(mubar, Jbar, logZ)

    out = util.where(J.is_zero|Jx.is_inf, other.cast(out), out)
    out = util.where(Jx.is_zero|J.is_inf, self.cast(out), out)

    out = eqx.tree_at(lambda x: x.logZ, out, self.logZ + other.logZ)
    return out

  @auto_vmap
  @dispatch
  def __add__(self, other: 'StandardGaussian') -> 'MixedGaussian':
    return self + other.to_mixed()

  @auto_vmap
  @dispatch
  def __add__(self, other: 'NaturalGaussian') -> 'MixedGaussian':
    return self + other.to_mixed()

  @auto_vmap
  def normalizing_constant(self):
    """Compute the normalizing constant, which is
    \int exp{-0.5*x^T Sigma^{-1} x + x^T Sigma^{-1}mu} dx. This is different
    than logZ which can be an arbitrary scalar."""
    Jmu = self.J@self.mu
    dim = self.mu.shape[-1]
    logZ = 0.5*vdot(Jmu, self.mu)
    logZ -= 0.5*self.J.get_log_det()
    logZ += 0.5*dim*jnp.log(2*jnp.pi)

    return util.where(self.J.is_inf|self.J.is_zero, logZ * 0.0, logZ)

  @auto_vmap
  def __call__(self, x: Array):
    Jx = self.J@x
    return -0.5*vdot(x, Jx) + vdot(self.mu, Jx) - self.logZ

  @auto_vmap
  def log_prob(self, x: Float[Array, 'D']):
    nc = self.normalizing_constant()
    Jx = self.J@x
    return -0.5*vdot(x, Jx) + vdot(self.mu, Jx) - nc

  @auto_vmap
  def score(self, x: Array) -> Array:
    """Score function of the Gaussian potential"""
    return self.J@(self.mu - x)

  @auto_vmap
  def sample(self, key: PRNGKeyArray):
    eps = random.normal(key, self.mu.shape)
    return self._sample(eps)

  if USE_CHOLESKY_SAMPLING:

    @auto_vmap
    def _sample(self, eps: Float[Array, 'D']):
      J = self.J# + self.J.eye(self.J.shape[0])*1e-6
      L_chol = J.get_cholesky()
      out = J.solve(L_chol@eps) + self.mu
      out = util.where(self.J.is_inf, self.mu, out)
      return out

    @auto_vmap
    def get_noise(self, x: Float[Array, 'D']):
      L_chol = self.J.get_cholesky()
      return L_chol.solve(self.J@x) - L_chol.T@self.mu

  else:

    @auto_vmap
    def _sample(self, eps: Float[Array, 'D']):
      U, S, V = self.J.get_svd()
      out = U@S.get_cholesky().solve(eps) + self.mu
      out = util.where(self.J.is_inf, self.mu, out)
      return out

    @auto_vmap
    def get_noise(self, x: Float[Array, 'D']):
      U, S, V = self.J.get_svd()
      S_chol = S.get_cholesky()
      return S_chol@U.T@(x - self.mu)

  def make_deterministic(self) -> 'MixedGaussian':
    new_J = self.J.set_inf()
    return MixedGaussian(self.mu, new_J, self.logZ)

################################################################################################################

class GaussianStatistics(AbstractBatchableObject):
  Ex: Float[Array, 'D']
  ExxT: Float[Array, 'D D']

  @property
  def batch_size(self):
    if self.Ex.ndim > 2:
      return self.Ex.shape[:-1]
    elif self.Ex.ndim == 2:
      return self.Ex.shape[0]
    elif self.Ex.ndim == 1:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.Ex.ndim}")

  def __add__(self, other: 'GaussianStatistics') -> 'GaussianStatistics':
    return GaussianStatistics(self.Ex + other.Ex, self.ExxT + other.ExxT)

  def to_std(self):
    # This is the m-step for the standard gaussian
    cov = self.ExxT - jnp.outer(self.Ex, self.Ex)
    return StandardGaussian(self.Ex, DenseMatrix(cov, tags=TAGS.no_tags))

  def to_nat(self):
    return self.to_std().to_nat()

def gaussian_e_step(dist: AbstractPotential) -> GaussianStatistics:
  return dist.to_ess()

def gaussian_m_step(stats: GaussianStatistics) -> AbstractPotential:
  return stats.to_std()

################################################################################################################

def check_distribution(dist: AbstractPotential):
  from linsdex.util.misc import empirical_dist, w2_distance
  # Sample from the gaussian and then check that the empirical distribution
  # is close to the true distribution
  key = random.PRNGKey(0)
  keys = random.split(key, 50000)
  dist.sample(key)
  samples = jax.vmap(dist.sample)(keys)
  empirical_dist = empirical_dist(samples)
  assert w2_distance(dist, empirical_dist) < 1e-3

  # Check that the normalizing constant is correct
  try:
    dist = dist.to_nat()
  except:
    pass
  def get_normalizing_constant(dist):
    return dist.normalizing_constant()
  dist_grad = eqx.filter_grad(get_normalizing_constant)(dist)
  dist_grad = eqx.tree_at(lambda x: x.J.tags, dist_grad, dist.J.tags)
  dist_grad = eqx.tree_at(lambda x: x.J, dist_grad, -2*dist_grad.J)
  comp = dist.sufficient_statistics(samples)
  assert w2_distance(dist_grad, comp) < 1e-3

  print('Distribution check passed')

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *
  jax.config.update('jax_enable_x64', True)

  ############################################
  # Joint natural gaussian tests

  dim = 4
  key = random.PRNGKey(0)

  def make_matrix(key, kind='nat', matrix='dense'):
    k1, k2, k3 = random.split(key, 3)
    mat = random.normal(k1, (dim, dim))
    mat = mat.T@mat
    J = DenseMatrix(mat, tags=TAGS.no_tags)
    h = random.normal(k2, (dim,))
    logZ = random.normal(k3, ())
    out = NaturalGaussian(J, h, logZ)
    if kind == 'nat':
      return out
    elif kind == 'std':
      return out.to_std()
    elif kind == 'mixture':
      std = out.to_std()
      return MixedGaussian(std.mu, out.J, out.logZ)

  k1, k2 = random.split(key, 2)
  dist1 = make_matrix(k1, kind='std', matrix='dense')
  dist2 = make_matrix(k2, kind='std', matrix='dense')
  dist_sum = dist1 + dist2

  nat1 = make_matrix(k1, kind='nat', matrix='dense')
  nat2 = make_matrix(k2, kind='nat', matrix='dense')
  nat_sum = (nat1 + nat2).to_std()

  mix1 = make_matrix(k1, kind='mixture', matrix='dense')
  mix2 = make_matrix(k2, kind='mixture', matrix='dense')
  mix_sum = (mix1 + mix2).to_std()

  # Check that the m-step is correct
  stats1 = gaussian_e_step(nat1)
  nat_m_step = gaussian_m_step(stats1)

  def loss(dist):
    keys = random.split(key, 10000000)
    samples = jax.vmap(nat1.sample)(keys)
    return -jax.vmap(dist.log_prob)(samples).mean()

  grad = eqx.filter_grad(loss)(nat_m_step)
  import pdb; pdb.set_trace()

  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_sum.Sigma, nat_sum.Sigma))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_sum.mu, nat_sum.mu))
  assert jnp.allclose(dist_sum.logZ, nat_sum.logZ)

  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_sum.Sigma, mix_sum.Sigma))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_sum.mu, mix_sum.mu))
  assert jnp.allclose(dist_sum.logZ, mix_sum.logZ)

  dist_std = make_matrix(key, kind='std', matrix='dense')
  dist_nat = make_matrix(key, kind='nat', matrix='dense')
  dist_mix = make_matrix(key, kind='mixture', matrix='dense')

  # Check that we can evaluate the log prob of a sample and have stable gradients
  x = dist_std.sample(key)
  def get_log_prob(dist, x):
    return dist.log_prob(x)
  log_prob_grad_std = eqx.filter_grad(get_log_prob)(dist_std, x)
  log_prob_grad_nat = eqx.filter_grad(get_log_prob)(dist_nat, x)
  log_prob_grad_mix = eqx.filter_grad(get_log_prob)(dist_mix, x)

  # import pdb; pdb.set_trace()

  # Check that the distributions agree
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_std.to_nat().J, dist_nat.J))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_std.to_nat().h, dist_nat.h))
  assert jnp.allclose(dist_std.to_nat().logZ, dist_nat.logZ)

  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_mix.to_nat().J, dist_nat.J))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, dist_mix.to_nat().h, dist_nat.h))
  assert jnp.allclose(dist_mix.to_nat().logZ, dist_nat.logZ)

  # Check that adding a zero potential doesn't change the distribution
  zero = dist_std.total_uncertainty_like(dist_std)
  sum_std = dist_std + zero
  sum_nat = dist_nat + zero
  sum_mix = dist_mix + zero
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std, dist_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, sum_nat, dist_nat))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, sum_mix, dist_mix))

  # Try a vectorized version
  zero = dist_mix.total_uncertainty_like(dist_mix)
  def make_zero(i):
    return dist_mix.total_uncertainty_like(dist_mix)

  zeros = eqx.filter_vmap(make_zero)(jnp.arange(3))
  node_potentials = jtu.tree_map(lambda xs, x: xs.at[0].set(x), zeros, dist_mix)
  check1 = node_potentials[0] + zeros[0]
  check2 = node_potentials[1] + zeros[1]
  check3 = node_potentials[2] + zeros[2]
  check = node_potentials + zeros


  # # Check that adding a deterministic potential leaves us with a deterministic distribution
  # zero = dist_std.total_uncertainty_like(dist_std).make_deterministic() # This is a deterministic potential at x=0
  # sum_std = dist_std + zero
  # sum_nat = dist_nat + zero
  # sum_mix = dist_mix + zero
  # assert jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std, dist_std))
  # assert jtu.tree_all(jtu.tree_map(jnp.allclose, sum_nat, dist_nat))
  # assert jtu.tree_all(jtu.tree_map(jnp.allclose, sum_mix, dist_mix))

  check_distribution(dist_std)
  check_distribution(dist_nat)
  check_distribution(dist_mix)

  import pdb; pdb.set_trace()


