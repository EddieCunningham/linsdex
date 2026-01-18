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
from linsdex.potential.gaussian.dist import AbstractGaussianPotential, MixedGaussian, NaturalGaussian, StandardGaussian, NaturalJointGaussian, GaussianStatistics
from plum import dispatch
import linsdex.util as util
from linsdex.matrix.tags import Tags, TAGS
from linsdex.util.parallel_scan import parallel_scan
import jax.tree_util as jtu
from linsdex.potential.gaussian.config import USE_CHOLESKY_SAMPLING
from linsdex.linear_functional.functional_ops import vdot, zeros_like
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.quadratic_form import QuadraticForm

__all__ = ['GaussianTransition',
           'max_likelihood_gaussian_transition',
           'GaussianJointStatistics',
           'gaussian_joint_e_step',
           'gaussian_joint_m_step']

class GaussianTransition(AbstractTransition):

  A: AbstractSquareMatrix
  u: Union[Float[Array, 'D'], LinearFunctional]
  Sigma: AbstractSquareMatrix
  logZ: Union[Float[Array, ''], QuadraticForm]

  def __init__(self,
    A: AbstractSquareMatrix,
    u: Union[Float[Array, 'D'], LinearFunctional],
    Sigma: AbstractSquareMatrix,
    logZ: Optional[Union[Float[Array, ''], QuadraticForm]] = None
  ):
    self.A = A
    self.u = u
    Sigma = util.psd_check(Sigma)
    self.Sigma = 0.5*(Sigma + Sigma.T)

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  @classmethod
  @dispatch
  def no_op_like(cls, other: 'GaussianTransition'):
    out = super().zeros_like(other)

    # A will be the identity matrix (don't change the dynamics)
    out = eqx.tree_at(lambda x: x.A, out, out.A.set_eye())

    # Set u to 0
    out = eqx.tree_at(lambda x: x.u, out, out.u*0.0)

    # Sigma will be 0 (don't add noise when we sample)
    out = eqx.tree_at(lambda x: x.Sigma, out, out.Sigma.set_zero())

    return out

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.A.batch_size

  @auto_vmap
  @dispatch
  def swap_variables(self) -> 'GaussianTransition':
    Ainv = self.A.get_inverse()
    u = -Ainv@self.u
    Sigma = Ainv@self.Sigma@Ainv.T
    Sigma = Sigma.set_symmetric()
    return GaussianTransition(Ainv, u, Sigma, self.logZ)

  @auto_vmap
  def marginalize_out_y(self) -> StandardGaussian:
    zeromu = zeros_like(self.u)
    zerocov = self.Sigma.zeros_like(self.Sigma)
    zerocov = zerocov.set_inf()
    nc = self.normalizing_constant()
    return StandardGaussian(zeromu, zerocov, self.logZ - nc)

  @auto_vmap
  def normalizing_constant(self):
    Sigmainv_u = self.Sigma.solve(self.u) # <----- This is the source of instability, but is 0 in LTI-SDEs because u=0!
    dim = self.u.shape[0]

    logZ = 0.5*vdot(Sigmainv_u, self.u)
    logZ += 0.5*self.Sigma.get_log_det()
    logZ += 0.5*dim*jnp.log(2*jnp.pi)

    return util.where(self.Sigma.is_zero, zeros_like(logZ), logZ)

  @auto_vmap
  def __call__(self, y: Float[Array, 'Dy'], x: Float[Array, 'Dx']) -> Float[Array, '']:
    return self.condition_on_x(x)(y)

  @auto_vmap
  def log_prob(self, y: Float[Array, 'Dy'], x: Float[Array, 'Dx']) -> Float[Array, '']:
    return self.condition_on_x(x).log_prob(y)

  @auto_vmap
  @dispatch
  def condition_on_x(self, x: Union[Float[Array, 'D'], LinearFunctional]) -> StandardGaussian:
    Ax = self.A@x

    muy = Ax + self.u
    Sigmay = self.Sigma

    logZ = vdot(Ax, self.Sigma.solve(0.5*Ax + self.u))
    logZ = util.where(self.Sigma.is_zero, zeros_like(logZ), logZ)
    return StandardGaussian(muy, Sigmay, logZ + self.logZ)

  @auto_vmap
  @dispatch
  def update_y(
    self,
    potential: NaturalGaussian,
    only_return_transition: bool = False
  ) -> Union['JointPotential', 'GaussianTransition']:
    """Incorporate a potential over y into the joint potential"""
    A, u, Sigma = self.A, self.u, self.Sigma
    Jx, hx, logZx = potential.J, potential.h, potential.logZ
    I = Sigma.set_eye()

    SigmaJ = Sigma@Jx
    I_plus_SigmaJ = I + SigmaJ
    S_ = I_plus_SigmaJ.T.solve(Jx).T
    T = I - Sigma@S_
    # If potential is zero (Jx = zero), then S_ = 0. Handled naturally
    # If potential is inf (Jx = inf), then T = 0.  Can't handle this because what hx has no meaning!
    T = util.where(Jx.tags.is_inf, T.set_zero(), T)
    T = util.where(Jx.tags.is_zero, T.set_eye(), T)

    # Same for both standard and natural:
    Sigmabar = T@Sigma
    Sigmabar = Sigmabar.set_symmetric()
    Abar = T@A
    ubar = T@u + Sigmabar@hx
    new_transition = GaussianTransition(Abar, ubar, Sigmabar, self.logZ)
    if only_return_transition:
      return new_transition

    Jbar = A.T@S_.T@A
    Jbar = Jbar.set_symmetric()
    hbar = Abar.T@hx - A.T@S_@u

    new_prior = NaturalGaussian(Jbar, hbar, logZx)
    return JointPotential(new_transition, new_prior)

  @auto_vmap
  @dispatch
  def update_y(
    self,
    potential: StandardGaussian,
    only_return_transition: bool = False
  ) -> Union['JointPotential', 'GaussianTransition']:
    A, u, Sigma = self.A, self.u, self.Sigma
    Sigmax, mux, logZx = potential.Sigma, potential.mu, potential.logZ
    I = Sigma.set_eye()

    Sigma_plus_Sigmax = Sigma + Sigmax
    S = Sigma_plus_Sigmax.T.solve(Sigma).T # Sigma@(Sigma + Sigmax)^{-1}
    S = util.where(Sigmax.tags.is_inf, S.set_zero(), S) # Since we're not fixing to tags anymore, we need to handle this manually

    T = I - S
    # If potential is zero (Sigmax = inf), then S = 0.  Handled by inf.solve(.) = 0
    # If potential is inf (Sigmax = zero), then T = 0.  Handled naturally, but enforce here for safety
    T = util.where(Sigmax.tags.is_zero, T.set_zero(), T)

    # Same for both standard and natural:
    Sigmabar = T@Sigma
    Sigmabar = Sigmabar.set_symmetric()
    Abar = T@A
    ubar = T@u + S@mux
    new_transition = GaussianTransition(Abar, ubar, Sigmabar, self.logZ)
    if only_return_transition:
      return new_transition

    Ainv = self.A.get_inverse() # Get this for free in LTI-SDEs
    Pbar = Ainv@Sigma_plus_Sigmax@Ainv.T
    Pbar = Pbar.set_symmetric()
    mbar = Ainv@(mux - u)
    mbar = util.where(Sigmax.tags.is_inf, zeros_like(mbar), mbar)

    new_prior = StandardGaussian(mbar, Pbar, logZx)
    return JointPotential(new_transition, new_prior)

  @auto_vmap
  @dispatch
  def update_y(
    self,
    potential: MixedGaussian,
    only_return_transition: bool = False
  ) -> Union['JointPotential', 'GaussianTransition']:
    """Incorporate a potential over y into the joint potential"""
    A, u, Sigma = self.A, self.u, self.Sigma
    Jy, muy, logZy = potential.J, potential.mu, potential.logZ
    I = Sigma.set_eye()

    SigmaJ = Sigma@Jy
    I_plus_SigmaJ = I + SigmaJ
    R = I_plus_SigmaJ.T.solve(Jy).T # Jy@(I + Sigma@Jy)^{-1}
    S = Sigma@R                     # Sigma@Jy@(I + Sigma@Jy)^{-1}
    T = I - S
    # If potential is zero (Jy = zero, total uncertainty), then R = 0. Handled naturally
    # If potential is inf (Jy = inf, total certainty), then T = 0 and S = I. Handled naturally, but enforce here for safety
    T, S = util.where(Jy.tags.is_inf, (T.set_zero(), S.set_eye()), (T, S))
    T, S = util.where(Jy.tags.is_zero, (T.set_eye(), S.set_zero()), (T, S))

    # Same for both standard and natural:
    Sigmabar = T@Sigma
    Sigmabar = Sigmabar.set_symmetric()
    Abar = T@A
    ubar = T@u + S@muy
    new_transition = GaussianTransition(Abar, ubar, Sigmabar, self.logZ)
    if only_return_transition:
      return new_transition

    # If Jy is inf, then R is Sigma^{-1}.
    # Can't do anything about this inversion of Sigma!
    RTA = util.where(Jy.tags.is_inf, Sigma.solve(A).cast_like(R), R.T@A)

    Jbar = A.T@RTA
    Jbar = Jbar.set_symmetric()
    Ainv = self.A.get_inverse() # Get this for free in LTI-SDEs
    mbar = Ainv@(muy - u)
    mbar = util.where(Jy.tags.is_zero, zeros_like(mbar), mbar)

    new_prior = MixedGaussian(mbar, Jbar, logZy)
    return JointPotential(new_transition, new_prior)

  @auto_vmap
  def chain(self, other: 'GaussianTransition') -> 'GaussianTransition':
    Ak, uk, Sigmak = other.A, other.u, other.Sigma    # Ax, ux, Sigmax
    Akm1, ukm1, Sigmakm1 = self.A, self.u, self.Sigma # Az, uz, Sigmaz

    A = Ak@Akm1
    u = Ak@ukm1 + uk
    Sigma = Sigmak + Ak@Sigmakm1@Ak.T
    Sigma = Sigma.set_symmetric()

    # Compute the log partition function.  This seems to be the most stable way even
    # though it requires inverting three matrices.
    Akinv = Ak.get_inverse()
    Sigmakinv = Sigmak.get_inverse()
    Sigmakm1inv = Sigmakm1.get_inverse()
    I = Sigmak.set_eye()
    dim = Sigmak.shape[0]

    Ax, ux, Sigmax = Ak, uk, Sigmak
    Az, uz, Sigmaz = Akm1, ukm1, Sigmakm1

    Axinv = Ax.get_inverse()
    Sigmaxinv = Sigmax.get_inverse()
    Sigmazinv = Sigmaz.get_inverse()

    T = (Sigmax + Ax@Sigmaz@Ax.T).get_inverse()

    matxx_inv = T - Sigmax.get_inverse()
    matxz = Axinv@Sigmax + Sigmaz@Ax.T
    matzz_inv = Ax.T@T@Ax - Sigmazinv
    new_term1 = 0.5*vdot(ux, matxx_inv@ux)
    new_term2 = vdot(ux, matxz.solve(uz))
    new_term3 = 0.5*vdot(uz, matzz_inv@uz)

    new_term4 = 0.5*(Sigma.get_log_det() - Sigmaz.get_log_det() - Sigmax.get_log_det()) # Seems right
    new_term5 = -0.5*dim*jnp.log(2*jnp.pi) # Seems right
    new_term6 = self.logZ + other.logZ # Seems right

    logZ = new_term1 + new_term2 + new_term3 + new_term4 + new_term5 + new_term6
    return GaussianTransition(A, u, Sigma, logZ)

  @auto_vmap
  def to_nat(self) -> NaturalJointGaussian:
    """For checking correctness"""
    J11 = self.Sigma.get_inverse()
    J12 = -self.Sigma.solve(self.A)
    J22 = -self.A.T@J12
    J22 = J22.set_symmetric()
    h1 = self.Sigma.solve(self.u)
    h2 = -self.A.T@h1
    return NaturalJointGaussian(J11, J12, J22, h1, h2, self.logZ)

  @auto_vmap
  def update_and_marginalize_out_x(self, potential: AbstractGaussianPotential) -> AbstractGaussianPotential:
    std_potential = potential.to_std()
    mu, Sigma = std_potential.mu, std_potential.Sigma

    new_mean = self.A@mu + self.u
    new_cov = self.Sigma + self.A@Sigma@self.A.T
    new_cov = new_cov.set_symmetric()

    # The new distribution is normalized, so just add a correction term
    new_dist = StandardGaussian(new_mean, new_cov)

    correction = potential.logZ - potential.normalizing_constant()
    correction += self.logZ - self.normalizing_constant()
    logZ = new_dist.logZ + correction
    out_std = StandardGaussian(new_mean, new_cov, logZ)

    if isinstance(potential, StandardGaussian):
      return out_std
    elif isinstance(potential, NaturalGaussian):
      return out_std.to_nat()
    elif isinstance(potential, MixedGaussian):
      return out_std.to_mixed()
    else:
      raise ValueError(f"Unknown potential type: {type(potential)}")

################################################################################################################

@auto_vmap
def functional_potential_to_transition(potential: AbstractGaussianPotential) -> GaussianTransition:
  """Convert a functional potential to a transition"""
  potential_std: StandardGaussian = potential.to_std()

  A = potential_std.mu.A
  u = potential_std.mu.b
  Sigma = potential_std.Sigma
  logZ = potential_std.logZ

  return GaussianTransition(A, u, Sigma, logZ)

################################################################################################################

def gaussian_chain_parallel_sample(transitions: GaussianTransition,
                                   x0: Float[Array, 'D'],
                                   keys: Float[Array, 'N-1 2']):
  """Parallel algorithm for sampling from a chain of Gaussian transitions.
  See appendix H here https://arxiv.org/pdf/2208.04933.

  Recurrence is x_{k+1} = A_k x_k + u_k + \Sigma_k^{1/2} \epsilon_k
                        = A_k x_k + v_k

  The chain operation is (A_k, v_k)*(A_{k-1}, v_{k-1}) = (A_k A_{k-1}, A_k v_{k-1} + v_k)
  """

  class Elements(AbstractBatchableObject):
    A: DenseMatrix
    v: Float[Array, 'D']

    @property
    def batch_size(self):
      return self.A.batch_size

  def make_elements(transition, key):
    eps = random.normal(key, transition.u.shape)
    v = transition.u
    if USE_CHOLESKY_SAMPLING:
      v += transition.Sigma.get_cholesky()@eps
    else:
      U, S, V = transition.Sigma.get_svd()
      S_chol = S.get_cholesky()
      v += U@S_chol@eps
    return Elements(transition.A, v)
  elements = jax.vmap(make_elements)(transitions, keys)
  A0 = elements[0].A
  I, zero = A0.eye(A0.shape[-1]), zeros_like(x0)
  elements = jtu.tree_map(lambda x, y: jnp.concatenate([x[None], y], axis=0), Elements(I, zero), elements)

  def chain(left, right):
    Akm1, ukm1 = left.A, left.v
    Ak, uk = right.A, right.v
    A = Ak@Akm1
    u = Ak@ukm1 + uk
    return Elements(A, u)
  result = parallel_scan(chain, elements, reverse=False)

  def get_sample(elements):
    A, u = elements.A, elements.v
    return A@x0 + u
  xts = jax.vmap(get_sample)(result)
  return xts

################################################################################################################

def max_likelihood_gaussian_transition(xts: Float[Array, 'B Dx'], yts: Float[Array, 'B Dy']) -> GaussianTransition:
  assert xts.ndim == 2

  # Compute the covariance of the data
  xt_xtT = jnp.einsum('bi,bj->ij', xts, xts)
  xt_ytT = jnp.einsum('bi,bj->ij', xts, yts)
  A = xt_ytT@jnp.linalg.inv(xt_xtT)

  dx = xts - jnp.einsum('ij,bj->bi', A, yts)
  Sigma = jnp.einsum('bi,bj->ij', dx, dx)/xts.shape[0]

  u = jnp.mean(dx, axis=0)

  return GaussianTransition(DenseMatrix(A, tags=TAGS.no_tags),
                            u,
                            DenseMatrix(Sigma, tags=TAGS.no_tags))

################################################################################################################

class GaussianJointStatistics(GaussianStatistics):
  Ex: Float[Array, 'Dx']
  ExxT: Float[Array, 'Dx Dx']
  ExyT: Float[Array, 'Dx Dy']
  Ey: Float[Array, 'Dy']
  EyyT: Float[Array, 'Dy Dy']

  @auto_vmap
  def augment(self):
    # Pad with an extra dimension so that we can
    # solve for the optimal transition parameters
    # during the M-step
    Exhat = jnp.pad(self.Ex, (0, 1), constant_values=1.0)
    ExhatxhatT = jnp.block([[self.ExxT, self.Ex[:,None]],
                            [self.Ex[None,:], jnp.array([1.0])]])
    ExhatyT = jnp.block([[self.ExyT],
                         [self.Ey.T]])
    return GaussianJointStatistics(Exhat, ExhatxhatT, ExhatyT, self.Ey, self.EyyT)

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

  @auto_vmap
  def to_block_stats(self):
    # Place the statistics into the block matrix that they came from
    Xi = jnp.block([[self.EyyT, self.EyxT],
                    [self.ExyT, self.ExxT]])
    mu = jnp.concatenate([self.Ey, self.Ex], axis=-1)
    return GaussianStatistics(mu, Xi)

  @classmethod
  def from_block_stats(cls, stats: GaussianStatistics):
    mu_reshaped = einops.rearrange(stats.Ex, '(two d) -> two d', two=2)
    Xi_reshaped = einops.rearrange(stats.ExxT, '(two i) (dos j) -> (two dos) i j', two=2, dos=2)

    Ex = mu_reshaped[1]
    Ey = mu_reshaped[0]
    ExxT = Xi_reshaped[3]
    ExyT = Xi_reshaped[2]
    EyyT = Xi_reshaped[0]
    return GaussianJointStatistics(Ex, ExxT, ExyT, Ey, EyyT)

  @property
  def EyxT(self):
    return self.ExyT.mT

def gaussian_joint_e_step(joint: JointPotential) -> GaussianJointStatistics:
  transition, prior = joint.transition, joint.prior
  transition_nat = transition.to_nat()
  joint_nat = transition_nat.update_x(prior.to_nat())
  joint_ess = joint_nat.to_ess()
  return GaussianJointStatistics.from_block_stats(joint_ess)

def gaussian_joint_m_step(statistics: GaussianJointStatistics) -> JointPotential:
  """Returns the solution to argmin_{q(y|x)}E_{p(x,y)}[\log q(y|x)] given the statistics for p(x,y)
  """
  dim = statistics.Ex.shape[-1]
  stats = statistics.to_block_stats()

  if stats.batch_size is not None:
    assert 0, 'Need to revisit this to see if it is correct'
    # Sum out the leading dimension
    N = stats.batch_size
    mu = stats.Ex.mean(axis=0)
    Sigma = stats.ExxT.sum(axis=0) - N*jnp.outer(mu, mu)
    stats = GaussianStatistics(mu, Sigma)

  nat = stats.to_nat().to_joint(dim=dim)
  J11, J12, J22 = nat.J11, nat.J12, nat.J22
  h1, h2 = nat.h1, nat.h2

  # Extract the transition
  Sigma = J11.get_inverse()
  A = -Sigma@J12
  u = Sigma@h1
  transition = GaussianTransition(A, u, Sigma)

  # Extract the prior
  J = J22 - A.T@J11@A
  h = h2 + A.T@h1
  prior = NaturalGaussian(J, h)
  return JointPotential(transition, prior)

################################################################################################################
