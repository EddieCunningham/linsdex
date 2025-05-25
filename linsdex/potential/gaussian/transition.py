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
from linsdex.potential.gaussian.dist import MixedGaussian, NaturalGaussian, StandardGaussian, NaturalJointGaussian, GaussianStatistics
from plum import dispatch
import linsdex.util as util
from linsdex.matrix.tags import Tags, TAGS
from linsdex.util.parallel_scan import parallel_scan
import jax.tree_util as jtu


__all__ = ['GaussianTransition',
           'max_likelihood_gaussian_transition',
           'GaussianJointStatistics',
           'gaussian_joint_e_step',
           'gaussian_joint_m_step']

class GaussianTransition(AbstractTransition):

  A: AbstractSquareMatrix
  u: Float[Array, 'D']
  Sigma: AbstractSquareMatrix
  logZ: Float[Array, '']

  def __init__(self,
    A: AbstractSquareMatrix,
    u: Float[Array, 'D'],
    Sigma: AbstractSquareMatrix,
    logZ: Optional[Float[Array, '']] = None
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
    out = eqx.tree_at(lambda x: x.u, out, jnp.zeros_like(other.u))

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
    zeromu = jnp.zeros_like(self.u)
    zerocov = self.Sigma.zeros_like(self.Sigma)
    zerocov = zerocov.set_inf()
    nc = self.normalizing_constant()
    return StandardGaussian(zeromu, zerocov, self.logZ - nc)

  @auto_vmap
  def normalizing_constant(self):
    Sigmainv_u = self.Sigma.solve(self.u) # <----- This is the source of instability, but is 0 in LTI-SDEs because u=0!
    dim = self.u.shape[0]

    logZ = 0.5*jnp.vdot(Sigmainv_u, self.u)
    logZ += 0.5*self.Sigma.get_log_det()
    logZ += 0.5*dim*jnp.log(2*jnp.pi)

    return util.where(self.Sigma.is_zero, jnp.array(0.0), logZ)

  @auto_vmap
  def __call__(self, y: Float[Array, 'Dy'], x: Float[Array, 'Dx']) -> Float[Array, '']:
    return self.condition_on_x(x)(y)

  @auto_vmap
  def log_prob(self, y: Float[Array, 'Dy'], x: Float[Array, 'Dx']) -> Float[Array, '']:
    return self.condition_on_x(x).log_prob(y)

  @auto_vmap
  @dispatch
  def condition_on_x(self, x: Float[Array, 'D']) -> StandardGaussian:
    Ax = self.A@x

    muy = Ax + self.u
    Sigmay = self.Sigma

    logZ = jnp.vdot(Ax, self.Sigma.solve(0.5*Ax + self.u))
    logZ = util.where(self.Sigma.is_zero, jnp.array(0.0), logZ)
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
    mbar = util.where(Sigmax.tags.is_inf, jnp.zeros_like(mbar), mbar)

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
    mbar = util.where(Jy.tags.is_zero, jnp.zeros_like(mbar), mbar)

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
    new_term1 = 0.5*jnp.vdot(ux, matxx_inv@ux)
    new_term2 = jnp.vdot(ux, matxz.solve(uz))
    new_term3 = 0.5*jnp.vdot(uz, matzz_inv@uz)

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

  _USE_CHOL = False # This global is in gaussian/dist.py as well.  Returns zeros if Sigma is 0 whereas cholesky returns inf.

  def make_elements(transition, key):
    eps = random.normal(key, transition.u.shape)
    v = transition.u
    if _USE_CHOL:
      v += transition.Sigma.get_cholesky()@eps
    else:
      U, S, V = transition.Sigma.get_svd()
      S_chol = S.get_cholesky()
      v += U@S_chol@eps
    return Elements(transition.A, v)
  elements = jax.vmap(make_elements)(transitions, keys)
  A0 = elements[0].A
  I, zero = A0.eye(A0.shape[-1]), jnp.zeros_like(x0)
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

def tmp_test():
  from linsdex import TimeSeries, GaussianPotentialSeries, StochasticHarmonicOscillator
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
  inverse_covs = inverse_covs + 0.01
  latent_potentials = GaussianPotentialSeries(series.times, means, certainty=inverse_covs)

  sde = StochasticHarmonicOscillator(freq=1.0, coeff=0.0, sigma=0.1, observation_dim=series.dim)
  cond_sde = sde.condition_on(latent_potentials)
  crf = cond_sde.discretize()
  first_message = crf.base_transitions[-1].update_and_marginalize_out_y(crf.node_potentials[-1])
  import pdb; pdb.set_trace()

def correctness_tests():

  # Turn on x64
  jax.config.update('jax_enable_x64', True)
  key = random.PRNGKey(0)

  k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)
  x_dim = 4
  x, y = random.normal(key, (2, x_dim))
  A = random.normal(k1, (x_dim, x_dim))
  A = DenseMatrix(A, tags=TAGS.no_tags)

  u = random.normal(k2, (x_dim,))
  Sigma = random.normal(k3, (x_dim, x_dim))
  Sigma = Sigma@Sigma.T
  Sigma = DenseMatrix(Sigma, tags=TAGS.no_tags)

  mux = random.normal(k4, (x_dim,))
  Sigmax = random.normal(k5, (x_dim, x_dim))
  Sigmax = Sigmax@Sigmax.T
  Sigmax = DenseMatrix(Sigmax, tags=TAGS.no_tags)

  logZ = jnp.array(1.3)
  # transition = GaussianTransition(A, u, Sigma)
  potential = StandardGaussian(mux, Sigmax)
  transition = GaussianTransition(A, u, Sigma, logZ)
  comp = transition.to_nat()

  zero_potential = potential.total_uncertainty_like(potential)
  out = transition.update_y(zero_potential.to_nat())
  # out = transition.update_y(zero_potential.to_std())
  out = transition.update_y(zero_potential.to_mixed())

  # Check that we can evaluate the log prob of a sample and have stable gradients
  def get_log_prob(transition, x, y):
    return transition.condition_on_x(x).log_prob(y)
  y2 = transition.condition_on_x(x).sample(key)
  log_prob_grad = eqx.filter_grad(get_log_prob)(transition, x, y2)

  # Check that the maximum likelihood solution is correct
  def sample(key, x):
    return transition.condition_on_x(x).sample(key)
  xs = random.normal(key, (1000, x_dim))
  keys = random.split(key, 1000)
  ys = jax.vmap(sample)(keys, xs)
  mle = max_likelihood_gaussian_transition(ys, xs)

  # Check that the e-step is correct
  joint = JointPotential(transition, potential)
  statistics = gaussian_joint_e_step(joint)
  keys = random.split(key, 50000)
  ys, xs = jax.vmap(joint.sample)(keys)
  Ex = xs.mean(axis=0)
  Ey = ys.mean(axis=0)
  B = xs.shape[0]
  ExxT = jnp.einsum('bi,bj->ij', xs, xs)/B
  ExyT = jnp.einsum('bi,bj->ij', xs, ys)/B
  EyyT = jnp.einsum('bi,bj->ij', ys, ys)/B
  assert jnp.allclose(statistics.Ex, Ex, atol=1e-1, rtol=1e-1)
  assert jnp.allclose(statistics.Ey, Ey, atol=1e-1, rtol=1e-1)
  assert jnp.allclose(statistics.ExxT, ExxT, atol=1e-1, rtol=1e-1)
  assert jnp.allclose(statistics.ExyT, ExyT, atol=1e-1, rtol=1e-1)
  assert jnp.allclose(statistics.EyyT, EyyT, atol=1e-1, rtol=1e-1)

  # Check that the m-step is correct
  joint = JointPotential(mle, potential)
  statistics = gaussian_joint_e_step(joint)
  updated_joint = gaussian_joint_m_step(statistics)
  updated_transition = updated_joint.transition

  params, static = eqx.partition(updated_transition, eqx.is_array)

  def loss(transition_params):
    transition = eqx.combine(transition_params, static)
    keys = random.split(key, 10000000)
    ys, xs = jax.vmap(joint.sample)(keys)
    log_probs = jax.vmap(lambda x, y: transition.condition_on_x(x).log_prob(y))(xs, ys)
    return -log_probs.mean()
  grad = eqx.filter_grad(loss)(params)
  grad = eqx.combine(grad, static)
  assert jnp.allclose(grad.A.as_matrix(), jnp.zeros_like(grad.A.as_matrix()), atol=1e-1, rtol=1e-1)
  assert jnp.allclose(grad.u, jnp.zeros_like(grad.u), atol=1e-1, rtol=1e-1)
  assert jnp.allclose(grad.Sigma.as_matrix(), jnp.zeros_like(grad.Sigma.as_matrix()), atol=1e-1, rtol=1e-1)

  def total_log_prob(transition, xs, ys):
    def log_prob(x, y):
      return transition.condition_on_x(x).log_prob(y)
    return jax.vmap(log_prob)(xs, ys).sum()

  key2, _ = random.split(key)
  xs = random.normal(key2, (1000, x_dim))
  keys = random.split(key2, 1000)
  ys = jax.vmap(sample)(keys, xs)
  mle_log_prob_grad = eqx.filter_grad(total_log_prob)(mle, xs, ys)

  # Check the chain operation
  chain = transition.chain(transition)
  chain_comp = comp.chain(comp)
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, chain.to_nat(), chain_comp))

  # Check swap variables
  swapped = transition.swap_variables()
  swapped_comp = comp.swap_variables()
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, swapped.to_nat(), swapped_comp))

  # Check condition_on_y
  conditioned_on_y = transition.condition_on_y(y)
  conditioned_on_y_true = comp.condition_on_y(y).to_std()
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, conditioned_on_y, conditioned_on_y_true))

  # Check condition_on_x
  conditioned_on_x = transition.condition_on_x(x)
  conditioned_on_x_true = comp.condition_on_x(x).to_std()
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, conditioned_on_x, conditioned_on_x_true))

  # Check condition_on_y with conditioning on x
  conditioned_on_x2 = transition.swap_variables().condition_on_y(x)
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, conditioned_on_x2, conditioned_on_x_true))

  # Compare the log probs of each
  conditioned_on_x = transition.condition_on_x(x)
  log_prob_transition = conditioned_on_x(y)
  log_prob_comp = comp(jnp.concatenate([y, x]))
  assert jnp.allclose(log_prob_transition, log_prob_comp)

  # Check update_y
  test = transition.update_y(potential.to_mixed())
  transition_nat = test.transition.to_nat()
  prior_nat = test.prior.to_nat()
  J11 = transition_nat.J11
  J12 = transition_nat.J12
  J22 = transition_nat.J22 + prior_nat.J
  h1 = transition_nat.h1
  h2 = transition_nat.h2 + prior_nat.h
  logZ = transition_nat.logZ + prior_nat.logZ
  truth = comp.update_y(potential.to_nat())
  assert jnp.allclose(J11._force_fix_tags().as_matrix(), truth.J11._force_fix_tags().as_matrix())
  assert jnp.allclose(J12._force_fix_tags().as_matrix(), truth.J12._force_fix_tags().as_matrix())
  assert jnp.allclose(J22._force_fix_tags().as_matrix(), truth.J22._force_fix_tags().as_matrix())
  assert jnp.allclose(h1, truth.h1)
  assert jnp.allclose(h2, truth.h2)
  assert jnp.allclose(logZ, truth.logZ)

  # Check update_y
  test = transition.update_y(potential)
  transition_nat = test.transition.to_nat()
  prior_nat = test.prior.to_nat()
  J11 = transition_nat.J11
  J12 = transition_nat.J12
  J22 = transition_nat.J22 + prior_nat.J
  h1 = transition_nat.h1
  h2 = transition_nat.h2 + prior_nat.h
  logZ = transition_nat.logZ + prior_nat.logZ
  truth = comp.update_y(potential.to_nat())
  assert jnp.allclose(J11._force_fix_tags().as_matrix(), truth.J11._force_fix_tags().as_matrix())
  assert jnp.allclose(J12._force_fix_tags().as_matrix(), truth.J12._force_fix_tags().as_matrix())
  assert jnp.allclose(J22._force_fix_tags().as_matrix(), truth.J22._force_fix_tags().as_matrix())
  assert jnp.allclose(h1, truth.h1)
  assert jnp.allclose(h2, truth.h2)
  assert jnp.allclose(logZ, truth.logZ)

  # Check update_y
  test = transition.update_y(potential.to_nat())
  transition_nat = test.transition.to_nat()
  prior_nat = test.prior
  J11 = transition_nat.J11
  J12 = transition_nat.J12
  J22 = transition_nat.J22 + prior_nat.J
  h1 = transition_nat.h1
  h2 = transition_nat.h2 + prior_nat.h
  logZ = transition_nat.logZ + prior_nat.logZ
  truth = comp.update_y(potential.to_nat())
  assert jnp.allclose(J11._force_fix_tags().as_matrix(), truth.J11._force_fix_tags().as_matrix())
  assert jnp.allclose(J12._force_fix_tags().as_matrix(), truth.J12._force_fix_tags().as_matrix())
  assert jnp.allclose(J22._force_fix_tags().as_matrix(), truth.J22._force_fix_tags().as_matrix())
  assert jnp.allclose(h1, truth.h1)
  assert jnp.allclose(h2, truth.h2)
  assert jnp.allclose(logZ, truth.logZ)

  # Run marginalize_out_y.  The ground truth here is zero
  marginal = transition.marginalize_out_y()

  # Check update_and_marginalize_out_x
  test_nat = transition.update_and_marginalize_out_x(potential.to_nat())
  test = transition.update_and_marginalize_out_x(potential)
  truth = comp.update_and_marginalize_out_x(potential.to_nat())
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, test.to_nat(), truth))

  # Check to see if we can handle deterministic potentials.
  # The ground truth here should be a deterministic potential
  # that has the same mean as the original potential
  potential = potential.make_deterministic()
  test = transition.update_y(potential)

  keys = random.split(key, 10)
  ys, xs = jax.vmap(test.sample)(keys)
  assert jnp.all(jax.vmap(jnp.allclose, in_axes=(None, 0))(potential.mu, ys))

  print("All tests passed!")


def performance_tests():
  """The key thing that we need to test here is that the functions
  update_and_marginalize_out_y and chain are as fast as latent_linear_sde
  versions."""

  # Turn on x64
  jax.config.update('jax_enable_x64', True)
  key = random.PRNGKey(0)

  k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)
  x_dim = 4

  # Create the transition
  x, y = random.normal(key, (2, x_dim))
  A_raw = random.normal(k1, (x_dim, x_dim))

  u = random.normal(k2, (x_dim,))
  Sigma_raw = random.normal(k3, (x_dim, x_dim))
  Sigma_raw = Sigma_raw@Sigma_raw.T

  mux = random.normal(k4, (x_dim,))
  Sigmax_raw = random.normal(k5, (x_dim, x_dim))
  Sigmax_raw = Sigmax_raw@Sigmax_raw.T

  logZ = jnp.array(1.3)


  # Test the current implementation
  with jax.profiler.trace("/tmp/tensorboard"):
    # Create the transition and node potential
    A = DenseMatrix(A_raw, tags=TAGS.no_tags)
    Sigma = DenseMatrix(Sigma_raw, tags=TAGS.no_tags)
    transition = GaussianTransition(A, u, Sigma, logZ)

    # Create the potential
    Sigmax = DenseMatrix(Sigmax_raw, tags=TAGS.no_tags)
    potential_std = StandardGaussian(mux, Sigmax)
    potential_nat = potential_std.to_nat()

    out = transition.update_and_marginalize_out_y(potential_nat)
    out = jtu.tree_map(lambda x: x.block_until_ready(), out)

    out = transition.update_and_marginalize_out_y(potential_nat)
    out = jtu.tree_map(lambda x: x.block_until_ready(), out)

    out = transition.update_and_marginalize_out_y(potential_nat)
    out = jtu.tree_map(lambda x: x.block_until_ready(), out)

  # Test the latent_linear_sde implementation
  from latent_linear_sde import TransitionDistribution as LatentLinearSDETransition
  from latent_linear_sde import StandardGaussian as StdGaussian
  from latent_linear_sde import MatrixEagerLinearOperator
  with jax.profiler.trace("/tmp/tensorboard"):
    A = MatrixEagerLinearOperator(A_raw)
    Sigma = MatrixEagerLinearOperator(Sigma_raw)
    transition_lt = LatentLinearSDETransition(A, u, Sigma)
    Sigmax_lls = MatrixEagerLinearOperator(Sigmax_raw)
    potential = StdGaussian(mux, Sigmax_lls).to_nat()

    out_lt = transition_lt.update_and_marginalize_out_x(potential)
    out_lt = jtu.tree_map(lambda x: x.block_until_ready(), out_lt)

    out_lt = transition_lt.update_and_marginalize_out_x(potential)
    out_lt = jtu.tree_map(lambda x: x.block_until_ready(), out_lt)

    out_lt = transition_lt.update_and_marginalize_out_x(potential)
    out_lt = jtu.tree_map(lambda x: x.block_until_ready(), out_lt)

  import pdb; pdb.set_trace()




if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *
  tmp_test()
  # correctness_tests()
  # performance_tests()