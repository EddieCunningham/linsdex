import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type, Dict
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
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.crf.crf import *
from linsdex.crf.continuous_crf import *
from linsdex.matrix.matrix_base import *
from linsdex.potential.gaussian.dist import *
from linsdex.potential.gaussian.transition import *
from linsdex.sde.sde_base import AbstractLinearSDE, AbstractLinearTimeInvariantSDE
from plum import dispatch
import linsdex.util as util
from linsdex.ssm import AbstractStateSpaceModel
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE

def tree_repeat(x: PyTree, length: int):
  return jax.vmap(lambda i: x)(jnp.arange(length))

class LinearDynamicalSystem(AbstractStateSpaceModel):
  prior: AbstractPotential
  transition_or_transitions: GaussianTransition
  emission_or_emissions: GaussianTransition
  parallel: bool = eqx.field(static=True)
  length: int = None

  def __init__(self, prior: AbstractPotential,
               transition_or_transitions: GaussianTransition,
               emission_or_emissions: GaussianTransition,
               length: int = None,
               parallel: bool = False):
    self.prior = prior

    if transition_or_transitions.batch_size is None and emission_or_emissions.batch_size is None:
      assert length is not None, "Must provide length if transitions and emissions are not batched"

    self.transition_or_transitions = transition_or_transitions
    self.emission_or_emissions = emission_or_emissions
    self.length = length

    self.parallel = parallel

  @property
  def transitions(self):

    if self.has_single_transition() and self.has_single_emission():
      transitions = tree_repeat(self.transition_or_transitions, self.length - 1)
    elif self.has_single_transition() == False:
      transitions = self.transition_or_transitions
    elif self.has_single_emission() == False:
      length = self.emission_or_emissions.batch_size
      transitions = tree_repeat(self.transition_or_transitions, length - 1)
    else:
      transitions = self.transition_or_transitions

    return transitions

  @property
  def emissions(self):

    if self.has_single_transition() and self.has_single_emission():
      emissions = tree_repeat(self.emission_or_emissions, self.length)
    elif self.has_single_transition() == False:
      length = self.transition_or_transitions.batch_size
      emissions = tree_repeat(self.emission_or_emissions, length + 1)
    elif self.has_single_emission() == False:
      emissions = self.emission_or_emissions
    else:
      emissions = self.emission_or_emissions

    return emissions

  def has_single_transition(self):
    return self.transition_or_transitions.batch_size is None

  def has_single_emission(self):
    return self.emission_or_emissions.batch_size is None

class LDSStatistics(AbstractBatchableObject):
  prior_stats: GaussianStatistics
  emission_stats: GaussianJointStatistics
  dynamics_stats: GaussianJointStatistics

  @property
  def batch_size(self):
    return self.prior_stats.batch_size

################################################################################################################

def lds_e_step(lds: LinearDynamicalSystem,
               yts: Float[Array, 'B N Dy']) -> LDSStatistics:

  if yts.ndim == 2:
    yts = yts[None]

  def get_statistics(yk: Float[Array, 'N Dy']) -> LDSStatistics:
    # Get p(X_{1:N} | Y_{1:N})
    crf = lds.get_posterior(yk)

    # Get p(x_{k+1}, x_k | Y_{1:N})
    joints = crf.get_joints()

    # Get the statistics for the dynamics
    dynamics_stats = jax.vmap(gaussian_joint_e_step)(joints)

    # Extract the statistics for the prior
    prior_stats = GaussianStatistics(dynamics_stats.Ex[0], dynamics_stats.ExxT[0])

    # Construct the statistics for the emissions
    Ex0 = dynamics_stats.Ex[0]
    Exk = jnp.concatenate([Ex0[None], dynamics_stats.Ey], axis=0)
    ExkxkT = jnp.concatenate([dynamics_stats.ExxT[0][None], dynamics_stats.EyyT], axis=0)
    ExkykT = jnp.einsum('...i,...j->...ij', Exk, yk)
    ykykT = jnp.einsum('...i,...j->...ij', yk, yk)

    emission_stats = GaussianJointStatistics(Exk, ExkxkT, ExkykT, yk, ykykT)
    return LDSStatistics(prior_stats, emission_stats, dynamics_stats)

  statistics = jax.vmap(get_statistics)(yts)
  return statistics

def lds_m_update(lds: LinearDynamicalSystem, statistics: LDSStatistics) -> LinearDynamicalSystem:

  # Sum out the batch dimension
  statistics = jtu.tree_map(lambda x: x.sum(axis=0), statistics)
  prior_stats, emission_stats, dynamics_stats = statistics.prior_stats, statistics.emission_stats, statistics.dynamics_stats

  # Get the new prior
  prior_new = gaussian_m_step(prior_stats)

  # Get the new emissions.  Using the Gaussian joint m-step doesn't seem to work,
  # so we'll just do this by hand.

  # Augment the statistics so that we can solve for the optimal transition parameters
  emission_stats_aug = emission_stats.augment()
  dynamics_stats_aug = dynamics_stats.augment()

  def solve_for_params(transition_aug_stats: GaussianJointStatistics) -> Tuple[Float[Array, 'Dx Dx'],
                                                                               Float[Array, 'Dx'],
                                                                               Float[Array, 'Dx Dx']]:
    assert transition_aug_stats.batch_size is None
    EyyT = transition_aug_stats.EyyT
    EyxT = transition_aug_stats.EyxT
    ExyT = transition_aug_stats.ExyT
    ExxT = transition_aug_stats.ExxT

    W = EyxT@jnp.linalg.inv(ExxT)
    Sigma = EyyT - EyxT@W.mT - W@ExyT + W@ExxT@W.T
    A = W[:,:-1]
    u = W[:,-1]
    return A, u, Sigma

  def update_gaussian_transition(transition_aug_stats: GaussianJointStatistics,
                                 single_transition: bool) -> GaussianTransition:
    if single_transition:
      reduced_dynamics_stats = jtu.tree_map(lambda x: x.sum(axis=0), transition_aug_stats)
      A_new, u_new, Sigma_new = solve_for_params(reduced_dynamics_stats)
      Sigma_new = Sigma_new/transition_aug_stats.batch_size
      dynamics_new = GaussianTransition(DenseMatrix(A_new, tags=TAGS.no_tags),
                                        u_new,
                                        DenseMatrix(Sigma_new, tags=TAGS.symmetric_tags))
    else:
      def create_transition(A, u, Sigma):
        return GaussianTransition(DenseMatrix(A, tags=TAGS.no_tags),
                                 u,
                                 DenseMatrix(Sigma, tags=TAGS.symmetric_tags))
      (A, u, Sigma) = eqx.filter_vmap(solve_for_params)(transition_aug_stats)
      dynamics_new = eqx.filter_vmap(create_transition)(A, u, Sigma)

    return dynamics_new

  dynamics_new = update_gaussian_transition(dynamics_stats_aug, lds.has_single_transition())
  emissions_new = update_gaussian_transition(emission_stats_aug, lds.has_single_emission())

  assert jtu.tree_structure(prior_new) == jtu.tree_structure(lds.prior)
  assert jtu.tree_structure(dynamics_new) == jtu.tree_structure(lds.transitions)
  assert jtu.tree_structure(emissions_new) == jtu.tree_structure(lds.emissions)

  return LinearDynamicalSystem(prior_new, dynamics_new, emissions_new, length=lds.length)

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *
  from linsdex.sde.sde_base import linear_sde_test
  import matplotlib.pyplot as plt
  import linsdex.util as util
  from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian, NaturalGaussian
  from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior
  from linsdex.ssm.simple_decoder import PaddingLatentVariableDecoder
  from linsdex.sde import *
  from linsdex.matrix import *
  jax.config.update('jax_enable_x64', True)

  import pickle
  series = pickle.load(open('series.pkl', 'rb'))[:10]
  ts = jnp.array(series.times)*1.0
  yts = jnp.array(series.values)[...,:2].astype(jnp.float64)
  yts = yts[None]

  # series = TimeSeries(ts, yts).make_windowed_batches(window_size=10)[:1]
  # ts, yts = series.times, series.values
  N = ts.shape[-1]


  y_dim = yts.shape[-1]
  sde = BrownianMotion(sigma=0.1, dim=y_dim)
  # sde = CriticallyDampedLangevinDynamics(mass=0.1, beta=0.1, dim=y_dim)
  # sde = HigherOrderTrackingModel(sigma=0.1, position_dim=y_dim, order=2)
  x_dim = sde.dim

  # Construct some parameters for the model
  key = random.PRNGKey(0)
  k1, k2, k3 = random.split(key, 3)
  H, A = random.normal(k1, (2, N, x_dim, x_dim))
  u = random.normal(k2, (N, x_dim))
  R, Sigma = random.normal(k3, (2, N, x_dim, x_dim))

  def make_transition(H, u, R):
    return GaussianTransition(util.to_matrix(H), u, util.to_matrix(R, symmetric=True))
  emissions = eqx.filter_vmap(make_transition)(H, u, R)
  transition = make_transition(A[0], u[0], Sigma[0])

  # Construct the prior
  mat = emissions.A[0]
  I = mat.eye(mat.shape[-1])
  prior = StandardGaussian(mu=jnp.zeros(x_dim), Sigma=I)

  if False:
    # Dynamax model for comparison
    from dynamax import linear_gaussian_ssm
    model = linear_gaussian_ssm.models.LinearGaussianSSM(state_dim=x_dim,
                                                        emission_dim=y_dim)
    params, props = model.initialize()
    mu0, Sigma0 = params.initial.mean, params.initial.cov
    prior = StandardGaussian(mu=mu0, Sigma=DenseMatrix(Sigma0, tags=TAGS.symmetric_tags))
    A, u, Sigma = params.dynamics.weights, params.dynamics.bias, params.dynamics.cov
    transition = GaussianTransition(DenseMatrix(A, tags=TAGS.no_tags),
                                    u,
                                    DenseMatrix(Sigma, tags=TAGS.symmetric_tags))
    H, v, R = params.emissions.weights, params.emissions.bias, params.emissions.cov
    emissions = GaussianTransition(DenseMatrix(H, tags=TAGS.no_tags),
                                    v,
                                    DenseMatrix(R, tags=TAGS.symmetric_tags))
    log_py = model.marginal_log_prob(params, yts[0])
    (init_stats, dynamics_stats, emission_stats), marginal_loglik = jax.vmap(model.e_step, in_axes=(None, 0))(params, yts)
    Ex0, Ex0x0T, _ = init_stats
    sum_zzT, sum_zyT, sum_yyT, num_timesteps = emission_stats
    sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timestepsm1 = dynamics_stats

    batched_stats = jtu.tree_map(lambda x: jnp.array(x)[None], (init_stats, dynamics_stats, emission_stats))
    updated_params, _ = model.m_step(params, props, batched_stats, None)
    updated_log_py = model.marginal_log_prob(updated_params, yts)

    # sum_xnxnT == sum_zzT[:-1,:-1] == dynamics_stats.EyyT.sum(axis=1) == emission_stats.ExxT.sum(axis=1)
    # sum_zpxnT[:-1] == dynamics_stats.ExyT.sum(axis=1)
    # sum_zpzpT[:-1,:-1] == dynamics_stats.ExxT.sum(axis=1)

  # Initialize the LinearDynamicalSystem
  lds = LinearDynamicalSystem(prior, transition, emissions, length=yts.shape[-2])

  # @eqx.filter_jit
  def em_step(lds, yts):
    stats = lds_e_step(lds, yts)
    lds = lds_m_update(lds, stats)
    def get_ll(y):
      return lds.get_posterior(y).get_marginal_log_likelihood()
    log_py = jax.vmap(get_ll)(yts).mean()

    import pdb; pdb.set_trace()
    return lds, log_py

  lds_updated, log_py_updated = em_step(lds, yts)
  samples1 = lds.sample(key)
  samples2 = lds_updated.sample(key)
  # import pdb; pdb.set_trace()

  import pdb; pdb.set_trace()