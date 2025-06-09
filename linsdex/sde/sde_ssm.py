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
from linsdex.potential.gaussian.dist import *
from linsdex.potential.gaussian.transition import *
from linsdex.matrix.matrix_with_inverse import MatrixWithInverse
from linsdex.sde.sde_base import AbstractLinearSDE, AbstractLinearTimeInvariantSDE
from plum import dispatch
import linsdex.util as util
from linsdex.ssm import AbstractStateSpaceModel
from linsdex.ssm.lds import LinearDynamicalSystem
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE

class StochasticSSM(AbstractStateSpaceModel):
  """A state space model with stochastic transitions and emissions."""

  t0: Float[Array, 'D']
  prior: AbstractPotential

  sde: AbstractLinearSDE
  emissions: GaussianTransition
  ts: Float[Array, 'N'] # The times where we have emissions

  parallel: bool = eqx.field(static=True)

  def __init__(
    self,
    sde: AbstractLinearSDE,
    prior: AbstractPotential,
    ts: Float[Array, 'N'],
    emissions: GaussianTransition,
    parallel: Optional[bool] = None
  ):
    self.sde = sde
    self.t0 = ts[0]
    self.prior = prior
    self.times = ts

    assert emissions.batch_size == ts.shape[0], "This model assumes that we have different emissions for each time point"
    self.emissions = emissions
    if parallel is None:
      parallel = jax.devices()[0].platform == 'gpu'
    self.parallel = parallel

  @property
  def transitions(self):
    # Construct the transitions
    s, t = self.times[:-1], self.times[1:]
    def make_transition_potential(s, t):
      return self.sde.get_transition_distribution(s, t)
    return eqx.filter_vmap(make_transition_potential)(s, t)

################################################################################################################

def em_update(sde_ssm: StochasticSSM, yts: Float[Array, 'B N Dy']) -> StochasticSSM:
  """Update sde using EM given the data yts.  This function makes the HUGE assumption that
  sde_ssm.times is a regular grid of times!!!!"""

  # Check that the time intervals are uniform
  dt = jnp.diff(sde_ssm.times)
  dt_std = jnp.std(dt)
  if dt_std > 1e-10:
    raise ValueError("Time intervals must be uniform for EM updates")

  transition = sde_ssm.sde.get_transition_distribution(sde_ssm.times[0], sde_ssm.times[1])

  from linsdex.ssm.lds import lds_e_step, lds_m_update
  lds = LinearDynamicalSystem(sde_ssm.prior, transition, sde_ssm.emissions)
  stats = lds_e_step(lds, yts)
  updated_lds = lds_m_update(lds, stats)

  import pdb; pdb.set_trace()



  def get_joint_distributions(yts):
    cond_sde = sde_ssm.get_posterior(yts)
    crf = cond_sde.discretize()
    return crf.get_joints()

  joints = jax.vmap(get_joint_distributions)(yts)
  import pdb; pdb.set_trace()

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *
  from linsdex.sde.sde_base import linear_sde_test
  import matplotlib.pyplot as plt
  import linsdex.util as util
  from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian, NaturalGaussian
  from linsdex.timeseries import TimeSeries
  from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior
  from linsdex.ssm.simple_decoder import PaddingLatentVariableDecoder
  from linsdex.sde import *
  from linsdex.matrix import *
  jax.config.update('jax_enable_x64', True)

  import pickle
  series = pickle.load(open('series.pkl', 'rb'))[:10]
  ts = jnp.array(series.times)*1.0
  yts = jnp.array(series.values)[...,:2]
  N = ts.shape[0]


  y_dim = yts.shape[-1]
  sde = BrownianMotion(sigma=0.1, dim=y_dim)
  # sde = CriticallyDampedLangevinDynamics(mass=0.1, beta=0.1, dim=y_dim)
  # sde = HigherOrderTrackingModel(sigma=0.1, position_dim=y_dim, order=2)
  x_dim = sde.dim

  # Construct some parameters for the model
  key = random.PRNGKey(0)
  k1, k2, k3 = random.split(key, 3)
  H = random.normal(k1, (N, x_dim, x_dim))
  u = random.normal(k2, (N, x_dim))
  R = random.normal(k3, (N, x_dim, x_dim))

  def make_transition(H, u, R):
    return GaussianTransition(util.to_matrix(H), u, util.to_matrix(R, symmetric=True))
  emissions = eqx.filter_vmap(make_transition)(H, u, R)

  # Construct the prior
  mat = emissions.A[0]
  I = mat.eye(mat.shape[-1])
  prior = StandardGaussian(mu=jnp.zeros(x_dim), Sigma=I)

  sde_ssm = StochasticSSM(sde, prior, ts, emissions, parallel=True)

  em_update(sde_ssm, yts)

  import pdb; pdb.set_trace()