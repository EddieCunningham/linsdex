import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import lineax as lx
import abc
import warnings
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix, TAGS
from linsdex.matrix.matrix_with_inverse import MatrixWithInverse
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.block.block_3x3 import Block3x3Matrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.potential.gaussian import GaussianTransition
from plum import dispatch
import linsdex.util as util
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries

__all__ = ['AbstractSDE',
           'AbstractLinearSDE',
           'AbstractLinearTimeInvariantSDE',
           'TimeScaledLinearTimeInvariantSDE']

################################################################################################################

class AbstractSDE(AbstractBatchableObject, abc.ABC):
  """An abstract SDE does NOT support sampling.  We need to incorporate a potential (or initial point)."""

  @abc.abstractmethod
  def get_drift(self, t: Scalar,  xt: Float[Array, 'D']) -> Float[Array, 'D']:
    pass

  @abc.abstractmethod
  def get_diffusion_coefficient(self, t: Scalar, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    pass

  def get_transition_distribution(self, s: Scalar, t: Scalar) -> AbstractTransition:
    raise NotImplementedError

################################################################################################################

class AbstractLinearSDE(AbstractSDE, abc.ABC):

  @abc.abstractmethod
  def get_params(self, t: Scalar) -> Tuple[AbstractSquareMatrix,
                                           Float[Array, 'D'],
                                           AbstractSquareMatrix]:
    """Get F, u, and L at time t
    """
    pass

  def get_diffusion_coefficient(self, t: Scalar, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    _, _, L = self.get_params(t)
    return L

  def get_drift(self, t: Scalar, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    F, u, _ = self.get_params(t)
    return F@xt + u

  def get_transition_distribution(
    self,
    s: Scalar,
    t: Scalar,
    ode_solver_params: Optional['ODESolverParams'] = None,
  ) -> GaussianTransition:
    """Get the transition parameters from time s to time t. See section
    6.1 of Särkkä's book (https://users.aalto.fi/~asolin/sde-book/sde-book.pdf)
    for the math details.  This class solves everything in reverse in order to
    only need to solve one ODE.

    TODO: Speed this up with a parallel algorithm.

    This solves for the transition parameters, A_{t,s}, u_{t,s}, and Sigma_{t,s} so that
    for a starting point x_s, the transition distribution p(x_t | x_s) is Gaussian
    N(x_t | A_{t,s} x_s + u_{t,s}, Sigma_{t,s})

    **Arguments**

    - `s`: The start time
    - `t`: The end time

    **Returns**

    - `A`: The transition matrix
    - `u`: The input
    - `Sigma`: The transition covariance
    """
    # Call get params to get the data types
    F, u, L = self.get_params(s)
    I = (F.T@L).set_eye() # Initialize with identity matrix.  Do it this way to get the right data type

    D = self.dim
    psi_TT = I # Initialize with identity matrix
    uT = jnp.zeros(D)
    SigmaT = I.set_zero() # Initialize with 0 matrix

    # Remove the tags from the matrices so that we avoid symbolic computation
    psi_TT = eqx.tree_at(lambda x: x.tags, psi_TT, TAGS.no_tags)
    SigmaT = eqx.tree_at(lambda x: x.tags, SigmaT, TAGS.no_tags)

    # Initialize the ODE state
    yT = (psi_TT, uT, SigmaT)

    # The ODE solver should not try to update the tags, so need to partition
    yT_params, yT_static = eqx.partition(yT, eqx.is_inexact_array)

    def reverse_dynamics(tau, ytau_params):
      ytau = eqx.combine(ytau_params, yT_static)
      psi_Ttau, _, _ = ytau

      Ftau, utau, L = self.get_params(tau)
      LLT = L@L.T

      dpsi_Ttau = -psi_Ttau@Ftau
      du = -psi_Ttau@utau # The negative sign comes from reversing the ODE
      dSigma = -psi_Ttau@LLT@psi_Ttau.T # The negative sign comes from reversing the ODE

      dytau = (dpsi_Ttau, du, dSigma)
      dytau_params, _ = eqx.partition(dytau, eqx.is_inexact_array)
      return dytau_params

    # Solve the ODE backwards in time
    from linsdex.sde.ode_sde_simulation import ode_solve, ODESolverParams
    save_times = jnp.array([t, s])
    if ode_solver_params is None:
      ode_solver_params = ODESolverParams()
    y0_params = ode_solve(reverse_dynamics, yT_params, save_times, params=ode_solver_params)

    # Extract the final time point and combine with the static data
    y0_params = jtu.tree_map(lambda x: x[-1], y0_params)
    y0 = eqx.combine(y0_params, yT_static)
    A, u, Sigma = y0

    # If all of Sigma has elements close to 0, symbolically set it to 0.
    Sigma = util.where(jnp.abs(Sigma.elements).max() < 1e-8, Sigma.set_zero(), Sigma)

    return GaussianTransition(A, u, Sigma)

  def condition_on(self, evidence: GaussianPotentialSeries) -> 'ConditionedLinearSDE':
    from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
    return ConditionedLinearSDE(self, evidence)

  def condition_on_starting_point(self, t0: Scalar, x0: Float[Array, 'D']) -> 'ConditionedLinearSDE':
    t0 = jnp.array(t0)
    evidence = GaussianPotentialSeries(t0, x0)
    return self.condition_on(evidence)

################################################################################################################

class AbstractLinearTimeInvariantSDE(AbstractLinearSDE, abc.ABC):

  F: eqx.AbstractVar[AbstractSquareMatrix]
  L: eqx.AbstractVar[AbstractSquareMatrix]

  @property
  def u(self) -> Float[Array, 'D']:
    return jnp.zeros((self.dim,))

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.F.batch_size

  @property
  def dim(self) -> int:
    return self.F.shape[0]

  def get_params(self, t: Scalar) -> Tuple[AbstractSquareMatrix,
                                           Float[Array, 'D'],
                                           AbstractSquareMatrix]:
    return self.F, self.u, self.L

  def get_transition_distribution(self,
                           s: Scalar,
                           t: Scalar) -> GaussianTransition:
    """Compute the covariance of the transition distribution from
       time s to time t.  We'll use the matrix fraction decomposition
       approach from section 6.3 of Särkkä's book (also detailed here
       https://arxiv.org/pdf/2302.07261)
    """
    D = self.F.shape[0]
    dt = t - s

    if isinstance(self.L, DiagonalMatrix):

      if isinstance(self.F, DiagonalMatrix):
        # This is simple
        A = (self.F*dt).get_exp()
        AinvT = (-self.F.T*dt).get_exp()
        A = MatrixWithInverse(A, AinvT.T)
        Sigma_AinvT = self.L@self.L.T*dt
        Sigma = self.L@self.L.T@A.T*dt

      elif isinstance(self.F, Block2x2Matrix) or \
           isinstance(self.F, Block3x3Matrix) or \
           isinstance(self.F, DenseMatrix) or \
           isinstance(self.F, Block2x2Matrix):

        zero = jnp.zeros((D, D))
        F, L = self.F.as_matrix(), self.L.as_matrix()
        X = jnp.block([[F, L@L.T], [zero, -F.T]])*dt
        Phi = jax.scipy.linalg.expm(X)

        A = Phi[:D,:D] # Top left
        Sigma_AinvT = Phi[:D,D:] # Top right
        Sigma = Sigma_AinvT@A.T # Bottom left
        AinvT = Phi[D:,D:] # Bottom right

        A = DenseMatrix(A, tags=TAGS.no_tags)
        AinvT = DenseMatrix(AinvT, tags=TAGS.no_tags)
        Sigma = DenseMatrix(Sigma, tags=TAGS.no_tags)

        if isinstance(self.F, Block2x2Matrix) or isinstance(self.F, Block3x3Matrix):
          # Don't have a proof for this, but seems like if L is diagonal and F has diagonal blocks,
          # then A and Sigma also have diagonal blocks.
          # TODO: Find proof of correctness and figure out more efficient implementation that doesn't require
          # taking matrix exponential of full X matrix.
          A = self.F.project_dense(A)
          AinvT = self.F.project_dense(AinvT)
          Sigma = self.F.project_dense(Sigma)

        A = MatrixWithInverse(A, AinvT.T)

      else:
        raise ValueError('Invalid F type')

    else:
      zero = jnp.zeros((D, D))
      F, L = self.F.as_matrix(), self.L.as_matrix()
      X = jnp.block([[F, L@L.T], [zero, -F.T]])*dt
      Phi = jax.scipy.linalg.expm(X)

      A = Phi[:D,:D] # Top left
      Sigma_AinvT = Phi[:D,D:] # Top right
      Sigma = Sigma_AinvT@A.T # Bottom left
      AinvT = Phi[D:,D:] # Bottom right

      A = DenseMatrix(A, tags=TAGS.no_tags)
      AinvT = DenseMatrix(AinvT, tags=TAGS.no_tags)
      A = MatrixWithInverse(A, AinvT.T)
      Sigma = DenseMatrix(Sigma, tags=TAGS.no_tags)

    u = jnp.zeros((D,))
    Sigma = Sigma.set_symmetric()

    # If all of Sigma has elements close to 0, symbolically set it to 0.
    Sigma = util.where(jnp.abs(Sigma.elements).mean() < 1e-8, Sigma.set_zero(), Sigma)

    return GaussianTransition(A, u, Sigma)

################################################################################################################

class TimeScaledLinearTimeInvariantSDE(AbstractLinearTimeInvariantSDE):
  """If sde represents dx_s = Fx_s ds + LdW_s, then this represents reparametrizing time
  as t = \gamma*s and also x_s = \gamma*tilde{x}_s
  """

  sde: AbstractLinearTimeInvariantSDE
  time_scale: Scalar

  def __init__(self,
               sde: AbstractLinearTimeInvariantSDE,
               time_scale: Scalar):
    self.sde = sde
    self.time_scale = time_scale

  @property
  def F(self) -> Float[Array, 'D D']:
    return self.sde.F*self.time_scale

  @property
  def L(self) -> Float[Array, 'D D']:
    return self.sde.L*jnp.sqrt(self.time_scale)

  @property
  def order(self) -> int:
    """To be compatible with HigherOrderTracking.  There is definitely a better
    way to access member variables of self.sde"""
    try:
      return self.sde.order
    except AttributeError:
      raise AttributeError(f'SDE of type {type(self.sde)} does not have an order')

  def get_transition_distribution(self,
                                  s: Scalar,
                                  t: Scalar) -> GaussianTransition:
    return self.sde.get_transition_distribution(s*self.time_scale, t*self.time_scale)
