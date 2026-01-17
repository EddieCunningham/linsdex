import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type, TYPE_CHECKING, List
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
from functools import wraps
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

if TYPE_CHECKING:
  from linsdex.sde.ode_sde_simulation import ODESolverParams
  from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE

__all__ = ['AbstractSDE',
           'AbstractLinearSDE',
           'AbstractLinearTimeInvariantSDE',
           'TimeScaledLinearTimeInvariantSDE',
           'vectorize_sde_transition']

################################################################################################################

def vectorize_sde_transition(f: Callable):
  """Decorator that automatically handles multiple start times `s` for SDE transition distributions.

  If `s` is an array, it applies `eqx.filter_vmap` to the function call.

  Args:
    f: The function to be vectorized. It must take `s` and `t` as arguments.

  Returns:
    A wrapped function that automatically handles vector inputs for `s`.
  """
  @wraps(f)
  def f_wrapper(self, s: Union[Scalar, Float[Array, 'N']], t: Scalar, *args, **kwargs):
    if jnp.ndim(s) > 0:
      return eqx.filter_vmap(lambda s_val: f(self, s_val, t, *args, **kwargs))(s)
    return f(self, s, t, *args, **kwargs)
  return f_wrapper

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
    s: Union[Scalar, Float[Array, 'N']],
    t: Scalar,
    ode_solver_params: Optional["ODESolverParams"] = None,
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

    - `s`: The start time(s). If this is an array, we assume it is sorted ascending
           and all values are less than t.
    - `t`: The end time

    **Returns**

    - `GaussianTransition`: The transition distribution(s). If s is an array,
                            this will be a batched GaussianTransition object.
    """
    # Call get params to get the data types. If s is a vector, use the first element
    s_is_scalar = jnp.ndim(s) == 0
    s_val = s if s_is_scalar else s[0]
    F, u, L = self.get_params(s_val)
    I = (F.T@L).set_eye() # Initialize with identity matrix.  Do it this way to get the right data type

    D = self.dim
    A_TT = I # Initialize with identity matrix
    uT = jnp.zeros(D)
    SigmaT = I.set_zero() # Initialize with 0 matrix

    # Remove the tags from the matrices so that we avoid symbolic computation
    A_TT = eqx.tree_at(lambda x: x.tags, A_TT, TAGS.no_tags)
    SigmaT = eqx.tree_at(lambda x: x.tags, SigmaT, TAGS.no_tags)

    # Initialize the ODE state
    yT = (A_TT, uT, SigmaT)

    # The ODE solver should not try to update the tags, so need to partition
    yT_params, yT_static = eqx.partition(yT, eqx.is_inexact_array)

    def reverse_dynamics(tau, ytau_params):
      ytau = eqx.combine(ytau_params, yT_static)
      A_Ttau, _, _ = ytau

      Ftau, utau, L = self.get_params(tau)
      LLT = L@L.T

      dA_Ttau = -A_Ttau@Ftau
      du = -A_Ttau@utau # The negative sign comes from reversing the ODE
      dSigma = -A_Ttau@LLT@A_Ttau.T # The negative sign comes from reversing the ODE

      dytau = (dA_Ttau, du, dSigma)
      dytau_params, _ = eqx.partition(dytau, eqx.is_inexact_array)
      return dytau_params

    # Solve the ODE backwards in time
    from linsdex.sde.ode_sde_simulation import ode_solve
    from linsdex.series.series import TimeSeries
    if s_is_scalar:
      save_times = jnp.array([t, s])
    else:
      # Assume s is sorted ascending and all less than t.
      # ode_solve requires monotonic save_times. Since we solve backwards,
      # we go from t down to min(s).
      save_times = jnp.concatenate([jnp.array([t]), s[::-1]])

    if ode_solver_params is None:
      from linsdex.sde.ode_sde_simulation import ODESolverParams
      ode_solver_params = ODESolverParams()

    # ode_solve returns a TimeSeries object if the state is an array,
    # or a pytree of arrays if the state is a pytree.
    res = ode_solve(reverse_dynamics, yT_params, save_times, params=ode_solver_params)
    if isinstance(res, TimeSeries):
      y0_params_raw = res.values
    else:
      y0_params_raw = res # This is a pytree of (num_save_times, ...) arrays

    # Extract all points except the first one (which is t)
    y0_params = jtu.tree_map(lambda x: x[1:], y0_params_raw)

    # If s was a vector, we need to reverse back to match original s order
    if not s_is_scalar:
      y0_params = jtu.tree_map(lambda x: x[::-1], y0_params)

    # Combine with static data. We need to broadcast static parts if s is a vector
    # so that eqx.filter_vmap works correctly later.
    if s_is_scalar:
      y0_params = jtu.tree_map(lambda x: x[0], y0_params)
      y0 = eqx.combine(y0_params, yT_static)
    else:
      batch_size_val = jtu.tree_leaves(y0_params)[0].shape[0]
      def broadcast_static(x):
        return jnp.broadcast_to(x, (batch_size_val,) + x.shape) if eqx.is_array(x) else x
      yT_static_batched = jtu.tree_map(broadcast_static, yT_static)
      y0 = eqx.combine(y0_params, yT_static_batched)

    A, u, Sigma = y0

    # If all of Sigma has elements close to 0, symbolically set it to 0.
    def finalize_sigma(S):
      return util.where(jnp.abs(S.elements).max() < 1e-8, S.set_zero(), S)

    if s_is_scalar:
      Sigma = finalize_sigma(Sigma)
      return GaussianTransition(A, u, Sigma)
    else:
      # Use filter_vmap to construct batched GaussianTransition
      Sigma = eqx.filter_vmap(finalize_sigma)(Sigma)
      return eqx.filter_vmap(GaussianTransition)(A, u, Sigma)

  def condition_on(self, evidence: GaussianPotentialSeries) -> "ConditionedLinearSDE":
    from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
    return ConditionedLinearSDE(self, evidence)

  def condition_on_starting_point(self, t0: Scalar, x0: Float[Array, 'D']) -> "ConditionedLinearSDE":
    from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
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

  @vectorize_sde_transition
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

      if jnp.iscomplexobj(Phi):
        Phi = Phi.real

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
  r"""If sde represents dx_s = Fx_s ds + LdW_s, then this represents reparametrizing time
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
