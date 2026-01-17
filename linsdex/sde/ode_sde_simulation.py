import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
import diffrax
import plum
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, PyTree
from linsdex.sde.sde_base import AbstractSDE
from linsdex.series.series import TimeSeries
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import abc

"""
This module provides utilities for numerically solving and sampling from ODEs and SDEs.
It offers:

1. ODESolverParams and ode_solve - Configuration and solver for probability flow ODEs
   with support for various numerical integration schemes and adjoint methods

2. SDESolverParams and sde_sample - Configuration and sampler for stochastic differential
   equations, enabling generation of sample paths with controlled numerical accuracy

These functions leverage the diffrax library for high-performance differential equation
solving in JAX, with support for automatic differentiation, vectorization, and various
numerical integration schemes. The module is particularly useful for simulating the
dynamics of diffusion models in the time series domain.
"""

__all__ = ['ode_solve',
           'ODESolverParams',
           'SDESolverParams',
           'sde_sample',
           'DiffraxSolverState']

class DiffraxSolverState(eqx.Module):
  """
  Container for diffrax solver and controller state.

  This class stores the internal state of a diffrax solver and controller,
  allowing solves to be continued from where they left off.

  Attributes:
    solver_state: The internal state of the diffrax solver
    controller_state: The internal state of the diffrax step-size controller
  """
  solver_state: Optional[PyTree] = None
  controller_state: Optional[PyTree] = None

class AbstractSolverParams(eqx.Module, abc.ABC):
  rtol: eqx.AbstractVar[float]
  atol: eqx.AbstractVar[float]
  solver: eqx.AbstractVar[str]
  adjoint: eqx.AbstractVar[str]
  stepsize_controller: eqx.AbstractVar[str]
  max_steps: eqx.AbstractVar[int]
  throw: eqx.AbstractVar[bool]
  progress_meter: eqx.AbstractVar[str]

  def to_dict(self) -> dict:
    """Convert solver parameters to a dictionary.

    Returns:
      A dictionary containing all the solver parameters.
    """
    return {
      "rtol": self.rtol,
      "atol": self.atol,
      "solver": self.solver,
      "adjoint": self.adjoint,
      "stepsize_controller": self.stepsize_controller,
      "max_steps": self.max_steps,
      "throw": self.throw,
      "progress_meter": self.progress_meter
    }

  def using_constant_step_size(self) -> bool:
    return self.stepsize_controller == 'none' or self.stepsize_controller == 'constant' or self.stepsize_controller is None

  @abc.abstractmethod
  def get_solver(self) -> diffrax.AbstractSolver:
    pass

  @abc.abstractmethod
  def get_adjoint(self) -> diffrax.AbstractAdjoint:
    pass

  @abc.abstractmethod
  def get_stepsize_controller(self) -> diffrax.AbstractStepSizeController:
    pass

  @abc.abstractmethod
  def get_progress_meter(self) -> diffrax.AbstractProgressMeter:
    pass

  @abc.abstractmethod
  def get_terms(self, sde: AbstractSDE) -> diffrax.AbstractTerm:
    pass

  def initialize_solve_state(self,
                             sde: AbstractSDE,
                             x0: Float[Array, 'D'],
                             t0: Scalar,
                             t1: Scalar,
                             key: PRNGKeyArray) -> DiffraxSolverState:
    """Initialize the solver state for the ODE/SDE solver."""
    # Initialise states
    solver = self.get_solver()
    stepsize_controller = self.get_stepsize_controller()

    if self.using_constant_step_size():
      dt0 = (t1 - t0)/self.max_steps
    else:
      dt0 = 0.01

    terms, args = self.get_terms(sde, t0, t1, x0, key, dt0)

    error_order = solver.error_order(terms)
    (tnext, controller_state) = stepsize_controller.init(
        terms, t0, t1, x0, dt0, args, solver.func, error_order
    )

    tnext = jnp.minimum(tnext, t1)
    solver_state = solver.init(terms, t0, tnext, x0, args)

    return DiffraxSolverState(solver_state=solver_state,
                              controller_state=controller_state)


class ODESolverParams(AbstractSolverParams):
  """
  Configuration parameters for ODE solving with diffrax.

  This class encapsulates all parameters needed to configure diffrax
  for solving ordinary differential equations, including tolerances,
  step sizes, solver algorithms, and adjoint methods.

  Attributes:
    rtol: Relative tolerance for adaptive step size controllers
    atol: Absolute tolerance for adaptive step size controllers
    solver: Name of the diffrax solver to use ('dopri5', 'euler', 'kvaerno5')
    adjoint: Method for computing gradients ('recursive_checkpoint', 'direct')
    stepsize_controller: Controller for step size ('pid', 'none', 'constant')
    max_steps: Maximum number of steps the solver is allowed to take
    throw: Whether to throw an exception if max_steps is exceeded
    progress_meter: Type of progress meter to use ('tqdm', 'text', 'none')
  """
  rtol: float = 1e-8
  atol: float = 1e-8
  solver: str = 'dopri5'
  adjoint: str = 'recursive_checkpoint'
  stepsize_controller: str = 'pid'
  max_steps: int = 65536
  throw: bool = True
  progress_meter: str = 'none'

  def get_solver(self) -> diffrax.AbstractSolver:
    """
    Get the diffrax solver object based on the configured solver name.

    Returns:
      A diffrax solver instance for ODE integration

    Raises:
      ValueError: If the configured solver name is not recognized
    """
    if self.solver == 'dopri5':
      solver = diffrax.Dopri5()
    elif self.solver == 'euler':
      solver = diffrax.Euler()
    elif self.solver == 'kvaerno5':
      solver = diffrax.Kvaerno5()
    else:
      raise ValueError(f"Unknown solver: {self.solver}")

    return solver

  def get_adjoint(self) -> diffrax.AbstractAdjoint:
    """
    Get the diffrax adjoint method based on the configured adjoint name.

    The adjoint method determines how gradients are computed when using
    the solver with automatic differentiation.

    Returns:
      A diffrax adjoint method instance

    Raises:
      ValueError: If the configured adjoint name is not recognized
    """
    if self.adjoint == 'recursive_checkpoint':
      return diffrax.RecursiveCheckpointAdjoint()
    elif self.adjoint == 'direct':
      return diffrax.DirectAdjoint()
    else:
      raise ValueError(f"Unknown adjoint: {self.adjoint}")

  def get_stepsize_controller(self) -> diffrax.AbstractStepSizeController:
    """
    Get the diffrax step size controller based on the configured controller name.

    Step size controllers determine how the integration step size changes during solving.

    Returns:
      A diffrax step size controller instance

    Raises:
      ValueError: If the configured controller name is not recognized
    """
    if self.stepsize_controller == 'pid':
      return diffrax.PIDController(rtol=self.rtol, atol=self.atol)
    elif self.stepsize_controller == 'none' or self.stepsize_controller == 'constant' or self.stepsize_controller is None:
      return diffrax.ConstantStepSize()
    else:
      raise ValueError(f"Unknown stepsize controller: {self.stepsize_controller}")

  def get_progress_meter(self) -> diffrax.AbstractProgressMeter:
    """
    Get the diffrax progress meter based on the configured progress meter name.

    Progress meters display information about the solver's progress.

    Returns:
      A diffrax progress meter instance
    """
    if self.progress_meter == 'tqdm':
      return diffrax.TqdmProgressMeter()
    elif self.progress_meter == 'text':
      return diffrax.TextProgressMeter()
    else:
      return diffrax.NoProgressMeter()

  def get_terms(self, sde: AbstractSDE, *args, **kwargs) -> Tuple[diffrax.AbstractTerm, PyTree]:
    """
    Get the diffrax terms for the ODE.  Returns the wrapped dynamics and the parameters.
    """
    sde_params, sde_static = eqx.partition(sde, eqx.is_inexact_array)

    @diffrax.ODETerm
    def wrapped_dynamics(t, xts, sde_params):
      sde = eqx.combine(sde_params, sde_static)
      return sde.get_flow(t, xts)

    return wrapped_dynamics, sde_params

################################################################################################################

class SDESolverParams(AbstractSolverParams):
  """
  Configuration parameters for SDE simulation with diffrax.

  This class encapsulates all parameters needed to configure diffrax
  for simulating stochastic differential equations, including tolerances,
  step sizes, solver algorithms, and adjoint methods.

  Attributes:
    rtol: Relative tolerance for adaptive step size controllers
    atol: Absolute tolerance for adaptive step size controllers
    solver: Name of the diffrax solver to use ('shark', 'euler_heun', 'reversible_heun', etc.)
    adjoint: Method for computing gradients ('recursive_checkpoint', 'direct')
    stepsize_controller: Controller for step size ('pid', 'none', 'constant')
    max_steps: Maximum number of steps the solver is allowed to take
    throw: Whether to throw an exception if max_steps is exceeded
    progress_meter: Type of progress meter to use ('tqdm', 'text', 'none')
    brownian_simulation_type: Type of Brownian motion simulation to use ('unsafe', 'virtual')
  """
  rtol: float = 1e-8
  atol: float = 1e-8
  solver: str = 'shark'
  adjoint: str = 'recursive_checkpoint'
  stepsize_controller: str = 'none'
  max_steps: int = 65536
  throw: bool = True
  progress_meter: str = 'none'

  brownian_simulation_type: str = 'virtual'

  def get_brownian_simulation_type(self,
                                   t0: Scalar,
                                   t1: Scalar,
                                   tol: Scalar,
                                   shape: tuple[int, ...] | PyTree,
                                   key: Array,
                                   levy_area: type['BrownianIncrement'] | type['SpaceTimeLevyArea'] | type['SpaceTimeTimeLevyArea'] = 'BrownianIncrement') -> Union[diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]:
    """
    Get the diffrax Brownian motion object based on the configured Brownian motion type.
    """
    if self.brownian_simulation_type == 'unsafe':
      return diffrax.UnsafeBrownianPath(shape, key, levy_area)
    elif self.brownian_simulation_type == 'virtual':
      return diffrax.VirtualBrownianTree(t0, t1, tol, shape, key, levy_area)
    else:
      raise ValueError(f"Unknown Brownian motion simulation type: {self.brownian_simulation_type}")

  def get_solver(self) -> diffrax.AbstractSolver:
    """
    Get the diffrax solver object based on the configured solver name.

    Returns:
      A diffrax solver instance for SDE integration

    Raises:
      ValueError: If the configured solver name is not recognized
    """
    if self.solver == 'euler_heun':
      solver = diffrax.EulerHeun()
    elif self.solver == 'shark':
      solver = diffrax.ShARK()
    elif self.solver == 'reversible_heun':
      solver = diffrax.ReversibleHeun()
    elif self.solver == 'ito_milstein':
      solver = diffrax.ItoMilstein()
    elif self.solver == 'stratonovich_milstein':
      solver = diffrax.StratonovichMilstein()
    elif self.solver == 'spark':
      solver = diffrax.SPaRK()
    elif self.solver == 'sea':
      solver = diffrax.SEA()
    else:
      raise ValueError(f"Unknown solver: {self.solver}")

    return solver

  def get_adjoint(self) -> diffrax.AbstractAdjoint:
    """
    Get the diffrax adjoint method based on the configured adjoint name.

    The adjoint method determines how gradients are computed when using
    the solver with automatic differentiation.

    Returns:
      A diffrax adjoint method instance

    Raises:
      ValueError: If the configured adjoint name is not recognized
    """
    if self.adjoint == 'recursive_checkpoint':
      return diffrax.RecursiveCheckpointAdjoint()
    elif self.adjoint == 'direct':
      return diffrax.DirectAdjoint()
    else:
      raise ValueError(f"Unknown adjoint: {self.adjoint}")

  def get_stepsize_controller(self) -> diffrax.AbstractStepSizeController:
    """
    Get the diffrax step size controller based on the configured controller name.

    Step size controllers determine how the integration step size changes during solving.

    Returns:
      A diffrax step size controller instance

    Raises:
      ValueError: If the configured controller name is not recognized
    """
    if self.stepsize_controller == 'pid':
      return diffrax.PIDController(rtol=self.rtol, atol=self.atol)
    elif self.stepsize_controller == 'none' or self.stepsize_controller == 'constant' or self.stepsize_controller is None:
      return diffrax.ConstantStepSize()
    else:
      raise ValueError(f"Unknown stepsize controller: {self.stepsize_controller}")

  def get_progress_meter(self) -> diffrax.AbstractProgressMeter:
    """
    Get the diffrax progress meter based on the configured progress meter name.

    Progress meters display information about the solver's progress.

    Returns:
      A diffrax progress meter instance
    """
    if self.progress_meter == 'tqdm':
      return diffrax.TqdmProgressMeter()
    elif self.progress_meter == 'text':
      return diffrax.TextProgressMeter()
    else:
      return diffrax.NoProgressMeter()

  def get_terms(self,
                sde: AbstractSDE,
                t0: Scalar,
                t1: Scalar,
                x0: Float[Array, 'D'],
                key: PRNGKeyArray,
                dt0: Scalar) -> Tuple[diffrax.AbstractTerm, PyTree]:
    """
    Get the diffrax terms for the ODE.  Returns the wrapped dynamics and the parameters.
    """

    sde_params, sde_static = eqx.partition(sde, eqx.is_inexact_array)

    @diffrax.ODETerm
    def wrapped_drift(t, xt, sde_params):
      sde = eqx.combine(sde_params, sde_static)
      return sde.get_drift(t, xt)

    def diffusion_fn(t, xt, sde_params):
      sde = eqx.combine(sde_params, sde_static)
      return sde.get_diffusion_coefficient(t, xt).as_matrix()

    # Create a Brownian motion process for the SDE
    bm = self.get_brownian_simulation_type(t0=t0,
                                            t1=t1,
                                            tol=dt0/2,
                                            shape=x0.shape,
                                            key=key,
                                            levy_area=diffrax.SpaceTimeLevyArea)

    # Combine drift and diffusion terms into a single SDE term
    diff = diffrax.ControlTerm(diffusion_fn, bm)
    terms = diffrax.MultiTerm(wrapped_drift, diff)

    return terms, sde_params

################################################################################################################

def _ode_sde_solve(sde: AbstractSDE,
                   x0: Array,
                   key: Optional[PRNGKeyArray],
                   save_times: Array,
                   params: SDESolverParams = SDESolverParams(),
                   diffrax_solver_state: Optional[DiffraxSolverState] = DiffraxSolverState(),
                   return_solve_solution: Optional[bool] = False) -> Union[TimeSeries, Tuple[TimeSeries, DiffraxSolverState]]:
  if key is not None and key.ndim > 1:
    raise ValueError("Can only call this with a single key!  We need a unique key for every data point.")

  # Get the solver object
  solver = params.get_solver()

  # Get the adjoint object
  adjoint = params.get_adjoint()

  # Get the stepsize controller object
  stepsize_controller = params.get_stepsize_controller()

  # Get the saveat object and the initial time and time step
  saveat = diffrax.SaveAt(ts=save_times, solver_state=True, controller_state=True)
  t0 = save_times[0]
  t1 = save_times[-1]

  if params.using_constant_step_size():
    dt0 = (t1 - t0)/params.max_steps
  else:
    dt0 = 0.01*jnp.sign(t1 - t0)

  terms, args = params.get_terms(sde, t0, t1, x0, key, dt0)

  # Get the progress meter object
  progress_meter = params.get_progress_meter()

  # Solve the SDE
  sol = diffrax.diffeqsolve(terms,
                            solver,
                            t0,
                            t1,
                            dt0=dt0,
                            y0=x0,
                            args=args,
                            saveat=saveat,
                            adjoint=adjoint,
                            stepsize_controller=stepsize_controller,
                            max_steps=params.max_steps + 1,
                            throw=params.throw,
                            progress_meter=progress_meter,
                            solver_state=diffrax_solver_state.solver_state,
                            controller_state=diffrax_solver_state.controller_state)

  from linsdex.series.series import TimeSeries
  if isinstance(sol.ys, TimeSeries):
    out = sol.ys
  elif isinstance(sol.ys, jnp.ndarray):
    out = TimeSeries(save_times, sol.ys)
  else:
    out = sol.ys
  if return_solve_solution:
    updated_diffrax_solver_state = DiffraxSolverState(solver_state=sol.solver_state,
                                                      controller_state=sol.controller_state)
    return out, updated_diffrax_solver_state
  else:
    return out

################################################################################################################

class DummySDE(AbstractSDE):

  flow_function: Optional[Callable[[Scalar, Float[Array, 'D']], Float[Array, 'D']]] = None
  drift_function: Optional[Callable[[Scalar, Float[Array, 'D']], Float[Array, 'D']]] = None
  diffusion_function: Optional[Callable[[Scalar, Float[Array, 'D']], Float[Array, 'D']]] = None

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return None

  def get_flow(self, t: Scalar, xt: Float[Array, 'D']) -> Float[Array, 'D']:
    if self.flow_function is None:
      raise ValueError("flow_function is not defined in this DummySDE")
    return self.flow_function(t, xt)

  def get_drift(self, t: Scalar,  xt: Float[Array, 'D']) -> Float[Array, 'D']:
    if self.drift_function is None:
      raise ValueError("drift_function is not defined in this DummySDE")
    return self.drift_function(t, xt)

  def get_diffusion_coefficient(self, t: Scalar, xt: Float[Array, 'D']):
    if self.diffusion_function is None:
      raise ValueError("diffusion_function is not defined in this DummySDE")
    return self.diffusion_function(t, xt)

def ode_solve(sde: Union[AbstractSDE, Callable[[Scalar, Float[Array, 'D']], Float[Array, 'D']]],
              x0: PyTree,
              save_times: Array,
              params: ODESolverParams = ODESolverParams(),
              diffrax_solver_state: Optional[DiffraxSolverState] = DiffraxSolverState(),
              return_solve_solution: Optional[bool] = False) -> Union[TimeSeries, Tuple[TimeSeries, DiffraxSolverState]]:
  """Solve the probability flow ODE of the input SDE

  This function numerically integrates the probability flow ODE derived from an SDE.
  The probability flow ODE shares the same marginal distributions as the SDE but follows
  a deterministic path, making it useful for tasks like sampling from diffusion models.

  **Arguments**:

  - sde: The AbstractSDE instance defining the stochastic differential equation
  - x0: The initial condition of the ODE (state at time t0)
  - save_times: Array of times at which to save the solution
  - params: Parameters for the ODE solver
  - diffrax_solver_state: The state of the solver. If provided, the solver will continue from the last saved time
  - return_solve_solution: Whether to return the DiffraxSolverState object. This contains the solver
                           state which can be used to continue the solve from the last saved time

  **Returns**:

  - TimeSeries: The solution trajectory at the save times
  - (Optional) DiffraxSolverState: The full diffrax solver state object if return_solve_solution=True
  """
  if isinstance(sde, AbstractSDE):
    pass
  else:
    sde = DummySDE(sde, None, None)

  return _ode_sde_solve(sde, x0, None, save_times, params, diffrax_solver_state, return_solve_solution)

def sde_sample(sde: Union[AbstractSDE, Tuple[Callable[[Scalar, Float[Array, 'D']], Float[Array, 'D']], Callable[[Scalar, Float[Array, 'D']], AbstractSquareMatrix]]],
               x0: Array,
               key: PRNGKeyArray,
               save_times: Array,
               params: SDESolverParams = SDESolverParams(),
               diffrax_solver_state: Optional[DiffraxSolverState] = DiffraxSolverState(),
               return_solve_solution: Optional[bool] = False) -> Union[TimeSeries, Tuple[TimeSeries, DiffraxSolverState]]:
  """Sample from an SDE dx/dt = f(t, x) + g(t, x) dW_t.

  This function generates sample paths from a stochastic differential equation.
  It supports both scalar and vector SDEs, handles batched initial conditions,
  and can vectorize over multiple random keys.

  **Arguments**:

  - sde: The SDE to sample from. An instance of AbstractSDE that defines drift and diffusion
  - x0: The initial condition (state at time t0)
  - key: JAX random key for generating the Brownian motion
  - save_times: Array of times at which to save the solution
  - params: Configuration parameters for the SDE solver
  - diffrax_solver_state: The state of the solver. If provided, the solver will continue from the last saved time
  - return_solve_solution: Whether to return the DiffraxSolverState object. This contains the solver
                           state which can be used to continue the solve from the last saved time

  **Returns**:

  - TimeSeries: The sampled trajectory at the save times
  - (Optional) DiffraxSolverState: The full diffrax solver state object if return_solve_solution=True
  """
  if isinstance(sde, AbstractSDE):
    pass
  else:
    drift_fn, diffusion_fn = sde
    sde = DummySDE(None, drift_fn, diffusion_fn)

  return _ode_sde_solve(sde, x0, key, save_times, params, diffrax_solver_state, return_solve_solution)