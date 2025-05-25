import jax
import jax.numpy as jnp
from jax import random
import pytest
import equinox as eqx

from linsdex.sde.ode_sde_simulation import (
  ODESolverParams,
  SDESolverParams,
  DiffraxSolverState,
  ode_solve,
  sde_sample,
  DummySDE
)
from linsdex.sde.sde_examples import BrownianMotion, OrnsteinUhlenbeck
from linsdex.series.series import TimeSeries
import linsdex.util as util


class TestDiffraxSolverState:
  """Test DiffraxSolverState container"""

  def test_initialization(self):
    """Test basic initialization of solver state"""
    state = DiffraxSolverState()
    assert state.solver_state is None
    assert state.controller_state is None

    # Test with custom states
    custom_solver_state = {"step": 0}
    custom_controller_state = {"dt": 0.01}
    state = DiffraxSolverState(
      solver_state=custom_solver_state,
      controller_state=custom_controller_state
    )
    assert state.solver_state == custom_solver_state
    assert state.controller_state == custom_controller_state


class TestODESolverParams:
  """Test ODE solver parameters"""

  def test_default_initialization(self):
    """Test default parameter values"""
    params = ODESolverParams()

    assert params.rtol == 1e-8
    assert params.atol == 1e-8
    assert params.solver == 'dopri5'
    assert params.adjoint == 'recursive_checkpoint'
    assert params.stepsize_controller == 'pid'
    assert params.max_steps == 8192
    assert params.throw is True
    assert params.progress_meter == 'tqdm'

  def test_custom_initialization(self):
    """Test custom parameter values"""
    params = ODESolverParams(
      rtol=1e-6,
      atol=1e-6,
      solver='euler',
      adjoint='direct',
      stepsize_controller='none',
      max_steps=1000,
      throw=False,
      progress_meter='none'
    )

    assert params.rtol == 1e-6
    assert params.atol == 1e-6
    assert params.solver == 'euler'
    assert params.adjoint == 'direct'
    assert params.stepsize_controller == 'none'
    assert params.max_steps == 1000
    assert params.throw is False
    assert params.progress_meter == 'none'

  def test_to_dict(self):
    """Test conversion to dictionary"""
    params = ODESolverParams()
    param_dict = params.to_dict()

    expected_keys = {
      'rtol', 'atol', 'solver', 'adjoint', 'stepsize_controller',
      'max_steps', 'throw', 'progress_meter'
    }
    assert set(param_dict.keys()) == expected_keys

  def test_using_constant_step_size(self):
    """Test constant step size detection"""
    # PID controller - not constant
    params_pid = ODESolverParams(stepsize_controller='pid')
    assert not params_pid.using_constant_step_size()

    # None controller - constant
    params_none = ODESolverParams(stepsize_controller='none')
    assert params_none.using_constant_step_size()

    # Constant controller - constant
    params_const = ODESolverParams(stepsize_controller='constant')
    assert params_const.using_constant_step_size()

  def test_get_solver(self):
    """Test solver instantiation"""
    # Test different solvers
    params_dopri5 = ODESolverParams(solver='dopri5')
    solver = params_dopri5.get_solver()
    assert solver.__class__.__name__ == 'Dopri5'

    params_euler = ODESolverParams(solver='euler')
    solver = params_euler.get_solver()
    assert solver.__class__.__name__ == 'Euler'

    params_kvaerno = ODESolverParams(solver='kvaerno5')
    solver = params_kvaerno.get_solver()
    assert solver.__class__.__name__ == 'Kvaerno5'

    # Test invalid solver
    params_invalid = ODESolverParams(solver='invalid_solver')
    with pytest.raises(ValueError, match="Unknown solver"):
      params_invalid.get_solver()

  def test_get_adjoint(self):
    """Test adjoint method instantiation"""
    params_recursive = ODESolverParams(adjoint='recursive_checkpoint')
    adjoint = params_recursive.get_adjoint()
    assert 'RecursiveCheckpoint' in adjoint.__class__.__name__

    params_direct = ODESolverParams(adjoint='direct')
    adjoint = params_direct.get_adjoint()
    assert 'DirectAdjoint' in adjoint.__class__.__name__

    # Test invalid adjoint
    params_invalid = ODESolverParams(adjoint='invalid_adjoint')
    with pytest.raises(ValueError, match="Unknown adjoint"):
      params_invalid.get_adjoint()

  def test_get_stepsize_controller(self):
    """Test stepsize controller instantiation"""
    params_pid = ODESolverParams(stepsize_controller='pid')
    controller = params_pid.get_stepsize_controller()
    assert 'PIDController' in controller.__class__.__name__

    params_none = ODESolverParams(stepsize_controller='none')
    controller = params_none.get_stepsize_controller()
    assert 'ConstantStepSize' in controller.__class__.__name__

  def test_get_progress_meter(self):
    """Test progress meter instantiation"""
    params_tqdm = ODESolverParams(progress_meter='tqdm')
    meter = params_tqdm.get_progress_meter()
    assert 'TqdmProgressMeter' in meter.__class__.__name__

    params_none = ODESolverParams(progress_meter='none')
    meter = params_none.get_progress_meter()
    assert 'NoProgressMeter' in meter.__class__.__name__


class TestSDESolverParams:
  """Test SDE solver parameters"""

  def test_default_initialization(self):
    """Test default parameter values"""
    params = SDESolverParams()

    assert params.rtol == 1e-8
    assert params.atol == 1e-8
    assert params.solver == 'shark'
    assert params.adjoint == 'recursive_checkpoint'
    assert params.stepsize_controller == 'none'
    assert params.max_steps == 8192
    assert params.throw is True
    assert params.progress_meter == 'none'
    assert params.brownian_simulation_type == 'virtual'

  def test_get_solver(self):
    """Test SDE solver instantiation"""
    params_shark = SDESolverParams(solver='shark')
    solver = params_shark.get_solver()
    assert 'ShARK' in solver.__class__.__name__

    params_euler = SDESolverParams(solver='euler_heun')
    solver = params_euler.get_solver()
    assert 'EulerHeun' in solver.__class__.__name__

    # Test invalid solver
    params_invalid = SDESolverParams(solver='invalid_solver')
    with pytest.raises(ValueError, match="Unknown solver"):
      params_invalid.get_solver()

  def test_get_brownian_simulation_type(self):
    """Test Brownian simulation type instantiation"""
    params = SDESolverParams()

    t0, t1 = 0.0, 1.0
    tol = 1e-3
    shape = (2,)
    key = random.PRNGKey(42)

    # Test virtual Brownian tree
    params_virtual = SDESolverParams(brownian_simulation_type='virtual')
    brownian = params_virtual.get_brownian_simulation_type(t0, t1, tol, shape, key)
    assert 'VirtualBrownianTree' in brownian.__class__.__name__

    # Test unsafe Brownian path
    params_unsafe = SDESolverParams(brownian_simulation_type='unsafe')
    brownian = params_unsafe.get_brownian_simulation_type(t0, t1, tol, shape, key)
    assert 'UnsafeBrownianPath' in brownian.__class__.__name__


class TestDummySDE:
  """Test dummy SDE wrapper"""

  def test_initialization(self):
    """Test dummy SDE initialization"""
    def flow_fn(t, xt):
      return -xt  # Simple exponential decay

    dummy_sde = DummySDE(flow_function=flow_fn)
    assert dummy_sde.flow_function is flow_fn

  def test_get_flow(self):
    """Test flow computation"""
    def flow_fn(t, xt):
      return -xt

    dummy_sde = DummySDE(flow_function=flow_fn)

    t = 1.0
    xt = jnp.array([1.0, 2.0])

    flow = dummy_sde.get_flow(t, xt)
    expected = -xt
    assert jnp.allclose(flow, expected)

class TestODESolve:
  """Test ODE solving functionality"""

  def test_ode_solve_with_function(self):
    """Test ODE solving with a function"""
    def dynamics(t, x):
      return -x  # dx/dt = -x, solution: x(t) = x0 * exp(-t)

    x0 = jnp.array([1.0, 2.0])
    save_times = jnp.array([0.0, 0.5, 1.0])

    params = ODESolverParams(max_steps=100)
    result = ode_solve(dynamics, x0, save_times, params)

    assert isinstance(result, TimeSeries)
    assert result.times.shape == save_times.shape
    assert result.values.shape == (len(save_times), len(x0))

    # Check analytical solution at t=1
    expected_at_1 = x0 * jnp.exp(-1.0)
    assert jnp.allclose(result.values[-1], expected_at_1, rtol=1e-3)

  def test_ode_solve_return_solution(self):
    """Test ODE solving with solution return"""
    def dynamics(t, x):
      return jnp.zeros_like(x)  # Constant solution

    x0 = jnp.array([1.0])
    save_times = jnp.array([0.0, 1.0])

    result, solution = ode_solve(dynamics, x0, save_times, return_solve_solution=True)

    assert isinstance(result, TimeSeries)
    # solution should be a diffrax.Solution object (can't test exact type without diffrax import)
    assert solution is not None


class TestSDESample:
  """Test SDE sampling functionality"""

  def test_sde_sample_basic(self):
    """Test basic SDE sampling"""
    sde = BrownianMotion(sigma=0.1, dim=2)

    x0 = jnp.array([0.0, 0.0])
    key = random.PRNGKey(42)
    save_times = jnp.array([0.0, 0.5, 1.0])

    params = SDESolverParams(max_steps=100)
    result = sde_sample(sde, x0, key, save_times, params)

    assert isinstance(result, TimeSeries)
    assert result.times.shape == save_times.shape
    assert result.values.shape == (len(save_times), len(x0))

    # First time point should be the initial condition
    assert jnp.allclose(result.values[0], x0)

  def test_sde_sample_deterministic(self):
    """Test SDE sampling with deterministic process (sigma=0)"""
    sde = OrnsteinUhlenbeck(sigma=0.0, lambda_=0.5, dim=1)

    x0 = jnp.array([1.0])
    key = random.PRNGKey(42)
    save_times = jnp.array([0.0, 1.0])

    params = SDESolverParams(max_steps=100)
    result = sde_sample(sde, x0, key, save_times, params)

    # Should be deterministic (analytical solution)
    expected_final = x0 * jnp.exp(-0.5 * 1.0)
    assert jnp.allclose(result.values[-1], expected_final, rtol=1e-3)

  def test_sde_sample_reproducibility(self):
    """Test that same key gives same result"""
    sde = BrownianMotion(sigma=1.0, dim=1)

    x0 = jnp.array([0.0])
    key = random.PRNGKey(123)
    save_times = jnp.array([0.0, 1.0])

    params = SDESolverParams(max_steps=100)

    result1 = sde_sample(sde, x0, key, save_times, params)
    result2 = sde_sample(sde, x0, key, save_times, params)

    assert jnp.allclose(result1.values, result2.values)

  def test_sde_sample_different_keys(self):
    """Test that different keys give different results"""
    sde = BrownianMotion(sigma=1.0, dim=1)

    x0 = jnp.array([0.0])
    key1 = random.PRNGKey(1)
    key2 = random.PRNGKey(2)
    save_times = jnp.array([0.0, 1.0])

    params = SDESolverParams(max_steps=100)

    result1 = sde_sample(sde, x0, key1, save_times, params)
    result2 = sde_sample(sde, x0, key2, save_times, params)

    # Results should be different (with high probability)
    assert not jnp.allclose(result1.values[1:], result2.values[1:], atol=1e-6)

  def test_sde_sample_return_solution(self):
    """Test SDE sampling with solution return"""
    sde = BrownianMotion(sigma=0.1, dim=1)

    x0 = jnp.array([0.0])
    key = random.PRNGKey(42)
    save_times = jnp.array([0.0, 1.0])

    result, solution = sde_sample(sde, x0, key, save_times, return_solve_solution=True)

    assert isinstance(result, TimeSeries)
    assert solution is not None


class TestConvergence:
  """Test numerical convergence"""

  def test_brownian_motion_variance(self):
    """Test that Brownian motion has correct variance scaling"""
    sigma = 1.0
    sde = BrownianMotion(sigma=sigma, dim=1)

    x0 = jnp.array([0.0])
    save_times = jnp.array([0.0, 1.0])

    # Generate multiple samples
    keys = random.split(random.PRNGKey(42), 1000)

    def sample_path(key):
      return sde_sample(sde, x0, key, save_times, SDESolverParams(max_steps=50))

    results = jax.vmap(sample_path)(keys)
    final_values = results.values[:, -1, 0]  # Final time, dimension 0

    # Variance should be approximately sigma^2 * T = 1.0 * 1.0 = 1.0
    empirical_variance = jnp.var(final_values)
    expected_variance = sigma**2 * 1.0

    assert jnp.abs(empirical_variance - expected_variance) < 0.1


if __name__ == "__main__":
  pytest.main([__file__])