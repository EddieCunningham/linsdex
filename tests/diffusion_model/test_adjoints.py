import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import pytest

from typing import Callable, Tuple

from jaxtyping import Array, Float, Scalar, PRNGKeyArray
from linsdex.diffusion_model.adjoints import (
  sde_simulation_with_internals,
  ode_simulation_with_internals,
  adjoint_simulation_from_sim_internals,
  SimulationState,
  AdjointSimulationState,
)


def make_linear_drift_term() -> diffrax.ODETerm:
  def vf(t: Scalar, x: Float[Array, "D"], theta: Float[Array, "1"]) -> Float[Array, "D"]:
    return theta * x
  return diffrax.ODETerm(vf)


class ConstCholesky:
  def __init__(self, L: Array):
    self._L = L
  def as_matrix(self) -> Array:
    return self._L
  @property
  def T(self):
    return self
  def solve(self, v: Array) -> Array:
    return jnp.linalg.solve(self._L.T, v)


def make_const_diffusion_fn(sigma: float, dim: int) -> Callable:
  L = jnp.eye(dim) * sigma
  chol = ConstCholesky(L)
  def diffusion_fn(t: Scalar, x: Float[Array, "D"], theta) -> Array:
    return chol.as_matrix()
  return diffusion_fn


def running_cost_fn(t: Scalar, x: Float[Array, "D"], theta: Float[Array, "1"]) -> Scalar:
  # Simple quadratic running cost in x plus small theta regularizer
  return 0.5 * jnp.sum(x**2) + 0.1 * jnp.sum(theta**2)


def terminal_cost_fn(xT: Float[Array, "D"]) -> Scalar:
  return 0.5 * jnp.sum(xT**2)


def total_loss_from_sim(sim: SimulationState, theta: Float[Array, "1"]) -> Scalar:
  ts = sim.ts
  xs = sim.xts
  dt = ts[1] - ts[0]
  running = jnp.sum(jax.vmap(lambda x, t: running_cost_fn(t, x, theta))(xs[1:], ts[1:])) * dt
  term = terminal_cost_fn(xs[-1])
  return running + term


def test_sde_simulation_manual_shapes():
  dim = 2
  theta = jnp.array(0.3)
  x0 = jnp.ones((dim,))
  key = jax.random.PRNGKey(0)
  t0, t1 = 0.0, 1.0
  n_steps = 16

  solver = diffrax.ShARK()
  drift_term = make_linear_drift_term()
  diffusion_fn = make_const_diffusion_fn(0.2, dim)

  sim = sde_simulation_with_internals(
    solver=solver,
    x0=x0,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta,
  )

  assert sim.ts.shape == (n_steps + 1,)
  assert sim.xts.shape == (n_steps + 1, dim)


def test_ode_simulation_manual_shapes_and_adjoint():
  dim = 2
  theta = jnp.array(0.3)
  x0 = jnp.ones((dim,))
  key = jax.random.PRNGKey(0)
  t0, t1 = 0.0, 1.0
  n_steps = 512

  solver = diffrax.Dopri5()
  drift_term = make_linear_drift_term()

  sim = ode_simulation_with_internals(
    solver=solver,
    x0=x0,
    ode_fn=drift_term,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta,
  )

  assert sim.ts.shape == (n_steps + 1,)
  assert sim.xts.shape == (n_steps + 1, dim)

  # Check equality to autodiff through the simulation (discretize-then-optimise)
  def loss_wrt_x0_fresh(x_init):
    sim2 = ode_simulation_with_internals(
      solver=solver,
      x0=x_init,
      ode_fn=drift_term,
      t0=t0,
      t1=t1,
      n_steps=n_steps,
      key=key,
      args=theta,
    )
    return total_loss_from_sim(sim2, theta)

  dL_dx0_fresh = jax.grad(loss_wrt_x0_fresh)(x0)
  adj = adjoint_simulation_from_sim_internals(sim, terminal_cost_fn, running_cost_fn, theta)
  assert jnp.allclose(adj.ats[0], dL_dx0_fresh, rtol=1e-6, atol=1e-8)

  # Intentionally only compare to autodiff through a fresh simulation

  # Cost trajectory checks
  assert adj.costs.shape == sim.ts.shape
  assert jnp.allclose(adj.costs[-1], terminal_cost_fn(sim.xts[-1]), rtol=1e-6, atol=1e-8)
  L_manual = total_loss_from_sim(sim, theta)
  assert jnp.allclose(adj.total_cost, L_manual, rtol=1e-6, atol=1e-8)

def test_adjoint_manual_matches_autodiff():
  dim = 2
  theta = jnp.array(0.3)
  x0 = jnp.array([0.7, -0.4])
  key = jax.random.PRNGKey(1)
  t0, t1 = 0.0, 1.0
  n_steps = 32

  solver = diffrax.ShARK()
  drift_term = make_linear_drift_term()
  diffusion_fn = make_const_diffusion_fn(0.25, dim)

  sim = sde_simulation_with_internals(
    solver=solver,
    x0=x0,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta,
  )

  # Define total loss for autodiff comparison
  def loss_wrt_x0(x_init):
    sim2 = sde_simulation_with_internals(
      solver=solver,
      x0=x_init,
      drift_fn=drift_term,
      diffusion_fn=diffusion_fn,
      t0=t0,
      t1=t1,
      n_steps=n_steps,
      key=key,
      args=theta,
    )
    return total_loss_from_sim(sim2, theta)

  def loss_wrt_theta(th):
    sim2 = sde_simulation_with_internals(
      solver=solver,
      x0=x0,
      drift_fn=drift_term,
      diffusion_fn=diffusion_fn,
      t0=t0,
      t1=t1,
      n_steps=n_steps,
      key=key,
      args=th,
    )
    return total_loss_from_sim(sim2, th)

  # Compute autodiff grads
  dL_dx0 = jax.grad(loss_wrt_x0)(x0)
  dL_dtheta = jax.grad(loss_wrt_theta)(theta)

  # Compute manual adjoint (returns AdjointSimulationState)
  adj = adjoint_simulation_from_sim_internals(sim, terminal_cost_fn, running_cost_fn, theta)
  assert isinstance(adj, AdjointSimulationState)

  # ats aligns with ascending time; a0 should be first element
  a0 = adj.ats[0]

  assert jnp.allclose(a0, dL_dx0, rtol=1e-5, atol=1e-6)
  # grad_theta may be a PyTree; reduce if scalar
  if isinstance(adj.grad_theta, dict) or isinstance(adj.grad_theta, tuple):
    gtheta_flat, _ = jax.tree_util.tree_flatten(adj.grad_theta)
    dtheta_flat, _ = jax.tree_util.tree_flatten(dL_dtheta)
    for g1, g2 in zip(gtheta_flat, dtheta_flat):
      assert jnp.allclose(g1, g2, rtol=1e-5, atol=1e-6)
  else:
    assert jnp.allclose(adj.grad_theta, dL_dtheta, rtol=1e-5, atol=1e-6)

  # Consistency checks on alignment:
  # - Last adjoint state corresponds to terminal time t1, so equals ∂Φ/∂x_T
  aT = jax.grad(terminal_cost_fn)(sim.xts[-1])
  assert jnp.allclose(adj.ats[-1], aT, rtol=1e-6, atol=1e-8)
  # - First adjoint state is at t0, matches ∂L/∂x0 (already checked as a0)
  assert adj.ts.shape == adj.xts.shape[:1] == adj.ats.shape[:1]
  # - Costs length matches and is cost-to-go: final equals terminal cost
  assert adj.costs.shape == adj.ts.shape
  assert jnp.allclose(adj.costs[-1], terminal_cost_fn(sim.xts[-1]), rtol=1e-6, atol=1e-8)
  # - total_cost is the earliest-time element
  assert jnp.allclose(adj.total_cost, adj.costs[0], rtol=1e-6, atol=1e-8)


def test_total_cost_fd_gradient_matches_adjoint():
  dim = 2
  theta = jnp.array(0.3)
  x0 = jnp.array([0.5, -0.2])
  key = jax.random.PRNGKey(42)
  t0, t1 = 0.0, 1.0
  n_steps = 64

  solver = diffrax.ShARK()
  drift_term = make_linear_drift_term()
  diffusion_fn = make_const_diffusion_fn(0.2, dim)

  # Base run
  sim0 = sde_simulation_with_internals(
    solver=solver,
    x0=x0,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta,
  )
  adj0 = adjoint_simulation_from_sim_internals(sim0, terminal_cost_fn, running_cost_fn, theta)

  # Check that reported total_cost matches manual computation on the same grid
  L_manual = total_loss_from_sim(sim0, theta)
  assert jnp.allclose(adj0.total_cost, L_manual, rtol=1e-6, atol=1e-8)
  assert jnp.allclose(adj0.costs[0], L_manual, rtol=1e-6, atol=1e-8)

  # Finite-difference directional derivative w.r.t. x0
  v = jnp.array([1.0, -1.0])
  v = v / jnp.linalg.norm(v)
  eps = 1e-4

  sim_plus = sde_simulation_with_internals(
    solver=solver,
    x0=x0 + eps * v,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta,
  )
  L_plus = adjoint_simulation_from_sim_internals(sim_plus, terminal_cost_fn, running_cost_fn, theta).total_cost

  sim_minus = sde_simulation_with_internals(
    solver=solver,
    x0=x0 - eps * v,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta,
  )
  L_minus = adjoint_simulation_from_sim_internals(sim_minus, terminal_cost_fn, running_cost_fn, theta).total_cost

  fd_dir_x0 = (L_plus - L_minus) / (2 * eps)
  true_dir_x0 = jnp.dot(adj0.ats[0], v)
  assert jnp.allclose(fd_dir_x0, true_dir_x0, rtol=5e-3, atol=5e-4)

  # Finite-difference for theta (scalar)
  u = jnp.array(1.0)
  sim_plus_th = sde_simulation_with_internals(
    solver=solver,
    x0=x0,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta + eps * u,
  )
  L_plus_th = adjoint_simulation_from_sim_internals(sim_plus_th, terminal_cost_fn, running_cost_fn, theta + eps * u).total_cost

  sim_minus_th = sde_simulation_with_internals(
    solver=solver,
    x0=x0,
    drift_fn=drift_term,
    diffusion_fn=diffusion_fn,
    t0=t0,
    t1=t1,
    n_steps=n_steps,
    key=key,
    args=theta - eps * u,
  )
  L_minus_th = adjoint_simulation_from_sim_internals(sim_minus_th, terminal_cost_fn, running_cost_fn, theta - eps * u).total_cost

  fd_theta = (L_plus_th - L_minus_th) / (2 * eps)
  if isinstance(adj0.grad_theta, dict) or isinstance(adj0.grad_theta, tuple):
    gtheta_flat, _ = jax.tree_util.tree_flatten(adj0.grad_theta)
    true_theta = sum([jnp.sum(x) for x in gtheta_flat])
  else:
    true_theta = jnp.sum(adj0.grad_theta)
  assert jnp.allclose(fd_theta, true_theta, rtol=5e-3, atol=5e-4)


if __name__ == "__main__":
  from debug import *
  test_sde_simulation_manual_shapes()
  test_adjoint_manual_matches_autodiff()
