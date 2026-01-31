import jax
import jax.numpy as jnp
from typing import Annotated, Optional, Tuple, Union, Callable
import equinox as eqx
import diffrax
from jaxtyping import Array, PRNGKeyArray
import jax.tree_util as jtu
from typing import TypeVar
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.diffusion_model.probability_path import DiffusionModelComponents, ProbabilityPathSlice
from linsdex.matrix.matrix_base import AbstractSquareMatrix

_SolverState = TypeVar("_SolverState")

class SimulationState(eqx.Module):
  solver: diffrax.AbstractSolver
  terms: diffrax.AbstractTerm
  states_pre: Annotated[_SolverState, "T"]
  ts: Float[Array, "T"]
  xts: Float[Array, "T D"]

class AdjointSimulationState(eqx.Module):
  ts: Float[Array, "T"]
  xts: Float[Array, "T D"]
  ats: Float[Array, "T D"]
  costs: Float[Array, "T"]
  grad_theta: Optional[PyTree]
  total_cost: Scalar

def sde_simulation_with_internals(
  solver: diffrax.AbstractSolver,
  x0: Float[Array, "D"],
  drift_fn: diffrax.ODETerm,
  diffusion_fn: Union[Callable[[Scalar, Float[Array, "D"], PyTree], Float[Array, "D"]], None],
  t0: Scalar,
  t1: Scalar,
  *,
  key: PRNGKeyArray,
  args: PyTree,
  n_steps: Optional[int] = None,
  ts: Optional[Float[Array, "n_steps"]] = None,
) -> SimulationState:
  """Manual SDE simulation that also records solver internals per step.

  This is a convenience wrapper around a manual `lax.scan`-based Stratonovich
  SDE simulation using a Diffrax solver, that returns not only the step
  endpoints `(ts, xts)` but also the sequence of pre-step solver states
  (`states_pre`). These pre-step states allow reconstructing the exact
  one-step maps Φ_k used during the forward pass, which is required to compute
  a solver-matched discrete adjoint.

  Args:
    solver: A Diffrax SDE solver (e.g., ShARK).
    x0: Initial state at time `t0`.
    drift_fn: A `diffrax.ODETerm` for the drift (must follow (t, x, args)).
    diffusion_fn: A callable returning the diffusion matrix at (t, x, args).
    t0, t1: Start and end times.
    n_steps: Number of fixed steps to simulate; grid has length n_steps + 1.
    key: PRNGKey used to construct the Brownian control.
    args: PyTree of parameters passed as `args` to the solver.

  Returns:
    SimulationState containing:
      - solver, terms: the solver and combined terms used
      - states_pre: sequence of pre-step solver states of length n_steps
      - ts: step endpoints of length n_steps + 1
      - xts: states at those endpoints, length n_steps + 1
  """
  assert isinstance(drift_fn, diffrax.ODETerm), "drift_fn must be a diffrax.ODETerm"

  if ts is None:
    ts = jnp.linspace(t0, t1, n_steps + 1)
  else:
    n_steps = ts.shape[0] - 1

  dt0 = (t1 - t0) / n_steps
  if diffusion_fn is None:
    terms = drift_fn
  else:
    bm = diffrax.VirtualBrownianTree(t0, t1, dt0/2, x0.shape, key, diffrax.SpaceTimeLevyArea)
    diff = diffrax.ControlTerm(diffusion_fn, bm)
    terms = diffrax.MultiTerm(drift_fn, diff)

  state0 = solver.init(terms, ts[0], ts[1], x0, args=args)

  def body(
    carry: Tuple[_SolverState, Float[Array, "D"]],
    t_pair: Tuple[Scalar, Scalar]
  ) -> Tuple[Tuple[_SolverState, Float[Array, "D"]], Tuple[_SolverState, Float[Array, "D"]]]:
    state, y_curr = carry
    tprev, tnext = t_pair

    # Advance one the SDE solver step; save next state and emit pre-step solver state.
    y_next, _, _, state_after, _ = solver.step(
      terms, tprev, tnext, y_curr, args, state, False
    )
    # Emit tuple (pre_step_state, y_next) so we can reuse states in backward pass
    return (state_after, y_next), (state, y_next)

  t_pairs = (ts[:-1], ts[1:])
  _, (states_pre, ys) = jax.lax.scan(body, (state0, x0), t_pairs)
  xts = jnp.concatenate([x0[None], ys], axis=0)

  return SimulationState(solver, terms, states_pre, ts, xts)

def ode_simulation_with_internals(
  solver: diffrax.AbstractSolver,
  x0: Float[Array, "D"],
  ode_fn: diffrax.ODETerm,
  t0: Scalar,
  t1: Scalar,
  *,
  key: PRNGKeyArray,
  args: PyTree,
  n_steps: Optional[int] = None,
  ts: Optional[Float[Array, "n_steps"]] = None,
) -> SimulationState:
  return sde_simulation_with_internals(
    solver=solver,
    x0=x0,
    drift_fn=ode_fn,
    diffusion_fn=None,
    t0=t0,
    t1=t1,
    key=key,
    args=args,
    n_steps=n_steps,
    ts=ts,
  )

def adjoint_simulation_from_sim_internals(
  simulation_state: SimulationState,
  terminal_cost_fn: Callable[[Float[Array, "D"]], Scalar],
  running_cost_fn: Callable[[Scalar, Float[Array, "D"], PyTree], Scalar],
  args: PyTree,
) -> AdjointSimulationState:
  """Discrete adjoint simulation using recorded solver internals.

  Given a `SimulationState` produced by `sde_simulation_with_internals`, this
  computes the solver-matched discrete adjoint on the same time grid. It uses
  the pre-step solver states to define the same one-step maps Φ_k as forward,
  and applies the right-point rule for the running cost:

    a_k = (∂x_{k+1}/∂x_k)^T [a_{k+1} + Δt ∇_x f(x_{k+1}, t_{k+1}, θ)]
    ∂L/∂θ += (∂x_{k+1}/∂θ)^T [a_{k+1} + Δt ∇_x f(·)] + Δt ∂f/∂θ(·)

  Args:
    simulation_state: Forward simulation outputs including solver internals.
    terminal_cost_fn: Φ(x_T); returns scalar terminal cost.
    running_cost_fn: f(t, x, θ); running cost density used in the adjoint update.
    args: Parameters θ (PyTree) to be threaded through the solver and cost.

  Returns:
    AdjointSimulationState with aligned arrays:
      - ts: step endpoints (length n)
      - xts: forward states at ts (length n)
      - ats: adjoint states aligned with ts
             (ats[0] = ∂L/∂x0, ats[-1] = ∂Φ/∂x_T)
      - grad_theta: accumulated parameter gradient PyTree
  """

  # Terminal cost and adjoint from terminal cost Phi
  xT = simulation_state.xts[-1]
  cost_T, aT = jax.value_and_grad(terminal_cost_fn)(xT)

  # Accumulate ∂L/∂θ = Σ_k (∂Φ_k/∂θ)^T a_{k+1}
  theta = args
  grad_theta = jtu.tree_map(jnp.zeros_like, theta)

  # Reverse-time scan over step intervals and states
  rev_tprev = simulation_state.ts[:-1][::-1]
  rev_tnext = simulation_state.ts[1:][::-1]
  rev_xk = simulation_state.xts[:-1][::-1]
  rev_xnext = simulation_state.xts[1:][::-1]

  # Reverse the sequence of pre-step solver states to align with reverse-time scan
  rev_state_pre = jtu.tree_map(lambda a: a[::-1], simulation_state.states_pre)

  running_cost_value_and_grad = jax.value_and_grad(running_cost_fn, argnums=(1, 2))

  def body(carry, data):
    a_curr, gt, cost_curr = carry
    tprev, tnext, xk, xnext, solver_state_k = data
    dt = tnext - tprev

    def F(xt, theta):
      y, _, _, _, _ = simulation_state.solver.step(
        simulation_state.terms, tprev, tnext, xt, theta, solver_state_k, False
      )
      return y

    # Running cost value and gradients at the right-point
    running_val, (gxr, gtr) = running_cost_value_and_grad(tnext, xnext, theta)

    # Augment cotangent at x_{k+1} with running-cost gradient (right-point rule)
    a_aug = a_curr + dt * gxr
    # VJP w.r.t. (xk, theta) treating solver internals as constants from forward pass
    _, vjp = jax.vjp(F, xk, theta)
    gx, gtheta = vjp(a_aug)
    atk = gx
    # Always accumulate parameter gradient: include step-map term (if any) and running-cost term
    if gtheta is None:
      gtheta = jtu.tree_map(jnp.zeros_like, theta)
    grad_theta_contribution = jtu.tree_map(lambda a, b: a + dt * b, gtheta, gtr)
    grad_theta = jtu.tree_map(lambda a, b: a + b, gt, grad_theta_contribution)
    cost_prev = cost_curr + dt * running_val
    return (atk, grad_theta, cost_prev), (atk, cost_prev)

  # Initialize cost at terminal time (already computed with value_and_grad)
  (_, grad_theta, _), (ats, costs_prev) = jax.lax.scan(
    body,
    (aT, grad_theta, cost_T),
    (rev_tprev, rev_tnext, rev_xk, rev_xnext, rev_state_pre)
  )

  # Current order is [a_T, a_{N-1}, ..., a_0]; flip so index i matches ts[i]
  ats_rev = jnp.concatenate([aT[None], ats], axis=0)
  ats_aligned = ats_rev[::-1]

  # Build costs aligned with ts: include terminal cost at the end
  costs_rev = jnp.concatenate([cost_T[None], costs_prev], axis=0)
  costs_aligned = costs_rev[::-1]

  # Compute total cost (right-point rule for running cost) on the same grid
  ts = simulation_state.ts
  xts = simulation_state.xts
  dt = ts[1] - ts[0]
  running_vals = jax.vmap(lambda t, x: running_cost_fn(t, x, args))(ts[1:], xts[1:])
  total_cost = jnp.sum(running_vals) * dt + terminal_cost_fn(xts[-1])
  return AdjointSimulationState(
    simulation_state.ts,
    simulation_state.xts,
    ats_aligned,
    costs_aligned,
    grad_theta,
    total_cost
  )
