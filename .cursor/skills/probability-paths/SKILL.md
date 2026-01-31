---
name: probability-paths
description: Work with probability path distributions for diffusion models, including bridge path marginals, memoryless sampling, and efficient batch computation. Use when you need to sample from or evaluate the distribution p(x_t | y_1) at intermediate times.
---

# Probability Paths for Diffusion Models

This skill covers the core probability path framework in linsdex for working with the distributional quantities that arise in diffusion-based generative models. While the `/diffusion-conversions` skill focuses on converting between different parameterizations (y1, score, flow, drift), this skill focuses on the underlying probabilistic objects and efficient sampling.

## When to Use

- Sampling from the bridge path marginal p(x_t | y_1) at intermediate times
- Computing all flow quantities (xt, flow, score, drift, noise) jointly for training
- Efficient batch computation of probability path slices at multiple times
- Memoryless trajectory sampling for Reciprocal Adjoint Matching (RAM) training
- Understanding the mathematical structure of diffusion model distributions

## Mathematical Background

A diffusion model defines a stochastic bridge between a prior distribution at t=0 and terminal evidence at t=1. The key distributions are:

- **Forward marginal** p(x_t | x_0): The distribution of the noised state at time t given initial state x_0
- **Backward message** β_t(x_t) = p(y_1 | x_t): The likelihood of terminal evidence given current state
- **Bridge path marginal** p_t(x_t | y_1) = ∫ p(x_t | x_0, y_1) p(x_0) dx_0: The marginal of the stochastic bridge

The `ProbabilityPathSlice` class encapsulates all these quantities at a specific time t, enabling exact computation of scores, flows, and drifts without approximation.

## Key Classes

### DiffusionModelComponents

Container for the fundamental components defining a diffusion model:

```python
from linsdex.diffusion_model.probability_path import DiffusionModelComponents
from linsdex import BrownianMotion, StandardGaussian, DiagonalMatrix

dim = 64
sde = BrownianMotion(sigma=1.0, dim=dim)

components = DiffusionModelComponents(
  linear_sde=sde,
  t0=0.0,
  x_t0_prior=StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim)),
  t1=1.0,
  evidence_cov=DiagonalMatrix.eye(dim) * 1e-4
)
```

### ProbabilityPathSlice

Represents all probabilistic quantities at a specific time t:

```python
from linsdex.diffusion_model.probability_path import ProbabilityPathSlice

t = 0.5
path_slice = ProbabilityPathSlice(components, t)

# Access the bridge path marginal p(x_t | y_1) as a functional
functional_marginal = path_slice.functional_pt_given_y1

# Access the backward message β_t as a functional
functional_beta = path_slice.functional_beta_t

# Access precision matrices
beta_precision = path_slice.beta_precision
marginal_precision = path_slice.marginal_precision
```

## Sampling from the Probability Path

### Basic Sampling

```python
import jax
import jax.numpy as jnp

# Sample x_t given y_1 (returns a LinearFunctional that maps y_1 -> x_t)
key = jax.random.PRNGKey(0)
functional_xt = path_slice.sample(key)

# Resolve with a specific y_1 value
y1 = jnp.zeros(dim)
xt = functional_xt(y1)  # Concrete sample

# Or use reparameterization with explicit noise
epsilon = jax.random.normal(key, (dim,))
functional_xt = path_slice._sample(epsilon)
xt = functional_xt(y1)
```

### Sampling All Flow Quantities for Training

The `_sample_matching_items` method returns all quantities needed for training as `LinearFunctional` objects:

```python
from linsdex.linear_functional.functional_ops import resolve_functional

epsilon = jax.random.normal(key, (dim,))
functional_items = path_slice._sample_matching_items(epsilon)

# functional_items contains:
#   t: time
#   xt: sampled state (LinearFunctional)
#   flow: probability flow velocity (LinearFunctional)
#   score: score function value (LinearFunctional)
#   drift: SDE drift (LinearFunctional)
#   noise: the epsilon used (LinearFunctional)

# Resolve all quantities with a specific y_1
y1 = jnp.zeros(dim)
resolved_items = resolve_functional(functional_items, y1)

# Now resolved_items contains concrete arrays
xt = resolved_items.xt
flow = resolved_items.flow
score = resolved_items.score
drift = resolved_items.drift
```

### Converting to Transition Distribution

```python
# Convert to a GaussianTransition p(x_t | y_1)
epsilon = jax.random.normal(key, (dim,))
transition = path_slice.to_transition(epsilon)

# The transition maps y_1 to x_t
x_given_y1 = transition.condition_on_x(y1)  # StandardGaussian for x_t
```

## Efficient Batch Computation

When computing probability path slices at multiple times, use `get_probability_path` which requires only 2 ODE solves regardless of the number of times:

```python
from linsdex.diffusion_model.probability_path import get_probability_path

# Efficient: 2 ODE solves total
times = jnp.linspace(0.0, 1.0, 100)
path_slices = get_probability_path(components, times)

# path_slices is a batched ProbabilityPathSlice with batch_size = 100
print(path_slices.batch_size)  # 100

# Access individual slices by indexing
slice_at_t50 = path_slices[50]

# Or use with vmap for parallel operations
def process_slice(path_slice, y1):
  return path_slice.score(path_slice._sample(jax.random.normal(key, (dim,)))(y1))

scores = jax.vmap(process_slice, in_axes=(0, None))(path_slices, y1)
```

This is much more efficient than computing slices individually:

```python
# Inefficient: 2n ODE solves for n times
path_slices_slow = jax.vmap(lambda t: ProbabilityPathSlice(components, t))(times)
```

## Memoryless Trajectory Sampling

For efficient sampling of entire trajectories conditioned on terminal state y_1, use the memoryless utilities:

```python
from linsdex.diffusion_model.memoryless import (
  sample_memoryless_trajectory,
  MemorylessForwardSDE,
  get_memoryless_projection_adjoint_path,
)

# Sample trajectory X_t | X_1 efficiently
x1 = jnp.zeros(dim)  # Terminal state
ts = jnp.linspace(0.0, 0.99, 50)  # Times to sample (excluding t1)
key = jax.random.PRNGKey(0)

trajectory = sample_memoryless_trajectory(
  components, x1, ts, key,
  method="discretization"  # or "simulation"
)

# trajectory is a TimeSeries with trajectory.times and trajectory.values
```

### MemorylessForwardSDE

The reverse-time SDE that induces a memoryless path distribution:

```python
memoryless_sde = MemorylessForwardSDE(components)

# Get parameters at reverse time s = t1 - t
s = 0.5
F_s, u_s, L_s = memoryless_sde.get_params(s)

# The SDE is: dX_s = (F_s X_s + u_s) ds + L_s dW_s
# in reverse time s where s = 0 corresponds to t = t1
```

### Full Path for Reciprocal Adjoint Matching

For training with RAM, use the precomputed full path:

```python
from linsdex.diffusion_model.memoryless import get_memoryless_projection_adjoint_path

times = jnp.linspace(0.0, 0.99, 50)
full_path = get_memoryless_projection_adjoint_path(components, times)

# Sample a trajectory given y1
key = jax.random.PRNGKey(0)
y1 = jnp.zeros(dim)
trajectory = full_path.sample(key, y1)

# Access precomputed quantities
p_xt_given_y1 = full_path.p_xt_given_y1  # Batched GaussianTransition
p_y1_given_xt = full_path.p_y1_given_xt  # Batched GaussianTransition
base_drifts = full_path.base_drifts  # Batched LinearFunctional
diffusion_coefficients = full_path.diffusion_coefficients  # Batched matrices
```

## Adjoint Simulation Utilities

For computing gradients through SDE solvers using discrete adjoints:

```python
from linsdex.diffusion_model.adjoints import (
  sde_simulation_with_internals,
  ode_simulation_with_internals,
  adjoint_simulation_from_sim_internals,
)
import diffrax

# Simulate SDE while recording solver internals
solver = diffrax.ShARK()

@diffrax.ODETerm
def drift_fn(t, x, args):
  return -x  # Example drift

def diffusion_fn(t, x, args):
  return jnp.eye(dim) * 0.1

sim_state = sde_simulation_with_internals(
  solver, x0, drift_fn, diffusion_fn,
  t0=0.0, t1=1.0,
  key=key, args=None, n_steps=100
)

# sim_state contains:
#   ts: time grid
#   xts: states at each time
#   states_pre: solver internals for adjoint computation

# Compute discrete adjoint
def terminal_cost(xT):
  return jnp.sum(xT**2)

def running_cost(t, x, args):
  return 0.0

adjoint_state = adjoint_simulation_from_sim_internals(
  sim_state, terminal_cost, running_cost, args=None
)

# adjoint_state contains:
#   ats: adjoint states aligned with ts
#   grad_theta: parameter gradients
#   total_cost: accumulated cost
```

## Helper Classes

### Affine Mappings

```python
from linsdex.diffusion_model.probability_path import (
  Y1ToBwdMean,
  Y1ToMarginalMean,
  BwdMeanToMarginalMean,
)

# Linear functional mapping y_1 to backward message mean
y1_to_bwd = Y1ToBwdMean(components, t)
bwd_mean = y1_to_bwd(y1)  # Apply mapping

# Linear functional mapping y_1 to marginal mean
y1_to_marginal = Y1ToMarginalMean(components, t)
marginal_mean = y1_to_marginal(y1)

# Mapping between backward and marginal means
bwd_to_marginal = BwdMeanToMarginalMean(components, t)
marginal_mean = bwd_to_marginal(bwd_mean)
```

### Probability Path Transitions

Compute the transition distribution between two times on the probability path:

```python
from linsdex.diffusion_model.probability_path import probability_path_transition

# Compute p(x_t | x_s, y_1) for s < t
s, t = 0.3, 0.7
transition = probability_path_transition(components, components, t, s)

# transition is a GaussianTransition
x_s = jnp.zeros(dim)
p_xt_given_xs = transition.condition_on_x(x_s)  # StandardGaussian
```

### Noise Schedule Drift Correction

When changing noise schedules while preserving marginals:

```python
from linsdex.diffusion_model.probability_path import noise_schedule_drift_correction

def custom_noise_schedule(t, xt):
  return DiagonalMatrix.eye(dim) * 0.5

corrected_drift = noise_schedule_drift_correction(
  components, t, xt, original_drift,
  noise_schedule=custom_noise_schedule
)
```

## Key Imports

```python
# Core probability path classes
from linsdex.diffusion_model.probability_path import (
  DiffusionModelComponents,
  ProbabilityPathSlice,
  get_probability_path,
  probability_path_transition,
  noise_schedule_drift_correction,
  Y1ToBwdMean,
  Y1ToMarginalMean,
  BwdMeanToMarginalMean,
)

# Memoryless sampling utilities
from linsdex.diffusion_model.memoryless import (
  MemorylessForwardSDE,
  MemorylessFullPath,
  sample_memoryless_trajectory,
  get_memoryless_projection_adjoint_path,
  memoryless_noise_schedule,
)

# Adjoint simulation
from linsdex.diffusion_model.adjoints import (
  SimulationState,
  AdjointSimulationState,
  sde_simulation_with_internals,
  ode_simulation_with_internals,
  adjoint_simulation_from_sim_internals,
)

# Resolving LinearFunctional objects
from linsdex.linear_functional.functional_ops import resolve_functional
```

## Integration with Other Skills

- Use `/diffusion-conversions` for converting between y1, score, flow, and drift representations
- Use `/gaussian-distributions` for working with the underlying Gaussian distributions
- Use `/sde-conditioning` for conditioning SDEs on observations
- Use `/matrix-operations` for efficient linear algebra with structured matrices

## Tips

- Use `get_probability_path` instead of vmapping over `ProbabilityPathSlice` for efficiency
- The `_sample_matching_items` method is ideal for training as it returns all quantities jointly
- `LinearFunctional` objects defer computation until resolved with a specific y_1 value
- For RAM training, use `get_memoryless_projection_adjoint_path` to precompute all needed quantities
- The discrete adjoint utilities are useful when you need gradients through custom SDE solvers
- For memoryless sampling, avoid times very close to t0 or t1 (use ranges like 0.1 to 0.9) for numerical stability
- When `sample_memoryless_trajectory` hits ODE solver limits, try `method="simulation"` with `solver_name="euler"`
