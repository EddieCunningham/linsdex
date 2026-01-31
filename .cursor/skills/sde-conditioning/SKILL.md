---
name: sde-conditioning
description: Condition Linear SDEs on observations to interpolate sparse data, perform Bayesian inference on time series, or create bridges between boundary conditions. Use when working with time series interpolation, state estimation, or posterior sampling.
---

# SDE Conditioning

Condition Linear SDEs on Gaussian observations to sample from posterior distributions over trajectories.

## When to Use

- Interpolating sparse time series observations with uncertainty
- Creating Brownian bridges or other conditioned stochastic processes
- Performing Bayesian inference on latent states given noisy measurements
- State estimation with process and observation noise

## Key Concepts

### Available SDEs

linsdex provides several built-in Linear SDEs:

- `BrownianMotion(sigma, dim)` - Standard Brownian motion dx = σ dW
- `OrnsteinUhlenbeck(sigma, lambda_, dim)` - Mean-reverting process dx = -λx dt + σ dW
- `StochasticHarmonicOscillator(freq, coeff, sigma, observation_dim)` - 2D state with position and velocity

### Workflow

1. Define an SDE for the latent dynamics
2. Create observations as a `TimeSeries`
3. Convert observations to Gaussian potentials using an encoder
4. Condition the SDE on the potentials
5. Sample posterior trajectories

## Code Examples

### Basic Time Series Interpolation

```python
import jax
import jax.numpy as jnp
import jax.random as random
from linsdex import TimeSeries, StochasticHarmonicOscillator
from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior

# 1. Create sparse observations
obs_times = jnp.linspace(0, 10, 5)
obs_values = jnp.sin(obs_times)[:, None]
observations = TimeSeries(obs_times, obs_values)

# 2. Define the SDE for latent dynamics
sde = StochasticHarmonicOscillator(
    freq=1.0,      # Natural frequency
    coeff=0.1,     # Damping
    sigma=0.5,     # Process noise
    observation_dim=1
)

# 3. Convert observations to Gaussian potentials
encoder = PaddingLatentVariableEncoderWithPrior(
    y_dim=1,       # Observed dimension
    x_dim=2,       # Latent dimension (position + velocity)
    sigma=0.01     # Observation noise (tight fit to data)
)
potentials = encoder(observations)

# 4. Condition SDE on potentials
conditioned_sde = sde.condition_on(potentials)

# 5. Sample from the posterior on a dense grid
key = random.PRNGKey(0)
keys = random.split(key, 128)
dense_times = jnp.linspace(0, 10, 2000)

posterior_samples = jax.vmap(
    conditioned_sde.sample, in_axes=(0, None)
)(keys, dense_times)

# Extract positions (first component of 2D state)
positions = posterior_samples.values[:, :, 0]
mean_position = positions.mean(axis=0)
std_position = positions.std(axis=0)
```

### Brownian Bridge

Condition a process on both endpoints:

```python
import jax.numpy as jnp
import jax.random as random
from linsdex import BrownianMotion, TimeSeries
from linsdex.ssm.simple_encoder import IdentityEncoder

# Define Brownian motion
bm = BrownianMotion(sigma=1.0, dim=2)

# Define endpoints: x(0) = [0,0], x(1) = [1,1]
endpoint_times = jnp.array([0.0, 1.0])
endpoint_values = jnp.array([[0.0, 0.0], [1.0, 1.0]])
endpoints = TimeSeries(endpoint_times, endpoint_values)

# Create potentials with tight observation noise
encoder = IdentityEncoder(dim=2)
potentials = encoder(endpoints)

# Condition to create Brownian bridge
bridge = bm.condition_on(potentials)

# Sample bridge trajectories
key = random.PRNGKey(0)
times = jnp.linspace(0, 1, 100)
trajectory = bridge.sample(key, times)
```

### Conditioning on Starting Point Only

```python
from linsdex import OrnsteinUhlenbeck

# Create an OU process
ou = OrnsteinUhlenbeck(sigma=0.5, lambda_=1.0, dim=2)

# Condition to start at specific point
x0 = jnp.array([2.0, -1.0])
conditioned = ou.condition_on_starting_point(t0=0.0, x0=x0)

# Sample trajectories
key = random.PRNGKey(0)
times = jnp.linspace(0, 5, 500)
trajectory = conditioned.sample(key, times)
```

### Using GaussianPotentialSeries Directly

For more control over observation uncertainties:

```python
from linsdex import GaussianPotentialSeries, BrownianMotion

bm = BrownianMotion(sigma=0.1, dim=1)

# Define observations with varying certainty
times = jnp.array([0.0, 0.5, 1.0])
values = jnp.array([[0.0], [0.5], [1.0]])
certainty = jnp.array([[1000.0], [100.0], [1000.0]])  # Higher = more certain

potentials = GaussianPotentialSeries(
    ts=times,
    xts=values,
    certainty=certainty
)

conditioned = bm.condition_on(potentials)
```

## Key Classes

- `TimeSeries(times, values, mask=None)` - Time series data container
- `GaussianPotentialSeries` - Series of Gaussian observation potentials
- `IdentityEncoder` - Direct encoding when observation dim equals latent dim
- `PaddingLatentVariableEncoderWithPrior` - Pads observations to higher-dim latent space

## Tips

- Use `PaddingLatentVariableEncoderWithPrior` when the latent state has more dimensions than observations
- Lower `sigma` in the encoder means tighter fit to observations
- Sample many trajectories in parallel with `jax.vmap` for efficiency
- The conditioned SDE uses parallel message passing for O(log T) complexity on GPU
