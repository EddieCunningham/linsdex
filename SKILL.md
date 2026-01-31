---
name: linsdex
description: A JAX-based library for linear stochastic differential equations, state-space models, and Gaussian inference. Use when working with time series interpolation, diffusion models, Kalman filtering, or probabilistic modeling with linear-Gaussian systems.
---

# Linsdex

A high-performance JAX-based library for linear stochastic differential equations (SDEs), state-space models, and Gaussian inference.

## When to Use

- Time series interpolation and smoothing with uncertainty quantification
- Building and training diffusion-based generative models
- State-space model inference with Kalman filtering and smoothing
- Probabilistic modeling with chain-structured Gaussian CRFs
- Working with linear-Gaussian systems that require GPU acceleration

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

### Dependencies

The library requires:
- `jax` and `jaxlib` for array operations and automatic differentiation
- `equinox` for neural network and PyTree utilities
- `diffrax` for ODE/SDE solvers
- `jaxtyping` for type annotations
- `plum-dispatch` for multiple dispatch

## Key Features

- **Linear SDEs** with exact Gaussian transition distributions
- **Parallel message passing** with O(log T) complexity on GPU
- **Multiple Gaussian parameterizations** (Standard, Natural, Mixed) for numerical stability
- **Specialized matrix types** (Diagonal, Block, Dense) with symbolic optimization
- **Diffusion model utilities** for converting between scores, flows, and drifts
- **Full JAX compatibility** with `jax.vmap`, `jax.grad`, and `jax.jit`

## Quick Start

```python
import jax
import jax.numpy as jnp
from linsdex import TimeSeries, StochasticHarmonicOscillator
from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior

# 1. Create sparse observations
obs_times = jnp.linspace(0, 10, 5)
obs_values = jnp.sin(obs_times)[:, None]
observations = TimeSeries(obs_times, obs_values)

# 2. Define an SDE for latent dynamics
sde = StochasticHarmonicOscillator(freq=1.0, coeff=0.1, sigma=0.5, observation_dim=1)

# 3. Convert observations to Gaussian potentials
encoder = PaddingLatentVariableEncoderWithPrior(y_dim=1, x_dim=2, sigma=0.01)
potentials = encoder(observations)

# 4. Condition SDE and sample posterior trajectories
conditioned_sde = sde.condition_on(potentials)
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 100)
dense_times = jnp.linspace(0, 10, 1000)
samples = jax.vmap(conditioned_sde.sample, in_axes=(0, None))(keys, dense_times)
```

## Specialized Skills

This library includes detailed skills for specific capabilities. Invoke these skills for in-depth guidance:

### `/sde-conditioning`
Condition Linear SDEs on observations for time series interpolation, Brownian bridges, and posterior sampling. Use when you need to interpolate sparse data or perform Bayesian inference on trajectories.

### `/diffusion-conversions`
Convert between diffusion model representations (y1, score, flow, drift) for training and sampling generative models. Use when building diffusion-based neural networks.

### `/probability-paths`
Work with probability path distributions for diffusion models, including bridge path marginals, memoryless sampling, and efficient batch computation. Use when you need to sample from p(x_t | y_1), compute all flow quantities jointly for training, or use Reciprocal Adjoint Matching.

### `/crf-inference`
Perform inference in chain-structured Gaussian CRFs with efficient message passing. Use for discrete-time state estimation, computing marginals, or sampling joint distributions.

### `/gaussian-distributions`
Work with three Gaussian parameterizations (Standard, Natural, Mixed) for numerical stability. Use when combining observations or converting between mean/covariance and precision forms.

### `/matrix-operations`
Use specialized matrix types (Diagonal, Dense, Block) with symbolic tags. Use when working with structured covariances or optimizing linear algebra operations.

## Key Imports

```python
# SDEs
from linsdex import BrownianMotion, OrnsteinUhlenbeck, StochasticHarmonicOscillator

# Gaussian distributions
from linsdex import StandardGaussian, NaturalGaussian, MixedGaussian
from linsdex import GaussianTransition, GaussianPotentialSeries

# Matrix types
from linsdex import DiagonalMatrix, DenseMatrix, TAGS
from linsdex.matrix.block import Block2x2Matrix

# CRF
from linsdex import CRF

# Time series
from linsdex import TimeSeries

# Diffusion utilities
from linsdex.diffusion_model.probability_path import (
    DiffusionModelComponents,
    DiffusionModelConversions,
    ProbabilityPathSlice,
    get_probability_path  # Efficient batch computation at multiple times
)

# Memoryless sampling for diffusion models
from linsdex.diffusion_model.memoryless import (
    sample_memoryless_trajectory,
    get_memoryless_projection_adjoint_path
)

# Encoders
from linsdex.ssm.simple_encoder import (
    IdentityEncoder,
    PaddingLatentVariableEncoderWithPrior
)
```

## Documentation

- See `TUTORIAL.md` for comprehensive documentation with mathematical foundations
- See `README.md` for a quick overview and examples
- See `example_usage.md` for additional code patterns
