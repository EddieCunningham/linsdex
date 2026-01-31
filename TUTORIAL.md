# Linsdex Tutorial

A comprehensive guide to using the `linsdex` library for linear stochastic differential equations, Gaussian inference, and diffusion models.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Architecture Overview](#architecture-overview)
4. [Core Modules](#core-modules)
   - [Matrix Module](#matrix-module)
   - [Gaussian Potentials](#gaussian-potentials)
   - [Time Series](#time-series)
   - [Stochastic Differential Equations](#stochastic-differential-equations)
   - [Conditional Random Fields](#conditional-random-fields)
   - [Diffusion Model Conversions](#diffusion-model-conversions)
   - [State Space Models](#state-space-models)
5. [Complete Workflows](#complete-workflows)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Advanced Topics](#advanced-topics)
8. [Agent Skills](#agent-skills)

## Introduction

`linsdex` is a high-performance JAX-based library for working with linear stochastic differential equations (SDEs), state-space models, and Gaussian inference. It provides a modular framework for defining, simulating, and conditioning linear-Gaussian systems with support for parallelized inference on GPUs.

### Primary Use Cases

- **Time series interpolation and smoothing**: Condition SDEs on sparse observations to interpolate missing values
- **State-space model inference**: Perform filtering, smoothing, and sampling in linear-Gaussian systems
- **Diffusion model development**: Build and train diffusion-based generative models with unified conversion utilities
- **Probabilistic modeling**: Work with chain-structured Gaussian conditional random fields (CRFs)

### Key Features

- **Linear SDEs**: Comprehensive support for linear time-invariant (LTI) and time-varying SDEs with exact transition distributions
- **Efficient Inference**: Sequential and parallel message passing using parallel scan for O(log T) complexity
- **Probabilistic Primitives**: Multiple Gaussian parameterizations (Standard, Natural, Mixed) with numerically stable operations
- **Specialized Linear Algebra**: Custom matrix library with Diagonal, Block, and Dense types using symbolic tags for optimization
- **Diffusion Utilities**: Unified interface for mapping between clean data predictions, scores, and probability flow
- **JAX-Native**: Fully compatible with `jax.vmap`, `jax.grad`, and `jax.jit` for automatic vectorization and differentiation

### Prerequisites

This tutorial assumes familiarity with:
- Python and JAX basics
- Linear algebra (matrices, eigenvalues, matrix exponentials)
- Probability theory (Gaussian distributions, conditional distributions)
- Stochastic processes (SDEs, Brownian motion)

## Installation

Install `linsdex` using pip:

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

### Dependencies

The library depends on:
- `jax` and `jaxlib`: Core array operations and automatic differentiation
- `equinox`: Neural network and PyTree utilities
- `diffrax`: ODE/SDE solvers
- `jaxtyping`: Type annotations for JAX arrays
- `plum-dispatch`: Multiple dispatch for Python

## Architecture Overview

The `linsdex` library is organized into several interconnected modules, each handling a specific aspect of probabilistic modeling with linear-Gaussian systems.

```
linsdex/
├── matrix/           # Specialized matrix types with symbolic optimization
│   ├── diagonal.py   # DiagonalMatrix
│   ├── dense.py      # DenseMatrix
│   ├── block/        # Block2x2Matrix, Block3x3Matrix
│   └── tags.py       # Symbolic tags for optimization
├── potential/        # Probability distributions
│   ├── gaussian/     # Gaussian distributions and transitions
│   └── categorical/  # Categorical distributions
├── linear_functional/ # Linear transformations and quadratic forms
├── sde/              # Stochastic differential equations
├── crf/              # Conditional random fields
├── diffusion_model/  # Diffusion model utilities
├── series/           # Time series data structures
├── ssm/              # State space model encoders/decoders
└── util/             # Utilities (parallel scan, etc.)
```

### Module Relationships

The modules build on each other in a layered architecture:

1. **Foundation Layer**: `matrix` and `linear_functional` provide the linear algebra primitives
2. **Distribution Layer**: `potential` uses matrices to represent Gaussian and categorical distributions
3. **Dynamics Layer**: `sde` defines stochastic processes using distributions for transitions
4. **Inference Layer**: `crf` provides message passing algorithms using potentials and transitions
5. **Application Layer**: `diffusion_model` and `ssm` combine lower layers for specific applications

## Core Modules

### Matrix Module

The matrix module provides specialized matrix types that track structural properties for optimization. Instead of always using dense matrices, `linsdex` uses diagonal, block, and tagged matrices to avoid unnecessary computation.

#### Matrix Types

**DiagonalMatrix**: Stores only diagonal elements for O(n) operations instead of O(n²).

```python
import jax.numpy as jnp
from linsdex import DiagonalMatrix

# Create a diagonal matrix from diagonal elements
diag_elements = jnp.array([1.0, 2.0, 3.0])
D = DiagonalMatrix(diag_elements)

# Create an identity matrix
I = DiagonalMatrix.eye(3)

# Matrix operations are efficient
D_inv = D.get_inverse()  # O(n) instead of O(n³)
log_det = D.get_log_det()  # O(n) instead of O(n³)
```

**DenseMatrix**: General dense matrices when structure cannot be exploited.

```python
from linsdex import DenseMatrix, TAGS

# Create a dense matrix
elements = jnp.array([[1.0, 0.5], [0.5, 2.0]])
M = DenseMatrix(elements, tags=TAGS.no_tags)

# Operations
M_inv = M.get_inverse()
chol = M.get_cholesky()
log_det = M.get_log_det()
```

**Block Matrices**: For higher-order systems with natural block structure.

```python
from linsdex.matrix.block import Block2x2Matrix

# Create a 2x2 block matrix
A = DiagonalMatrix.eye(2)
B = DenseMatrix(jnp.zeros((2, 2)), tags=TAGS.zero_tags)
C = DenseMatrix(jnp.zeros((2, 2)), tags=TAGS.zero_tags)
D = DiagonalMatrix.eye(2)

block_matrix = Block2x2Matrix(A, B, C, D)
```

#### Matrix Tags

Tags track properties like zero and infinite values. This enables symbolic simplification before numerical computation.

```python
from linsdex import TAGS

# Available tag configurations
TAGS.no_tags      # Regular matrix (non-zero, non-infinite)
TAGS.zero_tags    # Matrix is zero
TAGS.inf_tags     # Matrix has infinite elements (represents total uncertainty)

# Tags propagate through operations
# 0 * A = 0 (detected symbolically, no computation needed)
# A + 0 = A (detected symbolically)
```

### Gaussian Potentials

The `potential.gaussian` module implements Gaussian distributions in three parameterizations, each optimal for different operations.

#### Three Parameterizations

**StandardGaussian**: Mean (μ) and covariance (Σ) form. Best for sampling and interpretation.

```python
import jax.numpy as jnp
from linsdex import StandardGaussian, DiagonalMatrix

dim = 3
mu = jnp.zeros(dim)
Sigma = DiagonalMatrix.eye(dim)

dist = StandardGaussian(mu, Sigma)

# Sample from the distribution
key = jax.random.PRNGKey(0)
sample = dist.sample(key)

# Evaluate log probability
log_p = dist.log_prob(sample)
```

**NaturalGaussian**: Precision (J) and precision-weighted mean (h) form. Best for combining distributions.

```python
from linsdex import NaturalGaussian

# Natural parameters: J = Sigma^{-1}, h = J @ mu
J = DiagonalMatrix.eye(dim)
h = jnp.zeros(dim)

nat_dist = NaturalGaussian(J, h)

# Adding natural Gaussians multiplies the densities (efficient!)
combined = nat_dist + nat_dist  # Product of two Gaussians
```

**MixedGaussian**: Mean (μ) and precision (J) form. A stable bridge between standard and natural.

```python
from linsdex import MixedGaussian

mixed_dist = MixedGaussian(mu, J)

# Useful for Kalman filter updates where we have mean but need precision
```

#### Converting Between Parameterizations

```python
# Start with standard form
std_dist = StandardGaussian(mu, Sigma)

# Convert to natural form
nat_dist = std_dist.to_nat()

# Convert to mixed form
mixed_dist = std_dist.to_mixed()

# Convert back
std_from_nat = nat_dist.to_std()
std_from_mixed = mixed_dist.to_std()
```

#### Gaussian Transitions

`GaussianTransition` represents conditional distributions p(y|x) = N(y; Ax + u, Σ).

```python
from linsdex import GaussianTransition

dim = 2
A = DiagonalMatrix(jnp.ones(dim) * 0.9)  # Decay factor (diagonal matrix)
u = jnp.array([0.1, -0.1])  # Offset
Sigma = DiagonalMatrix.eye(dim) * 0.01  # Noise covariance

transition = GaussianTransition(A, u, Sigma)

# Apply to a point to get the conditional distribution
x = jnp.ones(dim)
p_y_given_x = transition.condition_on_x(x)  # Returns StandardGaussian

# Chain transitions: p(z|x) from p(y|x) and p(z|y)
transition2 = GaussianTransition(A, u, Sigma)
chained = transition.chain(transition2)  # p(z|x)
```

#### Gaussian Potential Series

For time series of Gaussian observations:

```python
from linsdex import GaussianPotentialSeries

times = jnp.array([0.0, 1.0, 2.0, 3.0])
values = jnp.array([[1.0], [2.0], [3.0], [4.0]])
std_dev = jnp.array([[0.1], [0.1], [0.1], [0.1]])

# Create a series of Gaussian potentials
potentials = GaussianPotentialSeries(ts=times, xts=values, standard_deviation=std_dev)

# Or with certainty (inverse variance)
certainty = jnp.array([[100.0], [100.0], [100.0], [100.0]])
potentials = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)
```

### Time Series

The `series` module provides data structures for handling time series data with support for batching and missing values.

#### TimeSeries Class

```python
import jax.numpy as jnp
from linsdex import TimeSeries

# Create a simple time series
times = jnp.linspace(0, 10, 100)
values = jnp.sin(times)[:, None]  # Shape: (100, 1)

series = TimeSeries(times, values)

# Access properties
print(f"Length: {len(series)}")
print(f"Dimension: {series.values.shape[-1]}")
```

#### Handling Missing Data

```python
# Create a mask for missing values (True = observed, False = missing)
mask = jnp.ones(100, dtype=bool)
mask = mask.at[30:40].set(False)  # Missing values at indices 30-39

series_with_mask = TimeSeries(times, values, mask=mask)
print(f"Mask shape: {series_with_mask.mask.shape}")
print(f"Number missing: {(~series_with_mask.mask).sum()}")
```

#### Batching and Windowing

```python
# Create windowed batches for training
long_times = jnp.arange(1000) * 0.1
long_values = jnp.sin(long_times)[:, None]
long_series = TimeSeries(long_times, long_values)

# Create batches of windows
windowed = long_series.make_windowed_batches(window_size=50)
print(f"Number of batches: {windowed.batch_size}")
print(f"Values shape: {windowed.values.shape}")  # (num_batches, 50, 1)
```

#### Using with JAX vmap

```python
import jax

# Create batched series
B, T, D = 32, 64, 2
times = jnp.tile(jnp.linspace(0, 1, T), (B, 1))
values = jnp.zeros((B, T, D))
batched_series = TimeSeries(times, values)

# Process each series in the batch
def process_single(series):
    return series.values.mean(axis=0)

# vmap automatically handles the batch dimension
batch_means = jax.vmap(process_single)(batched_series)
```

### Stochastic Differential Equations

The `sde` module provides a hierarchy of SDE classes for defining and simulating stochastic processes.

#### SDE Hierarchy

```
AbstractSDE
    └── AbstractLinearSDE
            └── AbstractLinearTimeInvariantSDE
```

- **AbstractSDE**: Base class defining drift and diffusion
- **AbstractLinearSDE**: Linear SDEs with time-varying coefficients
- **AbstractLinearTimeInvariantSDE**: LTI SDEs with constant coefficients (most efficient)

#### Built-in SDEs

```python
from linsdex import BrownianMotion, OrnsteinUhlenbeck, StochasticHarmonicOscillator

# Brownian Motion: dx = σ dW
bm = BrownianMotion(sigma=1.0, dim=2)

# Ornstein-Uhlenbeck: dx = -λx dt + σ dW
ou = OrnsteinUhlenbeck(sigma=0.5, lambda_=1.0, dim=2)

# Stochastic Harmonic Oscillator (2D state: position + velocity)
sho = StochasticHarmonicOscillator(
    freq=1.0,      # Natural frequency ω
    coeff=0.1,     # Damping coefficient γ
    sigma=0.5,     # Diffusion strength
    observation_dim=1
)
```

#### Transition Distributions

Linear SDEs provide exact Gaussian transition distributions p(x_t | x_s):

```python
# Get transition from time s to time t
s, t = 0.0, 1.0
transition = ou.get_transition_distribution(s, t)

# transition is a GaussianTransition
# Apply to get p(x_t | x_s = x)
x_s = jnp.zeros(2)
p_xt = transition.condition_on_x(x_s)
```

#### Conditioning on Data

The key feature of `linsdex` is conditioning SDEs on observations:

```python
import jax
import jax.random as random
from linsdex import TimeSeries, GaussianPotentialSeries

# Sparse observations
obs_times = jnp.array([0.0, 0.5, 1.0])
obs_values = jnp.array([[0.0], [0.3], [0.0]])
obs_std = jnp.array([[0.01], [0.01], [0.01]])

# Create Gaussian potentials from observations
potentials = GaussianPotentialSeries(
    ts=obs_times,
    xts=obs_values,
    standard_deviation=obs_std
)

# Condition the SDE
conditioned_sde = bm.condition_on(potentials)

# Sample trajectories from the posterior
key = random.PRNGKey(0)
save_times = jnp.linspace(0, 1, 100)
trajectory = conditioned_sde.sample(key, save_times)
```

#### Batch Sampling

```python
# Sample many trajectories in parallel
keys = random.split(key, 1000)
trajectories = jax.vmap(conditioned_sde.sample, in_axes=(0, None))(keys, save_times)

# trajectories.values has shape (1000, 100, dim)
```

#### Conditioning on Starting Point

```python
# Condition to start at a specific point
x0 = jnp.zeros(2)
conditioned = ou.condition_on_starting_point(t0=0.0, x0=x0)

# Sample from this conditioned process
trajectory = conditioned.sample(key, jnp.linspace(0, 1, 100))
```

### Conditional Random Fields

The `crf` module implements chain-structured probabilistic models for efficient inference.

#### Creating a CRF

```python
import equinox as eqx
from linsdex import CRF, NaturalGaussian, GaussianTransition, DiagonalMatrix

dim = 2
seq_len = 10

# Node potentials (one per timestep)
J = DiagonalMatrix.eye(dim)
h = jnp.zeros(dim)
single_potential = NaturalGaussian(J, h)
node_potentials = eqx.filter_vmap(lambda _: single_potential)(jnp.arange(seq_len))

# Transitions between adjacent nodes
A = DiagonalMatrix.eye(dim)  # Identity transition matrix
u = jnp.zeros(dim)
Sigma = DiagonalMatrix.eye(dim) * 0.1
single_transition = GaussianTransition(A, u, Sigma)
transitions = eqx.filter_vmap(lambda _: single_transition)(jnp.arange(seq_len - 1))

# Create the CRF
crf = CRF(node_potentials, transitions)
```

#### Message Passing

```python
# Forward messages (filtering)
fwd_messages = crf.get_forward_messages()

# Backward messages (smoothing preparation)
bwd_messages = crf.get_backward_messages()

# Marginal distributions at each timestep
marginals = crf.get_marginals()
```

#### Sampling

```python
key = random.PRNGKey(0)

# Sample a sequence from the CRF
sample = crf.sample(key)  # Shape: (seq_len, dim)

# Evaluate log probability of a sequence
log_prob = crf.log_prob(sample)
```

#### Parallel vs Sequential Inference

```python
# By default, uses parallel scan for O(log T) complexity on GPU
marginals = crf.get_marginals()

# Can also use sequential scan
marginals_seq = crf.get_marginals(parallel=False)
```

### Diffusion Model Conversions

The `diffusion_model` module provides utilities for building and training diffusion-based generative models.

#### DiffusionModelComponents

First, define the components of your diffusion model:

```python
from linsdex import BrownianMotion, StandardGaussian, DiagonalMatrix
from linsdex.diffusion_model.probability_path import DiffusionModelComponents

dim = 10
sde = BrownianMotion(sigma=0.1, dim=dim)
t0, t1 = 0.0, 1.0

# Prior at t0 (data distribution approximation)
x_t0_prior = StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))

# Evidence covariance at t1 (observation noise)
evidence_cov = DiagonalMatrix.eye(dim) * 0.001

components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=x_t0_prior,
    t1=t1,
    evidence_cov=evidence_cov
)
```

#### DiffusionModelConversions

Convert between different diffusion model quantities:

```python
from linsdex.diffusion_model.probability_path import DiffusionModelConversions

t = 0.5  # Current time
conversions = DiffusionModelConversions(components, t)

# Neural network predicts clean data y1
y1_pred = jnp.ones(dim)
xt = jnp.zeros(dim)  # Current state

# Convert to different quantities for sampling
flow = conversions.y1_to_flow(y1_pred, xt)    # Probability flow ODE velocity
drift = conversions.y1_to_drift(y1_pred, xt)  # SDE drift
score = conversions.y1_to_score(xt, y1_pred)  # Score function

# Convert from epsilon (noise) prediction
epsilon_pred = jnp.zeros(dim)
flow_from_eps = conversions.epsilon_to_flow(epsilon_pred, xt)
```

#### ProbabilityPathSlice

For efficiency, cache intermediate quantities:

```python
from linsdex.diffusion_model.probability_path import ProbabilityPathSlice

# Compute and cache quantities at time t
path_slice = ProbabilityPathSlice(components, t)

# Access cached values
beta_precision = path_slice.beta_precision
marginal_precision = path_slice.marginal_precision
```

#### Inverse Conversions

Convert back from score/flow to y1:

```python
# From score to y1
y1_from_score = conversions.score_to_y1(xt, score)

# From flow to y1
y1_from_flow = conversions.flow_to_y1(xt, flow)

# From drift to y1
y1_from_drift = conversions.drift_to_y1(xt, drift)
```

### State Space Models

The `ssm` module provides encoders and decoders for state space model inference.

#### Encoders

Encoders convert observations to Gaussian potentials on the latent space:

```python
from linsdex import TimeSeries
from linsdex.ssm.simple_encoder import (
    IdentityEncoder,
    PaddingLatentVariableEncoderWithPrior
)

# Identity encoder: observations directly become potentials
encoder = IdentityEncoder(dim=2)

# Padding encoder: pads observations to higher-dimensional latent space
# Useful when latent has more dimensions than observed (e.g., position + velocity)
encoder = PaddingLatentVariableEncoderWithPrior(
    y_dim=1,    # Observation dimension
    x_dim=2,    # Latent dimension
    sigma=0.01  # Observation noise
)

# Create potentials from observations
times = jnp.linspace(0, 10, 5)
values = jnp.sin(times)[:, None]
series = TimeSeries(times, values)

potentials = encoder(series)
```

#### Complete SSM Workflow

```python
from linsdex import StochasticHarmonicOscillator, TimeSeries
from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior
import jax
import jax.random as random

# 1. Define the latent dynamics (SDE)
sde = StochasticHarmonicOscillator(
    freq=1.0,
    coeff=0.1,
    sigma=0.5,
    observation_dim=1
)

# 2. Define sparse observations
times = jnp.linspace(0, 10, 5)
values = jnp.sin(times)[:, None]
series = TimeSeries(times, values)

# 3. Create encoder and convert observations to potentials
encoder = PaddingLatentVariableEncoderWithPrior(y_dim=1, x_dim=2, sigma=0.01)
potentials = encoder(series)

# 4. Condition SDE on potentials
conditioned_sde = sde.condition_on(potentials)

# 5. Sample posterior trajectories
key = random.PRNGKey(0)
keys = random.split(key, 100)
save_times = jnp.linspace(0, 10, 500)
samples = jax.vmap(conditioned_sde.sample, in_axes=(0, None))(keys, save_times)
```

## Complete Workflows

### Workflow 1: Time Series Interpolation

This example shows how to interpolate sparse observations using a stochastic harmonic oscillator.

```python
import jax
import jax.numpy as jnp
import jax.random as random
from linsdex import TimeSeries, StochasticHarmonicOscillator
from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior

# Step 1: Create sparse observations
obs_times = jnp.linspace(0, 10, 5)
obs_values = jnp.sin(obs_times)[:, None]
observations = TimeSeries(obs_times, obs_values)

# Step 2: Define the SDE for latent dynamics
sde = StochasticHarmonicOscillator(
    freq=1.0,      # Natural frequency
    coeff=0.1,     # Damping
    sigma=0.5,     # Process noise
    observation_dim=1
)

# Step 3: Convert observations to Gaussian potentials
encoder = PaddingLatentVariableEncoderWithPrior(
    y_dim=1,       # Observed dimension
    x_dim=2,       # Latent dimension (position + velocity)
    sigma=0.01     # Observation noise (tight fit to data)
)
potentials = encoder(observations)

# Step 4: Condition SDE on potentials
conditioned_sde = sde.condition_on(potentials)

# Step 5: Sample from the posterior on a dense grid
key = random.PRNGKey(0)
keys = random.split(key, 128)
dense_times = jnp.linspace(0, 10, 2000)

posterior_samples = jax.vmap(
    conditioned_sde.sample, in_axes=(0, None)
)(keys, dense_times)

# posterior_samples.values has shape (128, 2000, 2)
# First dimension of values is position, second is velocity
positions = posterior_samples.values[:, :, 0]
mean_position = positions.mean(axis=0)
std_position = positions.std(axis=0)
```

### Workflow 2: CRF-Based Inference

This example builds a CRF for discrete-time inference.

```python
import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from linsdex import CRF, NaturalGaussian, GaussianTransition, DiagonalMatrix

# Step 1: Define the problem dimensions
dim = 3
seq_len = 20

# Step 2: Create node potentials (observations with uncertainty)
obs_means = jnp.sin(jnp.linspace(0, 4*jnp.pi, seq_len))
obs_precision = 10.0  # Higher = more certain observations

def make_node_potential(obs_mean):
    J = DiagonalMatrix.eye(dim) * obs_precision
    h = jnp.zeros(dim).at[0].set(obs_mean * obs_precision)
    return NaturalGaussian(J, h)

node_potentials = eqx.filter_vmap(make_node_potential)(obs_means)

# Step 3: Create transitions
def make_transition(_):
    A = DiagonalMatrix(jnp.ones(dim) * 0.95)
    u = jnp.zeros(dim)
    Sigma = DiagonalMatrix.eye(dim) * 0.1
    return GaussianTransition(A, u, Sigma)

transitions = eqx.filter_vmap(make_transition)(jnp.arange(seq_len - 1))

# Step 4: Create CRF and perform inference
crf = CRF(node_potentials, transitions)

# Get marginal distributions
marginals = crf.get_marginals()

# Sample from the joint distribution
key = random.PRNGKey(0)
sample = crf.sample(key)  # Shape: (seq_len, dim)

# Compute log probability
log_prob = crf.log_prob(sample)
```

### Workflow 3: Diffusion Model Setup

This example shows how to set up diffusion model conversions for training and sampling.

```python
import jax.numpy as jnp
from linsdex import BrownianMotion, StandardGaussian, DiagonalMatrix
from linsdex.diffusion_model.probability_path import (
    DiffusionModelComponents,
    DiffusionModelConversions
)

# Step 1: Define model components
dim = 64
sde = BrownianMotion(sigma=1.0, dim=dim)

components = DiffusionModelComponents(
    linear_sde=sde,
    t0=0.0,
    x_t0_prior=StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim)),
    t1=1.0,
    evidence_cov=DiagonalMatrix.eye(dim) * 1e-4
)

# Step 2: Training loop (pseudocode)
def compute_loss(params, key, x0_batch):
    # Sample time uniformly
    t = jax.random.uniform(key, minval=0.0, maxval=1.0)

    # Get conversions at this time
    conversions = DiffusionModelConversions(components, t)

    # Sample xt from the forward process
    # xt = ... (depends on your forward sampling strategy)

    # Neural network predicts y1
    # y1_pred = neural_net(params, xt, t)

    # Loss: MSE between predicted and true y1
    # loss = jnp.mean((y1_pred - y1_true) ** 2)
    pass

# Step 3: Sampling (pseudocode)
def sample(params, key):
    # Start from noise at t1
    xt = jax.random.normal(key, (dim,))

    # Integrate backwards using probability flow
    times = jnp.linspace(1.0, 0.0, 100)

    for i in range(len(times) - 1):
        t = times[i]
        conversions = DiffusionModelConversions(components, t)

        # Get y1 prediction from neural network
        # y1_pred = neural_net(params, xt, t)

        # Convert to flow velocity
        # flow = conversions.y1_to_flow(y1_pred, xt)

        # Euler step
        dt = times[i+1] - times[i]
        # xt = xt + flow * dt

    return xt
```

## Mathematical Foundations

### Linear SDEs

A linear SDE has the form:

$$dx_t = (F_t x_t + u_t) dt + L_t dW_t$$

where:
- $x_t \in \mathbb{R}^d$ is the state
- $F_t \in \mathbb{R}^{d \times d}$ is the drift matrix
- $u_t \in \mathbb{R}^d$ is the drift offset
- $L_t \in \mathbb{R}^{d \times m}$ is the diffusion matrix
- $W_t$ is an m-dimensional Wiener process

For linear SDEs, the transition distribution $p(x_t | x_s)$ is Gaussian:

$$p(x_t | x_s) = \mathcal{N}(x_t; A_{t,s} x_s + u_{t,s}, \Sigma_{t,s})$$

The transition parameters are computed by solving the matrix ODE:

$$\frac{d}{dt} \begin{bmatrix} A_t \\ u_t \\ \Sigma_t \end{bmatrix} = \begin{bmatrix} F_t A_t \\ F_t u_t \\ F_t \Sigma_t + \Sigma_t F_t^T + L_t L_t^T \end{bmatrix}$$

### Gaussian Parameterizations

**Standard Form**: $\mathcal{N}(\mu, \Sigma)$

$$p(x) = \frac{1}{Z} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

**Natural Form**: $\mathcal{N}^{-1}(J, h)$ where $J = \Sigma^{-1}$ and $h = \Sigma^{-1}\mu$

$$p(x) = \frac{1}{Z} \exp\left(-\frac{1}{2}x^T J x + h^T x\right)$$

**Mixed Form**: $(\mu, J)$ where $J = \Sigma^{-1}$

The natural form is preferred for combining distributions because adding natural parameters corresponds to multiplying densities:

$$p_1(x) \cdot p_2(x) \propto \mathcal{N}^{-1}(J_1 + J_2, h_1 + h_2)$$

### Message Passing in Gaussian CRFs

A chain-structured CRF has the joint distribution:

$$p(x_1, \ldots, x_T) \propto \prod_{t=1}^T \phi_t(x_t) \prod_{t=1}^{T-1} \psi_t(x_t, x_{t+1})$$

Forward messages (alpha) and backward messages (beta) are computed recursively:

$$\alpha_t(x_t) = \phi_t(x_t) \int \psi_{t-1}(x_{t-1}, x_t) \alpha_{t-1}(x_{t-1}) dx_{t-1}$$

$$\beta_t(x_t) = \int \psi_t(x_t, x_{t+1}) \phi_{t+1}(x_{t+1}) \beta_{t+1}(x_{t+1}) dx_{t+1}$$

The marginal at each node is:

$$p(x_t | y_{1:T}) \propto \alpha_t(x_t) \beta_t(x_t)$$

### Parallel Scan

For chain-structured models, message passing can be parallelized using the associative scan operation. If we define a binary operator $\oplus$ that combines adjacent messages:

$$(m_1 \oplus m_2)(x) = \int m_1(x') \psi(x', x) m_2(x) dx'$$

Then all messages can be computed in $O(\log T)$ parallel steps using the parallel scan algorithm.

## Advanced Topics

### Parallel Scan for Efficient Inference

The `parallel_scan` function enables O(log T) inference on GPUs:

```python
from linsdex.util import parallel_scan

def combine_messages(msg1, msg2):
    # Define how to combine two adjacent messages
    # This must be an associative operation
    return combined_msg

# Apply parallel scan
messages = parallel_scan(combine_messages, initial_messages)
```

### Custom Batchable Objects

Create your own batchable classes using `AbstractBatchableObject`:

```python
import equinox as eqx
from linsdex import AbstractBatchableObject, auto_vmap
from typing import Union, Tuple

class MyBatchableClass(AbstractBatchableObject):
    data: jnp.ndarray
    params: jnp.ndarray

    @property
    def batch_size(self) -> Union[None, int, Tuple[int, ...]]:
        if self.data.ndim == 1:
            return None  # Not batched
        return self.data.shape[0]  # Batch size

    @auto_vmap
    def process(self, x: jnp.ndarray) -> jnp.ndarray:
        # This method is written for unbatched inputs
        # @auto_vmap automatically handles batching
        return self.data * x + self.params
```

### Matrix Tags for Optimization

Tags enable symbolic simplification:

```python
from linsdex import DiagonalMatrix, DenseMatrix, TAGS
import jax.numpy as jnp

# Zero matrix operations are detected symbolically
zero_matrix = DenseMatrix(jnp.zeros((3, 3)), tags=TAGS.zero_tags)
nonzero_matrix = DenseMatrix(jnp.eye(3), tags=TAGS.no_tags)
result = zero_matrix @ nonzero_matrix  # Tags indicate result is zero

# Infinite matrices represent total uncertainty (precision = 0)
inf_matrix = DenseMatrix(jnp.zeros((3, 3)), tags=TAGS.inf_tags)
# Used in potentials to represent uninformative priors
```

### Handling Missing Data

Use masks to handle missing observations:

```python
from linsdex import TimeSeries, GaussianPotentialSeries

# Create series with missing data
times = jnp.linspace(0, 1, 10)
values = jnp.zeros((10, 2))
mask = jnp.array([True, True, False, True, True, False, True, True, True, True])

series = TimeSeries(times, values, mask=mask)

# When creating potentials, missing values get zero precision (total uncertainty)
# This is handled automatically by the encoder
```

### Linear Functionals

Linear functionals represent deferred linear transformations:

```python
import jax.numpy as jnp
from linsdex import LinearFunctional, DiagonalMatrix

# Create a linear functional f(x) = Ax + b
A = DiagonalMatrix(jnp.ones(3) * 2.0)  # A must be a matrix type
b = jnp.ones(3)
functional = LinearFunctional(A, b)

# Apply to a vector
x = jnp.array([1.0, 2.0, 3.0])
result = functional(x)  # = A @ x + b

# Compose functionals: f(g(x))
A2 = DiagonalMatrix(jnp.ones(3) * 0.5)
functional2 = LinearFunctional(A2, jnp.zeros(3))
composed = functional(functional2)  # Returns LinearFunctional for f(g(x))
composed_result = composed(x)  # Evaluate composed function

# Get inverse
inverse = functional.get_inverse()  # f^{-1}(y)
original = inverse(result)  # Should equal x
```

This is particularly useful in diffusion models where we need to track linear relationships between different quantities (y1, score, drift, flow) without immediately computing values.

## Summary

`linsdex` provides a comprehensive toolkit for working with linear-Gaussian probabilistic models:

1. **Matrix types** with symbolic tags avoid unnecessary computation
2. **Three Gaussian parameterizations** ensure numerical stability for different operations
3. **Linear SDEs** with exact transition distributions enable precise modeling
4. **CRF inference** with parallel scan provides efficient GPU-accelerated inference
5. **Diffusion utilities** unify different perspectives on generative modeling
6. **State space model components** simplify the encoding/decoding workflow

The library is designed for both research and production use, with careful attention to numerical stability, computational efficiency, and API design.

For more examples, see the test files in the `tests/` directory and the example scripts in `scripts/`.

## Agent Skills

This library includes Agent Skills for AI-assisted development. Skills provide focused guidance on specific capabilities and can be invoked in Cursor or other compatible AI coding assistants.

### Available Skills

The following skills are available in `.cursor/skills/`:

| Skill | Description |
|-------|-------------|
| `sde-conditioning` | Condition Linear SDEs on observations for time series interpolation, Brownian bridges, and posterior sampling |
| `diffusion-conversions` | Convert between diffusion model representations (y1, score, flow, drift) for training and sampling |
| `crf-inference` | Perform inference in chain-structured Gaussian CRFs with efficient message passing |
| `gaussian-distributions` | Work with three Gaussian parameterizations (Standard, Natural, Mixed) |
| `matrix-operations` | Use specialized matrix types (Diagonal, Dense, Block) with symbolic tags |

### Using Skills

Skills are automatically discovered by compatible AI assistants. You can:

1. **Let the agent decide**: The agent will automatically use relevant skills based on your task
2. **Invoke directly**: Type `/skill-name` (e.g., `/sde-conditioning`) to explicitly use a skill

### Top-Level Skill

The `SKILL.md` file at the project root provides an overview of the library and references all available skills. This is the main entry point for AI assistants working with linsdex.

### Skill Locations

```
linsdex/
├── SKILL.md                              # Main skill entry point
└── .cursor/skills/
    ├── sde-conditioning/SKILL.md         # SDE conditioning workflows
    ├── diffusion-conversions/SKILL.md    # Diffusion model utilities
    ├── crf-inference/SKILL.md            # CRF inference patterns
    ├── gaussian-distributions/SKILL.md   # Gaussian parameterizations
    └── matrix-operations/SKILL.md        # Matrix type system
```
