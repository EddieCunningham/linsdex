---
name: crf-inference
description: Perform inference in chain-structured Gaussian Conditional Random Fields using efficient message passing. Use for discrete-time probabilistic modeling, computing marginals, sampling joint distributions, or Kalman-style filtering and smoothing.
---

# CRF Inference

Chain-structured Gaussian Conditional Random Fields (CRFs) for efficient discrete-time probabilistic inference with O(log T) complexity on parallel hardware.

## When to Use

- Discrete-time state-space model inference
- Computing marginal distributions at each timestep
- Sampling from joint distributions over sequences
- Kalman filtering and smoothing
- Any chain-structured Gaussian graphical model

## Key Concepts

### CRF Structure

A chain CRF has the joint distribution:

p(x1, ..., xT) ∝ ∏ φt(xt) × ∏ ψt(xt, xt+1)

Where:
- φt(xt) are node potentials (observations/priors at each timestep)
- ψt(xt, xt+1) are transition potentials (dynamics between timesteps)

### Message Passing

The CRF performs message passing to compute:
- Forward messages (filtering)
- Backward messages (smoothing preparation)
- Marginals at each timestep
- Samples from the joint distribution

### Parallel Scan

For long sequences, linsdex uses parallel scan for O(log T) complexity instead of O(T) sequential message passing.

## Code Examples

### Building a CRF

```python
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from linsdex import CRF, NaturalGaussian, GaussianTransition, DiagonalMatrix

dim = 2
seq_len = 10

# Create node potentials (one per timestep)
J = DiagonalMatrix.eye(dim)
h = jnp.zeros(dim)
single_potential = NaturalGaussian(J, h)
node_potentials = eqx.filter_vmap(lambda _: single_potential)(jnp.arange(seq_len))

# Create transitions between adjacent nodes
A = DiagonalMatrix.eye(dim)  # Identity transition matrix
u = jnp.zeros(dim)
Sigma = DiagonalMatrix.eye(dim) * 0.1
single_transition = GaussianTransition(A, u, Sigma)
transitions = eqx.filter_vmap(lambda _: single_transition)(jnp.arange(seq_len - 1))

# Create the CRF
crf = CRF(node_potentials, transitions)
```

### Computing Marginals

```python
# Get marginal distributions at each timestep
marginals = crf.get_marginals()

# marginals is a batch of Gaussian distributions
# Access mean and covariance at each timestep
means = marginals.mu  # Shape: (seq_len, dim)
```

### Forward and Backward Messages

```python
# Forward messages (filtering)
fwd_messages = crf.get_forward_messages()

# Backward messages (smoothing preparation)
bwd_messages = crf.get_backward_messages()
```

### Sampling from the CRF

```python
key = random.PRNGKey(0)

# Sample a sequence from the joint distribution
sample = crf.sample(key)  # Shape: (seq_len, dim)

# Sample multiple sequences
keys = random.split(key, 100)
samples = jax.vmap(crf.sample)(keys)  # Shape: (100, seq_len, dim)
```

### Evaluating Log Probability

```python
# Compute log probability of a sequence
log_prob = crf.log_prob(sample)
```

### Parallel vs Sequential Inference

```python
# By default, uses parallel scan for O(log T) complexity
marginals = crf.get_marginals()

# Can also use sequential scan (useful for debugging or small sequences)
marginals_seq = crf.get_marginals(parallel=False)
```

### CRF with Varying Observation Precision

```python
# Create node potentials with varying observation certainty
obs_means = jnp.sin(jnp.linspace(0, 4*jnp.pi, seq_len))
obs_precision = 10.0  # Higher = more certain observations

def make_node_potential(obs_mean):
    J = DiagonalMatrix.eye(dim) * obs_precision
    h = jnp.zeros(dim).at[0].set(obs_mean * obs_precision)
    return NaturalGaussian(J, h)

node_potentials = eqx.filter_vmap(make_node_potential)(obs_means)

# Create transitions with decay
def make_transition(_):
    A = DiagonalMatrix(jnp.ones(dim) * 0.95)  # Decay factor
    u = jnp.zeros(dim)
    Sigma = DiagonalMatrix.eye(dim) * 0.1
    return GaussianTransition(A, u, Sigma)

transitions = eqx.filter_vmap(make_transition)(jnp.arange(seq_len - 1))

crf = CRF(node_potentials, transitions)
marginals = crf.get_marginals()
```

### Using NaturalGaussian for Combining Observations

```python
from linsdex import NaturalGaussian

# Natural parameterization is efficient for combining observations
# Adding natural Gaussians multiplies densities

dim = 3
J1 = DiagonalMatrix.eye(dim)
h1 = jnp.array([1.0, 0.0, 0.0])
obs1 = NaturalGaussian(J1, h1)

J2 = DiagonalMatrix.eye(dim) * 2.0
h2 = jnp.array([0.0, 2.0, 0.0])
obs2 = NaturalGaussian(J2, h2)

# Combine observations (product of Gaussians)
combined = obs1 + obs2  # J = J1 + J2, h = h1 + h2
```

## Key Classes

- `CRF(node_potentials, transitions)` - Chain-structured CRF
- `NaturalGaussian(J, h)` - Node potentials in natural parameterization
- `GaussianTransition(A, u, Sigma)` - Linear-Gaussian transitions p(xt+1 | xt)

## GaussianTransition Details

A `GaussianTransition` represents p(y | x) = N(y; Ax + u, Σ):

- `A` - Linear transformation matrix
- `u` - Offset/bias term
- `Sigma` - Noise covariance

```python
transition = GaussianTransition(A, u, Sigma)

# Apply to a point to get conditional distribution
x = jnp.ones(dim)
p_y_given_x = transition.condition_on_x(x)  # Returns StandardGaussian

# Chain transitions: p(z|x) from p(y|x) and p(z|y)
transition2 = GaussianTransition(A2, u2, Sigma2)
chained = transition.chain(transition2)  # p(z|x)
```

## Tips

- Use `NaturalGaussian` for node potentials because addition corresponds to multiplying densities
- The CRF automatically uses parallel scan on GPU for sequences longer than a threshold
- For very long sequences, the O(log T) complexity provides significant speedups
- Use `DiagonalMatrix` for covariances when dimensions are independent to save computation
- The `equinox.filter_vmap` pattern is useful for creating batched potentials and transitions
