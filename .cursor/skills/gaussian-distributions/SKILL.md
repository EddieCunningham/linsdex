---
name: gaussian-distributions
description: Work with Gaussian distributions in three parameterizations for numerical stability and efficiency. Use when you need to sample, combine distributions, or convert between mean/covariance and precision/natural forms.
---

# Gaussian Distributions

linsdex implements Gaussian distributions in three parameterizations, each optimal for different operations.

## When to Use

- Sampling from Gaussian distributions
- Combining multiple Gaussian observations
- Converting between parameterizations for numerical stability
- Building probabilistic models with Gaussian components

## Three Parameterizations

### StandardGaussian(μ, Σ)

Mean and covariance form. Best for sampling and interpretation.

```python
from linsdex import StandardGaussian, DiagonalMatrix
import jax
import jax.numpy as jnp

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

### NaturalGaussian(J, h)

Precision J = Σ⁻¹ and precision-weighted mean h = Jμ. Best for combining distributions.

```python
from linsdex import NaturalGaussian, DiagonalMatrix

dim = 3
J = DiagonalMatrix.eye(dim)  # Precision matrix
h = jnp.zeros(dim)           # Precision-weighted mean

nat_dist = NaturalGaussian(J, h)

# Adding natural Gaussians multiplies densities (efficient!)
combined = nat_dist + nat_dist  # Product of two Gaussians
```

### MixedGaussian(μ, J)

Mean and precision form. A stable bridge between standard and natural.

```python
from linsdex import MixedGaussian, DiagonalMatrix

dim = 3
mu = jnp.zeros(dim)
J = DiagonalMatrix.eye(dim)  # Precision

mixed_dist = MixedGaussian(mu, J)

# Useful for Kalman filter updates where we have mean but need precision
```

## Converting Between Parameterizations

```python
from linsdex import StandardGaussian, DiagonalMatrix

dim = 2
mu = jnp.array([1.0, 2.0])
Sigma = DiagonalMatrix(jnp.array([0.5, 1.0]))

# Start with standard form
std_dist = StandardGaussian(mu, Sigma)

# Convert to natural form
nat_dist = std_dist.to_nat()

# Convert to mixed form
mixed_dist = std_dist.to_mixed()

# Convert back to standard
std_from_nat = nat_dist.to_std()
std_from_mixed = mixed_dist.to_std()
```

## When to Use Each Parameterization

| Parameterization | Best For |
|-----------------|----------|
| StandardGaussian | Sampling, interpretation, visualization |
| NaturalGaussian | Combining observations, message passing, CRF inference |
| MixedGaussian | Kalman filter updates, bridging between forms |

## Code Examples

### Combining Gaussian Observations

```python
from linsdex import NaturalGaussian, DiagonalMatrix

# Two independent observations of the same quantity
obs1_precision = 10.0
obs1_mean = 1.0

obs2_precision = 5.0
obs2_mean = 1.5

dim = 1
J1 = DiagonalMatrix(jnp.array([obs1_precision]))
h1 = jnp.array([obs1_mean * obs1_precision])
obs1 = NaturalGaussian(J1, h1)

J2 = DiagonalMatrix(jnp.array([obs2_precision]))
h2 = jnp.array([obs2_mean * obs2_precision])
obs2 = NaturalGaussian(J2, h2)

# Combine observations (product of Gaussians)
combined = obs1 + obs2

# Convert to standard form to see combined mean
combined_std = combined.to_std()
print(f"Combined mean: {combined_std.mu}")
print(f"Combined variance: {combined_std.Sigma.get_elements()}")
```

### Gaussian Transitions

`GaussianTransition` represents conditional distributions p(y|x) = N(y; Ax + u, Σ).

```python
from linsdex import GaussianTransition, DiagonalMatrix

dim = 2
A = DiagonalMatrix(jnp.ones(dim) * 0.9)  # Decay factor
u = jnp.array([0.1, -0.1])                # Offset
Sigma = DiagonalMatrix.eye(dim) * 0.01    # Noise covariance

transition = GaussianTransition(A, u, Sigma)

# Apply to a point to get the conditional distribution
x = jnp.ones(dim)
p_y_given_x = transition.condition_on_x(x)  # Returns StandardGaussian

# Chain transitions: p(z|x) from p(y|x) and p(z|y)
transition2 = GaussianTransition(A, u, Sigma)
chained = transition.chain(transition2)  # p(z|x)
```

### Gaussian Potential Series

For time series of Gaussian observations:

```python
from linsdex import GaussianPotentialSeries

times = jnp.array([0.0, 1.0, 2.0, 3.0])
values = jnp.array([[1.0], [2.0], [3.0], [4.0]])
std_dev = jnp.array([[0.1], [0.1], [0.1], [0.1]])

# Create with standard deviation
potentials = GaussianPotentialSeries(ts=times, xts=values, standard_deviation=std_dev)

# Or with certainty (inverse variance)
certainty = jnp.array([[100.0], [100.0], [100.0], [100.0]])
potentials = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)
```

### Sampling and Log Probability

```python
import jax
import jax.random as random
from linsdex import StandardGaussian, DiagonalMatrix

dim = 5
mu = jnp.zeros(dim)
Sigma = DiagonalMatrix.eye(dim)
dist = StandardGaussian(mu, Sigma)

# Single sample
key = random.PRNGKey(0)
sample = dist.sample(key)

# Batch sampling
keys = random.split(key, 100)
samples = jax.vmap(dist.sample)(keys)

# Log probability
log_p = dist.log_prob(sample)

# Batch log probability
log_ps = jax.vmap(dist.log_prob)(samples)
```

## Mathematical Background

**Standard Form**: N(μ, Σ)

p(x) = (1/Z) exp(-½(x - μ)ᵀ Σ⁻¹ (x - μ))

**Natural Form**: N⁻¹(J, h) where J = Σ⁻¹ and h = Σ⁻¹μ

p(x) = (1/Z) exp(-½xᵀJx + hᵀx)

**Product of Gaussians**: In natural form, multiplication becomes addition:

p₁(x) × p₂(x) ∝ N⁻¹(J₁ + J₂, h₁ + h₂)

This is why natural parameterization is preferred for message passing and combining observations.

## Key Classes

- `StandardGaussian(mu, Sigma)` - Mean and covariance
- `NaturalGaussian(J, h)` - Precision and precision-weighted mean
- `MixedGaussian(mu, J)` - Mean and precision
- `GaussianTransition(A, u, Sigma)` - Linear-Gaussian conditional
- `GaussianPotentialSeries` - Time series of Gaussian potentials

## Tips

- Use `NaturalGaussian` when you need to combine multiple observations
- Use `StandardGaussian` for sampling and final results
- The conversions are exact and handle edge cases numerically
- All Gaussian types work with the matrix types (`DiagonalMatrix`, `DenseMatrix`, etc.)
- Use `DiagonalMatrix` for covariances when dimensions are independent
