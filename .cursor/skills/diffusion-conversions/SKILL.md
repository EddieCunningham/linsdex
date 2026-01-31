---
name: diffusion-conversions
description: Convert between diffusion model representations including clean data predictions (y1), scores, probability flows, and drifts. Use when building or training diffusion-based generative models.
---

# Diffusion Model Conversions

A unified interface for mapping between different mathematical representations of diffusion processes, essential for training and sampling from diffusion-based generative models.

## When to Use

- Building diffusion-based generative models
- Converting neural network predictions (y1, epsilon) to sampling quantities (flow, drift, score)
- Implementing custom samplers for diffusion models
- Understanding the relationships between different diffusion parameterizations

## Key Concepts

### Diffusion Model Components

A diffusion process is defined by:

- A Linear SDE that governs the forward noising process
- A prior distribution at t0 (approximating the data distribution)
- An evidence covariance at t1 (observation noise at the end)

### Conversions Available

From clean data prediction y1:
- `y1_to_flow(y1, xt)` - Probability flow ODE velocity
- `y1_to_drift(y1, xt)` - SDE drift term
- `y1_to_score(xt, y1)` - Score function ∇log p(xt)

From noise prediction epsilon:
- `epsilon_to_flow(epsilon, xt)` - Probability flow from noise prediction

Inverse conversions:
- `score_to_y1(xt, score)` - Recover y1 from score
- `flow_to_y1(xt, flow)` - Recover y1 from flow
- `drift_to_y1(xt, drift)` - Recover y1 from drift

## Code Examples

### Setting Up Diffusion Components

```python
import jax.numpy as jnp
from linsdex import BrownianMotion, StandardGaussian, DiagonalMatrix
from linsdex.diffusion_model.probability_path import DiffusionModelComponents

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

### Converting Predictions During Training

```python
from linsdex.diffusion_model.probability_path import DiffusionModelConversions

# At a specific time t during training
t = 0.5
conversions = DiffusionModelConversions(components, t)

# Neural network predicts clean data y1
y1_pred = model(xt, t)  # Your neural network

# Convert to different quantities for loss computation
xt = ...  # Current noisy state
flow = conversions.y1_to_flow(y1_pred, xt)
drift = conversions.y1_to_drift(y1_pred, xt)
score = conversions.y1_to_score(xt, y1_pred)
```

### Converting from Epsilon Prediction

```python
# If your model predicts noise instead of clean data
epsilon_pred = noise_model(xt, t)

flow = conversions.epsilon_to_flow(epsilon_pred, xt)
```

### Inverse Conversions

```python
# Convert back from score/flow/drift to y1
y1_from_score = conversions.score_to_y1(xt, score)
y1_from_flow = conversions.flow_to_y1(xt, flow)
y1_from_drift = conversions.drift_to_y1(xt, drift)
```

### Using ProbabilityPathSlice for Efficiency

Cache intermediate quantities when performing multiple conversions at the same time:

```python
from linsdex.diffusion_model.probability_path import ProbabilityPathSlice

# Compute and cache quantities at time t
path_slice = ProbabilityPathSlice(components, t)

# Access cached values
beta_precision = path_slice.beta_precision
marginal_precision = path_slice.marginal_precision
```

### Batch Computation with get_probability_path

When computing probability path slices at multiple times, use `get_probability_path` for efficiency. It computes all slices using only 2 ODE solves total, regardless of the number of times:

```python
from linsdex.diffusion_model.probability_path import get_probability_path

# Efficient: computes all slices with just 2 ODE solves
times = jnp.linspace(0.0, 1.0, 100)
path_slices = get_probability_path(components, times)

# path_slices is a batched ProbabilityPathSlice
# Access individual slices by indexing
slice_at_t50 = path_slices[50]

# Or use with vmap for parallel operations
def compute_flow(path_slice, xt, y1_pred):
    conversions = DiffusionModelConversions(path_slice.components, path_slice.t)
    return conversions.y1_to_flow(y1_pred, xt)

flows = jax.vmap(compute_flow)(path_slices, xts, y1_preds)
```

This is much more efficient than calling `ProbabilityPathSlice` individually for each time, which would require 2n ODE solves for n times.

### Sampling Loop (Pseudocode)

```python
import jax
import jax.numpy as jnp

def sample(params, key, components):
    dim = components.linear_sde.dim
    
    # Start from noise at t1
    xt = jax.random.normal(key, (dim,))
    
    # Integrate backwards using probability flow
    times = jnp.linspace(1.0, 0.0, 100)
    
    for i in range(len(times) - 1):
        t = times[i]
        conversions = DiffusionModelConversions(components, t)
        
        # Get y1 prediction from neural network
        y1_pred = neural_net(params, xt, t)
        
        # Convert to flow velocity
        flow = conversions.y1_to_flow(y1_pred, xt)
        
        # Euler step
        dt = times[i+1] - times[i]
        xt = xt + flow * dt
    
    return xt
```

### Training Loop (Pseudocode)

```python
def compute_loss(params, key, x0_batch, components):
    # Sample time uniformly
    t = jax.random.uniform(key, minval=0.0, maxval=1.0)
    
    # Get conversions at this time
    conversions = DiffusionModelConversions(components, t)
    
    # Forward diffuse x0 to get xt
    # (implementation depends on your forward sampling strategy)
    xt = forward_sample(x0_batch, t, components)
    
    # Neural network predicts y1 (clean data)
    y1_pred = neural_net(params, xt, t)
    
    # Loss: MSE between predicted and true clean data
    loss = jnp.mean((y1_pred - x0_batch) ** 2)
    
    return loss
```

## Key Classes and Functions

- `DiffusionModelComponents` - Encapsulates SDE, prior, and evidence covariance
- `DiffusionModelConversions` - Conversion methods at a specific time t
- `ProbabilityPathSlice` - Cached intermediate quantities for efficiency
- `get_probability_path(components, times)` - Efficiently compute slices at multiple times with only 2 ODE solves

## Mathematical Background

The relationships between quantities are:

- Score: ∇_xt log p(xt | y1)
- Flow: The velocity field for the probability flow ODE
- Drift: The drift term in the reverse-time SDE

These are all linear functions of y1 when using Linear SDEs, which is why conversions are exact and efficient.

## Tips

- Use `ProbabilityPathSlice` when computing multiple quantities at the same time step
- The conversions are exact for Linear SDEs (no approximations)
- For training, predict y1 (clean data) rather than epsilon for better interpretability
- The library handles all the linear algebra optimizations automatically
