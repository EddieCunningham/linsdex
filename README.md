# linsdex

`linsdex` is a high performance JAX-based library for linear stochastic differential equations (SDEs), state-space models, and Gaussian inference. It provides a modular and extensible framework for defining, simulating, and conditioning linear-Gaussian systems with support for parallelized inference on GPUs.

## Key Features

The library focuses on high performance and numerical stability.

*   **Linear SDEs**: Comprehensive support for linear time-invariant (LTI) and time-varying SDEs with exact transition distributions.
*   **Efficient Inference**: Sequential and parallel message passing (parallel scan) for filtering, smoothing, and sampling in chain-structured Gaussian CRFs.
*   **Probabilistic Primitives**: Multiple Gaussian parameterizations (Standard, Natural, Mixed) with numerically stable operations.
*   **Specialized Linear Algebra**: A custom matrix library with Diagonal, Block, and Dense types that leverage symbolic tags for optimization.
*   **Diffusion Utilities**: Unified interface for mapping between clean data predictions, scores, and probability flow for generative modeling.
*   **JAX-Native**: Fully compatible with `jax.vmap`, `jax.grad`, and `jax.jit` for automatic vectorization and differentiation.

## Quick Start

Conditioning a base SDE on noisy observations to create a bridge or performing interpolation is straightforward.

```python
import jax.numpy as jnp
import jax.random as random
from linsdex import StochasticHarmonicOscillator, TimeSeries, GaussianPotentialSeries

# 1. Define time series data
times = jnp.linspace(0, 4, 4)
values = jnp.cos(times)[:, None]
series = TimeSeries(times, values)

# 2. Define a linear SDE (base process)
sde = StochasticHarmonicOscillator(
    freq=0.1,
    coeff=0.0,
    sigma=0.2,
    observation_dim=1
)

# 3. Create potentials from data and condition the SDE
potentials = GaussianPotentialSeries(times, values, standard_deviation=0.1)
conditioned_sde = sde.condition_on(potentials)

# 4. Draw samples from the posterior
key = random.PRNGKey(0)
save_times = jnp.linspace(0, 4, 100)
samples = conditioned_sde.sample(key, save_times)
```

## Core Components

The library provides several layers of abstraction for probabilistic modeling.

### Stochastic Differential Equations

The library defines a hierarchy of SDEs starting from `AbstractSDE`.

Linear Time-Invariant (LTI) SDEs are models where the drift and diffusion coefficients are constant over time. Examples include `BrownianMotion`, `OrnsteinUhlenbeck`, and `StochasticHarmonicOscillator`.

Linear SDEs are models with time-varying coefficients, such as `VariancePreserving` SDEs used in diffusion models.

Conditioned SDEs allow for conditioning a process on any number of Gaussian potentials using `ConditionedLinearSDE`.

Forward SDEs represent a process that transforms a sample from an unknown distribution toward a Gaussian prior. The `ForwardSDE` class is essential for diffusion-based generative modeling.

#### Diffusion Conversions

`linsdex` provides a unified interface for working with different mathematical representations of a diffusion process. They allow for mapping between neural network predictions (such as the clean data $y_1$) and quantities required for sampling (such as the probability flow or the drift of an SDE).

The `DiffusionModelComponents` class encapsulates the objects that define a diffusion process, including the base linear SDE, the prior distribution at $t_0$, and the evidence covariance at $t_1$.

```python
from linsdex.diffusion_model.diffusion_conversion import DiffusionModelComponents
from linsdex import BrownianMotion, StandardGaussian, DiagonalMatrix

dim = 10
sde = BrownianMotion(sigma=0.1, dim=dim)
xt0_prior = StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))
evidence_cov = DiagonalMatrix.eye(dim) * 0.001

components = DiffusionModelComponents(
    linear_sde=sde,
    t0=0.0,
    x_t0_prior=xt0_prior,
    t1=1.0,
    evidence_cov=evidence_cov
)
```

The `DiffusionModelConversions` class maps between different parameterizations of the diffusion path, such as converting a prediction into the score function, probability flow, or drift.

```python
from linsdex.diffusion_model.diffusion_conversion import DiffusionModelConversions

conversions = DiffusionModelConversions(components, t=0.5)

# Map clean data prediction y1 to different sampling quantities
flow = conversions.y1_to_flow(y1_pred, xt)
drift = conversions.y1_to_drift(y1_pred, xt)
score = conversions.y1_to_score(xt, y1_pred)
```

The `DiffusionPathQuantities` class can be used to compute and cache time-dependent intermediate quantities, avoiding redundant computations when performing multiple conversions at the same time step. Additionally, `noise_schedule_drift_correction` allows for adjusting the drift when a different noise schedule is used at inference time compared to training.

### Gaussian Potentials

`linsdex` implements Gaussians in three forms to ensure stability across different operations.

`StandardGaussian` uses mean ($\mu$) and covariance ($\Sigma$) parameters. This form is best for sampling and interpreting results.

`NaturalGaussian` uses precision-mean ($h$) and precision ($J$) parameters. This form is best for multiplying densities and message passing.

`MixedGaussian` uses mean ($\mu$) and precision ($J$) parameters. It provides a stable bridge between standard and natural forms, which is particularly useful for Kalman filtering steps.

### Conditional Random Fields (CRF)

The `CRF` class represents a chain-structured probabilistic model. It serves as the engine for discrete-time inference.

```python
from linsdex import CRF

# Create a CRF from node potentials and transitions
crf = CRF(node_potentials, transitions)

# Perform inference
messages = crf.filter() # Forward pass
marginals = crf.get_marginals() # p(x_t | observations)
samples = crf.sample(key) # Draw joint samples
```

For long sequences, `linsdex` uses a parallel scan implementation of message passing to provide $O(\log T)$ complexity on parallel hardware.

### Specialized Matrix Library

To handle structured models efficiently, `linsdex` includes a matrix library that avoids expensive dense operations when possible.

`DiagonalMatrix` is used for decoupled systems or independent noise.

`Block2x2Matrix` and `Block3x3Matrix` are optimized for higher-order tracking models.

Matrices carry symbolic tags like `TAGS.zero_tags` and `TAGS.eye_tags`. These tags allow the library to symbolically simplify expressions like $0 \times A$ or $I \times B$ before they reach JAX.

## Installation

```bash
pip install .
```

## Citation

If you use `linsdex` in your research, please cite the following software.

```bibtex
@software{cunningham2025linsdex,
  author       = {Cunningham, Edmond},
  title        = {{linsdex}: A High-Performance JAX-based Library for Linear Stochastic Differential Equations},
  version      = {0.1.0},
  url          = {https://github.com/EddieCunningham/linsdex},
  note         = {Python package},
}
```
