# linsdex
`linsdex` is a high performance JAX-based Python package for linear time-invariant stochastic differential equations (LTI-SDEs).
```python
# Create time series data
times = jnp.linspace(0, 4, 4)
values = jnp.cos(times)[:,None]
series = TimeSeries(times, values)

# Create a linear SDE to interpolate the time series
sde = StochasticHarmonicOscillator(
  freq=0.1,
  coeff=0.0,
  sigma=0.2,
  observation_dim=1
)

# Create potential functions from the data to condition the SDE
potentials = GaussianPotentialSeries(times, values, standard_deviation=0.1)
conditioned_sde: ConditionedLinearSDE = sde.condition_on(potentials)

# Pull samples from the conditioned SDE
n_samples = 100
keys = random.split(key, n_samples)
save_times = jnp.linspace(0, 4, 1000)
vmapped_sample_fn = jax.vmap(conditioned_sde.sample, in_axes=(0, None))
samples: TimeSeries = vmapped_sample_fn(keys, save_times)

# Plot the results
samples.plot()
```

# Installation

```bash
pip install .
```

## SDEs implemented
Check out the [sde_examples.py](linsdex/sde/sde_examples.py) file for the SDEs that are implemented.  This list includes:

- Brownian motion
- Ornstein-Uhlenbeck
- Variance preserving SDE
- Wiener velocity model
- Critically damped Langevin dynamics
- Third order Langevin dynamics
- Stochastic harmonic oscillator

## How it works
Inference in `linsdex` is done by discretizing the SDE and potentials into a CRF, and then using message passing to perform inference.  This library only considers linear SDEs and Gaussian potentials, and so the CRF is a chain structured Gaussian CRF, which is a probabilitic model that admits fast, closed form inference.

## Citation
If you use linsdex in your research, please cite:

```bibtex
@software{cunningham2025linsdex,
  author       = {Cunningham, Edmond},
  title        = {{linsdex}: A High-Performance JAX-based Library for Linear Time-Invariant Stochastic Differential Equations},
  version      = {0.1.0},
  url          = {https://github.com/EddieCunningham/linsdex},
  note         = {Python package},
}
```