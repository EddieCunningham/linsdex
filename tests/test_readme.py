import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from linsdex import BrownianMotion, TimeSeries, GaussianPotentialSeries, StochasticHarmonicOscillator
from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior
import matplotlib.pyplot as plt
import jax

def test_quick_start():
  # 1. Define time series data
  # We create sparse 1D data for interpolation
  times = jnp.linspace(0, 10, 5)
  values = jnp.sin(times)[:, None]
  series = TimeSeries(times, values)

  # 2. Define a linear SDE (base process)
  # We use a Stochastic Harmonic Oscillator which has a 2D state (1D position, 1D velocity)
  sde = StochasticHarmonicOscillator(
      freq=1.0,
      coeff=0.1,
      sigma=0.5,
      observation_dim=1
  )

  # 3. Create potentials from data and condition the SDE
  # PaddingLatentVariableEncoderWithPrior pads the 1D observations to the 2D latent space
  encoder = PaddingLatentVariableEncoderWithPrior(
      y_dim=1,
      x_dim=2,
      sigma=0.01
  )
  potentials = encoder(series)
  conditioned_sde = sde.condition_on(potentials)

  # 4. Draw samples from the posterior
  key = random.PRNGKey(0)
  keys = random.split(key, 128)

  # Interpolate on a denser time grid
  save_times = jnp.linspace(0, 10, 2000)
  samples: TimeSeries = jax.vmap(conditioned_sde.sample, in_axes=(0, None))(keys, save_times)

  # 5. Plot the original time series and the posterior samples
  fig, axes = samples.plot(show_plot=False)
  plt.savefig("quick_start.png")
  plt.close()

def test_probability_path():
  from linsdex.diffusion_model.probability_path import DiffusionModelComponents
  from linsdex import BrownianMotion, StandardGaussian, DiagonalMatrix
  from linsdex.diffusion_model.probability_path import DiffusionModelConversions

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

  conversions = DiffusionModelConversions(components, t=0.5)

  y1_pred = jnp.zeros(dim)
  xt = jnp.zeros(dim)

  # Map clean data prediction y1 to different sampling quantities
  flow = conversions.y1_to_flow(y1_pred, xt)
  drift = conversions.y1_to_drift(y1_pred, xt)
  score = conversions.y1_to_score(xt, y1_pred)

  assert flow.shape == (dim,)
  assert drift.shape == (dim,)
  assert score.shape == (dim,)

def test_crf():
  from linsdex import CRF, NaturalGaussian, GaussianTransition, DiagonalMatrix
  import jax.random as random

  dim = 2
  seq_len = 5
  key = random.PRNGKey(0)

  # Create dummy node potentials
  J = DiagonalMatrix.eye(dim)
  h = jnp.zeros(dim)
  node_potentials = eqx.filter_vmap(lambda _: NaturalGaussian(J, h))(jnp.arange(seq_len))

  # Create dummy transitions
  A = DiagonalMatrix.eye(dim)
  u = jnp.zeros(dim)
  Sigma = DiagonalMatrix.eye(dim)
  transitions = eqx.filter_vmap(lambda _: GaussianTransition(A, u, Sigma))(jnp.arange(seq_len - 1))

  # Create a CRF from node potentials and transitions
  crf = CRF(node_potentials, transitions)

  # Perform inference
  messages = crf.get_forward_messages() # Forward pass
  marginals = crf.get_marginals() # p(x_t | observations)
  samples = crf.sample(key) # Draw joint samples

  assert marginals.batch_size == seq_len
  assert samples.shape == (seq_len, dim)
