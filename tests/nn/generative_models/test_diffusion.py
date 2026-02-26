import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Optional, Tuple, Dict, Any, Annotated
import equinox as eqx

from linsdex import TimeSeries
from linsdex.nn.generative_models.diffusion import DiffusionTimeSeriesModel, DiffusionTimeSeriesModelConfig, DiffusionModel, DiffusionModelConfig, ImprovedDiffusionTimeSeriesModel, ImprovedDiffusionTimeSeriesModelConfig


class DummyDatasetConfig(eqx.Module):
  """Dummy dataset configuration for testing."""
  name: str = "test_dataset"
  unique_name: str = "test_dataset_unique"
  n_features: int = 2
  train_proportion: float = 0.7
  val_proportion: float = 0.15
  test_proportion: float = 0.15

  seq_length: int = 50
  observation_noise: float = 0.1
  process_noise: float = 0.1
  time_scale_mult: float = 1.0
  condition_length: int = 20


@pytest.fixture
def test_time_series():
  """Create a test time series for DiffusionTimeSeriesModel testing."""
  times = jnp.linspace(0, 2, 50)
  values = jnp.stack([
    jnp.cos(2 * jnp.pi * times),
    jnp.sin(2 * jnp.pi * times)
  ], axis=-1)
  return TimeSeries(times, values)


@pytest.fixture
def test_dataset_config():
  """Create a test dataset configuration."""
  return DummyDatasetConfig()


def create_diffusion_time_series_config() -> DiffusionTimeSeriesModelConfig:
  """Helper function to create DiffusionTimeSeriesModelConfig."""
  return DiffusionTimeSeriesModelConfig(
    name="diffusion_time_series",
    process_noise=0.1,
    condition_length=20,
    sde_type='brownian',
    nn_type='time_dependent_gru_rnn',
    # Dynamic fields for GRU model
    hidden_size=10,
    n_layers=1,
    intermediate_channels=10,
    # Additional required fields for get_nn
    cond_in_channels=2,
    in_channels=2,
    out_channels=2
  )


class TestDiffusionTimeSeriesModelInitialization:
  """Test DiffusionTimeSeriesModel initialization."""

  def test_init_success(self, test_dataset_config):
    """Test successful initialization of DiffusionTimeSeriesModel."""
    model_config = create_diffusion_time_series_config()
    random_seed = 42

    model = DiffusionTimeSeriesModel(model_config, test_dataset_config, random_seed)

    assert model is not None
    assert model.condition_length == 20
    assert model.linear_sde.dim == 100  # n_features (2) * seq_length (50)
    assert model.encoder.y_dim == 2
    assert model.encoder.x_dim == 100
    assert model.nn is not None
    assert model.model_config == model_config
    assert model.dataset_config == test_dataset_config


class TestDiffusionTimeSeriesModelMethods:
  """Test DiffusionTimeSeriesModel methods (loss_fn and sample)."""

  @pytest.fixture
  def model(self, test_dataset_config):
    """Create a DiffusionTimeSeriesModel for testing."""
    model_config = create_diffusion_time_series_config()
    random_seed = 42
    return DiffusionTimeSeriesModel(model_config, test_dataset_config, random_seed)

  def test_loss_fn(self, model, test_time_series):
    """Test loss_fn method runs without errors."""
    key = random.PRNGKey(42)

    loss, loss_dict = model.loss_fn(key, test_time_series, debug=False)

    assert isinstance(loss, (float, jnp.ndarray))
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert isinstance(loss_dict, dict)
    assert 'flow_matching' in loss_dict

  def test_sample(self, model, test_time_series):
    """Test sample method runs without errors."""
    key = random.PRNGKey(42)

    sampled_series = model.sample(key, test_time_series, debug=False)

    assert isinstance(sampled_series, TimeSeries)
    assert sampled_series.times.shape == test_time_series.times.shape
    assert sampled_series.values.shape == test_time_series.values.shape
    assert jnp.all(jnp.isfinite(sampled_series.values))


class TestDiffusionTimeSeriesModelFactory:
  """Test the get_probabilistic_model factory function."""

  def test_get_probabilistic_model_diffusion_time_series(self, test_dataset_config):
    """Test that get_probabilistic_model correctly creates a DiffusionTimeSeriesModel."""
    from linsdex.nn.generative_models.get_probabilistic_model import get_probabilistic_model

    model_config = create_diffusion_time_series_config()
    random_seed = 42

    # Use the factory function
    model = get_probabilistic_model(model_config, test_dataset_config, random_seed)

    # Verify it's the correct type
    assert isinstance(model, DiffusionTimeSeriesModel)
    assert model.condition_length == 20
    assert model.model_config == model_config
    assert model.dataset_config == test_dataset_config
