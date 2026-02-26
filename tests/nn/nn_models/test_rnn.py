import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from dataclasses import dataclass
from linsdex.nn.nn_models.rnn import (
  StackedGRUSequenceHypers,
  StackedGRURNN,
  GRUSeq2SeqHypers,
  GRUSeq2SeqModel
)
from linsdex import TimeSeries
from linsdex.nn.nn_models.get_nn import get_nn
# from experiment.config_base import AbstractModelConfig, AbstractDatasetConfig, EnvironmentConfig, TrainerConfig, ExperimentConfig


class GRURNNModelConfig(eqx.Module):
  """Model config for GRU RNN models"""
  model_type: str
  cond_in_channels: int
  in_channels: int
  out_channels: int
  hidden_size: int
  n_layers: int
  intermediate_channels: int
  nn_type: str = "time_dependent_gru_rnn"
  name: str = "gru_rnn"


class DummyDatasetConfig(eqx.Module):
  """Dummy dataset config for testing purposes"""
  n_features: int = 4
  pass


@pytest.fixture
def key():
  return random.PRNGKey(42)


class TestStackedGRUSequenceHypers:
  """Test suite for StackedGRUSequenceHypers"""

  def test_stacked_gru_sequence_hypers_initialization(self):
    """Test StackedGRUSequenceHypers initialization."""
    hypers = StackedGRUSequenceHypers(
      hidden_size=16,
      intermediate_channels=32,
      num_layers=3
    )

    assert hypers.hidden_size == 16
    assert hypers.intermediate_channels == 32
    assert hypers.num_layers == 3

  def test_stacked_gru_sequence_hypers_different_values(self):
    """Test StackedGRUSequenceHypers with different parameter values."""
    test_configs = [
      (8, 16, 2),
      (32, 64, 4),
      (12, 24, 5)
    ]

    for hidden_size, intermediate_channels, num_layers in test_configs:
      hypers = StackedGRUSequenceHypers(
        hidden_size=hidden_size,
        intermediate_channels=intermediate_channels,
        num_layers=num_layers
      )

      assert hypers.hidden_size == hidden_size
      assert hypers.intermediate_channels == intermediate_channels
      assert hypers.num_layers == num_layers


class TestStackedGRURNN:
  """Test suite for StackedGRURNN"""

  def test_stacked_gru_rnn_initialization(self, key):
    """Test StackedGRURNN initialization."""
    in_channels = 4
    out_channels = 6
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    assert hasattr(model, 'gru_blocks')
    assert hasattr(model, 'initial_state')
    assert hasattr(model, 'in_proj')
    assert hasattr(model, 'out_proj')
    assert hasattr(model, 'hypers')

    # Check shapes
    assert model.initial_state.shape == (hypers.num_layers, hypers.hidden_size)
    assert model.in_proj.in_features == in_channels
    assert model.in_proj.out_features == hypers.intermediate_channels
    assert model.out_proj.in_features == hypers.intermediate_channels
    assert model.out_proj.out_features == out_channels

    # Check hypers
    assert model.hypers.hidden_size == hypers.hidden_size
    assert model.hypers.intermediate_channels == hypers.intermediate_channels
    assert model.hypers.num_layers == hypers.num_layers

  def test_stacked_gru_rnn_forward_pass(self, key):
    """Test StackedGRURNN forward pass."""
    in_channels = 4
    out_channels = 6
    seq_length = 10
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))
    output = model(x)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_stacked_gru_rnn_different_sizes(self, key):
    """Test StackedGRURNN with different input/output sizes."""
    test_configs = [
      (2, 4, 8, 16, 2),
      (6, 8, 12, 24, 3),
      (10, 5, 16, 32, 4)
    ]

    seq_length = 15

    for in_channels, out_channels, hidden_size, intermediate_channels, num_layers in test_configs:
      hypers = StackedGRUSequenceHypers(
        hidden_size=hidden_size,
        intermediate_channels=intermediate_channels,
        num_layers=num_layers
      )

      model = StackedGRURNN(
        in_channels=in_channels,
        out_channels=out_channels,
        hypers=hypers,
        key=key
      )

      x = random.normal(key, (seq_length, in_channels))
      output = model(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()

  def test_stacked_gru_rnn_different_sequence_lengths(self, key):
    """Test StackedGRURNN with different sequence lengths."""
    in_channels = 4
    out_channels = 6
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    for seq_length in [1, 5, 10, 25, 50]:
      x = random.normal(key, (seq_length, in_channels))
      output = model(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()

  def test_stacked_gru_rnn_different_num_layers(self, key):
    """Test StackedGRURNN with different numbers of layers."""
    in_channels = 6
    out_channels = 8
    seq_length = 12

    for num_layers in [1, 2, 3, 5]:
      hypers = StackedGRUSequenceHypers(
        hidden_size=10,
        intermediate_channels=16,
        num_layers=num_layers
      )

      model = StackedGRURNN(
        in_channels=in_channels,
        out_channels=out_channels,
        hypers=hypers,
        key=key
      )

      x = random.normal(key, (seq_length, in_channels))
      output = model(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()
      assert model.initial_state.shape == (num_layers, 10)

  def test_stacked_gru_rnn_with_global_context(self, key):
    """Test StackedGRURNN with global context."""
    in_channels = 4
    out_channels = 6
    seq_length = 10
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    k1, k2 = random.split(key, 2)
    x = random.normal(k1, (seq_length, in_channels))
    global_context = random.normal(k2, (hypers.num_layers, hypers.hidden_size))

    output_with_context = model(x, global_context)
    output_without_context = model(x)

    assert output_with_context.shape == (seq_length, out_channels)
    assert output_without_context.shape == (seq_length, out_channels)
    assert jnp.isfinite(output_with_context).all()
    assert jnp.isfinite(output_without_context).all()

    # Outputs should be different when using global context
    assert not jnp.allclose(output_with_context, output_without_context, atol=1e-6)

  def test_stacked_gru_rnn_different_global_contexts(self, key):
    """Test StackedGRURNN with different global contexts."""
    in_channels = 4
    out_channels = 6
    seq_length = 10
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    k1, k2, k3 = random.split(key, 3)
    x = random.normal(k1, (seq_length, in_channels))
    context1 = random.normal(k2, (hypers.num_layers, hypers.hidden_size))
    context2 = random.normal(k3, (hypers.num_layers, hypers.hidden_size))

    output1 = model(x, context1)
    output2 = model(x, context2)

    # Different global contexts should produce different outputs
    assert not jnp.allclose(output1, output2, atol=1e-6)

  def test_stacked_gru_rnn_batch_size_property(self, key):
    """Test StackedGRURNN batch_size property."""
    in_channels = 4
    out_channels = 6
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Should be None for unbatched
    assert model.batch_size is None

  def test_stacked_gru_rnn_consistency(self, key):
    """Test that StackedGRURNN produces consistent results."""
    in_channels = 4
    out_channels = 6
    seq_length = 10
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    # Create two identical models
    model1 = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    model2 = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))

    output1 = model1(x)
    output2 = model2(x)

    # Same input and same initialization should produce same output
    assert jnp.allclose(output1, output2)

  def test_stacked_gru_rnn_different_inputs(self, key):
    """Test that StackedGRURNN produces different outputs for different inputs."""
    in_channels = 4
    out_channels = 6
    seq_length = 10
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    k1, k2 = random.split(key, 2)
    x1 = random.normal(k1, (seq_length, in_channels))
    x2 = random.normal(k2, (seq_length, in_channels))

    output1 = model(x1)
    output2 = model(x2)

    # Different inputs should produce different outputs
    assert not jnp.allclose(output1, output2, atol=1e-6)

  def test_stacked_gru_rnn_temporal_dependencies(self, key):
    """Test that StackedGRURNN captures temporal dependencies."""
    in_channels = 4
    out_channels = 6
    seq_length = 20
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create input with impulse at different time steps
    x1 = jnp.zeros((seq_length, in_channels))
    x1 = x1.at[5, :].set(1.0)

    x2 = jnp.zeros((seq_length, in_channels))
    x2 = x2.at[15, :].set(1.0)

    output1 = model(x1)
    output2 = model(x2)

    # Outputs should be different due to temporal position of impulse
    assert not jnp.allclose(output1, output2, atol=1e-6)

  def test_stacked_gru_rnn_zero_input(self, key):
    """Test StackedGRURNN with zero input."""
    in_channels = 4
    out_channels = 6
    seq_length = 10
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    zero_input = jnp.zeros((seq_length, in_channels))
    output = model(zero_input)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_stacked_gru_rnn_single_timestep(self, key):
    """Test StackedGRURNN with single timestep input."""
    in_channels = 4
    out_channels = 6
    hypers = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    model = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, (1, in_channels))
    output = model(x)

    assert output.shape == (1, out_channels)
    assert jnp.isfinite(output).all()

  def test_stacked_gru_rnn_depth_effect(self, key):
    """Test that stacking layers affects the output."""
    in_channels = 4
    out_channels = 6
    seq_length = 10

    # Single layer
    hypers1 = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=1
    )

    # Multiple layers
    hypers3 = StackedGRUSequenceHypers(
      hidden_size=8,
      intermediate_channels=12,
      num_layers=3
    )

    k1, k2 = random.split(key, 2)

    model1 = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers1,
      key=k1
    )

    model3 = StackedGRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers3,
      key=k2
    )

    x = random.normal(key, (seq_length, in_channels))

    output1 = model1(x)
    output3 = model3(x)

    # Different depths should generally produce different outputs
    assert not jnp.allclose(output1, output3, atol=1e-6)


class TestGRUSeq2SeqHypers:
  """Test suite for GRUSeq2SeqHypers"""

  def test_gru_seq2seq_hypers_initialization(self):
    """Test GRUSeq2SeqHypers initialization."""
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=3,
      intermediate_channels=32
    )

    assert hypers.hidden_size == 16
    assert hypers.n_layers == 3
    assert hypers.intermediate_channels == 32

  def test_gru_seq2seq_hypers_different_values(self):
    """Test GRUSeq2SeqHypers with different parameter values."""
    test_configs = [
      (8, 2, 16),
      (32, 4, 64),
      (12, 5, 24)
    ]

    for hidden_size, n_layers, intermediate_channels in test_configs:
      hypers = GRUSeq2SeqHypers(
        hidden_size=hidden_size,
        n_layers=n_layers,
        intermediate_channels=intermediate_channels
      )

      assert hypers.hidden_size == hidden_size
      assert hypers.n_layers == n_layers
      assert hypers.intermediate_channels == intermediate_channels


class TestGRUSeq2SeqModel:
  """Test suite for GRUSeq2SeqModel"""

  def test_gru_seq2seq_model_initialization(self, key):
    """Test GRUSeq2SeqModel initialization."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder')
    assert hasattr(model, 'time_features')
    assert hasattr(model, 'hypers')

    # Check component types
    assert isinstance(model.encoder, StackedGRURNN)
    assert isinstance(model.decoder, StackedGRURNN)

    # Check hypers
    assert model.hypers.hidden_size == hypers.hidden_size
    assert model.hypers.n_layers == hypers.n_layers
    assert model.hypers.intermediate_channels == hypers.intermediate_channels

  def test_gru_seq2seq_model_different_sizes(self, key):
    """Test GRUSeq2SeqModel with different input/output sizes."""
    test_configs = [
      (2, 3, 4, 8, 2, 16),
      (4, 6, 2, 12, 3, 20),
      (5, 4, 8, 16, 4, 32)
    ]

    for cond_in_channels, in_channels, out_channels, hidden_size, n_layers, intermediate_channels in test_configs:
      hypers = GRUSeq2SeqHypers(
        hidden_size=hidden_size,
        n_layers=n_layers,
        intermediate_channels=intermediate_channels
      )

      model = GRUSeq2SeqModel(
        cond_in_channels=cond_in_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        hypers=hypers,
        key=key
      )

      # Should initialize without errors
      assert isinstance(model, GRUSeq2SeqModel)

  def test_gru_seq2seq_model_series_to_array(self, key):
    """Test GRUSeq2SeqModel series_to_array method."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 10
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create test time series
    times = jnp.linspace(0, 1, seq_length)
    values = random.normal(key, (seq_length, in_channels))
    series = TimeSeries(times, values)

    # Convert to array with time features
    array_output = model.series_to_array(series)

    # Should have original values plus time features
    expected_channels = in_channels + hypers.hidden_size  # time_feature_size
    assert array_output.shape == (seq_length, expected_channels)
    assert jnp.isfinite(array_output).all()

    # First part should match original values
    assert jnp.allclose(array_output[:, :in_channels], values)

  def test_gru_seq2seq_model_create_context(self, key):
    """Test GRUSeq2SeqModel create_context method."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 15
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create test condition series
    times = jnp.linspace(0, 1, seq_length)
    values = random.normal(key, (seq_length, cond_in_channels))
    condition_series = TimeSeries(times, values)

    context = model.create_context(condition_series)
    # Should create context matrix with one vector per decoder layer
    assert context.shape == (hypers.n_layers, hypers.hidden_size)
    assert jnp.isfinite(context).all()

  def test_gru_seq2seq_model_forward_pass(self, key):
    """Test GRUSeq2SeqModel forward pass."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 10
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create test time series
    k1, k2 = random.split(key, 2)
    times = jnp.linspace(0, 1, seq_length)

    condition_values = random.normal(k1, (seq_length, cond_in_channels))
    condition_series = TimeSeries(times, condition_values)

    input_values = random.normal(k2, (seq_length, in_channels))
    input_series = TimeSeries(times, input_values)

    output = model(input_series, condition_series)
    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_gru_seq2seq_model_forward_with_context(self, key):
    """Test GRUSeq2SeqModel forward pass with provided context."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 10
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create test time series
    k1, k2, k3 = random.split(key, 3)
    times = jnp.linspace(0, 1, seq_length)

    condition_values = random.normal(k1, (seq_length, cond_in_channels))
    condition_series = TimeSeries(times, condition_values)

    input_values = random.normal(k2, (seq_length, in_channels))
    input_series = TimeSeries(times, input_values)

    # Create custom context (one vector per decoder layer)
    context = random.normal(k3, (hypers.n_layers, hypers.hidden_size))

    output = model(input_series, condition_series, context)
    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_gru_seq2seq_model_batch_size_property(self, key):
    """Test GRUSeq2SeqModel batch_size property."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Should be None for unbatched
    assert model.batch_size is None

  def test_gru_seq2seq_model_different_sequence_lengths(self, key):
    """Test GRUSeq2SeqModel with different sequence lengths."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    for seq_length in [5, 10, 20]:
      k1, k2 = random.split(key, 2)
      times = jnp.linspace(0, 1, seq_length)

      condition_values = random.normal(k1, (seq_length, cond_in_channels))
      condition_series = TimeSeries(times, condition_values)

      input_values = random.normal(k2, (seq_length, in_channels))
      input_series = TimeSeries(times, input_values)

      output = model(input_series, condition_series)
      assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_gru_seq2seq_model_consistency(self, key):
    """Test that GRUSeq2SeqModel produces consistent results."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 8
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    # Create two identical models
    model1 = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    model2 = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create test time series
    k1, k2 = random.split(key, 2)
    times = jnp.linspace(0, 1, seq_length)

    condition_values = random.normal(k1, (seq_length, cond_in_channels))
    condition_series = TimeSeries(times, condition_values)

    input_values = random.normal(k2, (seq_length, in_channels))
    input_series = TimeSeries(times, input_values)

    output1 = model1(input_series, condition_series)
    output2 = model2(input_series, condition_series)

    # Same input and same initialization should produce same output
    assert jnp.allclose(output1, output2)

  def test_gru_seq2seq_model_different_inputs(self, key):
    """Test that GRUSeq2SeqModel produces different outputs for different inputs."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 8
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create different test time series
    k1, k2, k3, k4 = random.split(key, 4)
    times = jnp.linspace(0, 1, seq_length)

    condition_values1 = random.normal(k1, (seq_length, cond_in_channels))
    condition_series1 = TimeSeries(times, condition_values1)
    condition_values2 = random.normal(k2, (seq_length, cond_in_channels))
    condition_series2 = TimeSeries(times, condition_values2)

    input_values1 = random.normal(k3, (seq_length, in_channels))
    input_series1 = TimeSeries(times, input_values1)
    input_values2 = random.normal(k4, (seq_length, in_channels))
    input_series2 = TimeSeries(times, input_values2)

    output1 = model(input_series1, condition_series1)
    output2 = model(input_series2, condition_series2)

    # Different inputs should produce different outputs
    assert not jnp.allclose(output1, output2, atol=1e-6)

  def test_gru_seq2seq_model_time_features_effect(self, key):
    """Test that time features affect the output."""
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    seq_length = 8
    hypers = GRUSeq2SeqHypers(
      hidden_size=16,
      n_layers=2,
      intermediate_channels=24
    )

    model = GRUSeq2SeqModel(
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hypers=hypers,
      key=key
    )

    # Create time series with same values but different times
    k1, k2 = random.split(key, 2)

    times1 = jnp.linspace(0, 1, seq_length)
    times2 = jnp.linspace(0.5, 1.5, seq_length)  # Different time range

    values = random.normal(k1, (seq_length, in_channels))
    condition_values = random.normal(k2, (seq_length, cond_in_channels))

    series1 = TimeSeries(times1, values)
    series2 = TimeSeries(times2, values)
    condition_series1 = TimeSeries(times1, condition_values)
    condition_series2 = TimeSeries(times2, condition_values)

    # Convert to arrays to see time feature effect
    array1 = model.series_to_array(series1)
    array2 = model.series_to_array(series2)

    # Values should be same, but time features should be different
    assert jnp.allclose(array1[:, :in_channels], array2[:, :in_channels])
    assert not jnp.allclose(array1[:, in_channels:], array2[:, in_channels:], atol=1e-6)

  def test_gru_seq2seq_model_get_nn_initialization(self, key):
    """Test GRUSeq2SeqModel initialization using get_nn."""

    # Define model parameters
    cond_in_channels = 4
    in_channels = 4
    out_channels = 4
    hidden_size = 16
    n_layers = 2
    intermediate_channels = 24

    # Create model config
    model_config = GRURNNModelConfig(
      model_type="gru_rnn",
      cond_in_channels=cond_in_channels,
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      n_layers=n_layers,
      intermediate_channels=intermediate_channels
    )

    # Create dataset config
    dataset_config = DummyDatasetConfig(
      n_features=in_channels,
    )

    random_seed = 42

    # Get model using get_nn
    from linsdex.nn.nn_models.rnn import TimeDependentGRUSeq2SeqModel
    model = get_nn(model_config, dataset_config, random_seed)

    # Verify the model is correctly initialized
    assert isinstance(model, TimeDependentGRUSeq2SeqModel)
    assert isinstance(model.encoder, StackedGRURNN)
    assert isinstance(model.decoder, StackedGRURNN)

    # Check model hyperparameters
    assert model.hypers.hidden_size == hidden_size
    assert model.hypers.n_layers == n_layers
    assert model.hypers.intermediate_channels == intermediate_channels

    # Test that the model can perform forward pass
    seq_length = 10
    k1, k2 = random.split(key, 2)

    times = jnp.linspace(0, 1, seq_length)
    condition_values = random.normal(k1, (seq_length, cond_in_channels))
    condition_series = TimeSeries(times, condition_values)

    input_values = random.normal(k2, (seq_length, in_channels))
    input_series = TimeSeries(times, input_values)

    # Time-dependent model requires an explicit time argument
    output = model(0.5, input_series, condition_series)
    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

    # Test context creation
    context = model.create_context(condition_series)
    assert context.shape == (n_layers, hidden_size)
    assert jnp.isfinite(context).all()

    output = model(0.5, input_series, condition_series, context)
    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()