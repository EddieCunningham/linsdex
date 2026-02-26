import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from linsdex.nn.nn_models.wavenet import (
  CausalConv1d, CausalConv1dHypers,
  WaveNetResBlock, WaveNetResBlockHypers,
  WaveNet, WaveNetHypers
)


class TestCausalConv1d:
  """Test suite for CausalConv1d layer"""

  def test_causal_conv1d_initialization(self, key):
    """Test CausalConv1d layer initialization."""
    in_channels = 4
    out_channels = 8
    hypers = CausalConv1dHypers(kernel_width=3, stride=1, dilation=1)

    layer = CausalConv1d(in_channels, out_channels, hypers, key=key)

    assert layer.hypers.kernel_width == 3
    assert layer.hypers.stride == 1
    assert layer.hypers.dilation == 1
    assert layer.hypers.padding == 2  # kernel_width - 1

  def test_causal_conv1d_forward(self, key):
    """Test CausalConv1d forward pass."""
    in_channels = 4
    out_channels = 8
    seq_length = 10
    hypers = CausalConv1dHypers(kernel_width=3)

    layer = CausalConv1d(in_channels, out_channels, hypers, key=key)

    # Test single sample
    x = random.normal(key, (seq_length, in_channels))
    output = layer(x)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_causal_conv1d_batch_size(self, key):
    """Test CausalConv1d batch size property."""
    in_channels = 4
    out_channels = 8
    hypers = CausalConv1dHypers()

    layer = CausalConv1d(in_channels, out_channels, hypers, key=key)

    # Should be None for unbatched
    assert layer.batch_size is None


class TestWaveNetResBlock:
  """Test suite for WaveNetResBlock"""

  def test_wavenet_resblock_initialization(self, key):
    """Test WaveNetResBlock initialization."""
    in_channels = 8
    hypers = WaveNetResBlockHypers(
      kernel_width=2,
      dilation=2,
      hidden_channels=16
    )

    block = WaveNetResBlock(in_channels, hypers, key=key)

    assert block.gating_conv.hypers.kernel_width == 2
    assert block.gating_conv.hypers.dilation == 2
    assert block.filter_conv.hypers.kernel_width == 2
    assert block.filter_conv.hypers.dilation == 2

  def test_wavenet_resblock_forward(self, key):
    """Test WaveNetResBlock forward pass."""
    in_channels = 8
    seq_length = 20
    hypers = WaveNetResBlockHypers(hidden_channels=16)

    block = WaveNetResBlock(in_channels, hypers, key=key)

    x = random.normal(key, (seq_length, in_channels))
    new_hidden, skip = block(x)

    assert new_hidden.shape == (seq_length, in_channels)
    assert skip.shape == (seq_length, in_channels)
    assert jnp.isfinite(new_hidden).all()
    assert jnp.isfinite(skip).all()

  def test_wavenet_resblock_residual_connection(self, key):
    """Test that WaveNetResBlock implements residual connections."""
    in_channels = 8
    seq_length = 20
    hypers = WaveNetResBlockHypers(hidden_channels=16)

    block = WaveNetResBlock(in_channels, hypers, key=key)

    x = random.normal(key, (seq_length, in_channels))
    new_hidden, skip = block(x)

    # The residual connection should make new_hidden different from x
    # but related through the residual connection
    assert not jnp.allclose(new_hidden, x)
    # Check that the output has reasonable magnitude
    assert jnp.abs(new_hidden).mean() > 0.01


class TestWaveNet:
  """Test suite for WaveNet"""

  def test_wavenet_initialization(self, key):
    """Test WaveNet initialization."""
    in_channels = 4
    out_channels = 2
    dilations = jnp.array([1, 2, 4, 8])
    hypers = WaveNetHypers(
      dilations=dilations,
      initial_filter_width=4,
      filter_width=2,
      residual_channels=16,
      dilation_channels=32,
      skip_channels=16
    )

    wavenet = WaveNet(in_channels, out_channels, hypers, key)

    # blocks is a vmapped WaveNetResBlock, so check its batch_size instead
    assert wavenet.blocks.batch_size == len(dilations)
    assert wavenet.in_projection_conv.hypers.kernel_width == 4
    assert wavenet.strict_autoregressive is False

  def test_wavenet_forward(self, key):
    """Test WaveNet forward pass."""
    in_channels = 4
    out_channels = 2
    seq_length = 32
    dilations = jnp.array([1, 2, 4])

    hypers = WaveNetHypers(
      dilations=dilations,
      residual_channels=8,
      dilation_channels=16,
      skip_channels=8
    )

    wavenet = WaveNet(in_channels, out_channels, hypers, key)

    x = random.normal(key, (seq_length, in_channels))
    output = wavenet(x)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_wavenet_strict_autoregressive(self, key):
    """Test WaveNet with strict autoregressive mode."""
    in_channels = 4
    out_channels = 2
    seq_length = 16
    dilations = jnp.array([1, 2])

    hypers = WaveNetHypers(
      dilations=dilations,
      residual_channels=8,
      dilation_channels=16,
      skip_channels=8
    )

    wavenet = WaveNet(in_channels, out_channels, hypers, key, strict_autoregressive=True)

    x = random.normal(key, (seq_length, in_channels))
    output = wavenet(x)

    # Output should have same length as input even with strict autoregressive
    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_wavenet_different_dilations(self, key):
    """Test WaveNet with different dilation patterns."""
    in_channels = 4
    out_channels = 2
    seq_length = 64

    k1, k2, k3 = random.split(key, 3)

    # Test with exponential dilations
    dilations_exp = jnp.array([1, 2, 4, 8, 16])
    hypers_exp = WaveNetHypers(
      dilations=dilations_exp,
      residual_channels=8,
      dilation_channels=16,
      skip_channels=8
    )

    wavenet_exp = WaveNet(in_channels, out_channels, hypers_exp, k1)

    # Test with linear dilations
    dilations_lin = jnp.array([1, 2, 3, 4, 5])
    hypers_lin = WaveNetHypers(
      dilations=dilations_lin,
      residual_channels=8,
      dilation_channels=16,
      skip_channels=8
    )

    wavenet_lin = WaveNet(in_channels, out_channels, hypers_lin, k2)

    x = random.normal(k3, (seq_length, in_channels))

    output_exp = wavenet_exp(x)
    output_lin = wavenet_lin(x)

    assert output_exp.shape == (seq_length, out_channels)
    assert output_lin.shape == (seq_length, out_channels)
    assert jnp.isfinite(output_exp).all()
    assert jnp.isfinite(output_lin).all()

    # Outputs should be different due to different dilation patterns and different keys
    assert not jnp.allclose(output_exp, output_lin, atol=1e-6)

  def test_wavenet_receptive_field(self, key):
    """Test that WaveNet has appropriate receptive field."""
    in_channels = 2
    out_channels = 1
    seq_length = 100
    dilations = jnp.array([1, 2, 4, 8])

    hypers = WaveNetHypers(
      dilations=dilations,
      filter_width=2,
      residual_channels=8,
      dilation_channels=16,
      skip_channels=8
    )

    wavenet = WaveNet(in_channels, out_channels, hypers, key)

    # Create input with impulse at different positions
    x_zeros = jnp.zeros((seq_length, in_channels))

    # Impulse at beginning
    x_impulse_start = x_zeros.at[0, 0].set(1.0)
    output_start = wavenet(x_impulse_start)

    # Impulse at middle
    x_impulse_mid = x_zeros.at[seq_length//2, 0].set(1.0)
    output_mid = wavenet(x_impulse_mid)

    # The outputs should be different due to different input positions
    assert not jnp.allclose(output_start, output_mid)

    # Both outputs should be finite
    assert jnp.isfinite(output_start).all()
    assert jnp.isfinite(output_mid).all()