import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import equinox as eqx
from linsdex.nn.nn_layers.layers import *


class TestWeightNormDense:
  """Test suite for WeightNormDense layer"""

  def test_weight_norm_dense_initialization(self, key):
    """Test WeightNormDense initialization."""
    in_size = 10
    out_size = 5

    layer = WeightNormDense(in_size, out_size, key)

    assert layer.in_size == in_size
    assert layer.out_size == out_size
    assert layer.W.shape == (out_size, in_size)
    assert layer.b.shape == (out_size,)
    assert layer.g.shape == ()

  def test_weight_norm_dense_forward(self, key):
    """Test WeightNormDense forward pass."""
    in_size = 8
    out_size = 4

    layer = WeightNormDense(in_size, out_size, key)

    x = random.normal(key, (in_size,))
    output = layer(x)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_weight_norm_dense_data_dependent_init(self, key):
    """Test WeightNormDense data-dependent initialization."""
    in_size = 6
    out_size = 3
    batch_size = 10

    layer = WeightNormDense(in_size, out_size, key)

    # Create batch data
    x_batch = random.normal(key, (batch_size, in_size))

    # Initialize with data
    initialized_layer = layer.data_dependent_init(x_batch, key=key)

    assert initialized_layer.in_size == in_size
    assert initialized_layer.out_size == out_size

    # Test forward pass with initialized layer
    x_single = random.normal(key, (in_size,))
    output = initialized_layer(x_single)
    assert output.shape == (out_size,)

  def test_weight_norm_dense_with_conditioning(self, key):
    """Test WeightNormDense with conditioning input."""
    in_size = 4
    out_size = 2

    layer = WeightNormDense(in_size, out_size, key)

    x = random.normal(key, (in_size,))
    y = random.normal(key, (3,))  # Conditioning

    # Should still work (y is ignored)
    output = layer(x, y)
    assert output.shape == (out_size,)


class TestWeightNormConv:
  """Test suite for WeightNormConv layer"""

  def test_weight_norm_conv_initialization(self, key):
    """Test WeightNormConv initialization."""
    input_shape = (8, 8, 3)
    filter_shape = (3, 3)
    out_size = 16

    layer = WeightNormConv(input_shape, filter_shape, out_size, key=key)

    assert layer.input_shape == input_shape
    assert layer.filter_shape == filter_shape
    assert layer.out_size == out_size
    expected_weight_shape = filter_shape + (input_shape[2], out_size)
    assert layer.W.shape == expected_weight_shape

  def test_weight_norm_conv_forward(self, key):
    """Test WeightNormConv forward pass."""
    input_shape = (4, 4, 2)
    filter_shape = (3, 3)
    out_size = 8

    layer = WeightNormConv(input_shape, filter_shape, out_size, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()

  def test_weight_norm_conv_data_dependent_init(self, key):
    """Test WeightNormConv data-dependent initialization."""
    input_shape = (4, 4, 2)
    filter_shape = (3, 3)
    out_size = 6
    batch_size = 5

    layer = WeightNormConv(input_shape, filter_shape, out_size, key=key)

    # Create batch data
    x_batch = random.normal(key, (batch_size,) + input_shape)

    # Initialize with data
    initialized_layer = layer.data_dependent_init(x_batch, key=key)

    # Test forward pass
    x_single = random.normal(key, input_shape)
    output = initialized_layer(x_single)
    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape


class TestWeightStandardizedConv:
  """Test suite for WeightStandardizedConv layer"""

  def test_weight_standardized_conv_initialization(self, key):
    """Test WeightStandardizedConv initialization."""
    input_shape = (6, 6, 4)
    filter_shape = (3, 3)
    out_size = 8

    layer = WeightStandardizedConv(input_shape, filter_shape, out_size, key=key)

    assert layer.input_shape == input_shape
    assert layer.filter_shape == filter_shape
    assert layer.out_size == out_size

  def test_weight_standardized_conv_forward(self, key):
    """Test WeightStandardizedConv forward pass."""
    input_shape = (4, 4, 3)
    filter_shape = (3, 3)
    out_size = 6

    layer = WeightStandardizedConv(input_shape, filter_shape, out_size, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()


class TestChannelConvention:
  """Test suite for ChannelConvention wrapper"""

  def test_channel_convention_initialization(self, key):
    """Test ChannelConvention initialization."""
    # Create a simple module to wrap
    linear = eqx.nn.Linear(4, 2, key=key)
    wrapper = ChannelConvention(linear)

    assert wrapper.module is linear

  def test_channel_convention_forward(self, key):
    """Test ChannelConvention forward pass."""
    # Create a Conv2d layer that expects (channels, height, width)
    conv = eqx.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, key=key)
    wrapper = ChannelConvention(conv)

    # Input in (height, width, channels) format
    x = random.normal(key, (3, 3, 4))
    output = wrapper(x)

    # Should transpose to channels-first, apply module, then transpose back
    assert output.shape == (3, 3, 2)
    assert jnp.isfinite(output).all()


class TestConvAndGroupNorm:
  """Test suite for ConvAndGroupNorm layer"""

  def test_conv_and_group_norm_initialization(self, key):
    """Test ConvAndGroupNorm initialization."""
    input_shape = (8, 8, 4)
    filter_shape = (3, 3)
    out_size = 8
    groups = 2

    layer = ConvAndGroupNorm(input_shape, filter_shape, out_size, groups, key=key)

    assert layer.input_shape == input_shape
    assert hasattr(layer, 'conv')
    assert hasattr(layer, 'norm')

  def test_conv_and_group_norm_forward(self, key):
    """Test ConvAndGroupNorm forward pass."""
    input_shape = (4, 4, 4)
    filter_shape = (3, 3)
    out_size = 8
    groups = 2

    layer = ConvAndGroupNorm(input_shape, filter_shape, out_size, groups, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()


class TestUpsample:
  """Test suite for Upsample layer"""

  def test_upsample_initialization(self, key):
    """Test Upsample initialization."""
    input_shape = (4, 4, 8)
    out_size = 16

    layer = Upsample(input_shape, out_size, key=key)

    assert layer.input_shape == input_shape
    assert layer.out_size == out_size

  def test_upsample_forward(self, key):
    """Test Upsample forward pass."""
    input_shape = (4, 4, 6)
    out_size = 12

    layer = Upsample(input_shape, out_size, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should double spatial dimensions
    expected_shape = (input_shape[0] * 2, input_shape[1] * 2, out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()

  def test_upsample_default_out_size(self, key):
    """Test Upsample with default output size."""
    input_shape = (2, 2, 8)

    layer = Upsample(input_shape, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should keep same number of channels by default
    expected_shape = (input_shape[0] * 2, input_shape[1] * 2, input_shape[2])
    assert output.shape == expected_shape


class TestDownsample:
  """Test suite for Downsample layer"""

  def test_downsample_initialization(self, key):
    """Test Downsample initialization."""
    input_shape = (8, 8, 4)
    out_size = 8

    layer = Downsample(input_shape, out_size, key=key)

    assert layer.input_shape == input_shape
    assert layer.out_size == out_size

  def test_downsample_forward(self, key):
    """Test Downsample forward pass."""
    input_shape = (8, 8, 6)
    out_size = 12

    layer = Downsample(input_shape, out_size, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should halve spatial dimensions
    expected_shape = (input_shape[0] // 2, input_shape[1] // 2, out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()


class TestGatedGlobalContext:
  """Test suite for GatedGlobalContext layer"""

  def test_gated_global_context_initialization(self, key):
    """Test GatedGlobalContext initialization."""
    input_shape = (8, 8, 16)

    layer = GatedGlobalContext(input_shape, key=key)

    assert layer.input_shape == input_shape

  def test_gated_global_context_forward(self, key):
    """Test GatedGlobalContext forward pass."""
    input_shape = (4, 4, 8)

    layer = GatedGlobalContext(input_shape, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should preserve input shape
    assert output.shape == input_shape
    assert jnp.isfinite(output).all()


class TestAttention:
  """Test suite for Attention layer"""

  def test_attention_initialization(self, key):
    """Test Attention initialization."""
    input_shape = (8, 8, 12)
    heads = 3
    dim_head = 4

    layer = Attention(input_shape, heads, dim_head, key=key)

    assert layer.input_shape == input_shape
    assert layer.heads == heads
    assert layer.dim_head == dim_head

  def test_attention_forward(self, key):
    """Test Attention forward pass."""
    input_shape = (4, 4, 8)
    heads = 2
    dim_head = 4

    layer = Attention(input_shape, heads, dim_head, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should preserve input shape
    assert output.shape == input_shape
    assert jnp.isfinite(output).all()


class TestLinearAttention:
  """Test suite for LinearAttention layer"""

  def test_linear_attention_initialization(self, key):
    """Test LinearAttention initialization."""
    input_shape = (6, 6, 8)
    heads = 2
    dim_head = 4

    layer = LinearAttention(input_shape, heads, dim_head, key=key)

    assert layer.input_shape == input_shape
    assert layer.heads == heads
    assert layer.dim_head == dim_head

  def test_linear_attention_forward(self, key):
    """Test LinearAttention forward pass."""
    input_shape = (4, 4, 6)
    heads = 2
    dim_head = 3

    layer = LinearAttention(input_shape, heads, dim_head, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should preserve input shape
    assert output.shape == input_shape
    assert jnp.isfinite(output).all()


class TestAttentionBlock:
  """Test suite for AttentionBlock layer"""

  def test_attention_block_initialization(self, key):
    """Test AttentionBlock initialization."""
    input_shape = (8, 8, 16)
    heads = 4
    dim_head = 4

    layer = AttentionBlock(input_shape, heads, dim_head, key=key)

    assert layer.input_shape == input_shape

  def test_attention_block_forward(self, key):
    """Test AttentionBlock forward pass."""
    input_shape = (4, 4, 8)
    heads = 2
    dim_head = 4

    layer = AttentionBlock(input_shape, heads, dim_head, key=key)

    x = random.normal(key, input_shape)
    output = layer(x)

    # Should preserve input shape due to residual connection
    assert output.shape == input_shape
    assert jnp.isfinite(output).all()

  def test_attention_block_linear_vs_standard(self, key):
    """Test AttentionBlock with linear vs standard attention."""
    input_shape = (4, 4, 6)
    heads = 2
    dim_head = 3

    # Linear attention
    layer_linear = AttentionBlock(input_shape, heads, dim_head,
                                use_linear_attention=True, key=key)

    # Standard attention
    layer_standard = AttentionBlock(input_shape, heads, dim_head,
                                  use_linear_attention=False, key=key)

    x = random.normal(key, input_shape)

    output_linear = layer_linear(x)
    output_standard = layer_standard(x)

    assert output_linear.shape == input_shape
    assert output_standard.shape == input_shape
    assert jnp.isfinite(output_linear).all()
    assert jnp.isfinite(output_standard).all()

    # Should produce different outputs
    assert not jnp.allclose(output_linear, output_standard, atol=1e-6)