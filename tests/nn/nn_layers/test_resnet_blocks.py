import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import equinox as eqx
from linsdex.nn.nn_layers.resnet_blocks import *


class TestGatedResBlock:
  """Test suite for GatedResBlock"""

  def test_gated_res_block_initialization_1d(self, key):
    """Test GatedResBlock initialization for 1D data."""
    input_shape = (16,)
    hidden_size = 32

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      key=key
    )

    assert block.input_shape == input_shape
    assert block.hidden_size == hidden_size
    assert block.cond_shape is None
    assert block.groups is None

  def test_gated_res_block_initialization_3d(self, key):
    """Test GatedResBlock initialization for image data."""
    input_shape = (8, 8, 16)
    hidden_size = 32
    filter_shape = (3, 3)
    groups = 4

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      filter_shape=filter_shape,
      groups=groups,
      key=key
    )

    assert block.input_shape == input_shape
    assert block.hidden_size == hidden_size
    assert block.filter_shape == filter_shape
    assert block.groups == groups

  def test_gated_res_block_forward_1d(self, key):
    """Test GatedResBlock forward pass for 1D data."""
    input_shape = (12,)
    hidden_size = 24

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    assert output.shape == input_shape
    assert jnp.isfinite(output).all()

  def test_gated_res_block_forward_3d(self, key):
    """Test GatedResBlock forward pass for image data."""
    input_shape = (4, 4, 8)
    hidden_size = 16
    filter_shape = (3, 3)
    groups = 2

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      filter_shape=filter_shape,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    assert output.shape == input_shape
    assert jnp.isfinite(output).all()

  def test_gated_res_block_with_conditioning_1d(self, key):
    """Test GatedResBlock with conditioning for 1D data."""
    input_shape = (10,)
    hidden_size = 20
    cond_shape = (5,)

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      cond_shape=cond_shape,
      key=key
    )

    x = random.normal(key, input_shape)
    y = random.normal(key, cond_shape)
    output = block(x, y)

    assert output.shape == input_shape
    assert jnp.isfinite(output).all()

  def test_gated_res_block_with_conditioning_3d(self, key):
    """Test GatedResBlock with conditioning for image data."""
    input_shape = (4, 4, 6)
    hidden_size = 12
    cond_shape = (4, 4, 3)
    filter_shape = (3, 3)
    groups = 2

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      cond_shape=cond_shape,
      filter_shape=filter_shape,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    y = random.normal(key, cond_shape)
    output = block(x, y)

    assert output.shape == input_shape
    assert jnp.isfinite(output).all()

  def test_gated_res_block_residual_connection(self, key):
    """Test that GatedResBlock implements proper residual connections."""
    input_shape = (8,)
    hidden_size = 16

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    # Output should be different from input due to processing
    assert not jnp.allclose(output, x)
    # But should have reasonable magnitude due to gating
    assert jnp.abs(output).mean() > 0.01

  def test_gated_res_block_data_dependent_init_1d(self, key):
    """Test GatedResBlock data-dependent initialization for 1D data."""
    input_shape = (6,)
    hidden_size = 12
    batch_size = 8

    block = GatedResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      key=key
    )

    # Create batch data
    x_batch = random.normal(key, (batch_size,) + input_shape)

    # Initialize with data
    initialized_block = block.data_dependent_init(x_batch, key=key)

    # Test forward pass
    x_single = random.normal(key, input_shape)
    output = initialized_block(x_single)
    assert output.shape == input_shape


class TestBlock:
  """Test suite for Block (group norm + conv)"""

  def test_block_initialization(self, key):
    """Test Block initialization."""
    input_shape = (8, 8, 4)
    out_size = 8
    groups = 2

    block = Block(
      input_shape=input_shape,
      out_size=out_size,
      groups=groups,
      key=key
    )

    assert block.input_shape == input_shape

  def test_block_forward(self, key):
    """Test Block forward pass."""
    input_shape = (4, 4, 6)
    out_size = 12
    groups = 2

    block = Block(
      input_shape=input_shape,
      out_size=out_size,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()

  def test_block_with_shift_scale(self, key):
    """Test Block with shift and scale conditioning."""
    input_shape = (4, 4, 4)
    out_size = 8
    groups = 2

    block = Block(
      input_shape=input_shape,
      out_size=out_size,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    # Create shift_scale as a tuple of (shift, scale) arrays with input shape
    shift = random.normal(key, input_shape)
    scale = random.normal(key, input_shape)
    shift_scale = (shift, scale)

    output = block(x, shift_scale=shift_scale)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()


class TestImageResBlock:
  """Test suite for ImageResBlock"""

  def test_image_res_block_initialization(self, key):
    """Test ImageResBlock initialization."""
    input_shape = (8, 8, 16)
    hidden_size = 32
    out_size = 24
    groups = 4

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      key=key
    )

    assert block.input_shape == input_shape
    assert block.hidden_size == hidden_size
    assert block.out_size == out_size
    assert block.groups == groups

  def test_image_res_block_forward(self, key):
    """Test ImageResBlock forward pass."""
    input_shape = (4, 4, 8)
    hidden_size = 16
    out_size = 12
    groups = 2

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()

  def test_image_res_block_with_conditioning(self, key):
    """Test ImageResBlock with conditioning."""
    input_shape = (4, 4, 6)
    hidden_size = 12
    out_size = 8
    cond_shape = (10,)
    groups = 2

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      cond_shape=cond_shape,
      key=key
    )

    x = random.normal(key, input_shape)
    y = random.normal(key, cond_shape)
    output = block(x, y)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()

  def test_image_res_block_same_out_size(self, key):
    """Test ImageResBlock with same input and output size."""
    input_shape = (6, 6, 8)
    hidden_size = 16
    out_size = 8  # Same as input channels
    groups = 2

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    # Should preserve input shape
    assert output.shape == input_shape
    assert jnp.isfinite(output).all()

  def test_image_res_block_different_out_size(self, key):
    """Test ImageResBlock with different input and output size."""
    input_shape = (4, 4, 6)
    hidden_size = 12
    out_size = 10  # Different from input channels
    groups = 2

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape
    assert jnp.isfinite(output).all()

  def test_image_res_block_data_dependent_init(self, key):
    """Test ImageResBlock data-dependent initialization."""
    input_shape = (4, 4, 4)
    hidden_size = 8
    out_size = 6
    groups = 2
    batch_size = 5

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      key=key
    )

    # Create batch data
    x_batch = random.normal(key, (batch_size,) + input_shape)

    # Initialize with data
    initialized_block = block.data_dependent_init(x_batch, key=key)

    # Test forward pass
    x_single = random.normal(key, input_shape)
    output = initialized_block(x_single)
    expected_shape = (input_shape[0], input_shape[1], out_size)
    assert output.shape == expected_shape

  def test_image_res_block_residual_behavior(self, key):
    """Test that ImageResBlock exhibits residual behavior when appropriate."""
    input_shape = (4, 4, 8)
    hidden_size = 16
    out_size = 8  # Same as input for proper residual connection
    groups = 2

    block = ImageResBlock(
      input_shape=input_shape,
      hidden_size=hidden_size,
      out_size=out_size,
      groups=groups,
      key=key
    )

    x = random.normal(key, input_shape)
    output = block(x)

    # Output should be different from input but related
    assert not jnp.allclose(output, x)
    # Should have reasonable magnitude
    assert jnp.abs(output).mean() > 0.01