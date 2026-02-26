import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import equinox as eqx
from linsdex.nn.nn_layers.time_condition import GaussianFourierProjection, TimeFeatures


class TestGaussianFourierProjection:
  """Test suite for GaussianFourierProjection"""

  def test_gaussian_fourier_projection_initialization(self, key):
    """Test GaussianFourierProjection initialization."""
    embedding_size = 16

    gfp = GaussianFourierProjection(embedding_size=embedding_size, key=key)

    assert gfp.embedding_size == embedding_size
    assert gfp.W.in_features == 1
    assert gfp.W.out_features == embedding_size

  def test_gaussian_fourier_projection_forward(self, key):
    """Test GaussianFourierProjection forward pass."""
    embedding_size = 8

    gfp = GaussianFourierProjection(embedding_size=embedding_size, key=key)

    # Test with scalar time
    t = jnp.array(0.5)
    output = gfp(t)

    assert output.shape == (2 * embedding_size,)
    assert jnp.isfinite(output).all()

  def test_gaussian_fourier_projection_different_times(self, key):
    """Test GaussianFourierProjection with different time values."""
    embedding_size = 6

    gfp = GaussianFourierProjection(embedding_size=embedding_size, key=key)

    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    outputs = []

    for t in times:
      t_tensor = jnp.array(t)
      output = gfp(t_tensor)
      outputs.append(output)

      assert output.shape == (2 * embedding_size,)
      assert jnp.isfinite(output).all()

    # Different times should produce different outputs
    for i in range(len(outputs)):
      for j in range(i + 1, len(outputs)):
        assert not jnp.allclose(outputs[i], outputs[j])

  def test_gaussian_fourier_projection_periodicity(self, key):
    """Test that GaussianFourierProjection has some periodic properties."""
    embedding_size = 4

    gfp = GaussianFourierProjection(embedding_size=embedding_size, key=key)

    # Test with time values that should show periodic behavior
    t1 = jnp.array(0.0)
    t2 = jnp.array(1.0)  # After one full period for some frequencies

    output1 = gfp(t1)
    output2 = gfp(t2)

    assert output1.shape == output2.shape
    assert jnp.isfinite(output1).all()
    assert jnp.isfinite(output2).all()

  def test_gaussian_fourier_projection_negative_times(self, key):
    """Test GaussianFourierProjection with negative time values."""
    embedding_size = 5

    gfp = GaussianFourierProjection(embedding_size=embedding_size, key=key)

    # Test with negative time
    t_neg = jnp.array(-0.5)
    output_neg = gfp(t_neg)

    # Test with positive time
    t_pos = jnp.array(0.5)
    output_pos = gfp(t_pos)

    assert output_neg.shape == (2 * embedding_size,)
    assert output_pos.shape == (2 * embedding_size,)
    assert jnp.isfinite(output_neg).all()
    assert jnp.isfinite(output_pos).all()

    # Should produce different outputs for positive and negative times
    assert not jnp.allclose(output_neg, output_pos)


class TestTimeFeatures:
  """Test suite for TimeFeatures"""

  def test_time_features_initialization(self, key):
    """Test TimeFeatures initialization."""
    embedding_size = 8
    out_features = 12
    activation = jax.nn.gelu

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      activation=activation,
      key=key
    )

    assert tf.out_features == out_features
    assert tf.projection.embedding_size == embedding_size
    assert tf.activation == activation

  def test_time_features_forward(self, key):
    """Test TimeFeatures forward pass."""
    embedding_size = 6
    out_features = 10

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    # Test with scalar time
    t = jnp.array(0.3)
    output = tf(t)

    assert output.shape == (out_features,)
    assert jnp.isfinite(output).all()

  def test_time_features_different_times(self, key):
    """Test TimeFeatures with different time values."""
    embedding_size = 4
    out_features = 8

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    times = [0.0, 0.1, 0.5, 0.9, 1.0]
    outputs = []

    for t in times:
      t_tensor = jnp.array(t)
      output = tf(t_tensor)
      outputs.append(output)

      assert output.shape == (out_features,)
      assert jnp.isfinite(output).all()

    # Different times should produce different outputs
    for i in range(len(outputs)):
      for j in range(i + 1, len(outputs)):
        assert not jnp.allclose(outputs[i], outputs[j], atol=1e-6)

  def test_time_features_different_activations(self, key):
    """Test TimeFeatures with different activation functions."""
    embedding_size = 4
    out_features = 6

    activations = [jax.nn.relu, jax.nn.gelu, jax.nn.tanh, jax.nn.swish]

    for activation in activations:
      tf = TimeFeatures(
        embedding_size=embedding_size,
        out_features=out_features,
        activation=activation,
        key=key
      )

      t = jnp.array(0.5)
      output = tf(t)

      assert output.shape == (out_features,)
      assert jnp.isfinite(output).all()

  def test_time_features_reproducibility(self, key):
    """Test that TimeFeatures produces reproducible results."""
    embedding_size = 5
    out_features = 7

    # Create two identical TimeFeatures modules
    tf1 = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    tf2 = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    # Same input
    t = jnp.array(0.42)

    output1 = tf1(t)
    output2 = tf2(t)

    # Should produce identical results
    assert jnp.allclose(output1, output2)

  def test_time_features_gradient_flow(self, key):
    """Test that TimeFeatures allows gradient flow."""
    embedding_size = 4
    out_features = 6

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    def loss_fn(t):
      output = tf(t)
      return jnp.sum(output**2)

    t = jnp.array(0.5)
    loss_val = loss_fn(t)
    grad_val = jax.grad(loss_fn)(t)

    assert jnp.isfinite(loss_val)
    assert jnp.isfinite(grad_val)
    assert grad_val != 0.0  # Should have non-zero gradient

  def test_time_features_large_times(self, key):
    """Test TimeFeatures with large time values."""
    embedding_size = 6
    out_features = 8

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    # Test with large positive and negative times
    large_times = [100.0, -100.0, 1000.0, -1000.0]

    for t_val in large_times:
      t = jnp.array(t_val)
      output = tf(t)

      assert output.shape == (out_features,)
      assert jnp.isfinite(output).all()

  def test_time_features_zero_time(self, key):
    """Test TimeFeatures specifically with zero time."""
    embedding_size = 3
    out_features = 5

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    t = jnp.array(0.0)
    output = tf(t)

    assert output.shape == (out_features,)
    assert jnp.isfinite(output).all()

    # Output should not be all zeros (due to the network processing)
    assert not jnp.allclose(output, jnp.zeros_like(output))

  def test_time_features_small_embedding_size(self, key):
    """Test TimeFeatures with very small embedding size."""
    embedding_size = 1
    out_features = 2

    tf = TimeFeatures(
      embedding_size=embedding_size,
      out_features=out_features,
      key=key
    )

    t = jnp.array(0.5)
    output = tf(t)

    assert output.shape == (out_features,)
    assert jnp.isfinite(output).all()