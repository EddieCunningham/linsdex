import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import equinox as eqx
from linsdex.nn.nn_models.resnet import ResNet, TimeDependentResNet, ResNetHypers, TimeDependentResNetHypers
from linsdex.nn.nn_models.get_nn import get_nn
from dataclasses import dataclass

class TestResNetHypers:
  """Test suite for ResNetHypers"""

  def test_resnet_hypers_initialization(self):
    """Test ResNetHypers initialization."""
    hypers = ResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=3,
      filter_shape=(3, 3),
      groups=1,
      activation=jax.nn.swish
    )

    assert hypers.working_size == 32
    assert hypers.hidden_size == 64
    assert hypers.n_blocks == 3
    assert hypers.filter_shape == (3, 3)
    assert hypers.groups == 1
    assert hypers.activation == jax.nn.swish

  def test_resnet_hypers_different_values(self):
    """Test ResNetHypers with different parameter values."""
    test_configs = [
      (16, 32, 2, (5, 5), 2, jax.nn.relu),
      (64, 128, 4, None, None, jax.nn.gelu),
      (24, 48, 5, (7, 7), 4, jax.nn.swish)
    ]

    for working_size, hidden_size, n_blocks, filter_shape, groups, activation in test_configs:
      hypers = ResNetHypers(
        working_size=working_size,
        hidden_size=hidden_size,
        n_blocks=n_blocks,
        filter_shape=filter_shape,
        groups=groups,
        activation=activation
      )

      assert hypers.working_size == working_size
      assert hypers.hidden_size == hidden_size
      assert hypers.n_blocks == n_blocks
      assert hypers.filter_shape == filter_shape
      assert hypers.groups == groups
      assert hypers.activation == activation


class TestResNet:
  """Test suite for ResNet"""

  def test_resnet_1d_initialization(self, key):
    """Test ResNet initialization with 1D input."""
    input_shape = (10,)
    out_size = 5
    hypers = ResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=3,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    assert resnet.input_shape == input_shape
    assert resnet.hypers.working_size == 32
    assert resnet.hypers.hidden_size == 64
    assert resnet.hypers.n_blocks == 3
    assert resnet.hypers.filter_shape is None
    assert resnet.cond_shape is None

  def test_resnet_3d_initialization(self, key):
    """Test ResNet initialization with 3D (image) input."""
    input_shape = (32, 32, 3)
    out_size = 10
    hypers = ResNetHypers(
      working_size=64,
      hidden_size=128,
      n_blocks=4,
      filter_shape=(5, 5),
      groups=2,
      activation=jax.nn.relu
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    assert resnet.input_shape == input_shape
    assert resnet.hypers.working_size == 64
    assert resnet.hypers.hidden_size == 128
    assert resnet.hypers.n_blocks == 4
    assert resnet.hypers.filter_shape == (5, 5)

  def test_resnet_1d_forward_pass(self, key):
    """Test ResNet forward pass with 1D input."""
    input_shape = (8,)
    out_size = 4
    hypers = ResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, input_shape)
    output = resnet(x)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_resnet_3d_forward_pass(self, key):
    """Test ResNet forward pass with 3D input."""
    input_shape = (16, 16, 3)
    out_size = 8
    hypers = ResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=2,
      filter_shape=(3, 3),
      groups=1,
      activation=jax.nn.swish
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, input_shape)
    output = resnet(x)

    assert output.shape == (16, 16, out_size)
    assert jnp.isfinite(output).all()

  def test_resnet_with_conditioning(self, key):
    """Test ResNet with conditioning input."""
    input_shape = (10,)
    cond_shape = (5,)
    out_size = 6
    hypers = ResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    k1, k2 = random.split(key, 2)
    x = random.normal(k1, input_shape)
    y = random.normal(k2, cond_shape)
    output = resnet(x, y)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_resnet_batch_size_property(self, key):
    """Test ResNet batch_size property."""
    input_shape = (8,)
    out_size = 4
    hypers = ResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    # Should be None for unbatched
    assert resnet.batch_size is None

  def test_resnet_different_block_counts(self, key):
    """Test ResNet with different numbers of blocks."""
    input_shape = (6,)
    out_size = 3

    for n_blocks in [1, 2, 3, 5]:
      hypers = ResNetHypers(
        working_size=16,
        hidden_size=32,
        n_blocks=n_blocks,
        filter_shape=None,
        groups=None,
        activation=jax.nn.swish
      )

      resnet = ResNet(
        input_shape=input_shape,
        out_size=out_size,
        hypers=hypers,
        key=key
      )

      x = random.normal(key, input_shape)
      output = resnet(x)

      assert output.shape == (out_size,)
      assert jnp.isfinite(output).all()
      assert resnet.hypers.n_blocks == n_blocks

  def test_resnet_data_dependent_init_1d(self, key):
    """Test ResNet data-dependent initialization with 1D input."""
    input_shape = (8,)
    out_size = 4
    batch_size = 10
    hypers = ResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish
    )

    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    k1, k2 = random.split(key, 2)
    x_batch = random.normal(k1, (batch_size,) + input_shape)

    initialized_resnet = resnet.data_dependent_init(x_batch, key=k2)

    # Test that it's still a valid ResNet
    x = random.normal(key, input_shape)
    output = initialized_resnet(x)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_resnet_different_activations(self, key):
    """Test ResNet with different activation functions."""
    input_shape = (6,)
    out_size = 3

    activations = [jax.nn.swish, jax.nn.relu, jax.nn.gelu]

    for activation in activations:
      hypers = ResNetHypers(
        working_size=16,
        hidden_size=32,
        n_blocks=2,
        filter_shape=None,
        groups=None,
        activation=activation
      )

      resnet = ResNet(
        input_shape=input_shape,
        out_size=out_size,
        hypers=hypers,
        key=key
      )

      x = random.normal(key, input_shape)
      output = resnet(x)

      assert output.shape == (out_size,)
      assert jnp.isfinite(output).all()

  def test_resnet_3d_with_conditioning(self):
    """Test ResNet with 3D (image) input and conditioning."""

    input_shape = (16, 16, 3)  # Small image
    cond_shape = (5,)
    out_size = 8

    hypers = ResNetHypers(
      working_size = 32,
      hidden_size = 64,
      n_blocks = 2,
      filter_shape = (3, 3),  # Required for 3D input
      groups = 2,  # Groups are allowed for 3D
      activation = jax.nn.gelu
    )

    key = random.PRNGKey(42)
    resnet_3d = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # Verify setup
    assert resnet_3d.input_shape == input_shape
    assert resnet_3d.cond_shape == cond_shape
    assert resnet_3d.hypers.groups == 2
    assert resnet_3d.hypers.filter_shape == (3, 3)

    # Test forward pass with conditioning
    k1, k2 = random.split(key, 2)
    x = random.normal(k1, input_shape)
    cond = random.normal(k2, cond_shape)

    output = resnet_3d(x, cond)

    # Output should have same spatial dimensions but different channels
    expected_output_shape = (16, 16, out_size)
    assert output.shape == expected_output_shape
    assert jnp.isfinite(output).all()

    # Test without conditioning (pass None)
    output_no_cond = resnet_3d(x, None)
    assert output_no_cond.shape == expected_output_shape
    assert jnp.isfinite(output_no_cond).all()

    # Should be different with and without conditioning
    assert not jnp.allclose(output, output_no_cond, atol=1e-6)

    # Test with different conditioning values
    k3 = random.split(key, 1)[0]
    cond2 = random.normal(k3, cond_shape)
    output2 = resnet_3d(x, cond2)

    assert output2.shape == expected_output_shape
    assert jnp.isfinite(output2).all()

    # Should be different with different conditioning
    assert not jnp.allclose(output, output2, atol=1e-6)


class TestTimeDependentResNetHypers:
  """Test suite for TimeDependentResNetHypers"""

  def test_time_dependent_resnet_hypers_initialization(self):
    """Test TimeDependentResNetHypers initialization."""
    hypers = TimeDependentResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=3,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=16,
      out_features=8
    )

    assert hypers.working_size == 32
    assert hypers.hidden_size == 64
    assert hypers.n_blocks == 3
    assert hypers.filter_shape is None
    assert hypers.groups is None
    assert hypers.activation == jax.nn.swish
    assert hypers.embedding_size == 16
    assert hypers.out_features == 8

  def test_time_dependent_resnet_hypers_different_values(self):
    """Test TimeDependentResNetHypers with different parameter values."""
    test_configs = [
      (16, 32, 2, (5, 5), 2, jax.nn.relu, 12, 6),
      (64, 128, 4, None, None, jax.nn.gelu, 20, 10),
      (24, 48, 5, (7, 7), 4, jax.nn.swish, 8, 4)
    ]

    for working_size, hidden_size, n_blocks, filter_shape, groups, activation, embedding_size, out_features in test_configs:
      hypers = TimeDependentResNetHypers(
        working_size=working_size,
        hidden_size=hidden_size,
        n_blocks=n_blocks,
        filter_shape=filter_shape,
        groups=groups,
        activation=activation,
        embedding_size=embedding_size,
        out_features=out_features
      )

      assert hypers.working_size == working_size
      assert hypers.hidden_size == hidden_size
      assert hypers.n_blocks == n_blocks
      assert hypers.filter_shape == filter_shape
      assert hypers.groups == groups
      assert hypers.activation == activation
      assert hypers.embedding_size == embedding_size
      assert hypers.out_features == out_features

class TestTimeDependentResNet:
  """Test suite for TimeDependentResNet"""

  def test_time_dependent_resnet_initialization(self, key):
    """Test TimeDependentResNet initialization."""
    input_shape = (10,)
    out_size = 5
    hypers = TimeDependentResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=3,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=16,
      out_features=8
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    assert time_resnet.input_shape == input_shape
    assert time_resnet.hypers.working_size == 32
    assert time_resnet.hypers.hidden_size == 64
    assert time_resnet.hypers.n_blocks == 3
    # cond_shape should be (out_features,) since no additional conditioning
    assert time_resnet.cond_shape == (hypers.out_features,)

  def test_time_dependent_resnet_forward_pass(self, key):
    """Test TimeDependentResNet forward pass."""
    input_shape = (8,)
    out_size = 4
    hypers = TimeDependentResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=12,
      out_features=6
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    k1, k2 = random.split(key, 2)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, input_shape)
    output = time_resnet(t, x)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_time_dependent_resnet_with_conditioning(self, key):
    """Test TimeDependentResNet with additional conditioning."""
    input_shape = (10,)
    cond_shape = (5,)
    out_size = 6
    hypers = TimeDependentResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=16,
      out_features=8
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # cond_shape should be (out_features + cond_shape[0],)
    assert time_resnet.cond_shape == (hypers.out_features + cond_shape[0],)

    k1, k2, k3 = random.split(key, 3)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, input_shape)
    y = random.normal(k3, cond_shape)
    output = time_resnet(t, x, y)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_time_dependent_resnet_batch_size_property(self, key):
    """Test TimeDependentResNet batch_size property."""
    input_shape = (8,)
    out_size = 4
    hypers = TimeDependentResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=12,
      out_features=6
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    # Should be None for unbatched
    assert time_resnet.batch_size is None

  def test_time_dependent_resnet_different_times(self, key):
    """Test TimeDependentResNet with different time values."""
    input_shape = (6,)
    out_size = 3
    hypers = TimeDependentResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=12,
      out_features=6
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, input_shape)

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
      t = jnp.array(t_val)
      output = time_resnet(t, x)

      assert output.shape == (out_size,)
      assert jnp.isfinite(output).all()

  def test_time_dependent_resnet_data_dependent_init(self, key):
    """Test TimeDependentResNet data-dependent initialization."""
    input_shape = (8,)
    out_size = 4
    batch_size = 10
    hypers = TimeDependentResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=12,
      out_features=6
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    k1, k2, k3 = random.split(key, 3)
    t_batch = random.uniform(k1, (batch_size,), minval=0.0, maxval=1.0)
    x_batch = random.normal(k2, (batch_size,) + input_shape)

    initialized_time_resnet = time_resnet.data_dependent_init(t_batch, x_batch, key=k3)

    # Test that it's still a valid TimeDependentResNet
    t = random.uniform(key, (), minval=0.0, maxval=1.0)
    x = random.normal(key, input_shape)
    output = initialized_time_resnet(t, x)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

  def test_time_dependent_resnet_time_consistency(self, key):
    """Test that TimeDependentResNet produces different outputs for different times."""
    input_shape = (6,)
    out_size = 3
    hypers = TimeDependentResNetHypers(
      working_size=16,
      hidden_size=32,
      n_blocks=2,
      filter_shape=None,
      groups=None,
      activation=jax.nn.swish,
      embedding_size=12,
      out_features=6
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    x = random.normal(key, input_shape)

    t1 = jnp.array(0.2)
    t2 = jnp.array(0.8)

    output1 = time_resnet(t1, x)
    output2 = time_resnet(t2, x)

    # Outputs should be different for different times
    assert not jnp.allclose(output1, output2, atol=1e-6)
    assert output1.shape == output2.shape == (out_size,)
    assert jnp.isfinite(output1).all()
    assert jnp.isfinite(output2).all()

  def test_time_dependent_resnet_3d_initialization(self, key):
    """Test TimeDependentResNet initialization with 3D (image) input."""
    input_shape = (32, 32, 3)
    out_size = 10
    hypers = TimeDependentResNetHypers(
      working_size=64,
      hidden_size=128,
      n_blocks=4,
      filter_shape=(5, 5),
      groups=2,
      activation=jax.nn.relu,
      embedding_size=20,
      out_features=12
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    assert time_resnet.input_shape == input_shape
    assert time_resnet.hypers.working_size == 64
    assert time_resnet.hypers.hidden_size == 128
    assert time_resnet.hypers.n_blocks == 4
    assert time_resnet.hypers.filter_shape == (5, 5)
    assert time_resnet.hypers.groups == 2
    # cond_shape should be (out_features,) since no additional conditioning
    assert time_resnet.cond_shape == (hypers.out_features,)

  def test_time_dependent_resnet_3d_forward_pass(self, key):
    """Test TimeDependentResNet forward pass with 3D input."""
    input_shape = (16, 16, 3)
    out_size = 8
    hypers = TimeDependentResNetHypers(
      working_size=32,
      hidden_size=64,
      n_blocks=2,
      filter_shape=(3, 3),
      groups=1,
      activation=jax.nn.swish,
      embedding_size=16,
      out_features=8
    )

    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      hypers=hypers,
      key=key
    )

    k1, k2 = random.split(key, 2)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, input_shape)
    output = time_resnet(t, x)

    assert output.shape == (16, 16, out_size)
    assert jnp.isfinite(output).all()

  def test_time_dependent_resnet_3d_with_conditioning(self):
    """Test TimeDependentResNet with 3D (image) input and conditioning."""

    input_shape = (12, 12, 3)  # Small image
    cond_shape = (4,)
    out_size = 6

    hypers = TimeDependentResNetHypers(
      working_size = 24,
      hidden_size = 48,
      n_blocks = 2,
      filter_shape = (3, 3),  # Required for 3D input
      groups = 2,  # Groups are allowed for 3D
      activation = jax.nn.gelu,
      embedding_size = 12,
      out_features = 8
    )

    key = random.PRNGKey(42)
    time_resnet_3d = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # Verify setup
    assert time_resnet_3d.input_shape == input_shape
    assert time_resnet_3d.cond_shape == (hypers.out_features + cond_shape[0],)
    assert time_resnet_3d.hypers.groups == 2
    assert time_resnet_3d.hypers.filter_shape == (3, 3)

    # Test forward pass with time and conditioning
    k1, k2, k3 = random.split(key, 3)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, input_shape)
    cond = random.normal(k3, cond_shape)

    output = time_resnet_3d(t, x, cond)

    # Output should have same spatial dimensions but different channels
    expected_output_shape = (12, 12, out_size)
    assert output.shape == expected_output_shape
    assert jnp.isfinite(output).all()

    # Test with different conditioning
    k4 = random.split(key, 1)[0]
    cond2 = random.normal(k4, cond_shape)
    output2 = time_resnet_3d(t, x, cond2)

    assert output2.shape == expected_output_shape
    assert jnp.isfinite(output2).all()

    # Should be different with different conditioning
    assert not jnp.allclose(output, output2, atol=1e-6)

    # Test with different time
    k5 = random.split(key, 1)[0]
    t2 = random.uniform(k5, (), minval=0.0, maxval=1.0)
    if jnp.allclose(t, t2):
      t2 = t + 0.3  # Force different time

    output3 = time_resnet_3d(t2, x, cond)

    assert output3.shape == expected_output_shape
    assert jnp.isfinite(output3).all()

    # Outputs should be different when conditioning changes
    assert not jnp.allclose(output2, output3, atol=1e-6)

class TestResNetGetNNIntegration:
  """Test suite for ResNet integration with get_nn function"""

  def test_get_nn_resnet_creation(self):
    """Test creating ResNet through get_nn function."""
    from linsdex.nn.nn_models.get_nn import get_nn

    # Create mock configs directly
    class MockDatasetConfig:
      n_features = 6

    class MockModelConfig:
      nn_type = 'resnet'
      working_size = 32
      hidden_size = 64
      n_blocks = 3
      n_groups = 1
      embedding_size = 16
      out_features = 8

    model_config = MockModelConfig()
    dataset_config = MockDatasetConfig()
    random_seed = 42

    # Create the model through get_nn
    model = get_nn(model_config, dataset_config, random_seed)

    # Verify it's a ResNet
    assert isinstance(model, ResNet)
    assert model.input_shape == (6,)
    assert model.hypers.working_size == 32
    assert model.hypers.hidden_size == 64
    assert model.hypers.n_blocks == 3
    assert model.hypers.groups is None  # Groups should be None for 1D data
    assert model.hypers.activation == jax.nn.gelu

    # Test that it can be called
    key = random.PRNGKey(42)
    x = random.normal(key, (6,))
    output = model(x)

    assert output.shape == (6,)
    assert jnp.isfinite(output).all()

  def test_get_nn_time_dependent_resnet_creation(self):
    """Test creating TimeDependentResNet through get_nn function."""
    from linsdex.nn.nn_models.get_nn import get_nn

    # Create mock configs directly
    class MockDatasetConfig:
      n_features = 8

    class MockModelConfig:
      nn_type = 'time_dependent_resnet'
      working_size = 32
      hidden_size = 64
      n_blocks = 3
      n_groups = 1
      embedding_size = 16
      out_features = 8

    model_config = MockModelConfig()
    dataset_config = MockDatasetConfig()
    random_seed = 42

    # Create the model through get_nn
    model = get_nn(model_config, dataset_config, random_seed)

    # Verify it's a TimeDependentResNet
    assert isinstance(model, TimeDependentResNet)
    assert model.input_shape == (8,)
    assert model.hypers.working_size == 32
    assert model.hypers.hidden_size == 64
    assert model.hypers.n_blocks == 3
    assert model.hypers.groups is None  # Groups should be None for 1D data
    assert model.hypers.activation == jax.nn.gelu
    assert model.hypers.embedding_size == 16
    assert model.hypers.out_features == 8

    # Test that it can be called
    key = random.PRNGKey(42)
    k1, k2 = random.split(key, 2)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, (8,))
    output = model(t, x)

    assert output.shape == (8,)
    assert jnp.isfinite(output).all()

  def test_get_nn_invalid_nn_type(self):
    """Test that get_nn raises error for invalid nn_type."""
    from linsdex.nn.nn_models.get_nn import get_nn

    # Create mock configs directly
    class MockDatasetConfig:
      n_features = 4

    class MockModelConfig:
      nn_type = 'invalid_type'

    model_config = MockModelConfig()
    dataset_config = MockDatasetConfig()
    random_seed = 42

    with pytest.raises(ValueError, match="Invalid neural network type"):
      get_nn(model_config, dataset_config, random_seed)

  def test_get_nn_different_feature_sizes(self):
    """Test get_nn with different feature sizes."""
    from linsdex.nn.nn_models.get_nn import get_nn

    for n_features in [1, 3, 10, 20]:
      # Create mock configs for each iteration
      class MockDatasetConfig:
        pass
      MockDatasetConfig.n_features = n_features

      class MockModelConfig:
        nn_type = 'resnet'
        working_size = 32
        hidden_size = 64
        n_blocks = 3
        n_groups = 1
        embedding_size = 16
        out_features = 8

      model_config = MockModelConfig()
      dataset_config = MockDatasetConfig()
      random_seed = 42
      model = get_nn(model_config, dataset_config, random_seed)

      assert isinstance(model, ResNet)
      assert model.input_shape == (n_features,)

      # Test forward pass
      key = random.PRNGKey(42)
      x = random.normal(key, (n_features,))
      output = model(x)

      assert output.shape == (n_features,)
      assert jnp.isfinite(output).all()

  def test_get_nn_different_hyperparameters(self):
    """Test get_nn with different hyperparameter values."""
    from linsdex.nn.nn_models.get_nn import get_nn

    # Create mock configs directly
    class MockDatasetConfig:
      n_features = 5

    class MockModelConfig:
      nn_type = 'resnet'
      working_size = 16
      hidden_size = 32
      n_blocks = 2
      n_groups = 2
      embedding_size = 8
      out_features = 4

    model_config = MockModelConfig()
    dataset_config = MockDatasetConfig()
    random_seed = 123
    model = get_nn(model_config, dataset_config, random_seed)

    assert isinstance(model, ResNet)
    assert model.hypers.working_size == 16
    assert model.hypers.hidden_size == 32
    assert model.hypers.n_blocks == 2
    assert model.hypers.groups is None  # Groups should be None for 1D data

    # Test forward pass
    key = random.PRNGKey(123)
    x = random.normal(key, (5,))
    output = model(x)

    assert output.shape == (5,)
    assert jnp.isfinite(output).all()

  def test_get_nn_resnet_with_conditioning(self):
    """Test creating ResNet with conditioning through direct initialization."""
    # Note: get_nn doesn't support conditioning yet, so we test direct ResNet creation
    # with conditioning and show how it could work

    input_shape = (8,)
    cond_shape = (4,)
    out_size = 6

    hypers = ResNetHypers(
      working_size = 32,
      hidden_size = 64,
      n_blocks = 3,
      filter_shape = None,
      groups = None,
      activation = jax.nn.gelu
    )

    key = random.PRNGKey(42)
    resnet = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # Verify conditioning is set up
    assert resnet.input_shape == input_shape
    assert resnet.cond_shape == cond_shape

    # Test forward pass with conditioning
    k1, k2 = random.split(key, 2)
    x = random.normal(k1, input_shape)
    cond = random.normal(k2, cond_shape)

    output = resnet(x, cond)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

    # Test that output is different with different conditioning
    k3 = random.split(key, 1)[0]
    cond2 = random.normal(k3, cond_shape)
    output2 = resnet(x, cond2)

    # Should produce different outputs for different conditioning
    assert not jnp.allclose(output, output2, atol=1e-6)

  def test_get_nn_time_dependent_resnet_with_conditioning(self):
    """Test creating TimeDependentResNet with additional conditioning."""

    input_shape = (6,)
    cond_shape = (3,)
    out_size = 4

    hypers = TimeDependentResNetHypers(
      working_size = 24,
      hidden_size = 48,
      n_blocks = 2,
      filter_shape = None,
      groups = None,
      activation = jax.nn.gelu,
      embedding_size = 12,
      out_features = 6
    )

    key = random.PRNGKey(42)
    time_resnet = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # Verify conditioning is set up (should be time features + additional conditioning)
    assert time_resnet.input_shape == input_shape
    assert time_resnet.cond_shape == (hypers.out_features + cond_shape[0],)

    # Test forward pass with time and conditioning
    k1, k2, k3 = random.split(key, 3)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, input_shape)
    cond = random.normal(k3, cond_shape)

    output = time_resnet(t, x, cond)

    assert output.shape == (out_size,)
    assert jnp.isfinite(output).all()

    # Test that output is different with different conditioning
    k4 = random.split(key, 1)[0]
    cond2 = random.normal(k4, cond_shape)
    output2 = time_resnet(t, x, cond2)

    # Should produce different outputs for different conditioning
    assert not jnp.allclose(output, output2, atol=1e-6)

    # Test that output is different for different times (with same conditioning)
    # Generate a different time value
    k5 = random.split(key, 1)[0]
    t2 = random.uniform(k5, (), minval=0.0, maxval=1.0)
    # Make sure the times are actually different
    if jnp.allclose(t, t2):
      t2 = t + 0.5  # Force different time

    output3 = time_resnet(t2, x, cond)

    # Should produce different outputs for different times
    assert not jnp.allclose(output, output3, atol=1e-6)

  def test_resnet_conditioning_vs_no_conditioning(self):
    """Test that ResNet behaves consistently with and without conditioning."""

    input_shape = (5,)
    out_size = 3

    hypers = ResNetHypers(
      working_size = 16,
      hidden_size = 32,
      n_blocks = 2,
      filter_shape = None,
      groups = None,
      activation = jax.nn.gelu
    )

    key = random.PRNGKey(42)

    # Create ResNet without conditioning
    resnet_no_cond = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=None,
      hypers=hypers,
      key=key
    )

    # Create ResNet with conditioning
    cond_shape = (2,)
    resnet_with_cond = ResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # Test forward passes
    k1, k2 = random.split(key, 2)
    x = random.normal(k1, input_shape)
    cond = random.normal(k2, cond_shape)

    # Without conditioning
    output1 = resnet_no_cond(x)
    assert output1.shape == (out_size,)
    assert jnp.isfinite(output1).all()

    # With conditioning - pass None (should work)
    output2 = resnet_with_cond(x, None)
    assert output2.shape == (out_size,)
    assert jnp.isfinite(output2).all()

    # With conditioning - pass actual conditioning
    output3 = resnet_with_cond(x, cond)
    assert output3.shape == (out_size,)
    assert jnp.isfinite(output3).all()

    # Output with conditioning should be different from without
    assert not jnp.allclose(output2, output3, atol=1e-6)

  def test_time_dependent_resnet_conditioning_combinations(self):
    """Test TimeDependentResNet with various conditioning combinations."""

    input_shape = (4,)
    out_size = 2

    hypers = TimeDependentResNetHypers(
      working_size = 16,
      hidden_size = 32,
      n_blocks = 2,
      filter_shape = None,
      groups = None,
      activation = jax.nn.gelu,
      embedding_size = 8,
      out_features = 4
    )

    key = random.PRNGKey(42)

    # Test without additional conditioning
    time_resnet_no_cond = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=None,
      hypers=hypers,
      key=key
    )

    # Test with additional conditioning
    cond_shape = (3,)
    time_resnet_with_cond = TimeDependentResNet(
      input_shape=input_shape,
      out_size=out_size,
      cond_shape=cond_shape,
      hypers=hypers,
      key=key
    )

    # Check conditioning shapes
    assert time_resnet_no_cond.cond_shape == (hypers.out_features,)
    assert time_resnet_with_cond.cond_shape == (hypers.out_features + cond_shape[0],)

    # Test forward passes
    k1, k2, k3 = random.split(key, 3)
    t = random.uniform(k1, (), minval=0.0, maxval=1.0)
    x = random.normal(k2, input_shape)
    cond = random.normal(k3, cond_shape)

    # Without additional conditioning (only time)
    output1 = time_resnet_no_cond(t, x)
    assert output1.shape == (out_size,)
    assert jnp.isfinite(output1).all()

    # With additional conditioning - must pass actual conditioning
    # (TimeDependentResNet with cond_shape set expects conditioning)
    output2 = time_resnet_with_cond(t, x, cond)
    assert output2.shape == (out_size,)
    assert jnp.isfinite(output2).all()

    # Test different conditioning values
    k4 = random.split(key, 1)[0]
    cond2 = random.normal(k4, cond_shape)
    output3 = time_resnet_with_cond(t, x, cond2)
    assert output3.shape == (out_size,)
    assert jnp.isfinite(output3).all()

    # Outputs should be different when conditioning changes
    assert not jnp.allclose(output2, output3, atol=1e-6)


def test_get_nn_resnet_3d_data_default():
  """Test ResNet creation through get_nn with 3D data using default filter_shape and groups."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 3  # RGB channels
    height: int = 32
    width: int = 32

  @dataclass
  class MockModelConfig:
    nn_type: str = "resnet"
    working_size: int = 64
    hidden_size: int = 128
    n_blocks: int = 3

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, ResNet)
  assert model.input_shape == (32, 32, 3)

  # Check that default values were used
  assert model.hypers.filter_shape == (3, 3)  # Default 3x3 filters
  assert model.hypers.groups == 8  # Default groups

  # Test forward pass
  key = random.PRNGKey(0)
  x = random.normal(key, (32, 32, 3))
  out = model(x)
  assert out.shape == (32, 32, 3)  # Output should have same spatial dimensions, same channels


def test_get_nn_resnet_3d_data_custom():
  """Test ResNet creation through get_nn with 3D data using custom filter_shape and groups."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 1  # Grayscale
    height: int = 64
    width: int = 64

  @dataclass
  class MockModelConfig:
    nn_type: str = "resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2
    filter_shape: tuple = (5, 5)  # Custom 5x5 filters
    n_groups: int = 4  # Custom groups

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, ResNet)
  assert model.input_shape == (64, 64, 1)

  # Check that custom values were used
  assert model.hypers.filter_shape == (5, 5)
  assert model.hypers.groups == 4

  # Test forward pass
  key = random.PRNGKey(0)
  x = random.normal(key, (64, 64, 1))
  out = model(x)
  assert out.shape == (64, 64, 1)  # Output should preserve shape


def test_get_nn_time_dependent_resnet_3d_data_default():
  """Test TimeDependentResNet creation through get_nn with 3D data using defaults."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 4
    height: int = 28
    width: int = 28

  @dataclass
  class MockModelConfig:
    nn_type: str = "time_dependent_resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2
    embedding_size: int = 64
    out_features: int = 1

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, TimeDependentResNet)
  assert model.input_shape == (28, 28, 4)

  # Check that default values were used
  assert model.hypers.filter_shape == (3, 3)
  assert model.hypers.groups == 8

  # Test forward pass
  key = random.PRNGKey(0)
  x = random.normal(key, (28, 28, 4))
  time = 0.1  # Scalar time, not array
  out = model(time, x)
  assert out.shape == (28, 28, 4)  # Output should preserve shape


def test_get_nn_1d_vs_3d_behavior():
  """Test that get_nn correctly differentiates between 1D and 3D data."""
  @dataclass
  class Mock1DDatasetConfig:
    n_features: int = 10

  @dataclass
  class Mock3DDatasetConfig:
    n_features: int = 3
    height: int = 32
    width: int = 32

  @dataclass
  class MockModelConfig:
    nn_type: str = "resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2

  model_config = MockModelConfig()
  dataset_config_1d = Mock1DDatasetConfig()
  dataset_config_3d = Mock3DDatasetConfig()
  random_seed = 42

  # Test 1D configuration
  model_1d = get_nn(model_config, dataset_config_1d, random_seed)
  assert model_1d.input_shape == (10,)
  assert model_1d.hypers.filter_shape is None
  assert model_1d.hypers.groups is None

  # Test 3D configuration
  model_3d = get_nn(model_config, dataset_config_3d, random_seed)
  assert model_3d.input_shape == (32, 32, 3)
  assert model_3d.hypers.filter_shape == (3, 3)
  assert model_3d.hypers.groups == 8

  # Test that forward passes work correctly
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)

  # 1D forward pass
  x_1d = random.normal(k1, (10,))
  out_1d = model_1d(x_1d)
  assert out_1d.shape == (10,)

  # 3D forward pass
  x_3d = random.normal(k2, (32, 32, 3))
  out_3d = model_3d(x_3d)
  assert out_3d.shape == (32, 32, 3)


def test_get_nn_resnet_1d_with_conditioning():
  """Test ResNet creation through get_nn with 1D data and conditioning."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 8
    cond_shape: tuple = (4,)  # 4-dimensional conditioning

  @dataclass
  class MockModelConfig:
    nn_type: str = "resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, ResNet)
  assert model.input_shape == (8,)
  assert model.cond_shape == (4,)

  # Test forward pass with conditioning
  key = random.PRNGKey(0)
  k1, k2 = random.split(key)
  x = random.normal(k1, (8,))
  conditioning = random.normal(k2, (4,))

  out = model(x, conditioning)
  assert out.shape == (8,)

  # Test forward pass without conditioning (should still work)
  out_no_cond = model(x, None)
  assert out_no_cond.shape == (8,)


def test_get_nn_resnet_3d_with_conditioning():
  """Test ResNet creation through get_nn with 3D data and conditioning."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 3
    height: int = 32
    width: int = 32
    cond_shape: tuple = (6,)  # 6-dimensional conditioning

  @dataclass
  class MockModelConfig:
    nn_type: str = "resnet"
    working_size: int = 64
    hidden_size: int = 128
    n_blocks: int = 2

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, ResNet)
  assert model.input_shape == (32, 32, 3)
  assert model.cond_shape == (6,)

  # Test forward pass with conditioning
  key = random.PRNGKey(0)
  k1, k2 = random.split(key)
  x = random.normal(k1, (32, 32, 3))
  conditioning = random.normal(k2, (6,))

  out = model(x, conditioning)
  assert out.shape == (32, 32, 3)

  # Test forward pass without conditioning
  out_no_cond = model(x, None)
  assert out_no_cond.shape == (32, 32, 3)


def test_get_nn_time_dependent_resnet_1d_with_conditioning():
  """Test TimeDependentResNet creation through get_nn with 1D data and conditioning."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 5
    conditioning_shape: tuple = (3,)  # Test alternative conditioning field name

  @dataclass
  class MockModelConfig:
    nn_type: str = "time_dependent_resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2
    embedding_size: int = 64
    out_features: int = 1

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, TimeDependentResNet)
  assert model.input_shape == (5,)
  assert model.cond_shape == (4,)  # 3 + 1 for time features (out_features=1)

  # Test forward pass with conditioning
  key = random.PRNGKey(0)
  k1, k2 = random.split(key)
  x = random.normal(k1, (5,))
  conditioning = random.normal(k2, (3,))  # User conditioning size
  time = 0.5

  out = model(time, x, conditioning)
  assert out.shape == (5,)

  # Test that different conditioning produces different outputs
  k3 = random.split(k2, 1)[0]
  conditioning2 = random.normal(k3, (3,))
  out2 = model(time, x, conditioning2)
  assert not jnp.allclose(out, out2, atol=1e-6)


def test_get_nn_time_dependent_resnet_3d_with_conditioning():
  """Test TimeDependentResNet creation through get_nn with 3D data and conditioning."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 4
    height: int = 28
    width: int = 28
    cond_shape: tuple = (8,)  # 8-dimensional conditioning

  @dataclass
  class MockModelConfig:
    nn_type: str = "time_dependent_resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2
    embedding_size: int = 64
    out_features: int = 1

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, TimeDependentResNet)
  assert model.input_shape == (28, 28, 4)
  assert model.cond_shape == (9,)  # 8 + 1 for time features (out_features=1)

  # Test forward pass with conditioning
  key = random.PRNGKey(0)
  k1, k2 = random.split(key)
  x = random.normal(k1, (28, 28, 4))
  conditioning = random.normal(k2, (8,))  # User conditioning size
  time = 0.3

  out = model(time, x, conditioning)
  assert out.shape == (28, 28, 4)

  # Test that different conditioning produces different outputs
  k3 = random.split(k2, 1)[0]
  conditioning2 = random.normal(k3, (8,))
  out2 = model(time, x, conditioning2)
  assert not jnp.allclose(out, out2, atol=1e-6)


def test_get_nn_conditioning_behavior_differences():
  """Test that conditioning actually affects model behavior."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 4
    cond_shape: tuple = (2,)

  @dataclass
  class MockModelConfig:
    nn_type: str = "resnet"
    working_size: int = 16
    hidden_size: int = 32
    n_blocks: int = 1

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 123

  model = get_nn(model_config, dataset_config, random_seed)

  # Test that different conditioning values produce different outputs
  key = random.PRNGKey(42)
  k1, k2, k3 = random.split(key, 3)

  x = random.normal(k1, (4,))
  cond1 = random.normal(k2, (2,))
  cond2 = random.normal(k3, (2,))

  out1 = model(x, cond1)
  out2 = model(x, cond2)

  # Different conditioning should produce different outputs
  assert not jnp.allclose(out1, out2, atol=1e-6)

  # Same conditioning should produce same outputs
  out3 = model(x, cond1)
  assert jnp.allclose(out1, out3, atol=1e-6)


def test_get_nn_time_dependent_resnet_no_conditioning():
  """Test TimeDependentResNet creation through get_nn without conditioning."""
  @dataclass
  class MockDatasetConfig:
    n_features: int = 6
    # No cond_shape field - no conditioning

  @dataclass
  class MockModelConfig:
    nn_type: str = "time_dependent_resnet"
    working_size: int = 32
    hidden_size: int = 64
    n_blocks: int = 2
    embedding_size: int = 64
    out_features: int = 1

  model_config = MockModelConfig()
  dataset_config = MockDatasetConfig()
  random_seed = 42

  model = get_nn(model_config, dataset_config, random_seed)

  # Test model creation
  assert isinstance(model, TimeDependentResNet)
  assert model.input_shape == (6,)
  assert model.cond_shape == (1,)  # Only time features (out_features=1)

  # Test forward pass without conditioning
  key = random.PRNGKey(0)
  x = random.normal(key, (6,))
  time = 0.2

  out = model(time, x, None)
  assert out.shape == (6,)

  # Test that different times produce different outputs
  out2 = model(0.8, x, None)
  assert not jnp.allclose(out, out2, atol=1e-6)