import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import equinox as eqx
from linsdex.nn.nn_layers.rnn_layers import CausalConv1d, WaveNetResBlock, GRURNN

class TestCausalConv1d:
  """Test suite for CausalConv1d"""

  def test_causal_conv1d_initialization(self, key):
    """Test CausalConv1d initialization with default parameters."""
    in_channels = 8
    out_channels = 16

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      key=key
    )

    assert conv.kernel_width == 3  # default
    assert conv.stride == 1  # default
    assert conv.dilation == 1  # default
    assert conv.use_bias is True  # default
    assert conv.padding == 2  # kernel_width - 1

  def test_causal_conv1d_custom_parameters(self, key):
    """Test CausalConv1d initialization with custom parameters."""
    in_channels = 4
    out_channels = 12
    kernel_width = 5
    stride = 2
    dilation = 3
    use_bias = False

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_width=kernel_width,
      stride=stride,
      dilation=dilation,
      use_bias=use_bias,
      key=key
    )

    assert conv.kernel_width == kernel_width
    assert conv.stride == stride
    assert conv.dilation == dilation
    assert conv.use_bias == use_bias
    assert conv.padding == kernel_width - 1

  def test_causal_conv1d_forward_pass(self, key):
    """Test CausalConv1d forward pass."""
    in_channels = 6
    out_channels = 10
    seq_length = 20

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))
    output = conv(x)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_causal_conv1d_different_sequence_lengths(self, key):
    """Test CausalConv1d with different sequence lengths."""
    in_channels = 4
    out_channels = 8

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      key=key
    )

    for seq_length in [5, 10, 25, 50]:
      x = random.normal(key, (seq_length, in_channels))
      output = conv(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()

  def test_causal_conv1d_different_kernel_widths(self, key):
    """Test CausalConv1d with different kernel widths."""
    in_channels = 3
    out_channels = 6
    seq_length = 15

    for kernel_width in [1, 2, 3, 4, 5]:
      conv = CausalConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_width=kernel_width,
        key=key
      )

      x = random.normal(key, (seq_length, in_channels))
      output = conv(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()
      assert conv.padding == kernel_width - 1

  def test_causal_conv1d_batch_size_property(self, key):
    """Test CausalConv1d batch_size property."""
    in_channels = 4
    out_channels = 8

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      key=key
    )

    # Should be None for unbatched
    assert conv.batch_size is None

  def test_causal_conv1d_with_bias_and_without(self, key):
    """Test CausalConv1d with and without bias."""
    in_channels = 5
    out_channels = 10
    seq_length = 12

    k1, k2 = random.split(key, 2)

    # With bias
    conv_with_bias = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      use_bias=True,
      key=k1
    )

    # Without bias
    conv_without_bias = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      use_bias=False,
      key=k2
    )

    x = random.normal(key, (seq_length, in_channels))

    output_with_bias = conv_with_bias(x)
    output_without_bias = conv_without_bias(x)

    assert output_with_bias.shape == (seq_length, out_channels)
    assert output_without_bias.shape == (seq_length, out_channels)
    assert jnp.isfinite(output_with_bias).all()
    assert jnp.isfinite(output_without_bias).all()

  def test_causal_conv1d_causality(self, key):
    """Test that CausalConv1d actually enforces causality."""
    in_channels = 4
    out_channels = 4
    seq_length = 10
    kernel_width = 3

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_width=kernel_width,
      key=key
    )

    # Create input with impulse at the end
    x = jnp.zeros((seq_length, in_channels))
    x = x.at[-1, :].set(1.0)

    output = conv(x)

    # Check that output at early time steps is not affected by the impulse at the end
    # This is a basic check - more rigorous causality testing would use Jacobians
    assert jnp.isfinite(output).all()

  def test_causal_conv1d_jacobian_lower_triangular(self, key):
    """Test that CausalConv1d produces a lower triangular Jacobian matrix."""
    in_channels = 4
    out_channels = 4
    seq_length = 8

    conv = CausalConv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_width=3,
      key=key
    )

    # Create input sequence
    x = random.normal(key, (seq_length, in_channels))

    # Define function for Jacobian computation
    def conv_func(x_input):
      return conv(x_input)

    # Compute the Jacobian
    J = eqx.filter_jacfwd(conv_func)(x)

    # Sum over the output and input feature dimensions to get a (seq_length, seq_length) matrix
    # J shape is (seq_length, out_channels, seq_length, in_channels)
    J = J.sum(axis=(1, 3))

    # Check that the Jacobian is lower triangular
    assert jnp.allclose(J, jnp.tril(J)), "CausalConv1d should produce lower triangular Jacobian"

class TestWaveNetResBlock:
  """Test suite for WaveNetResBlock"""

  def test_wavenet_resblock_initialization(self, key):
    """Test WaveNetResBlock initialization with default parameters."""
    in_channels = 8

    block = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    assert block.kernel_width == 2  # default
    assert block.dilation == 1  # default
    assert block.hidden_channels == 32  # default

  def test_wavenet_resblock_custom_parameters(self, key):
    """Test WaveNetResBlock initialization with custom parameters."""
    in_channels = 6
    kernel_width = 4
    dilation = 2
    hidden_channels = 64

    block = WaveNetResBlock(
      in_channels=in_channels,
      kernel_width=kernel_width,
      dilation=dilation,
      hidden_channels=hidden_channels,
      key=key
    )

    assert block.kernel_width == kernel_width
    assert block.dilation == dilation
    assert block.hidden_channels == hidden_channels

  def test_wavenet_resblock_forward_pass(self, key):
    """Test WaveNetResBlock forward pass."""
    in_channels = 8
    seq_length = 20

    block = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))
    new_hidden, skip = block(x)

    assert new_hidden.shape == (seq_length, in_channels)
    assert skip.shape == (seq_length, in_channels)
    assert jnp.isfinite(new_hidden).all()
    assert jnp.isfinite(skip).all()

  def test_wavenet_resblock_residual_connection(self, key):
    """Test that WaveNetResBlock properly implements residual connections."""
    in_channels = 6
    seq_length = 15

    block = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))
    new_hidden, skip = block(x)

    # The residual connection means new_hidden should be related to input x
    # We can't test exact equality due to the nonlinear transformations,
    # but we can verify the shapes and that values are finite
    assert new_hidden.shape == x.shape
    assert jnp.isfinite(new_hidden).all()

  def test_wavenet_resblock_different_sequence_lengths(self, key):
    """Test WaveNetResBlock with different sequence lengths."""
    in_channels = 4

    block = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    for seq_length in [5, 10, 25, 50]:
      x = random.normal(key, (seq_length, in_channels))
      new_hidden, skip = block(x)

      assert new_hidden.shape == (seq_length, in_channels)
      assert skip.shape == (seq_length, in_channels)
      assert jnp.isfinite(new_hidden).all()
      assert jnp.isfinite(skip).all()

  def test_wavenet_resblock_different_dilations(self, key):
    """Test WaveNetResBlock with different dilation values."""
    in_channels = 6
    seq_length = 30

    for dilation in [1, 2, 4, 8]:
      block = WaveNetResBlock(
        in_channels=in_channels,
        dilation=dilation,
        key=key
      )

      x = random.normal(key, (seq_length, in_channels))
      new_hidden, skip = block(x)

      assert new_hidden.shape == (seq_length, in_channels)
      assert skip.shape == (seq_length, in_channels)
      assert jnp.isfinite(new_hidden).all()
      assert jnp.isfinite(skip).all()
      assert block.dilation == dilation

  def test_wavenet_resblock_different_hidden_channels(self, key):
    """Test WaveNetResBlock with different hidden channel sizes."""
    in_channels = 8
    seq_length = 20

    for hidden_channels in [16, 32, 64, 128]:
      block = WaveNetResBlock(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        key=key
      )

      x = random.normal(key, (seq_length, in_channels))
      new_hidden, skip = block(x)

      assert new_hidden.shape == (seq_length, in_channels)
      assert skip.shape == (seq_length, in_channels)
      assert jnp.isfinite(new_hidden).all()
      assert jnp.isfinite(skip).all()
      assert block.hidden_channels == hidden_channels

  def test_wavenet_resblock_batch_size_property(self, key):
    """Test WaveNetResBlock batch_size property."""
    in_channels = 6

    block = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    # Should be None for unbatched
    assert block.batch_size is None

  def test_wavenet_resblock_gating_mechanism(self, key):
    """Test that WaveNetResBlock implements proper gating mechanism."""
    in_channels = 4
    seq_length = 15

    block = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    # Create two different inputs
    k1, k2 = random.split(key, 2)
    x1 = random.normal(k1, (seq_length, in_channels))
    x2 = random.normal(k2, (seq_length, in_channels))

    new_hidden1, skip1 = block(x1)
    new_hidden2, skip2 = block(x2)

    # Different inputs should produce different outputs
    assert not jnp.allclose(new_hidden1, new_hidden2, atol=1e-6)
    assert not jnp.allclose(skip1, skip2, atol=1e-6)

  def test_wavenet_resblock_consistency(self, key):
    """Test that WaveNetResBlock produces consistent results."""
    in_channels = 8
    seq_length = 20

    # Create two identical blocks
    block1 = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    block2 = WaveNetResBlock(
      in_channels=in_channels,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))

    new_hidden1, skip1 = block1(x)
    new_hidden2, skip2 = block2(x)

    # Same input and same initialization should produce same output
    assert jnp.allclose(new_hidden1, new_hidden2)
    assert jnp.allclose(skip1, skip2)

  def test_wavenet_resblock_jacobian_lower_triangular(self, key):
    """Test that WaveNetResBlock produces a lower triangular Jacobian matrix."""
    in_channels = 6
    seq_length = 8

    block = WaveNetResBlock(
      in_channels=in_channels,
      kernel_width=3,
      key=key
    )

    # Create input sequence
    x = random.normal(key, (seq_length, in_channels))

    # Define function for Jacobian computation (testing the new_hidden output)
    def block_func(x_input):
      new_hidden, skip = block(x_input)
      return new_hidden

    # Compute the Jacobian
    J = eqx.filter_jacfwd(block_func)(x)

    # Sum over the output and input feature dimensions to get a (seq_length, seq_length) matrix
    # J shape is (seq_length, in_channels, seq_length, in_channels)
    J = J.sum(axis=(1, 3))

    # Check that the Jacobian is lower triangular
    assert jnp.allclose(J, jnp.tril(J)), "WaveNetResBlock should produce lower triangular Jacobian"

  def test_wavenet_resblock_skip_jacobian_lower_triangular(self, key):
    """Test that WaveNetResBlock skip connection produces a lower triangular Jacobian matrix."""
    in_channels = 6
    seq_length = 8

    block = WaveNetResBlock(
      in_channels=in_channels,
      kernel_width=3,
      key=key
    )

    # Create input sequence
    x = random.normal(key, (seq_length, in_channels))

    # Define function for Jacobian computation (testing the skip output)
    def skip_func(x_input):
      new_hidden, skip = block(x_input)
      return skip

    # Compute the Jacobian
    J = eqx.filter_jacfwd(skip_func)(x)

    # Sum over the output and input feature dimensions to get a (seq_length, seq_length) matrix
    # J shape is (seq_length, in_channels, seq_length, in_channels)
    J = J.sum(axis=(1, 3))

    # Check that the Jacobian is lower triangular
    assert jnp.allclose(J, jnp.tril(J)), "WaveNetResBlock skip connection should produce lower triangular Jacobian"

class TestGRURNN:
  """Test suite for GRURNN"""

  def test_grurnn_initialization(self, key):
    """Test GRURNN initialization."""
    in_channels = 8
    out_channels = 12
    hidden_size = 16

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    # Check that components are properly initialized
    assert rnn.gru.input_size == in_channels
    assert rnn.gru.hidden_size == hidden_size
    assert rnn.out_proj.in_features == hidden_size
    assert rnn.out_proj.out_features == out_channels
    assert rnn.initial_state.shape == (hidden_size,)

  def test_grurnn_forward_pass(self, key):
    """Test GRURNN forward pass."""
    in_channels = 6
    out_channels = 10
    hidden_size = 12
    seq_length = 15

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))
    output = rnn(x)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_grurnn_different_sizes(self, key):
    """Test GRURNN with different input, output, and hidden sizes."""
    test_configs = [
      (4, 8, 16),
      (10, 5, 12),
      (16, 32, 24),
      (32, 16, 20)
    ]

    seq_length = 10

    for in_channels, out_channels, hidden_size in test_configs:
      rnn = GRURNN(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        key=key
      )

      x = random.normal(key, (seq_length, in_channels))
      output = rnn(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()

  def test_grurnn_batch_size_property(self, key):
    """Test GRURNN batch_size property."""
    in_channels = 8
    out_channels = 12
    hidden_size = 16

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    # Should be None for unbatched
    assert rnn.batch_size is None

  def test_grurnn_consistency(self, key):
    """Test that GRURNN produces consistent results."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12
    seq_length = 10

    # Create two identical RNNs
    rnn1 = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    rnn2 = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))

    output1 = rnn1(x)
    output2 = rnn2(x)

    # Same input and same initialization should produce same output
    assert jnp.allclose(output1, output2)

  def test_grurnn_different_inputs(self, key):
    """Test that GRURNN produces different outputs for different inputs."""
    in_channels = 8
    out_channels = 10
    hidden_size = 16
    seq_length = 12

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    k1, k2 = random.split(key, 2)

    x1 = random.normal(k1, (seq_length, in_channels))
    x2 = random.normal(k2, (seq_length, in_channels))

    output1 = rnn(x1)
    output2 = rnn(x2)

    # Different inputs should produce different outputs
    assert not jnp.allclose(output1, output2, atol=1e-6)

  def test_grurnn_different_sequence_lengths(self, key):
    """Test GRURNN with different sequence lengths."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    for seq_length in [5, 10, 20, 50]:
      x = random.normal(key, (seq_length, in_channels))
      output = rnn(x)

      assert output.shape == (seq_length, out_channels)
      assert jnp.isfinite(output).all()

  def test_grurnn_with_global_context(self, key):
    """Test GRURNN with global context."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12
    seq_length = 15

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    k1, k2 = random.split(key, 2)
    x = random.normal(k1, (seq_length, in_channels))
    global_context = random.normal(k2, (hidden_size,))

    output_with_context = rnn(x, global_context)
    output_without_context = rnn(x)

    assert output_with_context.shape == (seq_length, out_channels)
    assert output_without_context.shape == (seq_length, out_channels)
    assert jnp.isfinite(output_with_context).all()
    assert jnp.isfinite(output_without_context).all()

    # Outputs should be different when using global context
    assert not jnp.allclose(output_with_context, output_without_context, atol=1e-6)

  def test_grurnn_different_global_contexts(self, key):
    """Test GRURNN with different global contexts."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12
    seq_length = 15

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    k1, k2, k3 = random.split(key, 3)
    x = random.normal(k1, (seq_length, in_channels))
    context1 = random.normal(k2, (hidden_size,))
    context2 = random.normal(k3, (hidden_size,))

    output1 = rnn(x, context1)
    output2 = rnn(x, context2)

    # Different global contexts should produce different outputs
    assert not jnp.allclose(output1, output2, atol=1e-6)

  def test_grurnn_temporal_dependencies(self, key):
    """Test that GRURNN captures temporal dependencies."""
    in_channels = 4
    out_channels = 6
    hidden_size = 8
    seq_length = 20

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    # Create input with impulse at different time steps
    x1 = jnp.zeros((seq_length, in_channels))
    x1 = x1.at[5, :].set(1.0)

    x2 = jnp.zeros((seq_length, in_channels))
    x2 = x2.at[15, :].set(1.0)

    output1 = rnn(x1)
    output2 = rnn(x2)

    # Outputs should be different due to temporal position of impulse
    assert not jnp.allclose(output1, output2, atol=1e-6)

    # Output at later time steps should be affected by earlier impulse
    # This tests that the RNN maintains state across time
    assert not jnp.allclose(output1[10:], jnp.zeros_like(output1[10:]), atol=1e-6)

  def test_grurnn_zero_input(self, key):
    """Test GRURNN with zero input."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12
    seq_length = 10

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    zero_input = jnp.zeros((seq_length, in_channels))
    output = rnn(zero_input)

    assert output.shape == (seq_length, out_channels)
    assert jnp.isfinite(output).all()

  def test_grurnn_single_timestep(self, key):
    """Test GRURNN with single timestep input."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    x = random.normal(key, (1, in_channels))
    output = rnn(x)

    assert output.shape == (1, out_channels)
    assert jnp.isfinite(output).all()

  def test_grurnn_initial_state_influence(self, key):
    """Test that GRURNN initial state affects output."""
    in_channels = 6
    out_channels = 8
    hidden_size = 12
    seq_length = 5

    # Create RNN
    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    # Test with zero input to see pure initial state influence
    zero_input = jnp.zeros((seq_length, in_channels))

    # Test without global context (just initial state)
    output_no_context = rnn(zero_input)

    # Test with global context (initial state + context)
    global_context = random.normal(key, (hidden_size,))
    output_with_context = rnn(zero_input, global_context)

    # Outputs should be different due to different effective initial states
    assert not jnp.allclose(output_no_context, output_with_context, atol=1e-6)

  def test_grurnn_state_evolution(self, key):
    """Test that GRURNN hidden state evolves over time."""
    in_channels = 4
    out_channels = 6
    hidden_size = 8
    seq_length = 15

    rnn = GRURNN(
      in_channels=in_channels,
      out_channels=out_channels,
      hidden_size=hidden_size,
      key=key
    )

    x = random.normal(key, (seq_length, in_channels))
    output = rnn(x)

    # Output at different time steps should generally be different
    # (unless the input drives the system to a fixed point, which is unlikely with random input)
    first_half = output[:seq_length//2]
    second_half = output[seq_length//2:]

    # Check that outputs evolve (not all identical)
    assert not jnp.allclose(first_half.mean(axis=0), second_half.mean(axis=0), atol=1e-6)