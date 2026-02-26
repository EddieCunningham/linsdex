import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import equinox as eqx
from linsdex.nn.nn_layers.attention import MultiheadAttention

class TestMultiheadAttention:
  """Test suite for MultiheadAttention"""

  def test_multihead_attention_initialization(self, key):
    """Test MultiheadAttention initialization."""
    num_heads = 4
    query_size = 16
    key_value_size = 12
    output_size = 20

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=key_value_size,
      output_size=output_size,
      causal=True,
      key=key
    )

    assert mha.num_heads == num_heads
    assert mha.query_size == query_size
    assert mha.key_value_size == key_value_size
    assert mha.output_size == output_size
    assert mha.causal is True

  def test_multihead_attention_forward_causal(self, key):
    """Test MultiheadAttention forward pass with causal attention."""
    num_heads = 2
    query_size = 8
    key_value_size = 6
    q_seq = 10
    kv_seq = 12

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=key_value_size,
      output_size=query_size,
      causal=True,
      key=key
    )

    k1, k2 = random.split(key, 2)
    query = random.normal(k1, (q_seq, query_size))
    key_and_value = random.normal(k2, (kv_seq, key_value_size))

    output = mha(query, key_and_value)

    assert output.shape == (q_seq, query_size)  # output_size defaults to query_size
    assert jnp.isfinite(output).all()

  def test_multihead_attention_forward_non_causal(self, key):
    """Test MultiheadAttention forward pass with non-causal attention."""
    num_heads = 3
    query_size = 12
    key_value_size = 9
    output_size = 15
    q_seq = 8
    kv_seq = 10

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=key_value_size,
      output_size=output_size,
      causal=False,
      key=key
    )

    k1, k2 = random.split(key, 2)
    query = random.normal(k1, (q_seq, query_size))
    key_and_value = random.normal(k2, (kv_seq, key_value_size))

    output = mha(query, key_and_value)

    assert output.shape == (q_seq, output_size)
    assert jnp.isfinite(output).all()

  def test_multihead_attention_self_attention(self, key):
    """Test MultiheadAttention with self-attention (query and key_value are the same)."""
    num_heads = 2
    query_size = 8
    seq_length = 6

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,  # Same as query for self-attention
      output_size=query_size,
      causal=False,
      key=key
    )

    # Use same sequence for query and key_value
    sequence = random.normal(key, (seq_length, query_size))

    output = mha(sequence, sequence)

    assert output.shape == (seq_length, query_size)
    assert jnp.isfinite(output).all()

  def test_multihead_attention_different_sequence_lengths(self, key):
    """Test MultiheadAttention with different query and key-value sequence lengths."""
    num_heads = 4
    query_size = 16
    key_value_size = 12
    q_seq = 5
    kv_seq = 8

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=key_value_size,
      output_size=query_size,
      causal=False,
      key=key
    )

    k1, k2 = random.split(key, 2)
    query = random.normal(k1, (q_seq, query_size))
    key_and_value = random.normal(k2, (kv_seq, key_value_size))

    output = mha(query, key_and_value)

    assert output.shape == (q_seq, query_size)
    assert jnp.isfinite(output).all()

  def test_multihead_attention_batch_size_property(self, key):
    """Test MultiheadAttention batch_size property."""
    num_heads = 2
    query_size = 8

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,
      output_size=query_size,
      causal=False,
      key=key
    )

    # Should be None for unbatched
    assert mha.batch_size is None

  def test_attention_causality(self, key):
    """Test that causal attention actually enforces causality."""
    num_heads = 1
    query_size = 4
    seq_length = 6

    mha_causal = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,
      output_size=query_size,
      causal=True,
      key=key
    )

    mha_non_causal = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,
      output_size=query_size,
      causal=False,
      key=key
    )

    # Create sequence with specific pattern to test causality
    sequence = jnp.zeros((seq_length, query_size))
    sequence = sequence.at[-1, :].set(10.0)  # Put large value at end

    output_causal = mha_causal(sequence, sequence)
    output_non_causal = mha_non_causal(sequence, sequence)

    # Both should be finite
    assert jnp.isfinite(output_causal).all()
    assert jnp.isfinite(output_non_causal).all()

    # Causal and non-causal should produce different results
    assert not jnp.allclose(output_causal, output_non_causal, atol=1e-6)

  def test_multihead_attention_reproducibility(self, key):
    """Test that MultiheadAttention produces reproducible results."""
    num_heads = 2
    query_size = 8
    seq_length = 5

    # Create two identical attention modules
    mha1 = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,
      output_size=query_size,
      causal=False,
      key=key
    )

    mha2 = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,
      output_size=query_size,
      causal=False,
      key=key
    )

    # Same input
    sequence = random.normal(key, (seq_length, query_size))

    output1 = mha1(sequence, sequence)
    output2 = mha2(sequence, sequence)

    # Should produce identical results
    assert jnp.allclose(output1, output2)

  def test_multihead_attention_head_scaling(self, key):
    """Test MultiheadAttention with different numbers of heads."""
    query_size = 12
    seq_length = 8

    # Test with different numbers of heads that divide query_size evenly
    for num_heads in [1, 2, 3, 4, 6, 12]:
      mha = MultiheadAttention(
        num_heads=num_heads,
        query_size=query_size,
        key_value_size=query_size,
        output_size=query_size,
        causal=False,
        key=key
      )

      sequence = random.normal(key, (seq_length, query_size))
      output = mha(sequence, sequence)

      assert output.shape == (seq_length, query_size)
      assert jnp.isfinite(output).all()

  def test_multihead_attention_causal_jacobian(self, key):
    """Test that causal attention produces a lower triangular Jacobian matrix."""
    num_heads = 2
    query_size = 8
    seq_length = 6

    mha = MultiheadAttention(
      num_heads=num_heads,
      query_size=query_size,
      key_value_size=query_size,
      output_size=query_size,
      causal=True,
      key=key
    )

    # Create input sequence
    sequence = random.normal(key, (seq_length, query_size))

    # Define function for Jacobian computation (self-attention case)
    def attention_func(seq_values):
      return mha(seq_values, seq_values)

    # Compute the Jacobian
    J = eqx.filter_jacfwd(attention_func)(sequence)

    # Sum over the output and input feature dimensions to get a (seq_length, seq_length) matrix
    # J shape is (seq_length, features, seq_length, features)
    J = J.sum(axis=(1, 3))

    # Check that the Jacobian is lower triangular
    assert jnp.allclose(J, jnp.tril(J)), "Causal attention should produce lower triangular Jacobian"
