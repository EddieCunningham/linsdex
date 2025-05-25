import jax
import jax.numpy as jnp
from functools import partial
from typing import TypeVar, Generic, Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Literal, Annotated
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree, Int
import abc
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject, get_pytree_batch_size

################################################################################################################

class TestBatchableObject(AbstractBatchableObject):
  """Concrete implementation of AbstractBatchableObject for testing."""
  data: Float[Array, "N"]
  other_data: Optional[Float[Array, "N"]] = None

  @property
  def batch_size(self) -> Union[Tuple[int], int, None]:
    if self.data.ndim == 1:
      return None
    elif self.data.ndim == 2:
      return self.data.shape[0]
    else:
      return self.data.shape[:-1]

class TestAbstractBatchableObject:
  """PyTest for AbstractBatchableObject"""

  def test_batch_size(self):
    """Test that the batch_size property returns the correct dimensions."""
    # Test scalar (no batch dimension)
    scalar_obj = TestBatchableObject(data=jnp.array([5.0]))
    assert scalar_obj.batch_size is None

    # Test single batch dimension
    single_batch = TestBatchableObject(data=jnp.ones((10, 5)))
    assert single_batch.batch_size == 10

    # Test multiple batch dimensions
    multi_batch = TestBatchableObject(data=jnp.ones((8, 6, 4)))
    assert multi_batch.batch_size == (8, 6)

  def test_zeros_like(self):
    """Test that zeros_like creates a new object with zero-valued arrays."""
    # Create test object with non-zero data
    test_obj = TestBatchableObject(
      data=jnp.ones((3, 4)),
      other_data=jnp.full((3, 2), 5.0)
    )

    # Create zeros_like object
    zeros_obj = TestBatchableObject.zeros_like(test_obj)

    # Check that the new object has the same structure but zero values
    assert zeros_obj.data.shape == test_obj.data.shape
    assert zeros_obj.other_data.shape == test_obj.other_data.shape
    assert jnp.all(zeros_obj.data == 0.0)
    assert jnp.all(zeros_obj.other_data == 0.0)

  def test_shape(self):
    """Test that the shape property returns the correct shapes."""
    # Create test object
    test_obj = TestBatchableObject(
      data=jnp.ones((5, 3)),
      other_data=jnp.ones((5, 7))
    )

    # Get shapes
    shapes = test_obj.shape

    # Check shapes
    assert shapes.data == (5, 3)
    assert shapes.other_data == (5, 7)

  def test_getitem(self):
    """Test that __getitem__ correctly slices all array parameters."""
    # Create test object
    test_obj = TestBatchableObject(
      data=jnp.arange(20).reshape((4, 5)),
      other_data=jnp.arange(12).reshape((4, 3))
    )

    # Test single index
    single_idx = test_obj[0]
    assert single_idx.data.shape == (5,)
    assert jnp.array_equal(single_idx.data, jnp.arange(5))
    assert single_idx.other_data.shape == (3,)
    assert jnp.array_equal(single_idx.other_data, jnp.arange(3))

    # Test slice
    sliced = test_obj[1:3]
    assert sliced.data.shape == (2, 5)
    assert jnp.array_equal(sliced.data, jnp.arange(5, 15).reshape((2, 5)))
    assert sliced.other_data.shape == (2, 3)
    assert jnp.array_equal(sliced.other_data, jnp.arange(3, 9).reshape((2, 3)))

    # Test boolean mask
    mask = jnp.array([True, False, True, False])
    masked = test_obj[mask]
    assert masked.data.shape == (2, 5)
    assert jnp.array_equal(masked.data, jnp.vstack([test_obj.data[0], test_obj.data[2]]))
    assert masked.other_data.shape == (2, 3)
    assert jnp.array_equal(masked.other_data, jnp.vstack([test_obj.other_data[0], test_obj.other_data[2]]))


class TestGetPytreeBatchSize:
  """PyTest for get_pytree_batch_size function."""

  def test_simple_pytree(self):
    """Test get_pytree_batch_size with a simple pytree with common batch dimensions."""
    # Create a simple pytree with common batch dimensions
    pytree = {
      "a": jnp.ones((5, 3, 2)),
      "b": jnp.ones((5, 3, 4)),
      "c": jnp.ones((5, 3, 7, 2))
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Check batch size
    assert batch_size == (5, 3)

  def test_nested_pytree(self):
    """Test get_pytree_batch_size with a nested pytree."""
    # Create a nested pytree
    pytree = {
      "a": jnp.ones((8, 4, 2)),
      "b": {
        "c": jnp.ones((8, 4, 3)),
        "d": jnp.ones((8, 4, 5, 2))
      }
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Check batch size
    assert batch_size == (8, 4)

  def test_partial_common_prefix(self):
    """Test get_pytree_batch_size with arrays having only a partial common prefix."""
    # Create a pytree with partial common prefix
    pytree = {
      "a": jnp.ones((5, 3, 2)),
      "b": jnp.ones((5, 4, 3)),  # Different second dimension
      "c": jnp.ones((5, 7, 2))   # Different second dimension
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Check batch size (only first dimension matches)
    assert batch_size == (5,)

  def test_no_common_prefix(self):
    """Test get_pytree_batch_size with arrays having no common prefix."""
    # Create a pytree with no common prefix
    pytree = {
      "a": jnp.ones((5, 3, 2)),
      "b": jnp.ones((7, 3, 2)),  # Different first dimension
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Check that batch_size is None
    assert batch_size == ()

  def test_mixed_dimensions(self):
    """Test get_pytree_batch_size with arrays of different dimensionality."""
    # Create a pytree with arrays of different dimensionality
    pytree = {
      "a": jnp.ones((5, 3, 2)),
      "b": jnp.ones((5, 3)),     # One fewer dimension
      "c": jnp.ones((5,))        # Two fewer dimensions
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Should return just the common prefix
    assert batch_size == (5,)

  def test_scalar_values(self):
    """Test get_pytree_batch_size with scalar values that have no batch dimensions."""
    # pytest tests/test_batchable_object.py::TestGetPytreeBatchSize::test_scalar_values

    # Create a pytree with scalar values
    pytree = {
      "a": jnp.array(5.0),
      "b": jnp.array(3.0),
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Should return an empty tuple
    assert batch_size == ()

  def test_custom_batchable_object(self):
    """Test get_pytree_batch_size with a pytree containing a custom batchable object."""
    # Create a pytree with a custom batchable object
    pytree = {
      "a": jnp.ones((5, 3, 2)),
      "b": TestBatchableObject(data=jnp.ones((5, 3, 4))),
    }

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Should return the common prefix
    assert batch_size == (5, 3)

  def test_empty_pytree(self):
    """Test get_pytree_batch_size with an empty pytree."""
    # Create an empty pytree
    pytree = {}

    # Get batch size
    batch_size = get_pytree_batch_size(pytree)

    # Should return None for an empty pytree
    assert batch_size == ()

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import jax
  import jax.numpy as jnp
  import jax.random as random
  import matplotlib.pyplot as plt
  import wadler_lindig as wl
  import pickle

  data = pickle.load(open("data_dump.pkl", "rb"))
  out = get_pytree_batch_size(data)
  import pdb; pdb.set_trace()
