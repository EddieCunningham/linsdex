import jax
import jax.numpy as jnp
from functools import partial
from typing import TypeVar, Generic, Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Literal, Annotated
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree, Int
from functools import wraps
import abc
import jax.tree_util as jtu

__all__ = ["AbstractBatchableObject",
           "get_pytree_batch_size",
           "auto_vmap"]

def auto_vmap(f):
  """Decorator that automatically vectorizes methods of AbstractBatchableObject.

  This decorator automatically applies JAX's vmap to methods of objects
  inheriting from AbstractBatchableObject, handling batched operations without
  explicit vectorization code. It checks if the object is batched and, if so,
  applies vmap to the method call.

  Args:
    f: The method to be vectorized

  Returns:
    A wrapped function that automatically handles batched inputs

  Example:
    ```python
    class MyDistribution(AbstractBatchableObject):
      @auto_vmap
      def log_prob(self, x):
        # Code assuming unbatched self and x
        # Will automatically handle batched inputs
    ```
  """
  @wraps(f)
  def f_wrapper(self, *args, **kwargs):
    if self.batch_size:
      return eqx.filter_vmap(lambda s, a: f_wrapper(s, *a, **kwargs))(self, args)
    return f(self, *args, **kwargs)
  return f_wrapper

class AbstractBatchableObject(eqx.Module, abc.ABC):
  """Base class for objects that support batched operations.

  AbstractBatchableObject serves as the foundation for all mathematical objects
  in the DiffusionCRF library that can be batched along one or more dimensions.
  It provides a consistent interface for handling batched computations, including
  properties and methods to query batch dimensions and perform batch operations.

  This abstraction allows objects (like potentials and transitions) to be treated
  independently or as batches with the same API, enabling efficient vectorized
  operations through JAX's vmap.
  """

  @property
  @abc.abstractmethod
  def batch_size(self) -> Union[Tuple[int],int,None]:
    """Get the batch dimensions of this object.

    Returns:
      - A tuple of the leading dimensions if batched multiple times
      - An int if batched along a single dimension
      - None if not batched (i.e., a single instance)
    """
    pass

  @classmethod
  def zeros_like(cls, other: "AbstractBatchableObject") -> "AbstractBatchableObject":
    """Create a new instance with the same structure but all array values set to zero.

    Args:
      other: The template object to copy the structure from

    Returns:
      A new instance with the same structure as `other` but with zero-valued arrays
    """
    params, static = eqx.partition(other, eqx.is_array)
    zero_params = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
    return eqx.combine(zero_params, static)

  @property
  def shape(self) -> PyTree:
    """Get the shapes of all array parameters in this object.

    Returns:
      A PyTree with the same structure as this object, containing
      the shapes of all array parameters
    """
    params, static = eqx.partition(self, eqx.is_array)
    shapes = jtu.tree_map(lambda x: x.shape, params)
    return shapes

  def __getitem__(self, idx: Any):
    """Extract a slice or subset of this batched object.

    Args:
      idx: The index, slice, or mask to apply to all array parameters

    Returns:
      A new instance with all array parameters indexed by `idx`
    """
    params, static = eqx.partition(self, eqx.is_array)
    sliced_params = jtu.tree_map(lambda x: x[idx], params)
    return eqx.combine(sliced_params, static)

def get_pytree_batch_size(pytree: PyTree) -> Tuple[int]:
  """Returns the shared common prefix of all the shapes in the pytree

  Args:
    pytree: The pytree to get the batch size of

  Returns:
    A tuple of the shared common prefix of all the shapes in the pytree
  """
  # Split the pytree into its numpy arrays and static components
  params, static = eqx.partition(pytree, eqx.is_array)

  # Create a new class that will let us keep track of the shapes
  class Shape(eqx.Module):
    shape: Tuple[int]
  shape_tree = jtu.tree_map(lambda x: Shape(x.shape), params)

  # Get the leaves of the shape tree
  is_leaf = lambda x: isinstance(x, Shape)
  tree_leaves = jtu.tree_leaves(shape_tree, is_leaf=is_leaf)

  # Get the batch sizes and find their common prefix
  batch_sizes = [x.shape for x in tree_leaves]

  if len(batch_sizes) == 0:
    return ()

  # Start with the first shape
  result = batch_sizes[0]

  # Compare with each other shape
  for shape in batch_sizes[1:]:
    # Find the minimum length to compare
    min_len = min(len(result), len(shape))

    # Find where they differ
    i = 0
    while i < min_len and result[i] == shape[i]:
      i += 1

    # Truncate result to common prefix
    result = result[:i]

    # If no common prefix, return None
    if not result:
      return ()

  return result

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
