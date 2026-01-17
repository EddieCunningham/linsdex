import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
import equinox.internal as eqxi
from abc import ABC, abstractmethod
import diffrax
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Int, PyTree, Bool
from jax._src.util import curry
import abc
import jax.tree_util as jtu
from jax._src.lax import lax
from linsdex.util.misc import where
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.quadratic_form import QuadraticForm

def _tree_concatenate(tree1: PyTree, tree2: PyTree, axis: int = 0) -> PyTree:
  """Concatenate two PyTrees along a given axis, handling structural mismatches
  where one tree has an array and the other has a functional object (LinearFunctional
  or QuadraticForm).
  """
  functional_types = (LinearFunctional, QuadraticForm)
  def is_functional(x):
    return isinstance(x, functional_types)

  def promote_to_match(target_functional, source_array):
    # Get a single-element version of target_functional (without batch dim)
    if target_functional.batch_size is not None:
      # If it's a tuple (multi-batch), we just take the first of the first...
      # but usually it's just an int.
      idx = (0,) * len(jnp.shape(target_functional.batch_size)) if isinstance(target_functional.batch_size, tuple) else 0
      single_functional = target_functional[idx]
    else:
      single_functional = target_functional

    # Zero it out
    zero_functional = single_functional.zeros_like(single_functional)

    # Broadcast to source_array's batch size
    batch_size = source_array.shape[axis]
    def broadcast(x):
      if eqx.is_array(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)
      return x
    batched_zero = jtu.tree_map(broadcast, zero_functional)

    # Set the constant part
    if isinstance(target_functional, LinearFunctional):
      return eqx.tree_at(lambda f: f.b, batched_zero, source_array)
    elif isinstance(target_functional, QuadraticForm):
      return eqx.tree_at(lambda f: f.c, batched_zero, source_array)
    return batched_zero

  # First, check structures match with custom leaves
  treedef1 = jtu.tree_structure(tree1, is_leaf=is_functional)
  treedef2 = jtu.tree_structure(tree2, is_leaf=is_functional)
  if treedef1 != treedef2:
    raise ValueError(f"Tree structures must match. Got {treedef1} and {treedef2}")

  leaves1 = jtu.tree_leaves(tree1, is_leaf=is_functional)
  leaves2 = jtu.tree_leaves(tree2, is_leaf=is_functional)

  new_leaves = []
  for l1, l2 in zip(leaves1, leaves2):
    if type(l1) == type(l2):
      if is_functional(l1):
        new_leaf = jtu.tree_map(lambda a, b: jnp.concatenate([a, b], axis=axis), l1, l2)
      else:
        new_leaf = jnp.concatenate([l1, l2], axis=axis)
    else:
      if is_functional(l1):
        l2_promoted = promote_to_match(l1, l2)
        new_leaf = jtu.tree_map(lambda a, b: jnp.concatenate([a, b], axis=axis), l1, l2_promoted)
      else:
        l1_promoted = promote_to_match(l2, l1)
        new_leaf = jtu.tree_map(lambda a, b: jnp.concatenate([a, b], axis=axis), l1_promoted, l2)
    new_leaves.append(new_leaf)

  return treedef1.unflatten(new_leaves)

def parallel_scan(operator: Callable[[eqx.Module,eqx.Module],eqx.Module],
                  elements: eqx.Module,
                  reverse: bool = False) -> eqx.Module:
  """Simplified parallel scan (using the same algorithm as jax.lax.associative_scan)
  over the first axis.  Gives same outputs as jax.lax.associative_scan and annotated
  to make the algorithm easier to understand.

  **Arguments**
  - `operator`: A function that takes two arguments and returns a single argument.
  - `elements`: A batched eqx.Module with the elements to scan over.
  - `reverse`: If True, then the scan is performed in reverse.  This matters for operations
               that are not commutative!

  **Returns**
  A batched eqx.Module with the same structure as `elements`
  """
  if reverse:
    def rev_operator(a, b):
      return operator(b, a)
    return parallel_scan(rev_operator, elements[::-1], reverse=False)[::-1]

  # We will be using vmap to apply the operator over the first axis
  vmapped_operator = jax.vmap(operator)

  # Create the recursive scan
  def _scan(elements):
    """
    elements is a batched eqx.Module with the elements

    If elements is an array
    [t_1, t_2, ..., t_n], then return
    [t_1, t_1*t_2, ..., t_1*...*t_n]
    """
    n_elements = elements.batch_size

    # Base case
    if n_elements < 2:
      return elements

    """Get the even items in the final answer cumulative products.  The terms
    that we form depend on if there is an even or odd number of total elements

                    Odd number of elements
     [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_{n-1}]

                   Even number of elements
     [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_{n}]

    This is done recursively.
    """

    # Combine adjacent pairs of elements
    #                             odd number                      even number
    odd = elements[:-1:2] # [t_1, t_3, ..., t_{n-2}]         [t_1, t_3, ..., t_{n-1}]
    even = elements[1::2] # [t_2, t_4, ..., t_{n-1}]         [t_2, t_4, ..., t_{n}  ]
    assert odd.batch_size == even.batch_size

    """Merge the odd and even elements.

       Odd number of elements:
       [t_1*t_2, t_3*t_4, ..., t_{n-2}*t_{n-1}]

       Even number of elements:
       [t_1*t_2, t_3*t_4, ..., t_{n-1}*t_n]
    """
    merged_pairs = vmapped_operator(odd, even)

    """
    Recursively perform the up-sweep. The outputs of the recursive scan
    are the new odd elements when we combine everything for the down sweep.

    Odd number of elements:
    [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_{n-1}]

    Even number of elements:
    [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_n]
    """
    even_cumulative = _scan(merged_pairs)

    """
    Now that we have all of the cumulative products for the even elements,
    get the cumulative products for the odd elements. So we want to get:

    Odd number of elements:
    [t_1*...*t_3, t_1*...*t_5, t_1*...*t_7, t_1*...*t_n]

    Even number of elements:
    [t_1*...*t_3, t_1*...*t_5, t_1*...*t_7, t_1*...*t_{n-1}]
    """
    odd_no_first = elements[2::2]
    if n_elements%2 == 0:
      odd_cumulative_except_first = vmapped_operator(even_cumulative[:-1], odd_no_first)
    else:
      odd_cumulative_except_first = vmapped_operator(even_cumulative, odd_no_first)

    # Tag on t_1 to the start of the odd cumulative products
    odd_cumulative = _tree_concatenate(elements[:1], odd_cumulative_except_first, axis=0)

    """
    Interleave the even and odd cumulative products.
    """
    # Interleave the elements.  First see how we should pad the arrays
    if odd_cumulative.batch_size == even_cumulative.batch_size:
      odd_pad = (0, 1, 1)
      even_pad = (1, 0, 1)
    elif odd_cumulative.batch_size == even_cumulative.batch_size + 1:
      odd_pad = (0, 0, 1)
      even_pad = (1, 1, 1)
    else:
      raise ValueError(f"Batch sizes must be equal or differ by one: {odd_cumulative.batch_size} and {even_cumulative.batch_size}")

    @curry
    def make_pad(pad, array):
      pad_config = [(0, 0, 0)] * array.ndim
      pad_config[0] = pad
      return jax.lax.pad(array, lax._const(array, 0), pad_config)

    """Pad the arrays to interleave them
       [t_1,      0     , t_1*...*t_3, ...,      0         , t_1*...*t_n]
       [0  , t_1*...*t_2,      0     , ..., t_1*...*t_{n-1},      0     ]
    """
    odd_padded = jtu.tree_map(make_pad(odd_pad), odd_cumulative)
    even_padded = jtu.tree_map(make_pad(even_pad), even_cumulative)

    def combine(a, b):
      # The elements might have booleans, so make sure that we use the correct operation
      op = lax.bitwise_or if a.dtype == jnp.bool_ else lax.add
      return op(a, b)

    out = jtu.tree_map(combine, odd_padded, even_padded)
    assert out.batch_size == n_elements
    return out

  return _scan(elements)

################################################################################################################

def parallel_segmented_scan(
  operator: Callable[[eqx.Module,eqx.Module],eqx.Module],
  elements: eqx.Module,
  reset_mask: Bool[Array, 'N'],
  reverse: bool = False
) -> eqx.Module:
  """Simplified segmented parallel scan over the first axis.  This function does
  a scan over elements using operator, but resets the value at the specified indices
  rather than using the accumulated value.

  **Arguments**
  - `operator`: A function that takes two arguments and returns a single argument.
  - `elements`: A batched eqx.Module with the elements to scan over.
  - `reset_mask`: A boolean array with the same shape as `elements` that indicates
    which elements to reset.
  - `reverse`: If True, then the scan is performed in reverse.  This matters for operations
               that are not commutative!

  **Returns**
  A batched eqx.Module with the same structure as `elements`
  """
  assert elements.batch_size == reset_mask.shape[-1]

  if reverse:
    def rev_operator(a, b):
      return operator(b, a)
    return parallel_segmented_scan(rev_operator, elements[::-1], reset_mask[::-1], reverse=False)[::-1]

  # We will be using vmap to apply the operator over the first axis
  vmapped_operator = jax.vmap(operator)

  # Create the recursive scan
  def _scan(elements, reset_mask):
    """
    elements is a batched eqx.Module with the elements

    If elements is an array
    [t_1, t_2, ..., t_n], then return
    [t_1, t_1*t_2, ..., t_1*...*t_n]
    """
    n_elements = elements.batch_size

    # Base case
    if n_elements < 2:
      return elements

    """Get the even items in the final answer cumulative products.  The terms
    that we form depend on if there is an even or odd number of total elements

                    Odd number of elements
     [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_{n-1}]

                   Even number of elements
     [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_{n}]

    This is done recursively.
    """

    # Combine adjacent pairs of elements
    #                             odd number                      even number
    odd = elements[:-1:2] # [t_1, t_3, ..., t_{n-2}]         [t_1, t_3, ..., t_{n-1}]
    even = elements[1::2] # [t_2, t_4, ..., t_{n-1}]         [t_2, t_4, ..., t_{n}  ]
    assert odd.batch_size == even.batch_size

    # Get the corresponding segment mask
    odd_mask = reset_mask[:-1:2]
    even_mask = reset_mask[1::2]
    assert odd_mask.shape == even_mask.shape

    """Merge the odd and even elements.

       Odd number of elements:
       [t_1*t_2, t_3*t_4, ..., t_{n-2}*t_{n-1}]

       Even number of elements:
       [t_1*t_2, t_3*t_4, ..., t_{n-1}*t_n]
    """
    merged_pairs = vmapped_operator(odd, even)

    # We need to override the even elements if the segment mask is true
    def fn(even, even_mask, merged):
      return where(even_mask, even, merged)
    merged_pairs = jax.vmap(fn)(even, even_mask, merged_pairs)

    """
    Recursively perform the up-sweep. The outputs of the recursive scan
    are the new odd elements when we combine everything for the down sweep.

    Odd number of elements:
    [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_{n-1}]

    Even number of elements:
    [t_1*t_2, t_1*...*t_4, t_1*...*t_6, t_1*...*t_n]
    """
    new_mask = odd_mask | even_mask
    even_cumulative = _scan(merged_pairs, new_mask)

    """
    Now that we have all of the cumulative products for the even elements,
    get the cumulative products for the odd elements. So we want to get:

    Odd number of elements:
    [t_1*...*t_3, t_1*...*t_5, t_1*...*t_7, t_1*...*t_n]

    Even number of elements:
    [t_1*...*t_3, t_1*...*t_5, t_1*...*t_7, t_1*...*t_{n-1}]
    """
    odd_no_first = elements[2::2]
    odd_mask_no_first = reset_mask[2::2]
    if n_elements%2 == 0:
      odd_cumulative_except_first = vmapped_operator(even_cumulative[:-1], odd_no_first)
    else:
      odd_cumulative_except_first = vmapped_operator(even_cumulative, odd_no_first)

    def fn(odd_mask, odd, merged):
      return where(odd_mask, odd, merged)
    odd_cumulative_except_first = jax.vmap(fn)(odd_mask_no_first, odd_no_first, odd_cumulative_except_first)

    # Tag on t_1 to the start of the odd cumulative products
    odd_cumulative = _tree_concatenate(elements[:1], odd_cumulative_except_first, axis=0)

    """
    Interleave the even and odd cumulative products.
    """
    # Interleave the elements.  First see how we should pad the arrays
    if odd_cumulative.batch_size == even_cumulative.batch_size:
      odd_pad = (0, 1, 1)
      even_pad = (1, 0, 1)
    elif odd_cumulative.batch_size == even_cumulative.batch_size + 1:
      odd_pad = (0, 0, 1)
      even_pad = (1, 1, 1)
    else:
      raise ValueError(f"Batch sizes must be equal or differ by one: {odd_cumulative.batch_size} and {even_cumulative.batch_size}")

    @curry
    def make_pad(pad, array):
      pad_config = [(0, 0, 0)] * array.ndim
      pad_config[0] = pad
      return jax.lax.pad(array, lax._const(array, 0), pad_config)

    """Pad the arrays to interleave them
       [t_1,      0     , t_1*...*t_3, ...,      0         , t_1*...*t_n]
       [0  , t_1*...*t_2,      0     , ..., t_1*...*t_{n-1},      0     ]
    """
    odd_padded = jtu.tree_map(make_pad(odd_pad), odd_cumulative)
    even_padded = jtu.tree_map(make_pad(even_pad), even_cumulative)

    def combine(a, b):
      # The elements might have booleans, so make sure that we use the correct operation
      op = lax.bitwise_or if a.dtype == jnp.bool_ else lax.add
      return op(a, b)

    out = jtu.tree_map(combine, odd_padded, even_padded)
    assert out.batch_size == n_elements
    return out

  return _scan(elements, reset_mask)

################################################################################################################

def segmented_scan(
  operator: Callable[[eqx.Module,eqx.Module],eqx.Module],
  elements: eqx.Module,
  reset_mask: Bool[Array, 'N'],
  reverse: bool = False
) -> eqx.Module:
  """Segmented scan over the first axis.  This function does
  a scan over elements using operator, but resets the value at the specified indices
  rather than using the accumulated value.

  **Arguments**
  - `operator`: A function that takes two arguments and returns a single argument.
  - `elements`: A batched eqx.Module with the elements to scan over.
  - `reset_mask`: A boolean array with the same shape as `elements` that indicates
    which elements to reset.
  - `reset_value`: The value to reset to.  If None, then the value is reset to the original value.

  **Returns**
  A batched eqx.Module with the same structure as `elements`
  """
  if reverse:
    def rev_operator(a, b):
      return operator(b, a)
    return segmented_scan(rev_operator, elements[::-1], reset_mask[::-1], reverse=False)[::-1]

  def body(carry, inputs):
    current_sum = carry
    value, flag = inputs
    new_sum = where(flag, value, operator(current_sum, value))
    return new_sum, new_sum

  # Promotion check for lax.scan carry
  first_op_result = operator(elements[0], elements[1])
  initial_carry = where(jnp.array(False), first_op_result, elements[0])

  _, cumsum_result = jax.lax.scan(body, initial_carry, (elements[1:], reset_mask[1:]))

  # Add the first element to the beginning of the result
  elements_0_batched = jtu.tree_map(lambda x: x[None] if eqx.is_array(x) else x, elements[0])
  result = _tree_concatenate(elements_0_batched, cumsum_result, axis=0)
  return result

################################################################################################################

if __name__ == '__main__':
  pass

