import jax.numpy as jnp
from typing import Union, Optional, Any
import equinox as eqx
from jaxtyping import Array, Float, Scalar, PyTree
from plum import dispatch
import jax.tree_util

from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap

__all__ = ['QuadraticForm']

NumericScalar = Union[Scalar, float, int]

class QuadraticForm(AbstractBatchableObject):
  """Represents an inhomogeneous quadratic form f(x) = 0.5 * x^T A x + b^T x + c.

  Note:
    The quadratic term has a factor of 0.5 for consistency with the
    exponent of a Gaussian distribution.

  Attributes:
    A: The symmetric matrix in the quadratic term.
    b: The vector in the linear part.
    c: The scalar constant.
  """
  A: AbstractSquareMatrix
  b: Float[Array, 'D']
  c: Scalar

  def __init__(self, A: AbstractSquareMatrix, b: Float[Array, 'D'], c: Scalar):
    self.A = 0.5 * (A + A.T)
    self.b = b
    self.c = c

  @property
  def batch_size(self) -> Optional[int]:
    return self.A.batch_size

  @auto_vmap
  def __call__(self, x: Float[Array, 'D']) -> Scalar:
    return 0.5 * jnp.vdot(x, self.A @ x) + jnp.vdot(self.b, x) + self.c

  @auto_vmap
  @dispatch
  def __add__(self, other: 'QuadraticForm') -> 'QuadraticForm':
    return QuadraticForm(self.A + other.A, self.b + other.b, self.c + other.c)

  @auto_vmap
  @dispatch
  def __add__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(self.A, self.b, self.c + other)

  @auto_vmap
  def __radd__(self, other: NumericScalar) -> 'QuadraticForm':
    return self + other

  @auto_vmap
  @dispatch
  def __sub__(self, other: 'QuadraticForm') -> 'QuadraticForm':
    return QuadraticForm(self.A - other.A, self.b - other.b, self.c - other.c)

  @auto_vmap
  @dispatch
  def __sub__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(self.A, self.b, self.c - other)

  @auto_vmap
  def __rsub__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(-self.A, -self.b, other - self.c)

  @auto_vmap
  def __neg__(self) -> 'QuadraticForm':
    return QuadraticForm(-self.A, -self.b, -self.c)

  @auto_vmap
  @dispatch
  def __mul__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(other * self.A, other * self.b, other * self.c)

  @auto_vmap
  def __rmul__(self, other: NumericScalar) -> 'QuadraticForm':
    return self * other

def resolve_quadratic_form(pytree: PyTree, x: Float[Array, 'D']) -> PyTree:
  """Recursively apply a vector `x` to all QuadraticForm leaves in a PyTree.

  Args:
    pytree: The PyTree to resolve.
    x: The vector to apply to the QuadraticForm leaves.

  Returns:
    The resolved PyTree.
  """
  def is_leaf(obj: Any) -> bool:
    return isinstance(obj, QuadraticForm)

  def resolve(obj: Any) -> Any:
    if isinstance(obj, QuadraticForm):
      return obj(x)
    return obj

  return jax.tree_util.tree_map(resolve, pytree, is_leaf=is_leaf)
