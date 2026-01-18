import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import lineax as lx
import abc
import warnings
import jax.tree_util as jtu
from plum import dispatch
import linsdex.util as util
from linsdex.matrix.matrix_base import AbstractSquareMatrix, mat_mul, matrix_solve
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.matrix.diagonal import DiagonalMatrix

__all__ = ['LinearFunctional', 'resolve_linear_functional']

NumericScalar = Union[Scalar, float, int]

################################################################################################################

class LinearFunctional(AbstractBatchableObject):
  """Represents a linear functional of the form f(x) = Ax + b.

  This class is used to represent vectors that are not known constants, but
  rather depend linearly on some other vector `x`. This is useful for delaying
  computation and for representing conditional distributions where the parameters
  depend on another variable.

  Attributes:
    A: The matrix in the linear functional.
    b: The offset vector in the linear functional.
  """
  A: AbstractSquareMatrix
  b: Float[Array, 'D']

  @classmethod
  def identity(cls, dim: int) -> 'LinearFunctional':
    return LinearFunctional(DiagonalMatrix.eye(dim), jnp.zeros(dim))

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.A.batch_size

  @property
  def ndim(self) -> int:
    return self.b.ndim

  @property
  def shape(self):
    """This is for compatability with code that expects a vector."""
    return self.b.shape

  @auto_vmap
  @dispatch
  def __call__(self, x: Float[Array, 'D']) -> Float[Array, 'D']:
    return self.A@x + self.b

  @auto_vmap
  @dispatch
  def __call__(self, other: "LinearFunctional") -> "LinearFunctional":
    """Compose this linear functional with another one.

    If this represents f(x) = Ax + b and other represents g(x) = Cx + d,
    this returns a new linear functional representing f(g(x)).
    f(g(x)) = A(Cx + d) + b = (AC)x + (Ad + b)
    """
    new_A = self.A @ other.A
    new_b = self.A @ other.b + self.b
    return LinearFunctional(new_A, new_b)

  @auto_vmap
  @dispatch
  def __add__(self, other: 'LinearFunctional') -> 'LinearFunctional':
    return LinearFunctional(self.A + other.A, self.b + other.b)

  @auto_vmap
  @dispatch
  def __add__(self, other: Float[Array, 'D']) -> 'LinearFunctional':
    return LinearFunctional(self.A, self.b + other)

  @auto_vmap
  def __radd__(self, other: Float[Array, 'D']) -> 'LinearFunctional':
    return self + other

  @auto_vmap
  @dispatch
  def __sub__(self, other: 'LinearFunctional') -> 'LinearFunctional':
    return LinearFunctional(self.A - other.A, self.b - other.b)

  @auto_vmap
  @dispatch
  def __sub__(self, other: Float[Array, 'D']) -> 'LinearFunctional':
    return LinearFunctional(self.A, self.b - other)

  @auto_vmap
  def __rsub__(self, other: Float[Array, 'D']) -> 'LinearFunctional':
    return LinearFunctional(-self.A, other - self.b)

  @auto_vmap
  def __neg__(self) -> 'LinearFunctional':
    return LinearFunctional(-self.A, -self.b)

  @auto_vmap
  @dispatch
  def __mul__(self, other: NumericScalar) -> 'LinearFunctional':
    return LinearFunctional(other * self.A, other * self.b)

  @auto_vmap
  def __rmul__(self, other: NumericScalar) -> 'LinearFunctional':
    return self * other

  @auto_vmap
  def __rmatmul__(self, other: AbstractSquareMatrix) -> 'LinearFunctional':
    return LinearFunctional(other @ self.A, other @ self.b)

  @auto_vmap
  def get_inverse(self) -> 'LinearFunctional':
    """Get the inverse linear functional.

    If this represents f(x) = Ax + b, then the inverse represents
    g(y) = A^{-1}y - A^{-1}b such that g(f(x)) = x.

    Returns:
      LinearFunctional: The inverse linear functional.
    """
    A_inv = self.A.get_inverse()
    b_inv = -A_inv @ self.b
    return LinearFunctional(A_inv, b_inv)


def resolve_linear_functional(pytree: PyTree, x: Float[Array, 'D']) -> PyTree:
  """Apply x to the leaves of pytree that are LinearFunctional objects.

  Args:
    pytree: The pytree to resolve.
    x: The vector to apply to the leaves of the pytree.

  Returns:
    The resolved pytree.
  """
  def is_leaf(obj: Any) -> bool:
    return isinstance(obj, LinearFunctional)

  def resolve(obj: Any) -> Any:
    if isinstance(obj, LinearFunctional):
      return obj(x)
    return obj

  return jax.tree_util.tree_map(resolve, pytree, is_leaf=is_leaf)

@auto_vmap
@dispatch
def mat_mul(A: AbstractSquareMatrix, B: LinearFunctional) -> LinearFunctional:
  return LinearFunctional(A @ B.A, A @ B.b)

@dispatch
def matrix_solve(A: AbstractSquareMatrix, B: LinearFunctional) -> LinearFunctional:
  Ab = A.solve(B.b)
  Ab = util.where(A.tags.is_inf, jnp.zeros_like(Ab), Ab)
  return LinearFunctional(A.solve(B.A), Ab)