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
from linsdex.matrix.tags import Tags, TAGS
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap

__all__ = ['AbstractSquareMatrix',
           'make_parametric_symmetric_matrix']

################################################################################################################

class AbstractSquareMatrix(AbstractBatchableObject, abc.ABC):
  tags: eqx.AbstractVar[Tags]

  __fix_tags__ = False

  def replace_tags(self, tags: Tags) -> 'AbstractSquareMatrix':
    return eqx.tree_at(lambda x: x.tags, self, tags)

  @auto_vmap
  def _force_fix_tags(self) -> 'AbstractSquareMatrix':
    # Create the zero matrix
    zero = self.zeros_like(self)
    mat = util.where(self.tags.is_zero, zero, self)

    # Create the inf matrix
    inf = self.inf_like(self)
    mat = util.where(self.tags.is_inf, inf, mat)
    return mat

  def fix_to_tags(self) -> 'AbstractSquareMatrix':
    """This can help debugging but is not actually necessary because
    symbolic evaluation will automatically fix the tags."""
    if self.__fix_tags__ == False:
      return self
    return self._force_fix_tags()

  def cast_like(self, other: 'AbstractSquareMatrix') -> 'AbstractSquareMatrix':
    """Cast this matrix to be the same type as another matrix.

    This is a simple implementation that uses matrix addition with zeros.
    For more sophisticated casting options, use the `cast_matrix` function.

    Args:
        other: The matrix to cast like

    Returns:
        This matrix cast to the same type as other
    """
    return self + self.zeros_like(other)

  @classmethod
  def zeros_like(cls, other: 'AbstractSquareMatrix') -> 'AbstractSquareMatrix':
    """Sets all of the values of this matrix to zero"""
    zero = super().zeros_like(other)
    return eqx.tree_at(lambda x: x.tags, zero, TAGS.zero_tags)

  @classmethod
  def inf_like(cls, other: 'AbstractSquareMatrix') -> 'AbstractSquareMatrix':
    """Sets all of the values of this matrix to inf"""
    # The matrix will mostly look like zeros
    zero = super().zeros_like(other)

    # Set all of the values to inf
    params, static = eqx.partition(zero, eqx.is_inexact_array)
    params = jtu.tree_map(lambda x: jnp.inf*jnp.ones_like(x), params)
    inf = eqx.combine(params, static)

    # Set the tags to inf
    inf = eqx.tree_at(lambda x: x.tags, inf, TAGS.inf_tags)
    return inf

  def set_eye(self) -> 'AbstractSquareMatrix':
    out = self.eye(self.shape[0])
    return out.fix_to_tags()

  def set_symmetric(self) -> 'AbstractSquareMatrix':
    out = 0.5*(self + self.T)
    return out.fix_to_tags()

  def set_zero(self) -> 'AbstractSquareMatrix':
    out = self.zeros_like(self)
    mat = eqx.tree_at(lambda x: x.tags, out, TAGS.zero_tags)
    return mat.fix_to_tags()

  def set_inf(self) -> 'AbstractSquareMatrix':
    return self.inf_like(self)

  @property
  def is_zero(self):
    return ~self.tags.is_nonzero

  @property
  def is_inf(self):
    return self.tags.is_inf

  @property
  def is_symmetric(self):
    return self.tags.is_symmetric

  @property
  def is_eye(self):
    return self.tags.is_eye

  @property
  @abc.abstractmethod
  def shape(self):
    pass

  @property
  def ndim(self):
    return len(self.shape)

  @classmethod
  @abc.abstractmethod
  def zeros(cls, shape: Tuple[int, ...]) -> 'AbstractSquareMatrix':
    pass

  @classmethod
  @abc.abstractmethod
  def eye(cls, dim: int) -> 'AbstractSquareMatrix':
    pass

  @abc.abstractmethod
  def as_matrix(self) -> Float[Array, "M N"]:
    pass

  @abc.abstractmethod
  def __neg__(self) -> 'AbstractSquareMatrix':
    pass

  def __repr__(self):
    return f'{type(self).__name__}(\n{self.as_matrix()}\n)'

  def __add__(self, other: 'AbstractSquareMatrix') -> 'AbstractSquareMatrix':
    return mat_add(self, other)

  def __sub__(self, other: 'AbstractSquareMatrix') -> 'AbstractSquareMatrix':
    return mat_add(self, -other)

  def __mul__(self, other: Scalar) -> 'AbstractSquareMatrix':
    other = jnp.array(other)
    return scalar_mul(self, other)

  def __rmul__(self, other: Scalar) -> 'AbstractSquareMatrix':
    other = jnp.array(other)
    return scalar_mul(self, other)

  def __matmul__(self, other: Union['AbstractSquareMatrix', Float[Array, 'N']]) -> 'AbstractSquareMatrix':
    return mat_mul(self, other)

  def __truediv__(self, other: Scalar) -> 'AbstractSquareMatrix':
    other = jnp.array(other)
    return scalar_mul(self, 1/other)

  @auto_vmap
  def transpose(self):
    return transpose(self)

  @property
  def T(self):
    return self.transpose()

  @auto_vmap
  def solve(self, other: Union['AbstractSquareMatrix', Float[Array, 'N']]) -> 'AbstractSquareMatrix':
    return matrix_solve(self, other)

  @auto_vmap
  def get_inverse(self) -> 'AbstractSquareMatrix':
    return get_matrix_inverse(self)

  @auto_vmap
  def get_log_det(self, mask: Optional[Bool[Array, 'D']] = None) -> Scalar:
    return get_log_det(self, mask=mask)

  @auto_vmap
  def get_cholesky(self) -> 'AbstractSquareMatrix':
    return get_cholesky(self)

  @auto_vmap
  def get_trace(self) -> Scalar:
    return get_trace(self)

  @auto_vmap
  def get_exp(self) -> 'AbstractSquareMatrix':
    return get_exp(self)

  @auto_vmap
  def get_svd(self) -> Tuple['AbstractSquareMatrix', 'AbstractSquareMatrix', 'AbstractSquareMatrix']:
    return get_svd(self)

################################################################################################################

@dispatch.abstract
def mat_add(A: AbstractSquareMatrix, B: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Add two matrices.

  **Arguments**:

  - `A` - First matrix
  - `B` - Second matrix

  **Returns**:

  - The sum of A and B
  """
  pass

@dispatch.abstract
def scalar_mul(A: AbstractSquareMatrix, s: Scalar) -> AbstractSquareMatrix:
  """Multiply a matrix by a scalar.

  **Arguments**:

  - `A` - Matrix to be multiplied
  - `s` - Scalar multiplier

  **Returns**:

  - The product of A and s
  """
  pass

@dispatch.abstract
def mat_mul(A: AbstractSquareMatrix, B: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Multiply two matrices.

  **Arguments**:

  - `A` - First matrix
  - `B` - Second matrix

  **Returns**:

  - The matrix product of A and B
  """
  pass

@dispatch.abstract
def transpose(A: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Compute the transpose of a matrix.

  **Arguments**:

  - `A` - Matrix to be transposed

  **Returns**:

  - The transpose of A
  """
  pass

@dispatch.abstract
def matrix_solve(A: AbstractSquareMatrix, B: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Solve the matrix equation AX = B for X.

  **Arguments**:

  - `A` - Coefficient matrix
  - `B` - Right-hand side matrix

  **Returns**:

  - The solution X to AX = B
  """
  pass

@dispatch.abstract
def get_matrix_inverse(A: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Compute the inverse of a matrix.

  **Arguments**:

  - `A` - Matrix to be inverted

  **Returns**:

  - The inverse of A
  """
  pass

@dispatch.abstract
def get_log_det(A: AbstractSquareMatrix, mask: Optional[Bool[Array, 'D']] = None) -> Scalar:
  """Compute the log determinant of a matrix.

  **Arguments**:

  - `A` - Matrix to compute the log determinant for
  - `mask` - Optional mask to apply before computation

  **Returns**:

  - The log determinant of A
  """
  pass

@dispatch.abstract
def get_cholesky(A: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Compute the Cholesky decomposition of a matrix.

  **Arguments**:

  - `A` - Positive definite matrix to decompose

  **Returns**:

  - The Cholesky decomposition of A
  """
  pass

@dispatch.abstract
def get_trace(A: AbstractSquareMatrix) -> Scalar:
  """Compute the trace of a matrix.

  **Arguments**:

  - `A` - Matrix to compute the trace for

  **Returns**:

  - The trace of A
  """
  pass

@dispatch.abstract
def get_exp(A: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Compute the matrix exponential of a matrix.
  """
  pass

@dispatch.abstract
def get_svd(A: AbstractSquareMatrix) -> Tuple['AbstractSquareMatrix', 'AbstractSquareMatrix', 'AbstractSquareMatrix']:
  """Compute the SVD of a matrix.
  """
  pass

@dispatch.abstract
def make_parametric_symmetric_matrix(matrix: AbstractSquareMatrix) -> AbstractSquareMatrix:
  """Convert the symmetric matrix into a parametric form so that its
  elements are unconstrained.

  **Arguments**:

  - `matrix` - Symmetric matrix to convert

  **Returns**:

  - The parametric symmetric matrix
  """
  pass
