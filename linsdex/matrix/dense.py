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
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from plum import dispatch, ModuleType
from linsdex.matrix.tags import Tags, TAGS
from linsdex.util.svd import my_svd
from linsdex.series.batchable_object import auto_vmap

class DenseMatrix(AbstractSquareMatrix):

  tags: Tags
  elements: Float[Array, 'M N'] # The elements of the matrix

  def __init__(
      self,
      elements: Float[Array, 'M N'],
      tags: Tags
  ):
    self.elements = elements
    self.tags = tags

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.elements.ndim > 3:
      return self.elements.shape[:-2]
    elif self.elements.ndim == 3:
      return self.elements.shape[0]
    elif self.elements.ndim == 2:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.elements.ndim}")

  @property
  def shape(self):
    return self.elements.shape

  @classmethod
  def zeros(cls, shape: Tuple[int, ...]) -> 'DenseMatrix':
    return DenseMatrix(jnp.zeros(shape), tags=TAGS.zero_tags)

  @classmethod
  def eye(cls, dim: int) -> 'DenseMatrix':
    return DenseMatrix(jnp.eye(dim), tags=TAGS.no_tags)

  def as_matrix(self) -> Float[Array, "M N"]:
    if self.tags.is_nonzero is None:
      fixed_self = self
    else:
      fixed_self = self._force_fix_tags()
    return fixed_self.elements

  @auto_vmap
  def __neg__(self) -> 'DenseMatrix':
    return DenseMatrix(-self.elements, tags=self.tags)

  def project_dense(self, dense: 'DenseMatrix') -> 'DenseMatrix':
    return dense

class ParametricSymmetricDenseMatrix(DenseMatrix):

  tags: Tags
  _elements: Float[Array, 'N N']

  def __init__(self, elements: Float[Array, 'N N']):
    self._elements = elements
    self.tags = TAGS.no_tags

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self._elements.ndim > 3:
      return self._elements.shape[:-2]
    elif self._elements.ndim == 3:
      return self._elements.shape[0]
    elif self._elements.ndim == 2:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self._elements.ndim}")

  @property
  def elements(self) -> Float[Array, 'N N']:
    # Make the diagonal elements of self._elements positive
    diag_idx = jnp.arange(self._elements.shape[-1])
    diag_elements = self._elements[diag_idx, diag_idx]
    _elements = self._elements.at[diag_idx, diag_idx].set(jnp.abs(diag_elements) + 1e-8)

    # Also make sure that the matrix is upper triangular
    _elements = jnp.triu(_elements)

    # Return _elements.T@_elements
    return jnp.einsum('...ji,...jk->...ik', _elements, _elements)

@dispatch
def make_parametric_symmetric_matrix(matrix: DenseMatrix) -> ParametricSymmetricDenseMatrix:
  return ParametricSymmetricDenseMatrix(matrix.get_cholesky().T.elements)

################################################################################################################

@dispatch
def mat_add(A: DenseMatrix, B: DenseMatrix) -> DenseMatrix:
  new_tags = A.tags.add_update(B.tags)
  return DenseMatrix(A.elements + B.elements, tags=new_tags).fix_to_tags()

################################################################################################################

@dispatch
def scalar_mul(A: DenseMatrix, s: Scalar) -> DenseMatrix:
  new_tags = A.tags.scalar_mul_update()
  return DenseMatrix(s*A.elements, tags=new_tags).fix_to_tags()

################################################################################################################

@dispatch
def mat_mul(A: DenseMatrix, B: DenseMatrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.elements@B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: DenseMatrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return A.elements@b

################################################################################################################

@dispatch
def transpose(A: DenseMatrix) -> DenseMatrix:
  return DenseMatrix(A.elements.swapaxes(-1, -2), tags=A.tags)

################################################################################################################

@dispatch
def matrix_solve(A: DenseMatrix, B: DenseMatrix) -> DenseMatrix:
  A_elements = A.elements
  out_elements = jnp.linalg.solve(A_elements, B.elements)
  out_tags = A.tags.solve_update(B.tags)
  return DenseMatrix(out_elements, tags=out_tags).fix_to_tags()

@dispatch
def matrix_solve(A: DenseMatrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return jnp.linalg.solve(A.elements, b)

################################################################################################################

@dispatch
def get_matrix_inverse(A: DenseMatrix) -> DenseMatrix:
  out_elements = jnp.linalg.inv(A.elements)
  out_tags = A.tags.inverse_update()
  return DenseMatrix(out_elements, tags=out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_log_det(A: DenseMatrix) -> Scalar:
  return jnp.linalg.slogdet(A.elements)[1]

################################################################################################################

@dispatch
def get_cholesky(A: DenseMatrix) -> DenseMatrix:
  chol = jnp.linalg.cholesky(A.elements)
  out_tags = A.tags.cholesky_update()
  return DenseMatrix(chol, tags=out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_exp(A: DenseMatrix) -> DenseMatrix:
  expA = jax.scipy.linalg.expm(A.elements)
  out_tags = A.tags.exp_update()
  return DenseMatrix(expA, tags=out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_svd(A: DenseMatrix) -> Tuple[DenseMatrix, Any, DenseMatrix]:
  from linsdex.matrix.diagonal import DiagonalMatrix
  U_elts, s_elts, V_elts = my_svd(A.elements)
  U = DenseMatrix(U_elts, tags=TAGS.no_tags).fix_to_tags()
  s = DiagonalMatrix(s_elts, tags=TAGS.no_tags).fix_to_tags()
  V = DenseMatrix(V_elts, tags=TAGS.no_tags).fix_to_tags()
  return U, s, V

################################################################################################################
