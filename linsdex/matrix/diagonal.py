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
from linsdex.series.batchable_object import auto_vmap
from plum import dispatch
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.tags import Tags, TAGS
from linsdex.matrix.matrix_base import AbstractSquareMatrix

class DiagonalMatrix(AbstractSquareMatrix):

  tags: Tags
  elements: Float[Array, 'N'] # The elements of the matrix

  def __init__(
      self,
      elements: Float[Array, 'N'],
      tags: Optional[Tags] = None
  ):
    self.elements = elements
    self.tags = tags if tags is not None else TAGS.no_tags

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.elements.ndim > 2:
      return self.elements.shape[:-1]
    elif self.elements.ndim == 2:
      return self.elements.shape[0]
    elif self.elements.ndim == 1:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.elements.ndim}")

  @property
  def shape(self):
    dim = self.elements.shape[-1]
    return self.elements.shape[:-1] + (dim, dim)

  @classmethod
  def zeros(cls, dim: int) -> 'DiagonalMatrix':
    return DiagonalMatrix(jnp.zeros(dim), tags=TAGS.zero_tags)

  @classmethod
  def eye(cls, dim: int) -> 'DiagonalMatrix':
    return DiagonalMatrix(jnp.ones(dim), tags=TAGS.no_tags)

  @auto_vmap
  def as_matrix(self) -> Float[Array, "N"]:
    if self.tags.is_nonzero is None:
      fixed_self = self
    else:
      fixed_self = self._force_fix_tags()
    return jnp.diag(fixed_self.elements)

  @auto_vmap
  def __neg__(self) -> 'DiagonalMatrix':
    return DiagonalMatrix(-self.elements, tags=self.tags)

  @auto_vmap
  def to_dense(self) -> DenseMatrix:
    return DenseMatrix(jnp.diag(self.elements), tags=self.tags)

  @auto_vmap
  def project_dense(self, dense: 'DenseMatrix') -> 'DenseMatrix':
    diag_elements = jnp.diag(dense.elements)
    return DiagonalMatrix(diag_elements, tags=dense.tags)

class ParametricSymmetricDiagonalMatrix(DiagonalMatrix):

  tags: Tags
  _elements: Float[Array, 'N']

  def __init__(self, elements: Float[Array, 'N']):
    self._elements = elements
    self.tags = TAGS.no_tags

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self._elements.ndim > 2:
      return self._elements.shape[:-1]
    elif self._elements.ndim == 2:
      return self._elements.shape[0]
    elif self._elements.ndim == 1:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self._elements.ndim}")

  @property
  def elements(self) -> Float[Array, 'N']:
    return jnp.abs(self._elements) + 1e-8

@dispatch
def make_parametric_symmetric_matrix(matrix: DiagonalMatrix) -> ParametricSymmetricDiagonalMatrix:
  return ParametricSymmetricDiagonalMatrix(matrix.get_cholesky().T.elements)

################################################################################################################

@dispatch
def mat_add(A: DiagonalMatrix, B: DiagonalMatrix) -> DiagonalMatrix:
  new_tags = A.tags.add_update(B.tags)
  return DiagonalMatrix(A.elements + B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_add(A: DiagonalMatrix, B: DenseMatrix) -> DenseMatrix:
  new_tags = A.tags.add_update(B.tags)
  return DenseMatrix(A.as_matrix() + B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_add(A: DenseMatrix, B: DiagonalMatrix) -> DenseMatrix:
  return mat_add(B, A)

################################################################################################################

@dispatch
def scalar_mul(A: DiagonalMatrix, s: Scalar) -> DiagonalMatrix:
  new_tags = A.tags.scalar_mul_update()
  return DiagonalMatrix(s*A.elements, tags=new_tags).fix_to_tags()

################################################################################################################

@dispatch
def mat_mul(A: DiagonalMatrix, B: DiagonalMatrix) -> DiagonalMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DiagonalMatrix(A.elements*B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: DiagonalMatrix, B: DenseMatrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.as_matrix()@B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: DenseMatrix, B: DiagonalMatrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.elements@B.as_matrix(), tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: DiagonalMatrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return A.elements*b

################################################################################################################

@dispatch
def transpose(A: DiagonalMatrix) -> DiagonalMatrix:
  return A

################################################################################################################

@dispatch
def matrix_solve(A: DiagonalMatrix, B: DiagonalMatrix) -> DiagonalMatrix:
  A_elements = A.elements
  out_elements = B.elements/A_elements
  out_tags = A.tags.solve_update(B.tags)
  return DiagonalMatrix(out_elements, tags=out_tags).fix_to_tags()

@dispatch
def matrix_solve(A: DiagonalMatrix, B: DenseMatrix) -> DenseMatrix:
  A_elements = A.elements
  out_elements = B.elements/A_elements[...,None,:]
  out_tags = A.tags.solve_update(B.tags)
  return DenseMatrix(out_elements, tags=out_tags).fix_to_tags()

@dispatch
def matrix_solve(A: DenseMatrix, B: DiagonalMatrix) -> DenseMatrix:
  return matrix_solve(A, B.to_dense())

@dispatch
def matrix_solve(A: DiagonalMatrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  A_elements = A.elements
  return b/A_elements

################################################################################################################

@dispatch
def get_matrix_inverse(A: DiagonalMatrix) -> DiagonalMatrix:
  out_elements = 1/A.elements
  out_tags = A.tags.inverse_update()
  return DiagonalMatrix(out_elements, tags=out_tags).fix_to_tags()

@dispatch
def get_log_det(A: DiagonalMatrix, mask: Optional[Bool[Array, 'D']] = None) -> Scalar:
  elements = A.elements
  if mask is not None:
    elements = jnp.where(mask, elements, 1.0)
  return jnp.sum(jnp.log(jnp.abs(elements)))

@dispatch
def get_cholesky(A: DiagonalMatrix) -> DiagonalMatrix:
  out_elements = jnp.sqrt(A.elements)
  out_tags = A.tags.cholesky_update()
  return DiagonalMatrix(out_elements, tags=out_tags).fix_to_tags()

@dispatch
def get_exp(A: DiagonalMatrix) -> DiagonalMatrix:
  expA = jnp.exp(A.elements)
  out_tags = A.tags.exp_update()
  return DiagonalMatrix(expA, tags=out_tags).fix_to_tags()

@dispatch
def get_svd(A: DiagonalMatrix) -> Tuple[DiagonalMatrix, 'DiagonalMatrix', DiagonalMatrix]:
  A_elts = jnp.abs(A.elements)
  S = DiagonalMatrix(A_elts, tags=TAGS.no_tags).fix_to_tags()
  A_signs = A_elts/jnp.abs(A_elts)
  A_signs = jnp.where(jnp.abs(A_signs) > 1e-8, A_signs, jnp.ones_like(A_signs))
  U = A.set_eye()
  V = DiagonalMatrix(A_signs, tags=TAGS.no_tags).fix_to_tags()
  return U, S, V
