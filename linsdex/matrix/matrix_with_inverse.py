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
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.series.batchable_object import auto_vmap
from plum import dispatch

class MatrixWithInverse(AbstractSquareMatrix):

  matrix: AbstractSquareMatrix
  inverse_matrix: AbstractSquareMatrix

  def __init__(self, matrix: AbstractSquareMatrix, inverse_matrix: AbstractSquareMatrix):
    if matrix.shape != inverse_matrix.shape:
      raise ValueError(f"Matrix and inverse matrix must have the same shape, got {matrix.shape} and {inverse_matrix.shape}")
    self.matrix = matrix
    self.inverse_matrix = inverse_matrix

  @property
  def elements(self):
    return self.matrix.elements

  @property
  def tags(self):
    return self.matrix.tags

  def set_eye(self) -> 'AbstractSquareMatrix':
    new_mat = self.matrix.set_eye()
    new_inv = self.inverse_matrix.set_eye()
    return MatrixWithInverse(new_mat, new_inv)

  def set_symmetric(self) -> 'AbstractSquareMatrix':
    new_mat = self.matrix.set_symmetric()
    new_inv = self.inverse_matrix.set_symmetric()
    return MatrixWithInverse(new_mat, new_inv)

  def set_zero(self) -> 'AbstractSquareMatrix':
    new_mat = self.matrix.set_zero()
    new_inv = self.inverse_matrix.set_inf()
    return MatrixWithInverse(new_mat, new_inv)

  def set_inf(self) -> 'AbstractSquareMatrix':
    new_mat = self.matrix.set_inf()
    new_inv = self.inverse_matrix.set_zero()
    return MatrixWithInverse(new_mat, new_inv)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.matrix.batch_size

  @property
  def shape(self):
    return self.matrix.shape

  @classmethod
  def zeros(cls, shape: Tuple[int, ...]) -> 'MatrixWithInverse':
    matrix = AbstractSquareMatrix.zeros(shape)
    inverse_matrix = AbstractSquareMatrix.zeros(shape)
    return MatrixWithInverse(matrix, inverse_matrix)

  @classmethod
  def eye(cls, dim: int) -> 'MatrixWithInverse':
    matrix = AbstractSquareMatrix.eye(dim)
    inverse_matrix = AbstractSquareMatrix.eye(dim)
    return MatrixWithInverse(matrix, inverse_matrix)

  @auto_vmap
  def as_matrix(self):
    return self.matrix.as_matrix()

  @auto_vmap
  def __neg__(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(-self.matrix, -self.inverse_matrix)

  @auto_vmap
  def to_dense(self) -> DenseMatrix:
    return DenseMatrix(self.as_matrix(), tags=self.tags)

################################################################################################################

@dispatch
def mat_add(A: MatrixWithInverse, B: MatrixWithInverse) -> AbstractSquareMatrix:
  return mat_add(A.matrix, B.matrix)

@dispatch
def mat_add(A: MatrixWithInverse, B: AbstractSquareMatrix) -> AbstractSquareMatrix:
  return mat_add(A.matrix, B)

@dispatch
def mat_add(A: AbstractSquareMatrix, B: MatrixWithInverse) -> AbstractSquareMatrix:
  return mat_add(A, B.matrix)

################################################################################################################

@dispatch
def scalar_mul(A: MatrixWithInverse, s: Scalar) -> MatrixWithInverse:
  new_tags = A.tags.scalar_mul_update()
  return MatrixWithInverse(s*A.matrix, 1/s*A.inverse_matrix)

################################################################################################################

@dispatch
def mat_mul(A: MatrixWithInverse, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return A.matrix@b

@dispatch
def mat_mul(A: MatrixWithInverse, B: MatrixWithInverse) -> MatrixWithInverse:
  return MatrixWithInverse(mat_mul(A.matrix, B.matrix), mat_mul(B.inverse_matrix, A.inverse_matrix))

@dispatch
def mat_mul(A: MatrixWithInverse, B: AbstractSquareMatrix) -> AbstractSquareMatrix:
  return mat_mul(A.matrix, B)

@dispatch
def mat_mul(A: AbstractSquareMatrix, B: MatrixWithInverse) -> AbstractSquareMatrix:
  return mat_mul(A, B.matrix)

################################################################################################################

@dispatch
def transpose(A: MatrixWithInverse) -> MatrixWithInverse:
  return MatrixWithInverse(transpose(A.matrix), transpose(A.inverse_matrix))

################################################################################################################

@dispatch
def matrix_solve(A: MatrixWithInverse, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return mat_mul(A.inverse_matrix, b)

@dispatch
def matrix_solve(A: MatrixWithInverse, B: MatrixWithInverse) -> MatrixWithInverse:
  sol = mat_mul(A.inverse_matrix, B.matrix)
  sol_inv = mat_mul(B.inverse_matrix, A.inverse_matrix)
  return MatrixWithInverse(sol, sol_inv)

@dispatch
def matrix_solve(A: MatrixWithInverse, B: AbstractSquareMatrix) -> AbstractSquareMatrix:
  return mat_mul(A.inverse_matrix, B)

@dispatch
def matrix_solve(A: AbstractSquareMatrix, B: MatrixWithInverse) -> AbstractSquareMatrix:
  return matrix_solve(A, B.matrix)

################################################################################################################

@dispatch
def get_matrix_inverse(A: MatrixWithInverse) -> MatrixWithInverse:
  return MatrixWithInverse(A.inverse_matrix, A.matrix)

################################################################################################################

@dispatch
def get_log_det(A: MatrixWithInverse, mask: Optional[Bool[Array, 'D']] = None) -> Scalar:
  return A.matrix.get_log_det(mask=mask)

@dispatch
def get_trace(A: MatrixWithInverse) -> Scalar:
  return A.matrix.get_trace()

@dispatch
def get_cholesky(A: MatrixWithInverse) -> AbstractSquareMatrix:
  return A.matrix.get_cholesky()

@dispatch
def get_exp(A: MatrixWithInverse) -> AbstractSquareMatrix:
  return A.matrix.get_exp()

@dispatch
def get_svd(A: MatrixWithInverse) -> Tuple[AbstractSquareMatrix, 'DiagonalMatrix', AbstractSquareMatrix]:
  return A.matrix.get_svd()
