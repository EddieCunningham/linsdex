import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type, Annotated
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import lineax as lx
import abc
import warnings
import jax.tree_util as jtu
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.series.batchable_object import auto_vmap
from plum import dispatch
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.tags import Tags, TAGS
from linsdex.util.svd import my_svd

class Block2x2Matrix(AbstractSquareMatrix):

  tags: Tags
  matrices: Annotated[AbstractSquareMatrix, '2 2'] # Must be doubly batched!

  def __init__(
      self,
      matrices: Annotated[AbstractSquareMatrix, '2 2'],
      tags: Optional[Tags] = None
  ):
    assert isinstance(matrices, AbstractSquareMatrix)
    assert matrices.batch_size == (2, 2)
    self.matrices = matrices
    self.tags = tags if tags is not None else TAGS.no_tags

  @property
  def elements(self):
    return self.matrices.elements

  @classmethod
  def from_blocks(
    cls,
    A: Annotated[AbstractSquareMatrix, '2 2'],
    B: Annotated[AbstractSquareMatrix, '2 2'],
    C: Annotated[AbstractSquareMatrix, '2 2'],
    D: Annotated[AbstractSquareMatrix, '2 2']
  ) -> 'Block2x2Matrix':
    """The logic we will follow is that a matrix is zero if all of the matrices are zero
    and also, even though this seems incorrect, a matrix is inf if all (not any!) of the matrices are inf"""

    # Construct the matrix
    def make_block(a, b, c, d):
      row1 = jnp.concatenate([a[None], b[None]], axis=0)
      row2 = jnp.concatenate([c[None], d[None]], axis=0)
      return jnp.concatenate([row1[None], row2[None]], axis=0)
    matrices = jtu.tree_map(make_block, A, B, C, D)

    # is 0 if all of the matrices are 0
    new_is_zero = matrices.tags.is_zero.prod(dtype=bool)
    new_is_inf = matrices.tags.is_inf.prod(dtype=bool)
    # new_is_inf = matrices.tags.is_inf.sum(dtype=bool)
    new_tags = Tags(is_nonzero=~new_is_zero, is_inf=new_is_inf)
    out = Block2x2Matrix(matrices, tags=new_tags)
    return out

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.matrices.batch_size is None:
      raise ValueError('Batch size must be specified for Block2x2Matrix')
    if isinstance(self.matrices.batch_size, int):
      raise ValueError('Batch size must be specified for Block2x2Matrix')
    elif len(self.matrices.batch_size) == 2:
      return None
    elif len(self.matrices.batch_size) == 3:
      return self.matrices.batch_size[0]
    else:
      return self.matrices.batch_size[:-2]

  @property
  def shape(self):
    Ho2, Wo2 = self.matrices.shape[-2:]
    return Ho2*2, Wo2*2

  def zeros(self, shape: Tuple[int, ...]) -> 'Block2x2Matrix':
    """Not making this a classmethod because we would need to specify what types
    each of the matrices should be."""
    dim = shape[-1]
    assert shape[-2] == shape[-1] == dim
    assert dim%2 == 0
    block_shape = (dim//2, dim//2)
    def make_zero(mat):
      return mat.zeros(block_shape)
    matrices = eqx.filter_vmap(eqx.filter_vmap(make_zero))(self.matrices)
    return Block2x2Matrix(matrices, tags=TAGS.zero_tags)

  @classmethod
  def eye(cls, dim: int) -> 'Block2x2Matrix':
    return cls.from_diagonal(DiagonalMatrix.eye(dim))

  @auto_vmap
  def set_eye(self) -> 'Block2x2Matrix':
    """Override set_eye to preserve the types of submatrices.

    Unlike the base implementation which calls self.eye(), this method
    preserves the original matrix types (DenseMatrix, DiagonalMatrix, etc.)
    of the submatrices while setting their values to form an identity matrix.
    """
    dim = self.shape[0]
    assert dim % 2 == 0, "Matrix dimension must be even for Block2x2Matrix"
    half_dim = dim // 2

    # Create identity values for each block
    eye_top_left = jnp.eye(half_dim)
    eye_bottom_right = jnp.eye(half_dim)
    zeros_block = jnp.zeros((half_dim, half_dim))

    # Set each submatrix to the appropriate identity/zero values while preserving types
    def set_block_values(block, target_values):
      # Use the block's own method to create a matrix with the target values
      # This preserves the matrix type (DenseMatrix, DiagonalMatrix, etc.)
      if hasattr(block, 'project_dense'):
        # For matrices that can project from dense
        dense_target = DenseMatrix(target_values, tags=TAGS.no_tags)
        return block.project_dense(dense_target)
      else:
        # Fallback: create a new matrix of the same type
        return type(block)(target_values, tags=TAGS.no_tags)

    # Set the values for each block
    new_top_left = set_block_values(self.matrices[0, 0], eye_top_left)
    new_top_right = set_block_values(self.matrices[0, 1], zeros_block)
    new_bottom_left = set_block_values(self.matrices[1, 0], zeros_block)
    new_bottom_right = set_block_values(self.matrices[1, 1], eye_bottom_right)

    # Create the new Block2x2Matrix with identity structure
    result = self.from_blocks(new_top_left, new_top_right, new_bottom_left, new_bottom_right)

    # Update tags to reflect that this is now an identity matrix
    eye_tags = TAGS.no_tags  # Could be TAGS.eye_tags if that exists
    return eqx.tree_at(lambda x: x.tags, result, eye_tags).fix_to_tags()

  @auto_vmap
  def as_matrix(self):
    if self.tags.is_nonzero is None:
      fixed_self = self
    else:
      fixed_self = self._force_fix_tags()
    def make_matrix(mat):
      return mat.as_matrix()
    elements = eqx.filter_vmap(eqx.filter_vmap(make_matrix))(fixed_self.matrices)
    return einops.rearrange(elements, 'D1 D2 H W -> (D1 H) (D2 W)', D1=2, D2=2)

  @auto_vmap
  def __neg__(self) -> 'Block2x2Matrix':
    return Block2x2Matrix(-self.matrices, tags=self.tags)

  @auto_vmap
  def to_dense(self) -> DenseMatrix:
    return DenseMatrix(self.as_matrix(), tags=self.tags)

  @classmethod
  def from_diagonal(cls, diagonal: DiagonalMatrix) -> 'Block2x2Matrix':
    # If diagonal is batched, then vmap over the batch dimension
    if diagonal.batch_size is not None:
      return jax.vmap(cls.from_diagonal)(diagonal)
    assert diagonal.shape[-1]%2 == 0
    dim = diagonal.shape[-1]//2
    A_elts, D_elts = diagonal.elements[:dim], diagonal.elements[dim:]
    B_elts, C_elts = jnp.zeros((dim,)), jnp.zeros((dim,))

    A = DiagonalMatrix(A_elts, tags=diagonal.tags)
    D = DiagonalMatrix(D_elts, tags=diagonal.tags)
    B = DiagonalMatrix.zeros(dim)
    C = DiagonalMatrix.zeros(dim)

    def make_block(a, b, c, d):
      row1 = jnp.concatenate([a[None], b[None]], axis=0)
      row2 = jnp.concatenate([c[None], d[None]], axis=0)
      return jnp.concatenate([row1[None], row2[None]], axis=0)

    matrices = jtu.tree_map(make_block, A, B, C, D)
    out = Block2x2Matrix(matrices, tags=diagonal.tags)
    return out

  @auto_vmap
  def project_dense(self, dense: DenseMatrix) -> 'DenseMatrix':
    elements = dense.elements
    assert dense.shape[0] == dense.shape[1]
    assert dense.shape[0]%2 == 0
    dim = dense.shape[0]//2

    elements = einops.rearrange(elements, '(D1 H) (D2 W) -> D1 D2 H W', D1=2, D2=2)
    def proj(mat, elements):
      return mat.project_dense(DenseMatrix(elements, tags=TAGS.no_tags))
    matrices = jax.vmap(jax.vmap(proj))(self.matrices, elements)
    return Block2x2Matrix(matrices, tags=dense.tags)

# class ParametricSymmetric2x2BlockMatrix(Block2x2Matrix):

#   tags: Tags
#   _elements: Float[Array, '2 2 H W']

#   def __init__(self, elements: Float[Array, '2 2 H W']):
#     self._elements = elements
#     self.tags = TAGS.symmetric_tags

#   @property
#   def batch_size(self) -> Union[None,int,Tuple[int]]:
#     if self._elements.ndim > 4:
#       return self._elements.shape[:-1]
#     elif self._elements.ndim == 4:
#       return self._elements.shape[0]
#     elif self._elements.ndim == 3:
#       return None
#     else:
#       raise ValueError(f"Invalid number of dimensions: {self._elements.ndim}")

#   @property
#   @auto_vmap
#   def elements(self) -> Float[Array, '2 2 N']:
#     _elementsT = self._elements.swapaxes(-2, -3)
#     return einops.einsum(_elementsT, self._elements, 'i j a, j k a -> i k a')

# ################################################################################################################

# @dispatch
# def make_parametric_symmetric_matrix(matrix: Block2x2Matrix) -> ParametricSymmetric2x2BlockMatrix:
#   return ParametricSymmetric2x2BlockMatrix(matrix.get_cholesky().T.elements)

# @dispatch
# def mat_add(A: Block2x2Matrix, B: Union[Scalar, float]) -> Block2x2Matrix:
#   return Block2x2Matrix(A.elements + B, tags=A.tags).fix_to_tags()

@dispatch
def mat_add(A: Block2x2Matrix, B: Block2x2Matrix) -> Block2x2Matrix:
  out_matrices = A.matrices + B.matrices
  new_tags = A.tags.add_update(B.tags)
  return Block2x2Matrix(out_matrices, tags=new_tags)

@dispatch
def mat_add(A: Block2x2Matrix, B: DenseMatrix) -> DenseMatrix:
  dense_blocks = A.project_dense(B)
  return mat_add(A, dense_blocks)

@dispatch
def mat_add(A: DenseMatrix, B: Block2x2Matrix) -> DenseMatrix:
  return mat_add(B, A)

@dispatch
def mat_add(A: Block2x2Matrix, B: DiagonalMatrix) -> Block2x2Matrix:
  B_block = Block2x2Matrix.from_diagonal(B)
  return mat_add(A, B_block)

@dispatch
def mat_add(A: DiagonalMatrix, B: Block2x2Matrix) -> Block2x2Matrix:
  return mat_add(B, A)

################################################################################################################

@dispatch
def scalar_mul(A: Block2x2Matrix, s: Scalar) -> Block2x2Matrix:
  new_tags = A.tags.scalar_mul_update()
  return Block2x2Matrix(s*A.matrices, tags=new_tags).fix_to_tags()

################################################################################################################

@dispatch
def mat_mul(A: Block2x2Matrix, B: Block2x2Matrix) -> Block2x2Matrix:
  assert A.batch_size is None
  assert B.batch_size is None
  new_tags = A.tags.mat_mul_update(B.tags)

  m1 = A.matrices@B.matrices[:,0]
  m2 = A.matrices@B.matrices[:,1]

  t1 = m1[:,0] + m1[:,1]
  t2 = m2[:,0] + m2[:,1]

  C_matrices = jtu.tree_map(lambda x, y: jnp.stack([x, y], axis=1), t1, t2)
  C = Block2x2Matrix(C_matrices, tags=new_tags)
  return C

@dispatch
def mat_mul(A: Block2x2Matrix, B: DenseMatrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.as_matrix()@B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: DenseMatrix, B: Block2x2Matrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.elements@B.as_matrix(), tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: Block2x2Matrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  b_reshaped = b.reshape((2, -1))

  def matmul(x, y):
    return x@y

  c = jax.vmap(jax.vmap(matmul), in_axes=(0, None))(A.matrices, b_reshaped)
  elts = c.sum(axis=1).ravel()
  return elts

@dispatch
def mat_mul(A: Block2x2Matrix, B: DiagonalMatrix) -> Block2x2Matrix:
  B_block = Block2x2Matrix.from_diagonal(B)
  return mat_mul(A, B_block)

@dispatch
def mat_mul(A: DiagonalMatrix, B: Block2x2Matrix) -> Block2x2Matrix:
  A_block = Block2x2Matrix.from_diagonal(A)
  return mat_mul(A_block, B)

################################################################################################################

@dispatch
def transpose(mat: Block2x2Matrix) -> Block2x2Matrix:
  assert mat.batch_size is None, 'transpose expects unbatched inputs'

  matT = mat.matrices.transpose()

  def outer_transpose(x):
    return einops.rearrange(x, 'h w ... -> w h ...', h=2, w=2)

  transposed_matrices = jtu.tree_map(outer_transpose, matT)

  out_tags = mat.tags.transpose_update()
  out = Block2x2Matrix(transposed_matrices, tags=out_tags).fix_to_tags()
  return out

################################################################################################################

@dispatch
def matrix_solve(A: Block2x2Matrix,
                 b: Float[Array, 'N']) -> Float[Array, 'N']:
  Ainv = get_matrix_inverse(A)
  return mat_mul(Ainv, b)

@dispatch
def matrix_solve(A: Block2x2Matrix,
                 B: Union[Block2x2Matrix, DenseMatrix, DiagonalMatrix]) -> Union[Block2x2Matrix, DenseMatrix]:
  Ainv = get_matrix_inverse(A)
  return mat_mul(Ainv, B)

@dispatch
def matrix_solve(A: DenseMatrix, B: Block2x2Matrix) -> DenseMatrix:
  return matrix_solve(A, B.to_dense())

@dispatch
def matrix_solve(A: DiagonalMatrix, B: Block2x2Matrix) -> Block2x2Matrix:
  A_block = Block2x2Matrix.from_diagonal(A)
  return matrix_solve(A_block, B)

################################################################################################################

@dispatch
def get_matrix_inverse(A: Block2x2Matrix) -> Block2x2Matrix:
  a, b, c, d = A.matrices[0,0], A.matrices[0,1], A.matrices[1,0], A.matrices[1,1]

  top_left = (a - b@d.solve(c)).get_inverse()
  bottom_right = (d - c@a.solve(b)).get_inverse()

  top_right = -a.solve(b)@bottom_right
  bottom_left = -d.solve(c)@top_left

  A_inv = A.from_blocks(top_left, top_right, bottom_left, bottom_right)
  out_tags = A.tags.inverse_update()
  return eqx.tree_at(lambda x: x.tags, A_inv, out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_log_det(A: Block2x2Matrix, mask: Optional[Bool[Array, 'D']] = None) -> Scalar:
  if mask is not None:
    raise NotImplementedError("Masked log determinant not implemented for Block2x2Matrix")
  a, b, c, d = A.matrices[0,0], A.matrices[0,1], A.matrices[1,0], A.matrices[1,1]
  s = (d - c@a.solve(b))
  return a.get_log_det() + s.get_log_det()

################################################################################################################

@dispatch
def get_trace(A: Block2x2Matrix) -> Scalar:
  return A.matrices[0, 0].get_trace() + A.matrices[1, 1].get_trace()

################################################################################################################

@dispatch
def get_cholesky(A: Block2x2Matrix) -> Block2x2Matrix:
  a, b, bT, d = A.matrices[0,0], A.matrices[0,1], A.matrices[1,0], A.matrices[1,1]
  b = 0.5*(b + bT.T)
  s = (d - b.T@a.solve(b))

  La = a.get_cholesky()
  Ls = s.get_cholesky()
  L21 = La.solve(b).T

  zero = L21.zeros_like(L21.T)
  A_chol = A.from_blocks(La, zero, L21, Ls)
  out_tags = A.tags.cholesky_update()
  return eqx.tree_at(lambda x: x.tags, A_chol, out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_exp(A: Block2x2Matrix) -> Block2x2Matrix:
  warnings.warn('Using inefficient dense matrix exponential for Block2x2Matrix')
  A_values = jax.scipy.linalg.expm(A.as_matrix())
  A_elements_dense = einops.rearrange(A_values, '(A H) (B W) -> A B H W', A=2, B=2)

  matrix_type = type(A.matrices)
  def matrix_constructor(x):
    return matrix_type(x, tags=TAGS.no_tags)
  A_blocks = jax.vmap(jax.vmap(matrix_constructor))(A_elements_dense)
  return Block2x2Matrix(A_blocks, tags=A.tags.exp_update()).fix_to_tags()

################################################################################################################

@dispatch
def get_svd(A: Block2x2Matrix) -> Tuple[DenseMatrix, DiagonalMatrix, DenseMatrix]:
  U_elts, s_elts, V_elts = my_svd(A.as_matrix())
  U = A.project_dense(DenseMatrix(U_elts, tags=TAGS.no_tags).fix_to_tags())
  s = Block2x2Matrix.from_diagonal(DiagonalMatrix(s_elts, tags=TAGS.no_tags).fix_to_tags())
  V = A.project_dense(DenseMatrix(V_elts, tags=TAGS.no_tags).fix_to_tags())
  return U, s, V

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from debug import *
  from linsdex.matrix.matrix_base import matrix_tests

  # Turn on x64
  jax.config.update('jax_enable_x64', True)
  key = random.PRNGKey(0)

  # Dense matrix tests
  k1, k2 = random.split(key)

  half_dim = 2
  def init_matrix(key):
    return DenseMatrix(random.normal(key, (half_dim, half_dim)), tags=TAGS.no_tags)

  k1, k2 = random.split(key, 2)
  def make_block_matrix(key):
    keys = random.split(key, 4)
    matrices = jax.vmap(init_matrix)(keys)

    reshape = lambda x: einops.rearrange(x, '(h w) ... -> h w ...', h=2, w=2)
    matrices = jtu.tree_map(reshape, matrices)
    return Block2x2Matrix(matrices, tags=TAGS.no_tags)

  A = make_block_matrix(k1)
  B = make_block_matrix(k2)

  diag_elements = jnp.arange(half_dim*2)
  diagonal = DiagonalMatrix(diag_elements, tags=TAGS.symmetric_tags)
  C = Block2x2Matrix.from_diagonal(diagonal)

  D = A.project_dense(A.to_dense())

  matrix_tests(key, A, B)

  # Check that zero matrices are handled correctly
  A = Block2x2Matrix.zeros_like(A)
  matrix_tests(key, A, B)