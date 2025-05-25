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

class Block3x3Matrix(AbstractSquareMatrix):

  tags: Tags
  matrices: Annotated[AbstractSquareMatrix, '3 3'] # Must be doubly batched!

  def __init__(
      self,
      matrices: Annotated[AbstractSquareMatrix, '3 3'],
      tags: Tags
  ):
    assert isinstance(matrices, AbstractSquareMatrix)
    assert matrices.batch_size == (3, 3)
    self.matrices = matrices
    self.tags = tags

  @property
  def elements(self):
    return self.matrices.elements

  @classmethod
  def from_blocks(
    cls,
    A: Annotated[AbstractSquareMatrix, '3 3'],
    B: Annotated[AbstractSquareMatrix, '3 3'],
    C: Annotated[AbstractSquareMatrix, '3 3'],
    D: Annotated[AbstractSquareMatrix, '3 3'],
    E: Annotated[AbstractSquareMatrix, '3 3'],
    F: Annotated[AbstractSquareMatrix, '3 3'],
    G: Annotated[AbstractSquareMatrix, '3 3'],
    H: Annotated[AbstractSquareMatrix, '3 3'],
    I: Annotated[AbstractSquareMatrix, '3 3']
  ) -> 'Block3x3Matrix':
    """The logic we will follow is that a matrix is zero if all of the matrices are zero
    and also, even though this seems incorrect, a matrix is inf if all (not any!) of the matrices are inf"""

    # Construct the matrix
    def make_block(a, b, c, d, e, f, g, h, i):
      row1 = jnp.concatenate([a[None], b[None], c[None]], axis=0)
      row2 = jnp.concatenate([d[None], e[None], f[None]], axis=0)
      row3 = jnp.concatenate([g[None], h[None], i[None]], axis=0)
      return jnp.concatenate([row1[None], row2[None], row3[None]], axis=0)
    matrices = jtu.tree_map(make_block, A, B, C, D, E, F, G, H, I)

    # is 0 if all of the matrices are 0
    new_is_zero = matrices.tags.is_zero.prod(dtype=bool)
    new_is_inf = matrices.tags.is_inf.prod(dtype=bool)
    new_tags = Tags(is_nonzero=~new_is_zero, is_inf=new_is_inf)
    out = Block3x3Matrix(matrices, tags=new_tags)
    return out

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.matrices.batch_size is None:
      raise ValueError('Batch size must be specified for Block3x3Matrix')
    if isinstance(self.matrices.batch_size, int):
      raise ValueError('Batch size must be specified for Block3x3Matrix')
    elif len(self.matrices.batch_size) == 2:
      return None
    elif len(self.matrices.batch_size) == 3:
      return self.matrices.batch_size[0]
    else:
      return self.matrices.batch_size[:-2]

  @property
  def shape(self):
    Ho3, Wo3 = self.matrices.shape[-2:]
    return Ho3*3, Wo3*3

  def zeros(self, shape: Tuple[int, ...]) -> 'Block3x3Matrix':
    """Not making this a classmethod because we would need to specify what types
    each of the matrices should be."""
    dim = shape[-1]
    assert shape[-2] == shape[-1] == dim
    assert dim%3 == 0
    block_shape = (dim//3, dim//3)
    def make_zero(mat):
      return mat.zeros(block_shape)
    matrices = eqx.filter_vmap(eqx.filter_vmap(make_zero))(self.matrices)
    return Block3x3Matrix(matrices, tags=TAGS.zero_tags)

  @classmethod
  def eye(cls, dim: int) -> 'Block3x3Matrix':
    return cls.from_diagonal(DiagonalMatrix.eye(dim))

  @auto_vmap
  def set_eye(self) -> 'Block3x3Matrix':
    """Override set_eye to preserve the types of submatrices.

    Unlike the base implementation which calls self.eye(), this method
    preserves the original matrix types (DenseMatrix, DiagonalMatrix, etc.)
    of the submatrices while setting their values to form an identity matrix.
    """
    dim = self.shape[0]
    assert dim % 3 == 0, "Matrix dimension must be even for Block3x3Matrix"
    third_dim = dim // 3

    # Create identity values for diagonal blocks and zero values for off-diagonal blocks
    eye_block = jnp.eye(third_dim)
    zeros_block = jnp.zeros((third_dim, third_dim))

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

    # Set the values for each block (3x3 = 9 blocks total)
    # Diagonal blocks get identity, off-diagonal blocks get zeros
    new_00 = set_block_values(self.matrices[0, 0], eye_block)    # Top-left diagonal
    new_01 = set_block_values(self.matrices[0, 1], zeros_block)  # Top-middle off-diagonal
    new_02 = set_block_values(self.matrices[0, 2], zeros_block)  # Top-right off-diagonal
    new_10 = set_block_values(self.matrices[1, 0], zeros_block)  # Middle-left off-diagonal
    new_11 = set_block_values(self.matrices[1, 1], eye_block)    # Middle-middle diagonal
    new_12 = set_block_values(self.matrices[1, 2], zeros_block)  # Middle-right off-diagonal
    new_20 = set_block_values(self.matrices[2, 0], zeros_block)  # Bottom-left off-diagonal
    new_21 = set_block_values(self.matrices[2, 1], zeros_block)  # Bottom-middle off-diagonal
    new_22 = set_block_values(self.matrices[2, 2], eye_block)    # Bottom-right diagonal

    # Create the new Block3x3Matrix with identity structure
    result = self.from_blocks(new_00, new_01, new_02,
                            new_10, new_11, new_12,
                            new_20, new_21, new_22)

    # Update tags to reflect that this is now an identity matrix
    eye_tags = TAGS.no_tags  # Could be TAGS.eye_tags if that exists
    return eqx.tree_at(lambda x: x.tags, result, eye_tags).fix_to_tags()

  @auto_vmap
  def as_matrix(self):
    def make_matrix(mat):
      return mat.as_matrix()
    elements = eqx.filter_vmap(eqx.filter_vmap(make_matrix))(self.matrices)
    return einops.rearrange(elements, 'D1 D2 H W -> (D1 H) (D2 W)', D1=3, D2=3)

  @auto_vmap
  def __neg__(self) -> 'Block3x3Matrix':
    return Block3x3Matrix(-self.matrices, tags=self.tags)

  @auto_vmap
  def to_dense(self) -> DenseMatrix:
    return DenseMatrix(self.as_matrix(), tags=self.tags)

  @classmethod
  def from_diagonal(cls, diagonal: DiagonalMatrix) -> 'Block3x3Matrix':
    # If diagonal is batched, then vmap over the batch dimension
    if diagonal.batch_size is not None:
      return jax.vmap(cls.from_diagonal)(diagonal)
    assert diagonal.shape[-1]%3 == 0
    dim = diagonal.shape[-1]//3
    A_elts, E_elts, I_elts = diagonal.elements[:dim], diagonal.elements[dim:2*dim], diagonal.elements[2*dim:]
    B_elts, C_elts, D_elts, F_elts, G_elts, H_elts = [jnp.zeros((dim,)) for _ in range(6)]

    A = DiagonalMatrix(A_elts, tags=diagonal.tags)
    E = DiagonalMatrix(E_elts, tags=diagonal.tags)
    I = DiagonalMatrix(I_elts, tags=diagonal.tags)
    B, C, D, F, G, H = [DiagonalMatrix.zeros(dim) for _ in range(6)]

    def make_block(a, b, c, d, e, f, g, h, i):
      row1 = jnp.concatenate([a[None], b[None], c[None]], axis=0)
      row2 = jnp.concatenate([d[None], e[None], f[None]], axis=0)
      row3 = jnp.concatenate([g[None], h[None], i[None]], axis=0)
      return jnp.concatenate([row1[None], row2[None], row3[None]], axis=0)

    matrices = jtu.tree_map(make_block, A, B, C, D, E, F, G, H, I)
    out = Block3x3Matrix(matrices, tags=diagonal.tags)
    return out

  @auto_vmap
  def project_dense(self, dense: DenseMatrix) -> 'DenseMatrix':
    elements = dense.elements
    assert dense.shape[0] == dense.shape[1]
    assert dense.shape[0]%3 == 0
    dim = dense.shape[0]//3

    elements = einops.rearrange(elements, '(D1 H) (D2 W) -> D1 D2 H W', D1=3, D2=3)
    def proj(mat, elements):
      return mat.project_dense(DenseMatrix(elements, tags=TAGS.no_tags))
    matrices = jax.vmap(jax.vmap(proj))(self.matrices, elements)
    return Block3x3Matrix(matrices, tags=dense.tags)

@dispatch
def mat_add(A: Block3x3Matrix, B: Block3x3Matrix) -> Block3x3Matrix:
  out_matrices = A.matrices + B.matrices
  new_tags = A.tags.add_update(B.tags)
  return Block3x3Matrix(out_matrices, tags=new_tags)

@dispatch
def mat_add(A: Block3x3Matrix, B: DenseMatrix) -> DenseMatrix:
  dense_blocks = A.project_dense(B)
  return mat_add(A, dense_blocks)

@dispatch
def mat_add(A: DenseMatrix, B: Block3x3Matrix) -> DenseMatrix:
  return mat_add(B, A)

@dispatch
def mat_add(A: Block3x3Matrix, B: DiagonalMatrix) -> Block3x3Matrix:
  B_block = Block3x3Matrix.from_diagonal(B)
  return mat_add(A, B_block)

@dispatch
def mat_add(A: DiagonalMatrix, B: Block3x3Matrix) -> Block3x3Matrix:
  return mat_add(B, A)

################################################################################################################

@dispatch
def scalar_mul(A: Block3x3Matrix, s: Scalar) -> Block3x3Matrix:
  new_tags = A.tags.scalar_mul_update()
  return Block3x3Matrix(s*A.matrices, tags=new_tags).fix_to_tags()

################################################################################################################

@dispatch
def mat_mul(A: Block3x3Matrix, B: Block3x3Matrix) -> Block3x3Matrix:
  assert A.batch_size is None
  assert B.batch_size is None
  new_tags = A.tags.mat_mul_update(B.tags)

  m1 = A.matrices@B.matrices[:,0]
  m2 = A.matrices@B.matrices[:,1]
  m3 = A.matrices@B.matrices[:,2]

  t1 = m1[:,0] + m1[:,1] + m1[:,2]
  t2 = m2[:,0] + m2[:,1] + m2[:,2]
  t3 = m3[:,0] + m3[:,1] + m3[:,2]

  C_matrices = jtu.tree_map(lambda x, y, z: jnp.stack([x, y, z], axis=1), t1, t2, t3)
  C = Block3x3Matrix(C_matrices, tags=new_tags)
  return C

@dispatch
def mat_mul(A: Block3x3Matrix, B: DenseMatrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.as_matrix()@B.elements, tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: DenseMatrix, B: Block3x3Matrix) -> DenseMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return DenseMatrix(A.elements@B.as_matrix(), tags=new_tags).fix_to_tags()

@dispatch
def mat_mul(A: Block3x3Matrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  b_reshaped = b.reshape((3, -1))

  def matmul(x, y):
    return x@y

  c = jax.vmap(jax.vmap(matmul), in_axes=(0, None))(A.matrices, b_reshaped)
  elts = c.sum(axis=1).ravel()
  return elts

@dispatch
def mat_mul(A: Block3x3Matrix, B: DiagonalMatrix) -> Block3x3Matrix:
  B_block = Block3x3Matrix.from_diagonal(B)
  return mat_mul(A, B_block)

@dispatch
def mat_mul(A: DiagonalMatrix, B: Block3x3Matrix) -> Block3x3Matrix:
  A_block = Block3x3Matrix.from_diagonal(A)
  return mat_mul(A_block, B)

################################################################################################################

@dispatch
def transpose(mat: Block3x3Matrix) -> Block3x3Matrix:
  assert mat.batch_size is None, 'transpose expects unbatched inputs'

  matT = mat.matrices.transpose()

  def outer_transpose(x):
    return einops.rearrange(x, 'h w ... -> w h ...', h=3, w=3)

  transposed_matrices = jtu.tree_map(outer_transpose, matT)

  out_tags = mat.tags.transpose_update()
  out = Block3x3Matrix(transposed_matrices, tags=out_tags).fix_to_tags()
  return out

################################################################################################################

@dispatch
def matrix_solve(A: Block3x3Matrix,
                 b: Float[Array, 'N']) -> Float[Array, 'N']:
  Ainv = get_matrix_inverse(A)
  return mat_mul(Ainv, b)

@dispatch
def matrix_solve(A: Block3x3Matrix,
                 B: Union[Block3x3Matrix, DenseMatrix, DiagonalMatrix]) -> Union[Block3x3Matrix, DenseMatrix]:
  Ainv = get_matrix_inverse(A)
  return mat_mul(Ainv, B)

@dispatch
def matrix_solve(A: DenseMatrix, B: Block3x3Matrix) -> DenseMatrix:
  return matrix_solve(A, B.to_dense())

@dispatch
def matrix_solve(A: DiagonalMatrix, B: Block3x3Matrix) -> Block3x3Matrix:
  A_block = Block3x3Matrix.from_diagonal(A)
  return matrix_solve(A_block, B)

################################################################################################################

@dispatch
def get_matrix_inverse(A: Block3x3Matrix) -> Block3x3Matrix:
  # Extract block components
  A11, A12, A13 = A.matrices[0,0], A.matrices[0,1], A.matrices[0,2]
  A21, A22, A23 = A.matrices[1,0], A.matrices[1,1], A.matrices[1,2]
  A31, A32, A33 = A.matrices[2,0], A.matrices[2,1], A.matrices[2,2]

  # For general case, use 2x2 block partitioning: [A11 Y; Z W] where Y=[A12 A13], Z=[A21; A31], W=[A22 A23; A32 A33]

  # Step 1: Compute basic terms (reused throughout)
  A11_inv = A11.get_inverse()
  A11_inv_A12 = A11_inv @ A12
  A11_inv_A13 = A11_inv @ A13
  A21_A11_inv = A21 @ A11_inv
  A31_A11_inv = A31 @ A11_inv

  # Step 2: Compute Schur complement S = W - Z*A11^-1*Y
  S11 = A22 - A21 @ A11_inv_A12
  S12 = A23 - A21 @ A11_inv_A13
  S21 = A32 - A31 @ A11_inv_A12
  S22 = A33 - A31 @ A11_inv_A13

  # Step 3: Invert 2x2 Schur complement using nested block inversion
  S11_inv = S11.get_inverse()
  S11_inv_S12 = S11_inv @ S12
  S22_tilde = S22 - S21 @ S11_inv_S12  # Schur complement of S11 in S
  S22_tilde_inv = S22_tilde.get_inverse()

  # Compute S^-1 blocks
  temp1 = S11_inv_S12 @ S22_tilde_inv
  temp2 = S22_tilde_inv @ S21 @ S11_inv
  S_inv_11 = S11_inv + temp1 @ S21 @ S11_inv
  S_inv_12 = -temp1
  S_inv_21 = -temp2
  S_inv_22 = S22_tilde_inv

  # Step 4: Apply 2x2 block inversion formula: [A11 Y; Z W]^-1 = [A11^-1 + A11^-1*Y*S^-1*Z*A11^-1, -A11^-1*Y*S^-1; -S^-1*Z*A11^-1, S^-1]

  # Compute A11^-1*Y*S^-1 (conceptually 1x2 block)
  A11_inv_Y_S_inv_1 = A11_inv_A12 @ S_inv_11 + A11_inv_A13 @ S_inv_21
  A11_inv_Y_S_inv_2 = A11_inv_A12 @ S_inv_12 + A11_inv_A13 @ S_inv_22

  # Compute S^-1*Z*A11^-1 (conceptually 2x1 block)
  S_inv_Z_A11_inv_1 = S_inv_11 @ A21_A11_inv + S_inv_12 @ A31_A11_inv
  S_inv_Z_A11_inv_2 = S_inv_21 @ A21_A11_inv + S_inv_22 @ A31_A11_inv

  # Final inverse blocks
  B11 = A11_inv + A11_inv_Y_S_inv_1 @ A21_A11_inv + A11_inv_Y_S_inv_2 @ A31_A11_inv
  B12 = -A11_inv_Y_S_inv_1
  B13 = -A11_inv_Y_S_inv_2
  B21 = -S_inv_Z_A11_inv_1
  B22 = S_inv_11
  B23 = S_inv_12
  B31 = -S_inv_Z_A11_inv_2
  B32 = S_inv_21
  B33 = S_inv_22

  A_inv = A.from_blocks(B11, B12, B13,
                      B21, B22, B23,
                      B31, B32, B33)
  out_tags = A.tags.inverse_update()
  return eqx.tree_at(lambda x: x.tags, A_inv, out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_log_det(A: Block3x3Matrix) -> Scalar:
  # Extract block components
  A11, A12, A13 = A.matrices[0,0], A.matrices[0,1], A.matrices[0,2]
  A21, A22, A23 = A.matrices[1,0], A.matrices[1,1], A.matrices[1,2]
  A31, A32, A33 = A.matrices[2,0], A.matrices[2,1], A.matrices[2,2]

  # Use 2x2 block partitioning: [A11 Y; Z W] where Y=[A12 A13], Z=[A21; A31], W=[A22 A23; A32 A33]
  # det([A11 Y; Z W]) = det(A11) * det(W - Z * A11^-1 * Y)  [Schur complement formula]

  # Step 1: Compute det(A11)
  log_det_A11 = A11.get_log_det()

  # Step 2: Compute Schur complement S = W - Z * A11^-1 * Y
  A11_inv = A11.get_inverse()
  A11_inv_A12 = A11_inv @ A12
  A11_inv_A13 = A11_inv @ A13

  # S = [A22 A23; A32 A33] - [A21; A31] * A11^-1 * [A12 A13]
  S11 = A22 - A21 @ A11_inv_A12
  S12 = A23 - A21 @ A11_inv_A13
  S21 = A32 - A31 @ A11_inv_A12
  S22 = A33 - A31 @ A11_inv_A13

  # Step 3: Compute det(S) using 2x2 block determinant formula
  # det([S11 S12; S21 S22]) = det(S11) * det(S22 - S21 * S11^-1 * S12)
  log_det_S11 = S11.get_log_det()
  S11_inv = S11.get_inverse()
  S22_tilde = S22 - S21 @ S11_inv @ S12  # Schur complement of S11 in S
  log_det_S22_tilde = S22_tilde.get_log_det()

  log_det_S = log_det_S11 + log_det_S22_tilde

  # Step 4: Combine using logarithms
  return log_det_A11 + log_det_S

################################################################################################################

@dispatch
def get_cholesky(A: Block3x3Matrix) -> Block3x3Matrix:
  m11, m12, m13, m21, m22, m23, m31, m32, m33 = A.matrices[0,0], A.matrices[0,1], A.matrices[0,2], A.matrices[1,0], A.matrices[1,1], A.matrices[1,2], A.matrices[2,0], A.matrices[2,1], A.matrices[2,2]

  warnings.warn('Block3x3Matrix.get_cholesky is not passing autodiff tests!')

  # For a symmetric positive definite matrix, compute the Cholesky decomposition
  # Using the block Cholesky algorithm:
  # A = L L^T where L is lower triangular

  # For autodiff - symmetrize off-diagonal blocks
  # Since blocks are constructed independently, we need mutual symmetrization
  m12, m21 = 0.5*(m12 + m21.T), 0.5*(m21 + m12.T)
  m13, m31 = 0.5*(m13 + m31.T), 0.5*(m31 + m13.T)
  m23, m32 = 0.5*(m23 + m32.T), 0.5*(m32 + m23.T)

  def rev_solve(A, B):
    # BA^{-1} = ((A^{-1})^T B^T)^T
    return A.T.solve(B.T).T

  # LDL^T decomposition
  D1 = m11

  L21 = rev_solve(D1, m12.T)
  D2 = m22 - L21@D1@L21.T

  L31 = rev_solve(D1, m13.T)
  L32 = rev_solve(D2, m23.T - L31@D1@L21.T)
  D3 = m33 - L31@D1@L31.T - L32@D2@L32.T

  L11_chol = D1.get_cholesky()
  L21_chol = L21@L11_chol
  L31_chol = L31@L11_chol
  L22_chol = D2.get_cholesky()
  L32_chol = L32@L22_chol
  L33_chol = D3.get_cholesky()

  # Zeros blocks
  zero12 = L11_chol.zeros_like(m12)
  zero13 = L11_chol.zeros_like(m13)
  zero23 = L22_chol.zeros_like(m23)

  A_chol = A.from_blocks(L11_chol, zero12  , zero13,
                         L21_chol, L22_chol, zero23,
                         L31_chol, L32_chol, L33_chol)
  out_tags = A.tags.cholesky_update()
  return eqx.tree_at(lambda x: x.tags, A_chol, out_tags).fix_to_tags()

################################################################################################################

@dispatch
def get_exp(A: Block3x3Matrix) -> Block3x3Matrix:
  warnings.warn('Using inefficient dense matrix exponential for Block3x3Matrix')
  A_values = jax.scipy.linalg.expm(A.as_matrix())
  A_elements_dense = einops.rearrange(A_values, '(A H) (B W) -> A B H W', A=3, B=3)

  matrix_type = type(A.matrices)
  def matrix_constructor(x):
    return matrix_type(x, tags=TAGS.no_tags)
  A_blocks = jax.vmap(jax.vmap(matrix_constructor))(A_elements_dense)
  return Block3x3Matrix(A_blocks, tags=A.tags.exp_update()).fix_to_tags()

################################################################################################################

@dispatch
def get_svd(A: Block3x3Matrix) -> Tuple[DenseMatrix, DiagonalMatrix, DenseMatrix]:
  U_elts, s_elts, V_elts = my_svd(A.as_matrix())
  U = A.project_dense(DenseMatrix(U_elts, tags=TAGS.no_tags).fix_to_tags())
  s = Block3x3Matrix.from_diagonal(DiagonalMatrix(s_elts, tags=TAGS.no_tags).fix_to_tags())
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

  third_dim = 2
  def init_matrix(key):
    return DenseMatrix(random.normal(key, (third_dim, third_dim)), tags=TAGS.no_tags)

  k1, k2 = random.split(key, 2)
  def make_block_matrix(key):
    keys = random.split(key, 9)
    matrices = jax.vmap(init_matrix)(keys)

    reshape = lambda x: einops.rearrange(x, '(h w) ... -> h w ...', h=3, w=3)
    matrices = jtu.tree_map(reshape, matrices)
    return Block3x3Matrix(matrices, tags=TAGS.no_tags)

  A = make_block_matrix(k1)
  B = make_block_matrix(k2)

  diag_elements = jnp.arange(third_dim*3)
  diagonal = DiagonalMatrix(diag_elements, tags=TAGS.symmetric_tags)
  C = Block3x3Matrix.from_diagonal(diagonal)

  D = A.project_dense(A.to_dense())

  matrix_tests(key, A, B)

  # Check that zero matrices are handled correctly
  A = Block3x3Matrix.zeros_like(A)
  matrix_tests(key, A, B)
