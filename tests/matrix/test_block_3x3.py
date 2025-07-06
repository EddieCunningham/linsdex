import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
from linsdex.matrix.block.block_3x3 import Block3x3Matrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.tags import Tags, TAGS
from .shared import matrices_equal, matrix_tests, performance_tests
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import equinox as eqx
import jax.tree_util as jtu
import einops
from .base_for_tests import AbstractMatrixTest, autodiff_for_matrix_class, matrix_implementations_tests
# # turn on x64 precision for better numerical stability
# jax.config.update('jax_enable_x64', True)

class TestBlock3x3Matrix(unittest.TestCase, AbstractMatrixTest):
  matrix_class = Block3x3Matrix

  def setUp(self):
    # Set up common test fixtures
    self.key = random.PRNGKey(42)
    self.dim = 6  # This will be the total size (3x3 blocks of size 2x2)
    self.third_dim = self.dim // 3  # Size of each submatrix

    keys = random.split(self.key, 9)

    # Create nine submatrices for A
    A_elements = [
      random.normal(keys[i], (self.third_dim, self.third_dim))
      for i in range(9)
    ]

    # Create nine submatrices for B
    keys = random.split(random.PRNGKey(43), 9)
    B_elements = [
      random.normal(keys[i], (self.third_dim, self.third_dim))
      for i in range(9)
    ]

    # Create DenseMatrix instances
    A_submatrices = [DenseMatrix(e, tags=TAGS.no_tags) for e in A_elements]
    B_submatrices = [DenseMatrix(e, tags=TAGS.no_tags) for e in B_elements]

    # Create Block3x3Matrix instances
    self.A = Block3x3Matrix.from_blocks(*A_submatrices)
    self.B = Block3x3Matrix.from_blocks(*B_submatrices)

    # Create identity and zero matrices
    self.eye = Block3x3Matrix.eye(self.dim)

    # Create zero matrix - using our helper method
    self.zero = self.create_zeros_matrix()

    # Create symmetric matrices for testing
    A_sym_full = self.A.as_matrix()
    A_sym_full = A_sym_full @ A_sym_full.T
    self.A_sym = self.create_matrix(A_sym_full)

  # Helper to make batched matrices in the format expected by Block3x3Matrix
  def make_batched_matrices(self, a, b, c, d, e, f, g, h, i):
    """Create a batched matrix with shape (3, 3, dim, dim)"""
    # Stack into a 3x3 batch
    row1 = jnp.stack([a, b, c], axis=0)
    row2 = jnp.stack([d, e, f], axis=0)
    row3 = jnp.stack([g, h, i], axis=0)
    batched = jnp.stack([row1, row2, row3], axis=0)
    return batched

  def batched_dense_matrix(self, a, b, c, d, e, f, g, h, i):
    """Create a batched DenseMatrix with batch_size=(3,3)"""
    batched = self.make_batched_matrices(a, b, c, d, e, f, g, h, i)
    # Transform elements to a shape that can be used by DenseMatrix
    # with a batch size of (3, 3)
    return DenseMatrix(batched, tags=TAGS.no_tags)

  # Override factory methods
  def create_matrix(self, elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags

    # If elements is already a Block3x3Matrix, just update tags
    if isinstance(elements, Block3x3Matrix):
      return Block3x3Matrix(elements.matrices, tags=tags)

    # If elements is a 2D matrix, split it into 9 submatrices
    if len(elements.shape) == 2:
      assert elements.shape[0] == elements.shape[1], "Matrix must be square"
      assert elements.shape[0] % 3 == 0, "Matrix size must be divisible by 3"

      third_dim = elements.shape[0] // 3

      # Split into 9 submatrices
      A = elements[:third_dim, :third_dim]
      B = elements[:third_dim, third_dim:2*third_dim]
      C = elements[:third_dim, 2*third_dim:]
      D = elements[third_dim:2*third_dim, :third_dim]
      E = elements[third_dim:2*third_dim, third_dim:2*third_dim]
      F = elements[third_dim:2*third_dim, 2*third_dim:]
      G = elements[2*third_dim:, :third_dim]
      H = elements[2*third_dim:, third_dim:2*third_dim]
      I = elements[2*third_dim:, 2*third_dim:]

      # Create DenseMatrix instances
      A_mat = DenseMatrix(A, tags=tags)
      B_mat = DenseMatrix(B, tags=tags)
      C_mat = DenseMatrix(C, tags=tags)
      D_mat = DenseMatrix(D, tags=tags)
      E_mat = DenseMatrix(E, tags=tags)
      F_mat = DenseMatrix(F, tags=tags)
      G_mat = DenseMatrix(G, tags=tags)
      H_mat = DenseMatrix(H, tags=tags)
      I_mat = DenseMatrix(I, tags=tags)

      # Create Block3x3Matrix
      return Block3x3Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat, E_mat, F_mat, G_mat, H_mat, I_mat)

    # Handle other cases
    raise ValueError(f"Unsupported elements type: {type(elements)}")

  def create_random_matrix(self, key, shape=None, tags=None):
    if shape is None:
      shape = (self.dim, self.dim)
    if tags is None:
      tags = TAGS.no_tags

    third_dim = shape[0] // 3
    assert shape[0] == shape[1], "Matrix must be square"
    assert shape[0] % 3 == 0, "Matrix size must be divisible by 3"

    # Generate 9 random submatrices
    keys = random.split(key, 9)
    A = random.normal(keys[0], (third_dim, third_dim))
    B = random.normal(keys[1], (third_dim, third_dim))
    C = random.normal(keys[2], (third_dim, third_dim))
    D = random.normal(keys[3], (third_dim, third_dim))
    E = random.normal(keys[4], (third_dim, third_dim))
    F = random.normal(keys[5], (third_dim, third_dim))
    G = random.normal(keys[6], (third_dim, third_dim))
    H = random.normal(keys[7], (third_dim, third_dim))
    I = random.normal(keys[8], (third_dim, third_dim))

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=tags)
    B_mat = DenseMatrix(B, tags=tags)
    C_mat = DenseMatrix(C, tags=tags)
    D_mat = DenseMatrix(D, tags=tags)
    E_mat = DenseMatrix(E, tags=tags)
    F_mat = DenseMatrix(F, tags=tags)
    G_mat = DenseMatrix(G, tags=tags)
    H_mat = DenseMatrix(H, tags=tags)
    I_mat = DenseMatrix(I, tags=tags)

    # Create Block3x3Matrix
    return Block3x3Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat, E_mat, F_mat, G_mat, H_mat, I_mat)

  def create_random_symmetric_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags

    # Create a random matrix and make it symmetric
    mat = self.create_random_matrix(key, (dim, dim), tags)

    # Convert to dense, make symmetric, and convert back
    dense = mat.to_dense()
    sym_elements = dense.elements @ dense.elements.T

    return self.create_matrix(sym_elements, tags)

  def create_zeros_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.zero_tags

    third_dim = dim // 3

    # Create zero submatrices
    zero_mat = DenseMatrix(jnp.zeros((third_dim, third_dim)), tags=tags)

    # Create Block3x3Matrix with 9 zero blocks
    return Block3x3Matrix.from_blocks(
      zero_mat, zero_mat, zero_mat,
      zero_mat, zero_mat, zero_mat,
      zero_mat, zero_mat, zero_mat
    )

  def create_eye_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags

    # Use the class method to create identity matrix
    return Block3x3Matrix.eye(dim)

  def create_well_conditioned_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags

    # Create a well-conditioned matrix (symmetric positive definite)
    third_dim = dim // 3

    # Generate random keys
    keys = random.split(key, 3)

    # Create positive definite diagonal blocks
    A = random.normal(keys[0], (third_dim, third_dim))
    A = A @ A.T + third_dim * jnp.eye(third_dim)  # Make positive definite

    E = random.normal(keys[1], (third_dim, third_dim))
    E = E @ E.T + third_dim * jnp.eye(third_dim)  # Make positive definite

    I = random.normal(keys[2], (third_dim, third_dim))
    I = I @ I.T + third_dim * jnp.eye(third_dim)  # Make positive definite

    # Use zeros for off-diagonal blocks to ensure it's well-conditioned
    B = jnp.zeros((third_dim, third_dim))
    C = jnp.zeros((third_dim, third_dim))
    D = jnp.zeros((third_dim, third_dim))
    F = jnp.zeros((third_dim, third_dim))
    G = jnp.zeros((third_dim, third_dim))
    H = jnp.zeros((third_dim, third_dim))

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=tags)
    B_mat = DenseMatrix(B, tags=tags)
    C_mat = DenseMatrix(C, tags=tags)
    D_mat = DenseMatrix(D, tags=tags)
    E_mat = DenseMatrix(E, tags=tags)
    F_mat = DenseMatrix(F, tags=tags)
    G_mat = DenseMatrix(G, tags=tags)
    H_mat = DenseMatrix(H, tags=tags)
    I_mat = DenseMatrix(I, tags=tags)

    # Create Block3x3Matrix
    return Block3x3Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat, E_mat, F_mat, G_mat, H_mat, I_mat)

  def test_initialization(self):
    """Test initialization and shape properties."""
    # Test basic properties
    self.assertEqual(self.A.shape, (self.dim, self.dim))
    self.assertIsNone(self.A.batch_size)

    # Test submatrix access
    self.assertEqual(self.A.matrices.shape, (3, 3) + (self.third_dim, self.third_dim))

    # Test tags
    self.assertEqual(self.A.tags, TAGS.no_tags)
    self.assertEqual(self.zero.tags, TAGS.zero_tags)

  def test_from_blocks(self):
    """Test the from_blocks constructor."""
    # Create individual submatrices
    keys = random.split(self.key, 9)
    A = DenseMatrix(random.normal(keys[0], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    B = DenseMatrix(random.normal(keys[1], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    C = DenseMatrix(random.normal(keys[2], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    D = DenseMatrix(random.normal(keys[3], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    E = DenseMatrix(random.normal(keys[4], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    F = DenseMatrix(random.normal(keys[5], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    G = DenseMatrix(random.normal(keys[6], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    H = DenseMatrix(random.normal(keys[7], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)
    I = DenseMatrix(random.normal(keys[8], (self.third_dim, self.third_dim)), tags=TAGS.no_tags)

    # Create Block3x3Matrix
    block_mat = Block3x3Matrix.from_blocks(A, B, C, D, E, F, G, H, I)

    # Check submatrices
    self.assertTrue(matrices_equal(block_mat.matrices[0, 0].elements, A.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[0, 1].elements, B.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[0, 2].elements, C.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[1, 0].elements, D.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[1, 1].elements, E.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[1, 2].elements, F.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[2, 0].elements, G.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[2, 1].elements, H.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[2, 2].elements, I.elements))

    # Check shape
    self.assertEqual(block_mat.shape, (self.dim, self.dim))

  def test_from_diagonal(self):
    """Test the from_diagonal constructor."""
    # Create a diagonal matrix
    diag_elements = jnp.arange(1, self.dim + 1).astype(float)  # Non-zero diagonal
    diag_mat = DiagonalMatrix(diag_elements, tags=TAGS.no_tags)

    # Convert to Block3x3Matrix
    block_mat = Block3x3Matrix.from_diagonal(diag_mat)

    # Check shape
    self.assertEqual(block_mat.shape, (self.dim, self.dim))

    # Check that diagonal elements are preserved
    dense_block = block_mat.to_dense()
    self.assertTrue(jnp.allclose(jnp.diag(dense_block.elements), diag_elements))

    # Check that off-diagonal elements are zero
    diag_mask = jnp.eye(self.dim).astype(bool)
    self.assertTrue(jnp.allclose(dense_block.elements[~diag_mask], 0.0))

  def test_to_dense(self):
    """Test conversion to dense matrix."""
    # Create a Block3x3Matrix
    block_mat = self.create_random_matrix(self.key)

    # Convert to dense
    dense_mat = block_mat.to_dense()

    # Check shape
    self.assertEqual(dense_mat.shape, block_mat.shape)

    # Check that elements match the expected block structure
    dense_elements = dense_mat.elements

    # Extract blocks
    third_dim = self.dim // 3

    # Verify blocks match the original submatrices
    for i in range(3):
      for j in range(3):
        block = block_mat.matrices[i, j].as_matrix()
        dense_block = dense_elements[i*third_dim:(i+1)*third_dim, j*third_dim:(j+1)*third_dim]
        self.assertTrue(jnp.allclose(dense_block, block))

  def test_project_dense(self):
    """Test projection from dense to block matrix."""
    # Create a dense matrix
    dense_elements = random.normal(self.key, (self.dim, self.dim))
    dense_mat = DenseMatrix(dense_elements, tags=TAGS.no_tags)

    # Create a Block3x3Matrix to use for projection
    block_mat = self.create_random_matrix(self.key)

    # Project dense matrix to block structure
    projected = block_mat.project_dense(dense_mat)

    # Check that the result is a Block3x3Matrix
    self.assertIsInstance(projected, Block3x3Matrix)

    # Check that the projected matrix has the expected structure
    third_dim = self.dim // 3

    # For Block3x3Matrix, the projection should maintain the block structure
    # but use the dense matrix's values in each block
    reshaped_dense = einops.rearrange(
      dense_elements,
      '(r h) (c w) -> r c h w',
      r=3, c=3
    )

    for i in range(3):
      for j in range(3):
        self.assertTrue(
          matrices_equal(
            projected.matrices[i, j].as_matrix(),
            reshaped_dense[i, j]
          )
        )

  def test_get_inverse_block_formula(self):
    """Test that get_inverse uses the block matrix inversion formula."""
    # Create a well-conditioned block matrix
    block_mat = self.create_well_conditioned_matrix(self.key)

    # Compute inverse using Block3x3Matrix method
    block_inv = block_mat.get_inverse()

    # Compute inverse using standard matrix inversion
    dense_mat = block_mat.to_dense()
    dense_inv = jnp.linalg.inv(dense_mat.elements)

    # Check that the inverses match
    self.assertTrue(jnp.allclose(block_inv.as_matrix(), dense_inv, atol=1e-5))

  def test_get_cholesky_block_formula(self):
    """Test that get_cholesky uses the block Cholesky formula."""
    # Create a symmetric positive definite block matrix
    mat = self.create_random_symmetric_matrix(self.key)

    # Make it more diagonally dominant to ensure positive definiteness
    dense_mat = mat.to_dense()
    dense_elements = dense_mat.elements
    dense_elements = dense_elements + self.dim * jnp.eye(self.dim)
    mat = self.create_matrix(dense_elements)

    # Compute Cholesky using Block3x3Matrix method
    block_chol = mat.get_cholesky()

    # Compute Cholesky using standard method
    dense_mat = mat.to_dense()
    dense_chol = jnp.linalg.cholesky(dense_mat.elements)

    # Check that the result is lower triangular
    chol_dense = block_chol.to_dense().elements
    is_lower = jnp.allclose(jnp.triu(chol_dense, 1), 0.0, atol=1e-5)
    self.assertTrue(is_lower, "Cholesky result is not lower triangular")

    # Check that L @ L.T = original matrix
    recon = chol_dense @ chol_dense.T
    self.assertTrue(jnp.allclose(recon, dense_mat.elements, atol=1e-5))


def test_matrix_tests():
  """Test the matrix_tests function from shared.py with Block3x3Matrix."""
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)
  dim = 6
  third_dim = dim // 3

  # Function to create a Block3x3Matrix
  def create_block_matrix(key):
    keys = random.split(key, 9)

    # Create well-conditioned blocks
    blocks = []
    for i in range(9):
      if i in [0, 4, 8]:  # Diagonal blocks
        # Make diagonal blocks well-conditioned
        X = random.normal(keys[i], (third_dim, third_dim))
        X = X @ X.T + third_dim * jnp.eye(third_dim)
        blocks.append(DenseMatrix(X, tags=TAGS.no_tags))
      else:
        # Use small values for off-diagonal blocks
        X = 0.1 * random.normal(keys[i], (third_dim, third_dim))
        blocks.append(DenseMatrix(X, tags=TAGS.no_tags))

    return Block3x3Matrix.from_blocks(*blocks)

  A = create_block_matrix(k1)
  B = create_block_matrix(k2)

  # This should run without errors
  matrix_tests(key, A, B)


def test_performance():
  """Test the performance_tests function from shared.py with Block3x3Matrix."""
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)
  dim = 6
  third_dim = dim // 3

  # Function to create a Block3x3Matrix
  def create_block_matrix(key):
    keys = random.split(key, 9)

    # Create well-conditioned blocks
    blocks = []
    for i in range(9):
      if i in [0, 4, 8]:  # Diagonal blocks
        # Make diagonal blocks well-conditioned
        X = random.normal(keys[i], (third_dim, third_dim))
        X = X @ X.T + third_dim * jnp.eye(third_dim)
        blocks.append(DenseMatrix(X, tags=TAGS.no_tags))
      else:
        # Use small values for off-diagonal blocks
        X = 0.1 * random.normal(keys[i], (third_dim, third_dim))
        blocks.append(DenseMatrix(X, tags=TAGS.no_tags))

    return Block3x3Matrix.from_blocks(*blocks)

  A = create_block_matrix(k1)
  B = create_block_matrix(k2)

  # This should run without errors
  performance_tests(A, B)


def test_correctness_with_different_tags():
  """Test the correctness with different tag combinations."""
  key = random.PRNGKey(0)
  dim = 6
  third_dim = dim // 3

  # Custom function to create Block3x3Matrix
  def create_block_fn(elements, tags):
    if len(elements.shape) != 2 or elements.shape[0] != elements.shape[1]:
      raise ValueError("Elements must be a square matrix")

    if elements.shape[0] % 3 != 0:
      raise ValueError("Matrix size must be divisible by 3")

    third_dim = elements.shape[0] // 3

    # Split into 9 submatrices
    blocks = []
    for i in range(3):
      for j in range(3):
        block = elements[i*third_dim:(i+1)*third_dim, j*third_dim:(j+1)*third_dim]
        blocks.append(DenseMatrix(block, tags=tags))

    # Create Block3x3Matrix
    return Block3x3Matrix.from_blocks(*blocks)

  matrix_implementations_tests(
    key=key,
    create_matrix_fn=create_block_fn
  )


def test_autodiff():
  """Test that autodifferentiation works for Block3x3Matrix."""
  dim = 6
  third_dim = dim // 3

  # Custom function to create Block3x3Matrix
  def create_block_fn(elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags

    if len(elements.shape) != 2 or elements.shape[0] != elements.shape[1]:
      raise ValueError("Elements must be a square matrix")

    if elements.shape[0] % 3 != 0:
      raise ValueError("Matrix size must be divisible by 3")

    third_dim = elements.shape[0] // 3

    # Split into 9 submatrices
    blocks = []
    for i in range(3):
      for j in range(3):
        block = elements[i*third_dim:(i+1)*third_dim, j*third_dim:(j+1)*third_dim]
        blocks.append(DenseMatrix(block, tags=tags))

    # Create Block3x3Matrix
    return Block3x3Matrix.from_blocks(*blocks)

  autodiff_for_matrix_class(create_block_fn)


def test_diagonal_blocks():
  """Test Block3x3Matrix with DiagonalMatrix blocks."""
  key = random.PRNGKey(42)
  dim = 6
  third_dim = dim // 3

  # Create diagonal blocks
  keys = random.split(key, 9)
  diagonal_blocks = []
  for i in range(9):
    if i in [0, 4, 8]:  # Diagonal positions
      diag = random.normal(keys[i], (third_dim,))
      diagonal_blocks.append(DiagonalMatrix(diag, tags=TAGS.no_tags))
    else:  # Off-diagonal positions
      diagonal_blocks.append(DiagonalMatrix(jnp.zeros(third_dim), tags=TAGS.zero_tags))

  # Create Block3x3Matrix with diagonal blocks
  block_diag = Block3x3Matrix.from_blocks(*diagonal_blocks)

  # Test basic properties
  assert block_diag.shape == (dim, dim)

  # Test matrix-vector multiplication
  vector = jnp.ones(dim)
  result = block_diag @ vector
  assert result.shape == (dim,)

  # Create a block diagonal matrix for testing inverse
  # Make sure diagonal blocks have positive elements for stability
  keys = random.split(key, 3)
  A = DiagonalMatrix(jnp.abs(random.normal(keys[0], (third_dim,))) + 1.0, tags=TAGS.no_tags)
  E = DiagonalMatrix(jnp.abs(random.normal(keys[1], (third_dim,))) + 1.0, tags=TAGS.no_tags)
  I = DiagonalMatrix(jnp.abs(random.normal(keys[2], (third_dim,))) + 1.0, tags=TAGS.no_tags)

  zero_block = DiagonalMatrix(jnp.zeros(third_dim), tags=TAGS.zero_tags)

  blocks = [
    A, zero_block, zero_block,
    zero_block, E, zero_block,
    zero_block, zero_block, I
  ]

  true_block_diag = Block3x3Matrix.from_blocks(*blocks)

  # Test inverse
  inv = true_block_diag.get_inverse()
  assert inv.shape == (dim, dim)

  # Verify A * A^(-1) ≈ I
  identity = true_block_diag @ inv
  identity_dense = identity.to_dense().elements
  assert jnp.allclose(jnp.diag(identity_dense), jnp.ones(dim), atol=1e-5)

  # Test Cholesky decomposition
  chol = true_block_diag.get_cholesky()
  assert chol.shape == (dim, dim)

  # Verify L @ L.T = original matrix
  chol_dense = chol.to_dense().elements
  recon = chol_dense @ chol_dense.T
  assert jnp.allclose(recon, true_block_diag.to_dense().elements, atol=1e-5)


if __name__ == "__main__":
  # Enable x64 precision for better numerical stability
  jax.config.update('jax_enable_x64', True)
  unittest.main()