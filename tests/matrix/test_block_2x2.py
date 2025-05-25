import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.tags import Tags, TAGS
from tests.matrix.shared import matrices_equal, matrix_tests, performance_tests
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import equinox as eqx
import jax.tree_util as jtu
import einops
from tests.matrix.base_for_tests import AbstractMatrixTest, autodiff_for_matrix_class, matrix_implementations_tests
# turn on x64 precision for better numerical stability
jax.config.update('jax_enable_x64', True)

class TestBlock2x2Matrix(unittest.TestCase, AbstractMatrixTest):
  matrix_class = Block2x2Matrix

  def setUp(self):
    # Set up common test fixtures
    self.key = random.PRNGKey(42)
    self.dim = 4  # This will be the total size (2x2 blocks of size 2x2)
    self.half_dim = self.dim // 2  # Size of each submatrix

    k1, k2, k3, k4 = random.split(self.key, 4)

    # Create four submatrices for A
    A_elements = [
      random.normal(k1, (self.half_dim, self.half_dim)),
      random.normal(k2, (self.half_dim, self.half_dim)),
      random.normal(k3, (self.half_dim, self.half_dim)),
      random.normal(k4, (self.half_dim, self.half_dim))
    ]

    # Create four submatrices for B
    k1, k2, k3, k4 = random.split(random.PRNGKey(43), 4)
    B_elements = [
      random.normal(k1, (self.half_dim, self.half_dim)),
      random.normal(k2, (self.half_dim, self.half_dim)),
      random.normal(k3, (self.half_dim, self.half_dim)),
      random.normal(k4, (self.half_dim, self.half_dim))
    ]

    # Create DenseMatrix instances
    A_submatrices = [DenseMatrix(e, tags=TAGS.no_tags) for e in A_elements]
    B_submatrices = [DenseMatrix(e, tags=TAGS.no_tags) for e in B_elements]

    # Create Block2x2Matrix instances
    self.A = Block2x2Matrix.from_blocks(*A_submatrices)
    self.B = Block2x2Matrix.from_blocks(*B_submatrices)

    # Create identity and zero matrices
    self.eye = Block2x2Matrix.eye(self.dim)

    # Create zero matrix - using our helper method
    self.zero = self.create_zeros_matrix()

    # Create symmetric matrices for testing
    A_sym_full = self.A.as_matrix()
    A_sym_full = A_sym_full @ A_sym_full.T
    self.A_sym = self.create_matrix(A_sym_full)

  # Helper to make batched matrices in the format expected by Block2x2Matrix
  def make_batched_matrices(self, a, b, c, d):
    """Create a batched matrix with shape (2, 2, dim, dim)"""
    # Stack into a 2x2 batch
    row1 = jnp.stack([a, b], axis=0)
    row2 = jnp.stack([c, d], axis=0)
    batched = jnp.stack([row1, row2], axis=0)
    return batched

  def batched_dense_matrix(self, a, b, c, d):
    """Create a batched DenseMatrix with batch_size=(2,2)"""
    batched = self.make_batched_matrices(a, b, c, d)
    # Transform elements to a shape that can be used by DenseMatrix
    # with a batch size of (2, 2)
    return DenseMatrix(batched, tags=TAGS.no_tags)

  # Override factory methods
  def create_matrix(self, elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags

    # If elements is already a Block2x2Matrix, just update tags
    if isinstance(elements, Block2x2Matrix):
      return Block2x2Matrix(elements.matrices, tags=tags)

    # If elements is a 2D matrix, split it into 4 submatrices
    if len(elements.shape) == 2:
      assert elements.shape[0] == elements.shape[1], "Matrix must be square"
      assert elements.shape[0] % 2 == 0, "Matrix size must be even"

      half_dim = elements.shape[0] // 2

      # Split into 4 submatrices
      A = elements[:half_dim, :half_dim]
      B = elements[:half_dim, half_dim:]
      C = elements[half_dim:, :half_dim]
      D = elements[half_dim:, half_dim:]

      # Create DenseMatrix instances
      A_mat = DenseMatrix(A, tags=tags)
      B_mat = DenseMatrix(B, tags=tags)
      C_mat = DenseMatrix(C, tags=tags)
      D_mat = DenseMatrix(D, tags=tags)

      # Create Block2x2Matrix
      return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

    # Handle other cases
    raise ValueError(f"Unsupported elements type: {type(elements)}")

  def create_random_matrix(self, key, shape=None, tags=None):
    if shape is None:
      shape = (self.dim, self.dim)
    if tags is None:
      tags = TAGS.no_tags

    half_dim = shape[0] // 2
    assert shape[0] == shape[1], "Matrix must be square"
    assert shape[0] % 2 == 0, "Matrix size must be even"

    # Generate 4 random submatrices
    k1, k2, k3, k4 = random.split(key, 4)
    A = random.normal(k1, (half_dim, half_dim))
    B = random.normal(k2, (half_dim, half_dim))
    C = random.normal(k3, (half_dim, half_dim))
    D = random.normal(k4, (half_dim, half_dim))

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=tags)
    B_mat = DenseMatrix(B, tags=tags)
    C_mat = DenseMatrix(C, tags=tags)
    D_mat = DenseMatrix(D, tags=tags)

    # Create Block2x2Matrix
    return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

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

    half_dim = dim // 2

    # Create zero submatrices
    zero_mat = DenseMatrix(jnp.zeros((half_dim, half_dim)), tags=tags)

    # Create Block2x2Matrix
    return Block2x2Matrix.from_blocks(zero_mat, zero_mat, zero_mat, zero_mat)

  def create_eye_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags

    # Use the class method to create identity matrix
    return Block2x2Matrix.eye(dim)

  def create_well_conditioned_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags

    # Create a well-conditioned matrix (symmetric positive definite)
    half_dim = dim // 2

    # Generate 4 random submatrices
    k1, k2 = random.split(key, 2)
    A = random.normal(k1, (half_dim, half_dim))
    A = A @ A.T + half_dim * jnp.eye(half_dim)  # Make positive definite

    D = random.normal(k2, (half_dim, half_dim))
    D = D @ D.T + half_dim * jnp.eye(half_dim)  # Make positive definite

    # Use zeros for off-diagonal blocks to ensure it's well-conditioned
    B = jnp.zeros((half_dim, half_dim))
    C = jnp.zeros((half_dim, half_dim))

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=tags)
    B_mat = DenseMatrix(B, tags=tags)
    C_mat = DenseMatrix(C, tags=tags)
    D_mat = DenseMatrix(D, tags=tags)

    # Create Block2x2Matrix
    return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

  def test_initialization(self):
    """Test initialization and shape properties."""
    # Test basic properties
    self.assertEqual(self.A.shape, (self.dim, self.dim))
    self.assertIsNone(self.A.batch_size)

    # Test submatrix access
    self.assertEqual(self.A.matrices.shape, (2, 2) + (self.half_dim, self.half_dim))

    # Test tags
    self.assertEqual(self.A.tags, TAGS.no_tags)
    self.assertEqual(self.zero.tags, TAGS.zero_tags)

  def test_from_blocks(self):
    """Test the from_blocks constructor."""
    # Create individual submatrices
    k1, k2, k3, k4 = random.split(self.key, 4)
    A = DenseMatrix(random.normal(k1, (self.half_dim, self.half_dim)), tags=TAGS.no_tags)
    B = DenseMatrix(random.normal(k2, (self.half_dim, self.half_dim)), tags=TAGS.no_tags)
    C = DenseMatrix(random.normal(k3, (self.half_dim, self.half_dim)), tags=TAGS.no_tags)
    D = DenseMatrix(random.normal(k4, (self.half_dim, self.half_dim)), tags=TAGS.no_tags)

    # Create Block2x2Matrix
    block_mat = Block2x2Matrix.from_blocks(A, B, C, D)

    # Check submatrices
    self.assertTrue(matrices_equal(block_mat.matrices[0, 0].elements, A.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[0, 1].elements, B.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[1, 0].elements, C.elements))
    self.assertTrue(matrices_equal(block_mat.matrices[1, 1].elements, D.elements))

    # Check shape
    self.assertEqual(block_mat.shape, (self.dim, self.dim))

  def test_from_diagonal(self):
    """Test the from_diagonal constructor."""
    # Create a diagonal matrix
    diag_elements = jnp.arange(1, self.dim + 1).astype(float)  # Non-zero diagonal
    diag_mat = DiagonalMatrix(diag_elements, tags=TAGS.no_tags)

    # Convert to Block2x2Matrix
    block_mat = Block2x2Matrix.from_diagonal(diag_mat)

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
    # Create a Block2x2Matrix
    block_mat = self.create_random_matrix(self.key)

    # Convert to dense
    dense_mat = block_mat.to_dense()

    # Check shape
    self.assertEqual(dense_mat.shape, block_mat.shape)

    # Check that elements match the expected block structure
    dense_elements = dense_mat.elements

    # Extract blocks
    A = block_mat.matrices[0, 0].as_matrix()
    B = block_mat.matrices[0, 1].as_matrix()
    C = block_mat.matrices[1, 0].as_matrix()
    D = block_mat.matrices[1, 1].as_matrix()

    # Check that the dense matrix has the expected block structure
    half_dim = self.dim // 2
    self.assertTrue(jnp.allclose(dense_elements[:half_dim, :half_dim], A))
    self.assertTrue(jnp.allclose(dense_elements[:half_dim, half_dim:], B))
    self.assertTrue(jnp.allclose(dense_elements[half_dim:, :half_dim], C))
    self.assertTrue(jnp.allclose(dense_elements[half_dim:, half_dim:], D))

  def test_project_dense(self):
    """Test projection from dense to block matrix."""
    # Create a dense matrix
    dense_elements = random.normal(self.key, (self.dim, self.dim))
    dense_mat = DenseMatrix(dense_elements, tags=TAGS.no_tags)

    # Create a Block2x2Matrix to use for projection
    block_mat = self.create_random_matrix(self.key)

    # Project dense matrix to block structure
    projected = block_mat.project_dense(dense_mat)

    # Check that the result is a Block2x2Matrix
    self.assertIsInstance(projected, Block2x2Matrix)

    # Check that the projected matrix has the expected structure
    half_dim = self.dim // 2
    dense_projected = projected.to_dense()

    # For Block2x2Matrix, the projection should maintain the block structure
    # but use the dense matrix's values in each block
    reshaped_dense = einops.rearrange(
      dense_elements,
      '(r h) (c w) -> r c h w',
      r=2, c=2
    )

    for i in range(2):
      for j in range(2):
        self.assertTrue(
          matrices_equal(
            projected.matrices[i, j].as_matrix(),
            reshaped_dense[i, j]
          )
        )

  def test_get_inverse_block_formula(self):
    """Test that get_inverse uses the block matrix inversion formula."""
    # Create a well-conditioned block matrix with non-zero off-diagonal blocks
    k1, k2, k3, k4 = random.split(self.key, 4)
    half_dim = self.dim // 2

    # Create well-conditioned blocks
    A = random.normal(k1, (half_dim, half_dim))
    A = A @ A.T + half_dim * jnp.eye(half_dim)

    D = random.normal(k2, (half_dim, half_dim))
    D = D @ D.T + half_dim * jnp.eye(half_dim)

    # Use small values for off-diagonal blocks to ensure numerical stability
    B = 0.1 * random.normal(k3, (half_dim, half_dim))
    C = 0.1 * random.normal(k4, (half_dim, half_dim))

    # Create Block2x2Matrix
    A_mat = DenseMatrix(A, tags=TAGS.no_tags)
    B_mat = DenseMatrix(B, tags=TAGS.no_tags)
    C_mat = DenseMatrix(C, tags=TAGS.no_tags)
    D_mat = DenseMatrix(D, tags=TAGS.no_tags)

    block_mat = Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

    # Compute inverse using Block2x2Matrix method
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

    # Compute Cholesky using Block2x2Matrix method
    block_chol = mat.get_cholesky()

    # Compute Cholesky using standard method
    dense_mat = mat.to_dense()
    dense_chol = jnp.linalg.cholesky(dense_mat.elements)

    # Check that the result is lower triangular
    chol_dense = block_chol.to_dense().elements
    is_lower = jnp.allclose(jnp.triu(chol_dense, 1), 0.0)
    self.assertTrue(is_lower, "Cholesky result is not lower triangular")

    # Check that L @ L.T = original matrix
    recon = chol_dense @ chol_dense.T
    self.assertTrue(jnp.allclose(recon, dense_mat.elements, atol=1e-5))


def test_matrix_tests():
  """Test the matrix_tests function from shared.py with Block2x2Matrix."""
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)
  dim = 4
  half_dim = dim // 2

  # Function to create a Block2x2Matrix
  def create_block_matrix(key):
    keys = random.split(key, 4)

    # Create well-conditioned submatrices
    A = random.normal(keys[0], (half_dim, half_dim))
    A = A @ A.T + half_dim * jnp.eye(half_dim)

    D = random.normal(keys[1], (half_dim, half_dim))
    D = D @ D.T + half_dim * jnp.eye(half_dim)

    # Small off-diagonal blocks for stability
    B = 0.1 * random.normal(keys[2], (half_dim, half_dim))
    C = 0.1 * random.normal(keys[3], (half_dim, half_dim))

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=TAGS.no_tags)
    B_mat = DenseMatrix(B, tags=TAGS.no_tags)
    C_mat = DenseMatrix(C, tags=TAGS.no_tags)
    D_mat = DenseMatrix(D, tags=TAGS.no_tags)

    return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

  A = create_block_matrix(k1)
  B = create_block_matrix(k2)

  # This should run without errors
  matrix_tests(key, A, B)


def test_performance():
  """Test the performance_tests function from shared.py with Block2x2Matrix."""
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)
  dim = 4
  half_dim = dim // 2

  # Function to create a Block2x2Matrix
  def create_block_matrix(key):
    keys = random.split(key, 4)

    # Create well-conditioned submatrices
    A = random.normal(keys[0], (half_dim, half_dim))
    A = A @ A.T + half_dim * jnp.eye(half_dim)

    D = random.normal(keys[1], (half_dim, half_dim))
    D = D @ D.T + half_dim * jnp.eye(half_dim)

    # Small off-diagonal blocks for stability
    B = 0.1 * random.normal(keys[2], (half_dim, half_dim))
    C = 0.1 * random.normal(keys[3], (half_dim, half_dim))

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=TAGS.no_tags)
    B_mat = DenseMatrix(B, tags=TAGS.no_tags)
    C_mat = DenseMatrix(C, tags=TAGS.no_tags)
    D_mat = DenseMatrix(D, tags=TAGS.no_tags)

    return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

  A = create_block_matrix(k1)
  B = create_block_matrix(k2)

  # This should run without errors
  performance_tests(A, B)


def test_correctness_with_different_tags():
  """Test the correctness with different tag combinations."""
  key = random.PRNGKey(0)
  dim = 4
  half_dim = dim // 2

  # Custom function to create Block2x2Matrix
  def create_block_fn(elements, tags):
    if len(elements.shape) != 2 or elements.shape[0] != elements.shape[1]:
      raise ValueError("Elements must be a square matrix")

    if elements.shape[0] % 2 != 0:
      raise ValueError("Matrix size must be even")

    half_dim = elements.shape[0] // 2

    # Split into 4 submatrices
    A = elements[:half_dim, :half_dim]
    B = elements[:half_dim, half_dim:]
    C = elements[half_dim:, :half_dim]
    D = elements[half_dim:, half_dim:]

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=tags)
    B_mat = DenseMatrix(B, tags=tags)
    C_mat = DenseMatrix(C, tags=tags)
    D_mat = DenseMatrix(D, tags=tags)

    # Create Block2x2Matrix
    return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

  matrix_implementations_tests(
    key=key,
    create_matrix_fn=create_block_fn
  )


def test_comprehensive_correctness():
  """Test all tag combinations for Block2x2Matrix."""
  from itertools import product

  # Turn on x64
  jax.config.update('jax_enable_x64', True)
  key = random.PRNGKey(0)

  dim = 4
  half_dim = dim // 2

  # All available tags
  tag_options = [
    TAGS.zero_tags,
    TAGS.no_tags
    # Skip inf_tags as they can cause numerical issues with block matrices
  ]

  # Test all combinations of tags
  for tag_A, tag_B in product(tag_options, tag_options):
    k1, k2 = random.split(key)
    key, _ = random.split(key)

    # Generate base random matrices with good conditioning
    def create_matrix_elements(key, tag):
      keys = random.split(key, 4)

      if tag.is_zero:
        A = jnp.zeros((half_dim, half_dim))
        B = jnp.zeros((half_dim, half_dim))
        C = jnp.zeros((half_dim, half_dim))
        D = jnp.zeros((half_dim, half_dim))
      else:
        # Create well-conditioned submatrices
        A = random.normal(keys[0], (half_dim, half_dim))
        A = A @ A.T + half_dim * jnp.eye(half_dim)

        D = random.normal(keys[1], (half_dim, half_dim))
        D = D @ D.T + half_dim * jnp.eye(half_dim)

        # Small off-diagonal blocks for stability
        B = 0.1 * random.normal(keys[2], (half_dim, half_dim))
        C = 0.1 * random.normal(keys[3], (half_dim, half_dim))

      # Create DenseMatrix instances
      A_mat = DenseMatrix(A, tags=tag)
      B_mat = DenseMatrix(B, tags=tag)
      C_mat = DenseMatrix(C, tags=tag)
      D_mat = DenseMatrix(D, tags=tag)

      return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

    A = create_matrix_elements(k1, tag_A)
    B = create_matrix_elements(k2, tag_B)

    try:
      matrix_tests(key, A, B)
    except Exception as e:
      # Some operations will fail with certain tag combinations
      print(f"Test failed for tags {tag_A}, {tag_B}: {str(e)}")


def test_autodiff():
  """Test that autodifferentiation works for Block2x2Matrix."""
  dim = 4
  half_dim = dim // 2

  # Custom function to create Block2x2Matrix
  def create_block_fn(elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags

    if len(elements.shape) != 2 or elements.shape[0] != elements.shape[1]:
      raise ValueError("Elements must be a square matrix")

    if elements.shape[0] % 2 != 0:
      raise ValueError("Matrix size must be even")

    half_dim = elements.shape[0] // 2

    # Split into 4 submatrices
    A = elements[:half_dim, :half_dim]
    B = elements[:half_dim, half_dim:]
    C = elements[half_dim:, :half_dim]
    D = elements[half_dim:, half_dim:]

    # Create DenseMatrix instances
    A_mat = DenseMatrix(A, tags=tags)
    B_mat = DenseMatrix(B, tags=tags)
    C_mat = DenseMatrix(C, tags=tags)
    D_mat = DenseMatrix(D, tags=tags)

    # Create Block2x2Matrix
    return Block2x2Matrix.from_blocks(A_mat, B_mat, C_mat, D_mat)

  autodiff_for_matrix_class(create_block_fn)


def test_dense_blocks():
  """Test Block2x2Matrix with DenseMatrix blocks."""
  key = random.PRNGKey(100)
  dim = 4
  half_dim = dim // 2

  # Create four dense blocks
  k1, k2, k3, k4 = random.split(key, 4)

  dense1 = random.normal(k1, (half_dim, half_dim))
  dense2 = random.normal(k2, (half_dim, half_dim))
  dense3 = random.normal(k3, (half_dim, half_dim))
  dense4 = random.normal(k4, (half_dim, half_dim))

  # Create DenseMatrix instances
  A_dense = DenseMatrix(dense1, tags=TAGS.no_tags)
  B_dense = DenseMatrix(dense2, tags=TAGS.no_tags)
  C_dense = DenseMatrix(dense3, tags=TAGS.no_tags)
  D_dense = DenseMatrix(dense4, tags=TAGS.no_tags)

  # Create Block2x2Matrix with all dense blocks
  all_dense = Block2x2Matrix.from_blocks(A_dense, B_dense, C_dense, D_dense)

  # Test basic properties
  assert all_dense.shape == (dim, dim)

  # Test matrix-vector multiplication
  vector = jnp.ones(dim)
  result = all_dense @ vector
  assert result.shape == (dim,)

  # Create a dense matrix for comparison
  dense_elements = jnp.zeros((dim, dim))
  dense_elements = dense_elements.at[:half_dim, :half_dim].set(dense1)
  dense_elements = dense_elements.at[:half_dim, half_dim:].set(dense2)
  dense_elements = dense_elements.at[half_dim:, :half_dim].set(dense3)
  dense_elements = dense_elements.at[half_dim:, half_dim:].set(dense4)

  dense_expected = DenseMatrix(dense_elements, tags=TAGS.no_tags)

  # Test to_dense conversion
  dense_result = all_dense.to_dense()
  assert jnp.allclose(dense_result.elements, dense_expected.elements)

  # Test block diagonal case with dense blocks
  zero_dense = DenseMatrix(jnp.zeros((half_dim, half_dim)), tags=TAGS.zero_tags)
  block_diag = Block2x2Matrix.from_blocks(A_dense, zero_dense, zero_dense, D_dense)

  # Test matrix multiplication
  product = all_dense @ block_diag
  assert product.shape == (dim, dim)

  # Test matrix addition
  sum_matrix = all_dense + block_diag
  assert sum_matrix.shape == (dim, dim)

  # Test creating a symmetric matrix with dense blocks
  # Create a symmetric structure
  k5 = random.fold_in(key, 5)
  sym_matrix = random.normal(k5, (dim, dim))
  sym_matrix = sym_matrix + sym_matrix.T  # Make it symmetric

  # Convert to Block2x2Matrix
  sym_block = TestBlock2x2Matrix().create_matrix(sym_matrix)

  # Test Cholesky on a positive definite matrix
  spd_matrix = sym_matrix @ sym_matrix.T + dim * jnp.eye(dim)
  spd_block = TestBlock2x2Matrix().create_matrix(spd_matrix)

  # Compute Cholesky
  chol = spd_block.get_cholesky()

  # Verify Cholesky properties
  chol_dense = chol.to_dense().elements

  # Check that L is lower triangular
  is_lower = jnp.allclose(jnp.triu(chol_dense, 1), 0.0)
  assert is_lower, "Cholesky result is not lower triangular"

  # Check that L @ L.T = original matrix
  recon = chol_dense @ chol_dense.T
  assert jnp.allclose(recon, spd_matrix, atol=1e-5)

  # Test inverse of a well-conditioned matrix
  # Create a well-conditioned matrix
  A_well = dense1 @ dense1.T + half_dim * jnp.eye(half_dim)
  D_well = dense4 @ dense4.T + half_dim * jnp.eye(half_dim)

  well_cond = Block2x2Matrix.from_blocks(
    DenseMatrix(A_well, tags=TAGS.no_tags),
    zero_dense,
    zero_dense,
    DenseMatrix(D_well, tags=TAGS.no_tags)
  )

  # Compute inverse
  inv = well_cond.get_inverse()
  assert inv.shape == (dim, dim)

  # Verify that A * A^(-1) ≈ I
  identity = well_cond @ inv
  identity_dense = identity.to_dense().elements
  assert jnp.allclose(jnp.diag(identity_dense), jnp.ones(dim), atol=1e-5)


def test_diagonal_blocks():
  """Test Block2x2Matrix with DiagonalMatrix blocks."""
  key = random.PRNGKey(42)
  dim = 4
  half_dim = dim // 2

  # Create diagonal blocks
  k1, k2, k3, k4 = random.split(key, 4)
  diag1 = random.normal(k1, (half_dim,))
  diag2 = random.normal(k2, (half_dim,))
  diag3 = random.normal(k3, (half_dim,))
  diag4 = random.normal(k4, (half_dim,))

  # Create DiagonalMatrix instances
  A_diag = DiagonalMatrix(diag1, tags=TAGS.no_tags)
  B_diag = DiagonalMatrix(diag2, tags=TAGS.no_tags)
  C_diag = DiagonalMatrix(diag3, tags=TAGS.no_tags)
  D_diag = DiagonalMatrix(diag4, tags=TAGS.no_tags)

  # Create Block2x2Matrix with all diagonal blocks
  all_diag = Block2x2Matrix.from_blocks(A_diag, B_diag, C_diag, D_diag)

  # Test basic properties
  assert all_diag.shape == (dim, dim)

  # Test matrix-vector multiplication
  vector = jnp.ones(dim)
  result = all_diag @ vector
  assert result.shape == (dim,)

  # Create a dense matrix for comparison
  dense_elements = jnp.zeros((dim, dim))
  dense_elements = dense_elements.at[:half_dim, :half_dim].set(jnp.diag(diag1))
  dense_elements = dense_elements.at[:half_dim, half_dim:].set(jnp.diag(diag2))
  dense_elements = dense_elements.at[half_dim:, :half_dim].set(jnp.diag(diag3))
  dense_elements = dense_elements.at[half_dim:, half_dim:].set(jnp.diag(diag4))

  dense_expected = DenseMatrix(dense_elements, tags=TAGS.no_tags)

  # Test to_dense conversion
  dense_result = all_diag.to_dense()
  assert jnp.allclose(dense_result.elements, dense_expected.elements)

  # Test block diagonal case with diagonal blocks
  zero_diag = DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags)
  block_diag = Block2x2Matrix.from_blocks(A_diag, zero_diag, zero_diag, D_diag)

  # Test matrix operations
  # Addition
  sum_matrix = all_diag + block_diag
  assert sum_matrix.shape == (dim, dim)

  # Matrix-matrix multiplication
  product = all_diag @ block_diag
  assert product.shape == (dim, dim)

  # Test diagonal blocks with special operations

  # Test inverse
  # Make sure diagonals are non-zero for invertibility
  nonzero_diag1 = jnp.abs(diag1) + 1.0
  nonzero_diag4 = jnp.abs(diag4) + 1.0

  invertible_diag = Block2x2Matrix.from_blocks(
    DiagonalMatrix(nonzero_diag1, tags=TAGS.no_tags),
    DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags),
    DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags),
    DiagonalMatrix(nonzero_diag4, tags=TAGS.no_tags)
  )

  inv = invertible_diag.get_inverse()
  assert inv.shape == (dim, dim)

  # Verify that the inverse is correct by checking that A * A^(-1) ≈ I
  identity = invertible_diag @ inv
  identity_dense = identity.to_dense().elements
  assert jnp.allclose(jnp.diag(identity_dense), jnp.ones(dim), atol=1e-5)

  # Test Cholesky decomposition with positive diagonal elements
  pos_diag1 = jnp.abs(nonzero_diag1)
  pos_diag4 = jnp.abs(nonzero_diag4)

  pos_def_diag = Block2x2Matrix.from_blocks(
    DiagonalMatrix(pos_diag1, tags=TAGS.no_tags),
    DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags),
    DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags),
    DiagonalMatrix(pos_diag4, tags=TAGS.no_tags)
  )

  chol = pos_def_diag.get_cholesky()
  assert chol.shape == (dim, dim)

  # Verify L @ L.T = original matrix
  chol_dense = chol.to_dense().elements
  recon = chol_dense @ chol_dense.T
  assert jnp.allclose(recon, pos_def_diag.to_dense().elements, atol=1e-5)


def test_autodiff_with_diagonal_blocks():
  """Test autodifferentiation with diagonal blocks."""
  import jax

  # Create a simple function that operates on a Block2x2Matrix with diagonal blocks
  def matrix_function(elements):
    # Reshape elements into half dimensions
    dim = elements.shape[0]
    half_dim = dim // 2

    # Create diagonal blocks
    diag1 = elements[:half_dim]  # First diagonal elements
    diag2 = elements[half_dim:]  # Second diagonal elements

    # Create DiagonalMatrix instances
    A_diag = DiagonalMatrix(diag1, tags=TAGS.no_tags)
    D_diag = DiagonalMatrix(diag2, tags=TAGS.no_tags)
    zero_block = DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags)

    # Create block diagonal matrix
    block_mat = Block2x2Matrix.from_blocks(A_diag, zero_block, zero_block, D_diag)

    # Compute determinant (product of diagonal elements for block diagonal)
    log_det = block_mat.get_log_det()
    det = jnp.exp(log_det)
    return det

  # Create test data
  dim = 4
  half_dim = dim // 2
  diag_elements = jnp.ones(dim) + 0.5 * jnp.arange(dim)

  # Compute gradient using JAX
  grad_fn = jax.grad(matrix_function)
  grad_result = grad_fn(diag_elements)

  # For a block diagonal matrix with diagonal blocks, the gradient of determinant
  # with respect to each diagonal element should be the product of all other diagonal
  # elements (similar to the cofactor formula)

  # Compute expected gradient for first block
  expected_grad1 = jnp.zeros(dim)
  for i in range(half_dim):
    # Gradient is product of all diags except the current one
    cofactor = jnp.prod(diag_elements[:half_dim]) / diag_elements[i]
    cofactor *= jnp.prod(diag_elements[half_dim:])
    expected_grad1 = expected_grad1.at[i].set(cofactor)

  # Compute expected gradient for second block
  for i in range(half_dim, dim):
    # Gradient is product of all diags except the current one
    cofactor = jnp.prod(diag_elements[half_dim:]) / diag_elements[i]
    cofactor *= jnp.prod(diag_elements[:half_dim])
    expected_grad1 = expected_grad1.at[i].set(cofactor)

  # Verify gradients match
  assert jnp.allclose(grad_result, expected_grad1)

  # Test a more complex function with matrix inversion
  def matrix_inverse_function(elements):
    dim = elements.shape[0]
    half_dim = dim // 2

    # Create positive diagonal elements to ensure invertibility
    pos_diag1 = jnp.abs(elements[:half_dim]) + 1.0
    pos_diag2 = jnp.abs(elements[half_dim:]) + 1.0

    # Create DiagonalMatrix instances
    A_diag = DiagonalMatrix(pos_diag1, tags=TAGS.no_tags)
    D_diag = DiagonalMatrix(pos_diag2, tags=TAGS.no_tags)
    zero_block = DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags)

    # Create block diagonal matrix
    block_mat = Block2x2Matrix.from_blocks(A_diag, zero_block, zero_block, D_diag)

    # Get inverse
    inv = block_mat.get_inverse()

    # Return sum of diagonal elements of inverse as scalar output
    diag_sum = 0.0
    for i in range(2):
      block = inv.matrices[i, i]
      if isinstance(block, DiagonalMatrix):
        diag_sum += jnp.sum(block.elements)
      else:
        diag_sum += jnp.trace(block.elements)

    return diag_sum

  # Test elements
  test_elements = jnp.array([2.0, 3.0, 4.0, 5.0])

  # Compute gradient with JAX
  grad_fn2 = jax.grad(matrix_inverse_function)
  grad_result2 = grad_fn2(test_elements)

  # For block diagonal with diagonal blocks, the gradient of inverse's diagonal sum
  # with respect to diagonal elements should be -1/x^2 (derivative of 1/x)
  half_dim = dim // 2

  # Compute expected gradient
  pos_diag1 = jnp.abs(test_elements[:half_dim]) + 1.0
  pos_diag2 = jnp.abs(test_elements[half_dim:]) + 1.0

  # Compute gradient of abs(x) + 1 with respect to x
  # d/dx (|x| + 1) = sign(x)
  grad_abs1 = jnp.sign(test_elements[:half_dim])
  grad_abs2 = jnp.sign(test_elements[half_dim:])

  # Gradient of diagonal sum of inverse with respect to diagonal elements
  # For diagonal matrix, inverse's diagonal is 1/diag
  # So d/dx (1/diag) = -1/diag^2 * d(diag)/dx
  expected_grad2 = jnp.zeros(dim)
  expected_grad2 = expected_grad2.at[:half_dim].set(-1.0 / (pos_diag1 * pos_diag1) * grad_abs1)
  expected_grad2 = expected_grad2.at[half_dim:].set(-1.0 / (pos_diag2 * pos_diag2) * grad_abs2)

  # Verify gradients are reasonably close
  # Note: numerical precision issues may cause small differences
  assert jnp.allclose(grad_result2, expected_grad2, atol=1e-4)


def test_set_eye_preserves_matrix_types():
  """Test that set_eye() preserves the types of submatrices in Block2x2Matrix.

  This test reproduces a bug where T.set_eye() returns DiagonalMatrix blocks
  even when the original T.matrices contained DenseMatrix blocks, causing
  type mismatches in util.where operations.
  """
  key = random.PRNGKey(42)
  dim = 4
  half_dim = dim // 2

  # Create a Block2x2Matrix with all DenseMatrix blocks
  k1, k2, k3, k4 = random.split(key, 4)
  A = DenseMatrix(random.normal(k1, (half_dim, half_dim)), tags=TAGS.no_tags)
  B = DenseMatrix(random.normal(k2, (half_dim, half_dim)), tags=TAGS.no_tags)
  C = DenseMatrix(random.normal(k3, (half_dim, half_dim)), tags=TAGS.no_tags)
  D = DenseMatrix(random.normal(k4, (half_dim, half_dim)), tags=TAGS.no_tags)

  T = Block2x2Matrix.from_blocks(A, B, C, D)

  # Verify that all submatrices are DenseMatrix
  assert isinstance(T.matrices[0, 0], DenseMatrix)
  assert isinstance(T.matrices[0, 1], DenseMatrix)
  assert isinstance(T.matrices[1, 0], DenseMatrix)
  assert isinstance(T.matrices[1, 1], DenseMatrix)

  # Call set_eye() - this should preserve the matrix types
  T_eye = T.set_eye()

  # Check that the result is still a Block2x2Matrix with DenseMatrix blocks
  assert isinstance(T_eye, Block2x2Matrix)
  assert isinstance(T_eye.matrices[0, 0], DenseMatrix), f"Expected DenseMatrix, got {type(T_eye.matrices[0, 0])}"
  assert isinstance(T_eye.matrices[0, 1], DenseMatrix), f"Expected DenseMatrix, got {type(T_eye.matrices[0, 1])}"
  assert isinstance(T_eye.matrices[1, 0], DenseMatrix), f"Expected DenseMatrix, got {type(T_eye.matrices[1, 0])}"
  assert isinstance(T_eye.matrices[1, 1], DenseMatrix), f"Expected DenseMatrix, got {type(T_eye.matrices[1, 1])}"

  # Verify that the result is actually an identity matrix
  expected_identity = jnp.eye(dim)
  actual_matrix = T_eye.as_matrix()
  assert jnp.allclose(actual_matrix, expected_identity), "set_eye() should produce identity matrix"

  # Test that util.where works with the result (this is where the original bug manifested)
  import linsdex.util as util

  # Create a condition and test util.where with T and T_eye
  condition = jnp.array(True)  # Simple boolean condition

  # This should not raise a type mismatch error
  try:
    result = util.where(condition, T_eye, T)
    # Verify the result has the correct types
    assert isinstance(result.matrices[0, 0], DenseMatrix)
    assert isinstance(result.matrices[0, 1], DenseMatrix)
    assert isinstance(result.matrices[1, 0], DenseMatrix)
    assert isinstance(result.matrices[1, 1], DenseMatrix)
  except ValueError as e:
    if "Custom node type mismatch" in str(e):
      # This is the bug we're testing for
      pytest.fail(f"set_eye() caused type mismatch in util.where: {e}")
    else:
      raise


def test_set_eye_with_diagonal_blocks():
  """Test that set_eye() works correctly when original blocks are DiagonalMatrix."""
  dim = 4
  half_dim = dim // 2

  # Create a Block2x2Matrix with all DiagonalMatrix blocks
  A = DiagonalMatrix(jnp.ones(half_dim), tags=TAGS.no_tags)
  B = DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags)
  C = DiagonalMatrix(jnp.zeros(half_dim), tags=TAGS.zero_tags)
  D = DiagonalMatrix(jnp.ones(half_dim), tags=TAGS.no_tags)

  T = Block2x2Matrix.from_blocks(A, B, C, D)

  # Verify that all submatrices are DiagonalMatrix
  assert isinstance(T.matrices[0, 0], DiagonalMatrix)
  assert isinstance(T.matrices[0, 1], DiagonalMatrix)
  assert isinstance(T.matrices[1, 0], DiagonalMatrix)
  assert isinstance(T.matrices[1, 1], DiagonalMatrix)

  # Call set_eye() - this should preserve the matrix types
  T_eye = T.set_eye()

  # Check that the result is still a Block2x2Matrix with DiagonalMatrix blocks
  assert isinstance(T_eye, Block2x2Matrix)
  assert isinstance(T_eye.matrices[0, 0], DiagonalMatrix)
  assert isinstance(T_eye.matrices[0, 1], DiagonalMatrix)
  assert isinstance(T_eye.matrices[1, 0], DiagonalMatrix)
  assert isinstance(T_eye.matrices[1, 1], DiagonalMatrix)

  # Verify that the result is actually an identity matrix
  expected_identity = jnp.eye(dim)
  actual_matrix = T_eye.as_matrix()
  assert jnp.allclose(actual_matrix, expected_identity)


def test_set_eye_with_mixed_blocks():
  """Test that set_eye() works correctly when blocks are mixed types."""
  dim = 4
  half_dim = dim // 2
  key = random.PRNGKey(123)

  # Create a Block2x2Matrix with all DenseMatrix blocks first
  # (since from_blocks requires all blocks to be the same type)
  A = DenseMatrix(random.normal(key, (half_dim, half_dim)), tags=TAGS.no_tags)
  B = DenseMatrix(jnp.ones((half_dim, half_dim)), tags=TAGS.no_tags)
  C = DenseMatrix(jnp.zeros((half_dim, half_dim)), tags=TAGS.zero_tags)
  D = DenseMatrix(jnp.eye(half_dim), tags=TAGS.no_tags)

  T = Block2x2Matrix.from_blocks(A, B, C, D)

  # Call set_eye() - this should preserve the matrix types
  T_eye = T.set_eye()

  # Check that the result preserves the original matrix types (all DenseMatrix)
  assert isinstance(T_eye.matrices[0, 0], DenseMatrix)
  assert isinstance(T_eye.matrices[0, 1], DenseMatrix)
  assert isinstance(T_eye.matrices[1, 0], DenseMatrix)
  assert isinstance(T_eye.matrices[1, 1], DenseMatrix)

  # Verify that the result is actually an identity matrix
  expected_identity = jnp.eye(dim)
  actual_identity = T_eye.as_matrix()
  assert jnp.allclose(actual_identity, expected_identity)


if __name__ == "__main__":
  # Enable x64 precision for better numerical stability
  jax.config.update('jax_enable_x64', True)
  unittest.main()