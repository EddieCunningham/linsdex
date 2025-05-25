import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
from linsdex.matrix.diagonal import DiagonalMatrix, ParametricSymmetricDiagonalMatrix
from .shared import matrices_equal, matrix_tests, performance_tests
from linsdex.matrix.tags import Tags, TAGS
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import equinox as eqx
from .base_for_tests import AbstractMatrixTest, autodiff_for_matrix_class, matrix_implementations_tests


class TestDiagonalMatrix(unittest.TestCase, AbstractMatrixTest):
  matrix_class = DiagonalMatrix

  def setUp(self):
    # Set up common test fixtures
    self.key = random.PRNGKey(42)
    self.dim = 4

    k1, k2 = random.split(self.key)
    self.A_elements = random.normal(k1, (self.dim,))
    self.B_elements = random.normal(k2, (self.dim,))
    self.A = DiagonalMatrix(self.A_elements, tags=TAGS.no_tags)
    self.B = DiagonalMatrix(self.B_elements, tags=TAGS.no_tags)

    # Create special matrices
    self.zero = DiagonalMatrix(jnp.zeros((self.dim,)), tags=TAGS.zero_tags)
    self.eye = DiagonalMatrix(jnp.ones((self.dim,)), tags=TAGS.no_tags)

  # Override factory methods
  def create_matrix(self, elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags
    # If elements is a 2D matrix, extract the diagonal
    if len(elements.shape) == 2:
      elements = jnp.diag(elements)
    return DiagonalMatrix(elements, tags=tags)

  def create_random_matrix(self, key, shape=None, tags=None):
    if shape is None:
      dim = self.dim
    elif isinstance(shape, tuple) and len(shape) == 2:
      dim = shape[0]  # Assume square
    else:
      dim = shape

    if tags is None:
      tags = TAGS.no_tags

    elements = random.normal(key, (dim,))
    return DiagonalMatrix(elements, tags=tags)

  def create_random_symmetric_matrix(self, key, dim=None, tags=None):
    # For diagonal matrices, any diagonal matrix is symmetric
    return self.create_random_matrix(key, dim, tags)

  def create_zeros_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.zero_tags
    return DiagonalMatrix(jnp.zeros((dim,)), tags=tags)

  def create_eye_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags
    return DiagonalMatrix(jnp.ones((dim,)), tags=tags)

  def create_well_conditioned_matrix(self, key, dim=None, tags=None):
    # For diagonal matrix, ensure all diagonal elements are significant
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags

    # Create positive elements with good conditioning
    elements = jnp.abs(random.normal(key, (dim,))) + 1.0
    return DiagonalMatrix(elements, tags=tags)

  def test_initialization(self):
    # Test basic initialization
    A = DiagonalMatrix(self.A_elements, tags=TAGS.no_tags)
    self.assertTrue(matrices_equal(A.elements, self.A_elements))
    self.assertEqual(A.tags, TAGS.no_tags)

    # Test initialization with different tags
    B = DiagonalMatrix(self.B_elements, tags=TAGS.no_tags)
    self.assertTrue(matrices_equal(B.elements, self.B_elements))
    self.assertEqual(B.tags, TAGS.no_tags)

  def test_to_dense(self):
    # Test conversion to dense matrix
    A_dense = self.A.to_dense()
    self.assertTrue(matrices_equal(A_dense.elements, jnp.diag(self.A_elements)))
    self.assertEqual(A_dense.tags, self.A.tags)

  def test_project_dense(self):
    # Test projection from dense to diagonal
    from linsdex.matrix.dense import DenseMatrix

    # Create a dense matrix
    k1 = random.PRNGKey(101)
    dense_elements = random.normal(k1, (self.dim, self.dim))
    dense_matrix = DenseMatrix(dense_elements, tags=TAGS.no_tags)

    # Project to diagonal
    diag_matrix = self.A.project_dense(dense_matrix)
    expected_elements = jnp.diag(dense_elements)

    self.assertTrue(matrices_equal(diag_matrix.elements, expected_elements))
    self.assertEqual(diag_matrix.tags, dense_matrix.tags)

  def test_parametric_symmetric_diagonal_matrix(self):
    # Test initialization
    param_sym = ParametricSymmetricDiagonalMatrix(self.A_elements)

    # Check if diagonal elements are positive
    self.assertTrue(jnp.all(param_sym.elements > 0))

  def test_make_parametric_symmetric_matrix(self):
    from linsdex.matrix.diagonal import make_parametric_symmetric_matrix

    # Start with a positive diagonal matrix
    A_pos = self.create_well_conditioned_matrix(self.key)

    # Convert to parametric form
    param_A = make_parametric_symmetric_matrix(A_pos)

    # Check elements are positive
    self.assertTrue(jnp.all(param_A.elements > 0))


def test_matrix_tests():
  """Test the matrix_tests function from shared.py with DiagonalMatrix."""
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)
  dim = 4

  A_elements = random.normal(k1, (dim,))
  B_elements = random.normal(k2, (dim,))

  # Make them well-conditioned
  A_elements = jnp.abs(A_elements) + 1.0
  B_elements = jnp.abs(B_elements) + 1.0

  A = DiagonalMatrix(A_elements, tags=TAGS.no_tags)
  B = DiagonalMatrix(B_elements, tags=TAGS.no_tags)

  # This should run without errors
  matrix_tests(key, A, B)


def test_performance():
  """Test the performance_tests function from shared.py with DiagonalMatrix."""
  key = random.PRNGKey(42)
  k1, k2 = random.split(key)
  dim = 4

  A_elements = random.normal(k1, (dim,))
  B_elements = random.normal(k2, (dim,))

  A = DiagonalMatrix(A_elements, tags=TAGS.no_tags)
  B = DiagonalMatrix(B_elements, tags=TAGS.no_tags)

  # This should run without errors
  performance_tests(A, B)


def test_correctness_with_different_tags():
  """Test the correctness with different tag combinations."""
  key = random.PRNGKey(0)

  # Custom function to create diagonal matrix from dense elements
  def create_diagonal_fn(elements, tags):
    # If elements is a 2D matrix, extract the diagonal
    if len(elements.shape) == 2:
      elements = jnp.diag(elements)
    return DiagonalMatrix(elements, tags=tags)

  matrix_implementations_tests(
    key=key,
    create_matrix_fn=create_diagonal_fn
  )


def test_comprehensive_correctness():
  """Test all tag combinations for DiagonalMatrix."""
  from itertools import product

  # Turn on x64
  jax.config.update('jax_enable_x64', True)
  key = random.PRNGKey(0)

  # All available tags
  tag_options = [
    TAGS.zero_tags,
    TAGS.no_tags,
    TAGS.inf_tags
  ]

  # Test all combinations of tags
  failed_tests = []
  total_tests = 0

  for tag_A, tag_B in product(tag_options, tag_options):
    total_tests += 1

    k1, k2 = random.split(key)
    key, _ = random.split(key)

    # Generate base random matrices
    A_raw = random.normal(k1, (4,))
    B_raw = random.normal(k2, (4,))

    # Modify matrices according to tags
    if tag_A.is_zero:
      A_raw = jnp.zeros_like(A_raw)
    if tag_A.is_inf:
      A_raw = jnp.full_like(A_raw, jnp.inf)

    if tag_B.is_zero:
      B_raw = jnp.zeros_like(B_raw)
    if tag_B.is_inf:
      B_raw = jnp.full_like(B_raw, jnp.inf)

    A = DiagonalMatrix(A_raw, tags=tag_A)
    B = DiagonalMatrix(B_raw, tags=tag_B)

    matrix_tests(key, A, B)

  # In pytest, use assertions
  assert len(failed_tests) == 0, f"{len(failed_tests)} tag combinations failed tests"


def test_autodiff():
  """Test that autodifferentiation works for all matrix operations."""

  # Custom function to create diagonal matrix from dense elements
  def create_diagonal_fn(elements, tags):
    # If elements is a 2D matrix, extract the diagonal
    if len(elements.shape) == 2:
      elements = jnp.diag(elements)
    return DiagonalMatrix(elements, tags=tags)

  autodiff_for_matrix_class(create_diagonal_fn)


if __name__ == "__main__":
  # Enable x64 precision for better numerical stability
  jax.config.update('jax_enable_x64', True)
  unittest.main()