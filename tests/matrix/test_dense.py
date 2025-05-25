import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
from linsdex.matrix.dense import DenseMatrix, ParametricSymmetricDenseMatrix
from linsdex.matrix.tags import Tags, TAGS
from tests.matrix.shared import matrices_equal, matrix_tests, performance_tests
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import equinox as eqx
from tests.matrix.base_for_tests import AbstractMatrixTest, autodiff_for_matrix_class, matrix_implementations_tests


class TestDenseMatrix(unittest.TestCase, AbstractMatrixTest):
  matrix_class = DenseMatrix

  def setUp(self):
    # Set up common test fixtures
    self.key = random.PRNGKey(42)
    self.dim = 4

    k1, k2 = random.split(self.key)
    self.A_elements = random.normal(k1, (self.dim, self.dim))
    self.B_elements = random.normal(k2, (self.dim, self.dim))
    self.A = DenseMatrix(self.A_elements, tags=TAGS.no_tags)
    self.B = DenseMatrix(self.B_elements, tags=TAGS.no_tags)

    # Create symmetric matrices for testing
    self.A_sym_elements = self.A_elements @ self.A_elements.T
    self.B_sym_elements = self.B_elements @ self.B_elements.T
    self.A_sym = DenseMatrix(self.A_sym_elements, tags=TAGS.no_tags)
    self.B_sym = DenseMatrix(self.B_sym_elements, tags=TAGS.no_tags)

    # Create special matrices
    self.zero = DenseMatrix(jnp.zeros((self.dim, self.dim)), tags=TAGS.zero_tags)
    self.eye = DenseMatrix(jnp.eye(self.dim), tags=TAGS.no_tags)

  # Override factory methods
  def create_matrix(self, elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags
    return DenseMatrix(elements, tags=tags)

  def create_random_matrix(self, key, shape=None, tags=None):
    if shape is None:
      shape = (self.dim, self.dim)
    if tags is None:
      tags = TAGS.no_tags
    elements = random.normal(key, shape)
    return DenseMatrix(elements, tags=tags)

  def create_random_symmetric_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags
    elements = random.normal(key, (dim, dim))
    sym_elements = elements @ elements.T
    return DenseMatrix(sym_elements, tags=tags)

  def create_zeros_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.zero_tags
    return DenseMatrix(jnp.zeros((dim, dim)), tags=tags)

  def create_eye_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags
    return DenseMatrix(jnp.eye(dim), tags=tags)

  def create_well_conditioned_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags
    elements = random.normal(key, (dim, dim))
    # Make it positive definite and well-conditioned
    elements = elements @ elements.T + dim * jnp.eye(dim)
    return DenseMatrix(elements, tags=tags)

  def test_initialization(self):
    # Test basic initialization
    A = DenseMatrix(self.A_elements, tags=TAGS.no_tags)
    self.assertTrue(matrices_equal(A.elements, self.A_elements))
    self.assertEqual(A.tags, TAGS.no_tags)

    # Test initialization with different tags
    B = DenseMatrix(self.B_elements, tags=TAGS.no_tags)
    self.assertTrue(matrices_equal(B.elements, self.B_elements))
    self.assertEqual(B.tags, TAGS.no_tags)

  def test_as_matrix(self):
    mat = self.A.as_matrix()
    self.assertTrue(matrices_equal(mat, self.A_elements))

def test_correctness_with_different_tags():
  """Test the correctness with different tag combinations."""
  key = random.PRNGKey(0)

  # Custom function to create diagonal matrix from dense elements
  def create_diagonal_fn(elements, tags):
    # If elements is a 2D matrix, extract the diagonal
    return DenseMatrix(elements, tags=tags)

  matrix_implementations_tests(
    key=key,
    create_matrix_fn=create_diagonal_fn
  )


def test_comprehensive_correctness():
  """Test all tag combinations for DenseMatrix."""
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
    A_raw = random.normal(k1, (4, 4))
    B_raw = random.normal(k2, (4, 4))

    # Modify matrices according to tags
    if tag_A.is_zero:
      A_raw = jnp.zeros_like(A_raw)
    if tag_A.is_inf:
      A_raw = jnp.full_like(A_raw, jnp.inf)

    if tag_B.is_zero:
      B_raw = jnp.zeros_like(B_raw)
    if tag_B.is_inf:
      B_raw = jnp.full_like(B_raw, jnp.inf)

    A = DenseMatrix(A_raw, tags=tag_A)
    B = DenseMatrix(B_raw, tags=tag_B)

    matrix_tests(key, A, B)

  # In pytest, use assertions
  assert len(failed_tests) == 0, f"{len(failed_tests)} tag combinations failed tests"


def test_autodiff():
  """Test that autodifferentiation works for all matrix operations."""
  # Custom function to create diagonal matrix from dense elements
  def create_diagonal_fn(elements, tags):
    # If elements is a 2D matrix, extract the diagonal
    return DenseMatrix(elements, tags=tags)

  autodiff_for_matrix_class(create_diagonal_fn)


if __name__ == "__main__":
  # Enable x64 precision for better numerical stability
  jax.config.update('jax_enable_x64', True)
  unittest.main()