import jax
import jax.numpy as jnp
from jax import random
import pytest
import equinox as eqx
import jax.tree_util as jtu

from linsdex.linear_functional.linear_functional import LinearFunctional, resolve_linear_functional
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.matrix.tags import TAGS

def create_matrix(key, dim, matrix_type):
  """Helper to create different types of matrices."""
  if matrix_type == "dense":
    return DenseMatrix(random.normal(key, (dim, dim)))
  elif matrix_type == "diagonal":
    return DiagonalMatrix(random.normal(key, (dim,)))
  else:
    raise ValueError(f"Unknown matrix type: {matrix_type}")

def create_functional(key, dim, matrix_type="dense"):
  """Helper to create a LinearFunctional with random data."""
  k1, k2 = random.split(key)
  A = create_matrix(k1, dim, matrix_type)
  b = random.normal(k2, (dim,))
  return LinearFunctional(A, b)

@pytest.mark.parametrize("matrix_type", ["dense", "diagonal"])
class TestLinearFunctional:
  def test_initialization(self, matrix_type):
    """Test basic initialization of LinearFunctional."""
    dim = 3
    key = random.PRNGKey(0)
    A = create_matrix(key, dim, matrix_type)
    b = jnp.ones(dim)
    lf = LinearFunctional(A, b)
    assert lf.A is A
    assert lf.b is b

  def test_call_method(self, matrix_type):
    """Test the __call__ method for correct evaluation."""
    dim = 2
    key = random.PRNGKey(1)
    A = create_matrix(key, dim, matrix_type)
    b = jnp.array([5, 6], dtype=jnp.float32)
    lf = LinearFunctional(A, b)
    x = jnp.array([1, 1], dtype=jnp.float32)
    result = lf(x)
    expected = A@x + b
    assert jnp.allclose(result, expected)

  def test_addition(self, matrix_type):
    """Test addition of two LinearFunctional objects."""
    key = random.PRNGKey(0)
    dim = 3
    lf1 = create_functional(key, dim, matrix_type)
    lf2 = create_functional(random.split(key)[1], dim, matrix_type)

    lf_sum = lf1 + lf2
    assert isinstance(lf_sum, LinearFunctional)
    assert jnp.allclose((lf_sum.A).as_matrix(), (lf1.A + lf2.A).as_matrix())
    assert jnp.allclose(lf_sum.b, lf1.b + lf2.b)

    # Test adding a vector
    vec = jnp.ones(dim)
    lf_sum_vec = lf1 + vec
    assert jnp.allclose(lf_sum_vec.A.as_matrix(), lf1.A.as_matrix())
    assert jnp.allclose(lf_sum_vec.b, lf1.b + vec)

  def test_subtraction(self, matrix_type):
    """Test subtraction of two LinearFunctional objects."""
    key = random.PRNGKey(1)
    dim = 2
    lf1 = create_functional(key, dim, matrix_type)
    lf2 = create_functional(random.split(key)[1], dim, matrix_type)

    lf_sub = lf1 - lf2
    assert jnp.allclose((lf_sub.A).as_matrix(), (lf1.A - lf2.A).as_matrix())
    assert jnp.allclose(lf_sub.b, lf1.b - lf2.b)

  def test_negation(self, matrix_type):
    """Test negation of a LinearFunctional."""
    key = random.PRNGKey(2)
    dim = 4
    lf = create_functional(key, dim, matrix_type)
    neg_lf = -lf
    assert jnp.allclose(neg_lf.A.as_matrix(), -lf.A.as_matrix())
    assert jnp.allclose(neg_lf.b, -lf.b)

  def test_scalar_multiplication(self, matrix_type):
    """Test multiplication of a LinearFunctional by a scalar."""
    key = random.PRNGKey(3)
    dim = 3
    lf = create_functional(key, dim, matrix_type)
    scalar = 2.5

    scaled_lf = lf * scalar
    assert jnp.allclose(scaled_lf.A.as_matrix(), (lf.A * scalar).as_matrix())
    assert jnp.allclose(scaled_lf.b, lf.b * scalar)

    scaled_lf_r = scalar * lf
    assert jnp.allclose(scaled_lf_r.A.as_matrix(), (lf.A * scalar).as_matrix())
    assert jnp.allclose(scaled_lf_r.b, lf.b * scalar)

  def test_matrix_multiplication(self, matrix_type):
    """Test multiplication of a LinearFunctional by a matrix."""
    key = random.PRNGKey(4)
    dim = 2
    lf = create_functional(key, dim, matrix_type)
    M = create_matrix(random.split(key)[1], dim, matrix_type)

    new_lf = M @ lf
    assert jnp.allclose(new_lf.A.as_matrix(), (M @ lf.A).as_matrix())
    assert jnp.allclose(new_lf.b, M @ lf.b)

  def test_matrix_solve(self, matrix_type):
    """Test solving a linear system with a LinearFunctional."""
    key = random.PRNGKey(5)
    dim = 3
    k1, k2, k3 = random.split(key, 3)

    lf = create_functional(k1, dim, matrix_type)
    M = create_matrix(k2, dim, "dense") # Must be dense to guarantee invertibility

    solved_lf = M.solve(lf)
    x = random.normal(k3, (dim,))

    reconstructed_val = M @ solved_lf(x)
    original_val = lf(x)
    assert jnp.allclose(reconstructed_val, original_val, atol=1e-5)

  def test_get_inverse(self, matrix_type):
    """Test the get_inverse method for correctness."""
    key = random.PRNGKey(6)
    dim = 2
    k1, k2, k3 = random.split(key, 3)

    # Create an invertible matrix - for diagonal, ensure non-zero diagonal elements
    if matrix_type == "dense":
      A = create_matrix(k1, dim, "dense")
    else:  # diagonal
      diag_vals = random.normal(k1, (dim,))
      diag_vals = jnp.where(jnp.abs(diag_vals) < 1e-3, 1.0, diag_vals)  # Ensure non-zero
      A = DiagonalMatrix(diag_vals)

    b = random.normal(k2, (dim,))
    lf = LinearFunctional(A, b)

    # Get the inverse
    lf_inv = lf.get_inverse()

    # Test that inverse(f(x)) = x
    x = random.normal(k3, (dim,))
    y = lf(x)
    x_recovered = lf_inv(y)
    assert jnp.allclose(x_recovered, x, atol=1e-5)

    # Test that f(inverse(y)) = y
    y_test = random.normal(random.split(k3)[1], (dim,))
    x_from_y = lf_inv(y_test)
    y_recovered = lf(x_from_y)
    assert jnp.allclose(y_recovered, y_test, atol=1e-5)

    # Test mathematical properties: if f(x) = Ax + b, then f^{-1}(y) = A^{-1}y - A^{-1}b
    A_inv_expected = A.get_inverse()
    b_inv_expected = -A_inv_expected @ b

    assert jnp.allclose(lf_inv.A.as_matrix(), A_inv_expected.as_matrix(), atol=1e-5)
    assert jnp.allclose(lf_inv.b, b_inv_expected, atol=1e-5)


class TestLinearFunctionalWithSymbolicMatrices:
  def test_zero_matrix_multiplication(self):
    """Test multiplication by a symbolic zero matrix."""
    dim = 3
    key = random.PRNGKey(100)
    lf = create_functional(key, dim)
    zero_matrix = DenseMatrix(jnp.zeros((dim, dim)), tags=TAGS.zero_tags)

    result_lf = zero_matrix @ lf
    assert result_lf.A.tags.is_zero
    assert jnp.allclose(result_lf.b, 0.0)

  def test_identity_matrix_multiplication(self):
    """Test multiplication by an identity matrix."""
    dim = 4
    key = random.PRNGKey(101)
    lf = create_functional(key, dim)
    identity_matrix = DenseMatrix.eye(dim)

    result_lf = identity_matrix @ lf
    x = random.normal(random.split(key)[1], (dim,))

    assert jnp.allclose(result_lf(x), lf(x))

  def test_infinity_matrix_solve(self):
    """Test solving with a symbolic infinity matrix."""
    dim = 2
    key = random.PRNGKey(102)
    lf = create_functional(key, dim)

    # A.solve(B) where A is inf -> should be zero
    inf_matrix = DenseMatrix(jnp.eye(dim), tags=TAGS.inf_tags)

    solved_lf = inf_matrix.solve(lf)
    assert solved_lf.A.tags.is_zero
    assert jnp.allclose(solved_lf.b, 0.0)

class TestResolveLinearFunctional:
  def test_resolve_basic(self):
    """Test basic functionality of resolve_linear_functional."""
    dim = 3
    key = random.PRNGKey(10)
    lf = create_functional(key, dim)
    x = random.normal(random.split(key)[1], (dim,))

    resolved_pytree = resolve_linear_functional(lf, x)
    assert not isinstance(resolved_pytree, LinearFunctional)
    assert jnp.allclose(resolved_pytree, lf(x))

  def test_resolve_in_pytree(self):
    """Test resolving functionals nested in a larger pytree."""
    dim = 2
    key = random.PRNGKey(11)
    k1, k2, k3 = random.split(key, 3)

    lf1 = create_functional(k1, dim)
    lf2 = create_functional(k2, dim)
    x = random.normal(k3, (dim,))

    pytree = {
      'a': lf1,
      'b': [lf2, jnp.ones(dim, dtype=jnp.float32)],
      'c': {'d': 3.0}
    }

    resolved_pytree = resolve_linear_functional(pytree, x)

    assert not isinstance(resolved_pytree['a'], LinearFunctional)
    assert jnp.allclose(resolved_pytree['a'], lf1(x))
    assert not isinstance(resolved_pytree['b'][0], LinearFunctional)
    assert jnp.allclose(resolved_pytree['b'][0], lf2(x))
    assert jnp.allclose(resolved_pytree['b'][1], jnp.ones(dim, dtype=jnp.float32))
    assert resolved_pytree['c']['d'] == 3.0

  def test_vmapped_functional(self):
    """Test vmapping over a LinearFunctional."""
    dim = 2
    batch_size = 5
    key = random.PRNGKey(20)

    keys = random.split(key, batch_size)
    vmapped_lf = jax.vmap(create_functional, in_axes=(0, None))(keys, dim)

    assert vmapped_lf.batch_size == batch_size

    x = random.normal(key, (dim,))
    result = jax.vmap(lambda s: s(x))(vmapped_lf)
    assert result.shape == (batch_size, dim)

    # Test resolving a vmapped functional
    x_batch = random.normal(key, (batch_size, dim))

    # Manually vmap the __call__ to handle the batch dimensions correctly.
    resolved = jax.vmap(lambda f, _x: f(_x))(vmapped_lf, x_batch)
    assert resolved.shape == (batch_size, dim)

    # Manually compute expected result
    expected = jax.vmap(lambda f, _x: f.A@_x + f.b)(vmapped_lf, x_batch)
    assert jnp.allclose(resolved, expected)

if __name__ == "__main__":
  pytest.main([__file__])