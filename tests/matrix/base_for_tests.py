import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.matrix.tags import Tags, TAGS
from tests.matrix.shared import matrices_equal, matrix_tests, performance_tests
from jax import grad, jacfwd, value_and_grad
from typing import Type, Callable, Tuple, Any, Dict, List, Optional
from itertools import product
import abc
import equinox as eqx

class AbstractMatrixTest(abc.ABC):
    """Base test class for any AbstractSquareMatrix implementation.

    Subclasses must override the create_matrix and create_* methods to provide matrix instances
    of the specific type being tested.
    """

    # These properties should be set by subclasses
    matrix_class = None  # The matrix class to test

    # These are factory methods that subclasses must override
    @abc.abstractmethod
    def create_matrix(self, elements, tags=None):
        """Create a matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_matrix")

    @abc.abstractmethod
    def create_random_matrix(self, key, shape=None, tags=None):
        """Create a random matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_random_matrix")

    @abc.abstractmethod
    def create_random_symmetric_matrix(self, key, dim=None, tags=None):
        """Create a random symmetric matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_random_symmetric_matrix")

    @abc.abstractmethod
    def create_zeros_matrix(self, dim=None, tags=None):
        """Create a zeros matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_zeros_matrix")

    @abc.abstractmethod
    def create_eye_matrix(self, dim=None, tags=None):
        """Create an identity matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_eye_matrix")

    @abc.abstractmethod
    def create_well_conditioned_matrix(self, key, dim=None, tags=None):
        """Create a well-conditioned matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_well_conditioned_matrix")

    @abc.abstractmethod
    def test_initialization(self):
        raise NotImplementedError("Subclasses must implement test_initialization")

    # Common tests that work with any matrix type
    def test_shape_property(self):
        A = self.create_random_matrix(self.key)
        self.assertEqual(len(A.shape), 2)
        self.assertEqual(A.shape[0], self.dim)
        self.assertEqual(A.shape[1], self.dim)

        zero = self.create_zeros_matrix()
        self.assertEqual(zero.shape, (self.dim, self.dim))

    def test_batch_size(self):
        # Test single matrix (no batch)
        A = self.create_random_matrix(self.key)
        self.assertIsNone(A.batch_size)

    def test_class_methods(self):
        # Test zeros method
        zero_mat = self.create_zeros_matrix()
        zero_array = jnp.zeros((self.dim, self.dim))
        # For diagonal matrices, as_matrix() will create a diagonal matrix from elements
        self.assertTrue(matrices_equal(zero_mat.as_matrix(), zero_array))
        self.assertEqual(zero_mat.tags, TAGS.zero_tags)

        # Test eye method
        eye_mat = self.create_eye_matrix()
        self.assertTrue(matrices_equal(eye_mat.as_matrix(), jnp.eye(self.dim)))

    def test_neg(self):
        A = self.create_random_matrix(self.key)
        neg_A = -A
        self.assertTrue(matrices_equal(neg_A.as_matrix(), -A.as_matrix()))

    def test_addition(self):
        key1, key2 = random.split(self.key)
        A = self.create_random_matrix(key1)
        B = self.create_random_matrix(key2)

        # Test regular addition
        C = A + B
        self.assertTrue(matrices_equal(C.as_matrix(), A.as_matrix() + B.as_matrix()))

        # Test addition with zero
        zero = self.create_zeros_matrix()
        D = A + zero
        self.assertTrue(matrices_equal(D.as_matrix(), A.as_matrix()))

    def test_subtraction(self):
        key1, key2 = random.split(self.key)
        A = self.create_random_matrix(key1)
        B = self.create_random_matrix(key2)

        C = A - B
        self.assertTrue(matrices_equal(C.as_matrix(), A.as_matrix() - B.as_matrix()))

    def test_scalar_multiplication(self):
        A = self.create_random_matrix(self.key)
        scalar = 2.5

        # Test left multiplication
        C = scalar * A
        self.assertTrue(matrices_equal(C.as_matrix(), scalar * A.as_matrix()))

        # Test right multiplication
        D = A * scalar
        self.assertTrue(matrices_equal(D.as_matrix(), A.as_matrix() * scalar))

    def test_matrix_multiplication(self):
        key1, key2 = random.split(self.key)
        A = self.create_random_matrix(key1)
        B = self.create_random_matrix(key2)

        # Test matrix-matrix multiplication
        C = A @ B
        self.assertTrue(matrices_equal(C.as_matrix(), A.as_matrix() @ B.as_matrix()))

        # Test matrix-vector multiplication
        v = random.normal(self.key, (self.dim,))
        result = A @ v
        expected = A.as_matrix() @ v
        self.assertTrue(matrices_equal(result, expected))

    def test_division(self):
        A = self.create_random_matrix(self.key)
        scalar = 2.0

        C = A / scalar
        self.assertTrue(matrices_equal(C.as_matrix(), A.as_matrix() / scalar))

    def test_transpose(self):
        A = self.create_random_matrix(self.key)
        A_t = A.T
        self.assertTrue(matrices_equal(A_t.as_matrix(), A.as_matrix().T))

    def test_solve(self):
        A = self.create_well_conditioned_matrix(self.key)

        # Test matrix-vector solve
        b = random.normal(self.key, (self.dim,))
        x = A.solve(b)
        expected = jnp.linalg.solve(A.as_matrix(), b)
        self.assertTrue(matrices_equal(x, expected))

        # Test matrix-matrix solve
        key1, _ = random.split(self.key)
        B = self.create_random_matrix(key1)
        X = A.solve(B)
        expected_mat = jnp.linalg.solve(A.as_matrix(), B.as_matrix())
        self.assertTrue(matrices_equal(X.as_matrix(), expected_mat))

    def test_inverse(self):
        A = self.create_well_conditioned_matrix(self.key)

        A_inv = A.get_inverse()
        expected_inv = jnp.linalg.inv(A.as_matrix())
        self.assertTrue(matrices_equal(A_inv.as_matrix(), expected_inv))

    def test_log_det(self):
        A = self.create_well_conditioned_matrix(self.key)

        log_det = A.get_log_det()
        expected_log_det = jnp.linalg.slogdet(A.as_matrix())[1]
        self.assertTrue(jnp.allclose(log_det, expected_log_det))

    def test_cholesky(self):
        A = self.create_random_symmetric_matrix(self.key)

        chol = A.get_cholesky()
        expected_chol = jnp.linalg.cholesky(A.as_matrix())
        self.assertTrue(matrices_equal(chol.as_matrix(), expected_chol))

    def test_exp(self):
        A = self.create_random_matrix(self.key)

        A_exp = A.get_exp()
        expected_exp = jax.scipy.linalg.expm(A.as_matrix())
        self.assertTrue(matrices_equal(A_exp.as_matrix(), expected_exp))


def autodiff_for_matrix_class(create_diagonal_fn):
    """Test that autodifferentiation works for all matrix operations.

    Args:
        create_diagonal_fn: The matrix class to test
        batch_support: Whether the matrix class supports batched operations
    """
    # Enable x64 for better numerical stability
    jax.config.update('jax_enable_x64', True)

    # Set up test matrices
    key = random.PRNGKey(42)
    dim = 6

    # Create well-conditioned matrices for stable testing
    key1, key2 = random.split(key)
    A_raw = random.normal(key1, (dim, dim))
    A_raw = A_raw @ A_raw.T + dim * jnp.eye(dim)  # Make positive definite

    B_raw = random.normal(key2, (dim, dim))
    B_raw = B_raw @ B_raw.T + dim * jnp.eye(dim)

    v = random.normal(key, (dim,))
    scalar = 2.0

    # Create matrix factory functions
    create_A = lambda x: create_diagonal_fn(x, tags=TAGS.no_tags)
    create_B = lambda: create_diagonal_fn(B_raw, tags=TAGS.no_tags)

    # Cast to the correct structure
    A_raw = create_A(A_raw).as_matrix()
    B_raw = create_B().as_matrix()

    # Test core operations
    def neg_fn(x):
        A = create_A(x)
        return (-A).elements.sum()

    def direct_neg_fn(x):
        return (-x).sum()

    grad_neg_raw = grad(neg_fn)(A_raw)
    expected_grad_neg_raw = grad(direct_neg_fn)(A_raw)
    grad_neg = create_diagonal_fn(grad_neg_raw, TAGS.no_tags)
    expected_grad_neg = create_diagonal_fn(expected_grad_neg_raw, TAGS.no_tags)
    assert jnp.allclose(grad_neg.as_matrix(), expected_grad_neg.as_matrix())

    # Addition
    def add_fn(x):
        A = create_A(x)
        B = create_B()
        return (A + B).elements.sum()

    def direct_add_fn(x):
        return (x + B_raw).sum()

    grad_add_raw = grad(add_fn)(A_raw)
    expected_grad_add_raw = grad(direct_add_fn)(A_raw)
    grad_add = create_diagonal_fn(grad_add_raw, TAGS.no_tags)
    expected_grad_add = create_diagonal_fn(expected_grad_add_raw, TAGS.no_tags)
    assert jnp.allclose(grad_add.as_matrix(), expected_grad_add.as_matrix())

    # Subtraction
    def sub_fn(x):
        A = create_A(x)
        B = create_B()
        return (A - B).elements.sum()

    def direct_sub_fn(x):
        return (x - B_raw).sum()

    grad_sub_raw = grad(sub_fn)(A_raw)
    expected_grad_sub_raw = grad(direct_sub_fn)(A_raw)
    grad_sub = create_diagonal_fn(grad_sub_raw, TAGS.no_tags)
    expected_grad_sub = create_diagonal_fn(expected_grad_sub_raw, TAGS.no_tags)
    assert jnp.allclose(grad_sub.as_matrix(), expected_grad_sub.as_matrix())

    # Scalar multiplication
    def scalar_mul_fn(x):
        A = create_A(x)
        return (scalar * A).elements.sum()

    def direct_scalar_mul_fn(x):
        return (scalar * x).sum()

    grad_scalar_mul_raw = grad(scalar_mul_fn)(A_raw)
    expected_grad_scalar_mul_raw = grad(direct_scalar_mul_fn)(A_raw)
    grad_scalar_mul = create_diagonal_fn(grad_scalar_mul_raw, TAGS.no_tags)
    expected_grad_scalar_mul = create_diagonal_fn(expected_grad_scalar_mul_raw, TAGS.no_tags)
    assert jnp.allclose(grad_scalar_mul.as_matrix(), expected_grad_scalar_mul.as_matrix())

    # Matrix multiplication
    def matmul_fn(x):
        A = create_A(x)
        B = create_B()
        return (A @ B).elements.sum()

    def direct_matmul_fn(x):
        return (x @ B_raw).sum()

    grad_matmul_raw = grad(matmul_fn)(A_raw)
    expected_grad_matmul_raw = grad(direct_matmul_fn)(A_raw)
    grad_matmul = create_diagonal_fn(grad_matmul_raw, TAGS.no_tags)
    expected_grad_matmul = create_diagonal_fn(expected_grad_matmul_raw, TAGS.no_tags)
    assert jnp.allclose(grad_matmul.as_matrix(), expected_grad_matmul.as_matrix())

    # Matrix-vector multiplication
    def matvec_fn(x):
        A = create_A(x)
        return jnp.sum(A @ v)

    def direct_matvec_fn(x):
        return jnp.sum(x @ v)

    grad_matvec_raw = grad(matvec_fn)(A_raw)
    expected_grad_matvec_raw = grad(direct_matvec_fn)(A_raw)
    grad_matvec = create_diagonal_fn(grad_matvec_raw, TAGS.no_tags)
    expected_grad_matvec = create_diagonal_fn(expected_grad_matvec_raw, TAGS.no_tags)
    assert jnp.allclose(grad_matvec.as_matrix(), expected_grad_matvec.as_matrix())

    # Division
    def div_fn(x):
        A = create_A(x)
        return (A / scalar).elements.sum()

    def direct_div_fn(x):
        return (x / scalar).sum()

    grad_div_raw = grad(div_fn)(A_raw)
    expected_grad_div_raw = grad(direct_div_fn)(A_raw)
    grad_div = create_diagonal_fn(grad_div_raw, TAGS.no_tags)
    expected_grad_div = create_diagonal_fn(expected_grad_div_raw, TAGS.no_tags)
    assert jnp.allclose(grad_div.as_matrix(), expected_grad_div.as_matrix())

    # Transpose
    def transpose_fn(x):
        A = create_A(x)
        return A.T.elements.sum()

    def direct_transpose_fn(x):
        return x.T.sum()

    grad_transpose_raw = grad(transpose_fn)(A_raw)
    expected_grad_transpose_raw = grad(direct_transpose_fn)(A_raw)
    grad_transpose = create_diagonal_fn(grad_transpose_raw, TAGS.no_tags)
    expected_grad_transpose = create_diagonal_fn(expected_grad_transpose_raw, TAGS.no_tags)
    assert jnp.allclose(grad_transpose.as_matrix(), expected_grad_transpose.as_matrix())

    # Advanced operations

    # Matrix-vector solve
    def solve_vec_fn(x):
        A = create_A(x)
        return jnp.sum(A.solve(v))

    def direct_solve_vec_fn(x):
        return jnp.sum(jnp.linalg.solve(x, v))

    grad_solve_vec_raw = jax.jacobian(solve_vec_fn)(A_raw)
    expected_grad_solve_vec_raw = jax.jacobian(direct_solve_vec_fn)(A_raw)
    grad_solve_vec = create_diagonal_fn(grad_solve_vec_raw, TAGS.no_tags)
    expected_grad_solve_vec = create_diagonal_fn(expected_grad_solve_vec_raw, TAGS.no_tags)
    assert jnp.allclose(grad_solve_vec.as_matrix(), expected_grad_solve_vec.as_matrix())

    # Inverse
    def inverse_fn(x):
        A = create_A(x)
        return A.get_inverse().elements.sum()

    def direct_inverse_fn(x):
        return jnp.sum(jnp.linalg.inv(x))

    grad_inverse_raw = grad(inverse_fn)(A_raw)
    expected_grad_inverse_raw = grad(direct_inverse_fn)(A_raw)
    grad_inverse = create_diagonal_fn(grad_inverse_raw, TAGS.no_tags)
    expected_grad_inverse = create_diagonal_fn(expected_grad_inverse_raw, TAGS.no_tags)
    assert jnp.allclose(grad_inverse.as_matrix(), expected_grad_inverse.as_matrix())

    # Log determinant
    def logdet_fn(x):
        A = create_A(x)
        return A.get_log_det()

    def direct_logdet_fn(x):
        return jnp.linalg.slogdet(x)[1]

    grad_logdet_raw = grad(logdet_fn)(A_raw)
    expected_grad_logdet_raw = grad(direct_logdet_fn)(A_raw)
    grad_logdet = create_diagonal_fn(grad_logdet_raw, TAGS.no_tags)
    expected_grad_logdet = create_diagonal_fn(expected_grad_logdet_raw, TAGS.no_tags)
    assert jnp.allclose(grad_logdet.as_matrix(), expected_grad_logdet.as_matrix())

    # Cholesky
    def cholesky_fn(x):
        A = create_A(x)
        return A.get_cholesky().elements.sum()

    def direct_cholesky_fn(x):
        return jnp.sum(jnp.linalg.cholesky(x))

    grad_cholesky_raw = grad(cholesky_fn)(A_raw)
    expected_grad_cholesky_raw = grad(direct_cholesky_fn)(A_raw)
    grad_cholesky = create_diagonal_fn(grad_cholesky_raw, TAGS.no_tags)
    expected_grad_cholesky = create_diagonal_fn(expected_grad_cholesky_raw, TAGS.no_tags)
    assert jnp.allclose(grad_cholesky.as_matrix(), expected_grad_cholesky.as_matrix())

    # Matrix exponential
    def exp_fn(x):
        A = create_A(x)
        return A.get_exp().elements.sum()

    def direct_exp_fn(x):
        return jnp.sum(jax.scipy.linalg.expm(x))

    grad_exp_raw = grad(exp_fn)(A_raw)
    expected_grad_exp_raw = grad(direct_exp_fn)(A_raw)
    grad_exp = create_diagonal_fn(grad_exp_raw, TAGS.no_tags)
    expected_grad_exp = create_diagonal_fn(expected_grad_exp_raw, TAGS.no_tags)
    assert jnp.allclose(grad_exp.as_matrix(), expected_grad_exp.as_matrix())

    # SVD (just U component)
    def svd_u_fn(x):
        A = create_A(x)
        U, _, _ = A.get_svd()
        return U.elements.sum()

    def direct_svd_u_fn(x):
        U, _, _ = jnp.linalg.svd(x, full_matrices=False)
        return jnp.sum(U)

    # Use forward-mode autodiff for SVD
    grad_svd_u_raw = jacfwd(svd_u_fn)(A_raw)
    expected_grad_svd_u_raw = jacfwd(direct_svd_u_fn)(A_raw)
    grad_svd_u = create_diagonal_fn(grad_svd_u_raw, TAGS.no_tags)
    expected_grad_svd_u = create_diagonal_fn(expected_grad_svd_u_raw, TAGS.no_tags)
    assert jnp.allclose(grad_svd_u.as_matrix(), expected_grad_svd_u.as_matrix())


def matrix_implementations_tests(
    key: random.PRNGKey,
    create_matrix_fn: Callable[[jnp.ndarray, Tags], AbstractSquareMatrix]
):
    """Test the correctness of a matrix implementation with different tag combinations.

    Args:
        key: Random key
        matrix_class: The matrix class to test
    """
    # Test a subset of tag combinations
    tag_options = [
        TAGS.zero_tags,
        TAGS.no_tags
    ]

    for tag_A in tag_options:
        for tag_B in tag_options:
            k1, k2 = random.split(key)
            key, _ = random.split(key)

            A_raw = random.normal(k1, (6, 6))
            B_raw = random.normal(k2, (6, 6))

            # Modify matrices according to tags
            if tag_A.is_zero:
                A_raw = jnp.zeros_like(A_raw)

            if tag_B.is_zero:
                B_raw = jnp.zeros_like(B_raw)

            A = create_matrix_fn(A_raw, tag_A)
            B = create_matrix_fn(B_raw, tag_B)

            # Run the matrix tests - should not raise exceptions
            matrix_tests(key, A, B)