import jax
import jax.numpy as jnp
from jax import random
import pytest

from linsdex.linear_functional.quadratic_form import QuadraticForm, resolve_quadratic_form
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.functional_ops import vdot
from linsdex.matrix.dense import DenseMatrix


def create_quadratic_form(key, dim):
  k1, k2, k3 = random.split(key, 3)
  A = DenseMatrix(random.normal(k1, (dim, dim)))
  b = random.normal(k2, (dim,))
  c = random.normal(k3, ())
  return QuadraticForm(A, b, c)

def create_linear_functional(key, dim):
  k1, k2 = random.split(key)
  A = DenseMatrix(random.normal(k1, (dim, dim)))
  b = random.normal(k2, (dim,))
  return LinearFunctional(A, b)


class TestQuadraticForm:
  key = random.PRNGKey(0)
  dim = 4

  def test_init(self):
    qf = create_quadratic_form(self.key, self.dim)
    assert isinstance(qf, QuadraticForm)
    assert qf.A.shape == (self.dim, self.dim)
    assert qf.b.shape == (self.dim,)
    assert jnp.isscalar(qf.c)
    # check for symmetry
    assert jnp.allclose(qf.A.as_matrix(), qf.A.as_matrix().T)

  def test_call(self):
    qf = create_quadratic_form(self.key, self.dim)
    x = random.normal(self.key, (self.dim,))

    # Manual calculation
    expected = 0.5 * jnp.vdot(x, qf.A @ x) + jnp.vdot(qf.b, x) + qf.c

    # Using the __call__ method
    result = qf(x)

    assert jnp.allclose(result, expected)

  def test_addition(self):
    k1, k2 = random.split(self.key)
    qf1 = create_quadratic_form(k1, self.dim)
    qf2 = create_quadratic_form(k2, self.dim)

    qf_sum = qf1 + qf2

    assert jnp.allclose(qf_sum.A.as_matrix(), (qf1.A + qf2.A).as_matrix())
    assert jnp.allclose(qf_sum.b, qf1.b + qf2.b)
    assert jnp.allclose(qf_sum.c, qf1.c + qf2.c)

    # test with scalar
    scalar = 2.5
    qf_sum_scalar = qf1 + scalar
    assert jnp.allclose(qf_sum_scalar.A.as_matrix(), qf1.A.as_matrix())
    assert jnp.allclose(qf_sum_scalar.b, qf1.b)
    assert jnp.allclose(qf_sum_scalar.c, qf1.c + scalar)

    qf_sum_scalar_r = scalar + qf1
    assert jnp.allclose(qf_sum_scalar_r.A.as_matrix(), qf1.A.as_matrix())
    assert jnp.allclose(qf_sum_scalar_r.b, qf1.b)
    assert jnp.allclose(qf_sum_scalar_r.c, qf1.c + scalar)

  def test_subtraction(self):
    k1, k2 = random.split(self.key)
    qf1 = create_quadratic_form(k1, self.dim)
    qf2 = create_quadratic_form(k2, self.dim)

    qf_sub = qf1 - qf2

    assert jnp.allclose(qf_sub.A.as_matrix(), (qf1.A - qf2.A).as_matrix())
    assert jnp.allclose(qf_sub.b, qf1.b - qf2.b)
    assert jnp.allclose(qf_sub.c, qf1.c - qf2.c)

    # test with scalar
    scalar = 2.5
    qf_sub_scalar = qf1 - scalar
    assert jnp.allclose(qf_sub_scalar.A.as_matrix(), qf1.A.as_matrix())
    assert jnp.allclose(qf_sub_scalar.b, qf1.b)
    assert jnp.allclose(qf_sub_scalar.c, qf1.c - scalar)

    qf_sub_scalar_r = scalar - qf1
    assert jnp.allclose(qf_sub_scalar_r.A.as_matrix(), -qf1.A.as_matrix())
    assert jnp.allclose(qf_sub_scalar_r.b, -qf1.b)
    assert jnp.allclose(qf_sub_scalar_r.c, scalar - qf1.c)

  def test_negation(self):
    qf = create_quadratic_form(self.key, self.dim)
    qf_neg = -qf

    assert jnp.allclose(qf_neg.A.as_matrix(), -qf.A.as_matrix())
    assert jnp.allclose(qf_neg.b, -qf.b)
    assert jnp.allclose(qf_neg.c, -qf.c)

  def test_multiplication(self):
    qf = create_quadratic_form(self.key, self.dim)
    scalar = 2.5

    qf_mul = qf * scalar

    assert jnp.allclose(qf_mul.A.as_matrix(), (scalar * qf.A).as_matrix())
    assert jnp.allclose(qf_mul.b, scalar * qf.b)
    assert jnp.allclose(qf_mul.c, scalar * qf.c)

    qf_mul_r = scalar * qf

    assert jnp.allclose(qf_mul_r.A.as_matrix(), (scalar * qf.A).as_matrix())
    assert jnp.allclose(qf_mul_r.b, scalar * qf.b)
    assert jnp.allclose(qf_mul_r.c, scalar * qf.c)


class TestVdot:
  key = random.PRNGKey(0)
  dim = 4

  def test_vdot_lf_lf(self):
    k1, k2, k_x = random.split(self.key, 3)
    lf1 = create_linear_functional(k1, self.dim)
    lf2 = create_linear_functional(k2, self.dim)
    x = random.normal(k_x, (self.dim,))

    # expected value
    val1 = lf1(x)
    val2 = lf2(x)
    expected = jnp.vdot(val1, val2)

    # actual value
    qf = vdot(lf1, lf2)
    actual = qf(x)

    assert isinstance(qf, QuadraticForm)
    assert jnp.allclose(actual, expected)

  def test_vdot_lf_vec(self):
    k1, k_x = random.split(self.key)
    lf = create_linear_functional(k1, self.dim)
    vec = random.normal(k_x, (self.dim,))
    x = random.normal(self.key, (self.dim,))

    # expected value
    val_lf = lf(x)
    expected = jnp.vdot(val_lf, vec)

    # actual value
    qf = vdot(lf, vec)
    actual = qf(x)

    assert isinstance(qf, QuadraticForm)
    assert jnp.allclose(actual, expected)

class TestResolve:
  key = random.PRNGKey(0)
  dim = 4

  def test_resolve_quadratic_form(self):
    k1, k_x = random.split(self.key)
    qf = create_quadratic_form(k1, self.dim)
    x = random.normal(k_x, (self.dim,))

    tree = {"a": qf, "b": jnp.ones(1)}

    resolved_tree = resolve_quadratic_form(tree, x)

    assert jnp.allclose(resolved_tree["a"], qf(x))
    assert jnp.allclose(resolved_tree["b"], jnp.ones(1))