import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
from linsdex.matrix.tags import Tags, TAGS
from .shared import matrices_equal, matrix_tests, performance_tests
import numpy as np
from linsdex.matrix.matrix_base import AbstractSquareMatrix

def create_tag(is_zero: bool, is_inf: bool) -> Tags:
  assert isinstance(is_zero, bool)
  assert isinstance(is_inf, bool)
  return Tags(not is_zero, is_inf)

def assert_tags_equal(tag1: Tags, tag2: Tags):
  assert tag1.is_zero == tag2.is_zero, f"Zero mismatch: {tag1} vs {tag2}"
  assert tag1.is_inf == tag2.is_inf, f"Inf mismatch: {tag1} vs {tag2}"

def test_add_update():
  # Create all combinations
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)
  noninf = create_tag(is_zero=False, is_inf=False)

  # Test cases from the table
  test_cases = [
    # (A, B, expected_result)
    (zero, zero, create_tag(is_zero=True, is_inf=False)),
    (zero, nonzero, create_tag(is_zero=False, is_inf=False)),
    (zero, inf, create_tag(is_zero=False, is_inf=True)),
    (zero, noninf, create_tag(is_zero=False, is_inf=False)),

    (nonzero, nonzero, create_tag(is_zero=False, is_inf=False)),
    (nonzero, inf, create_tag(is_zero=False, is_inf=True)),
    (nonzero, noninf, create_tag(is_zero=False, is_inf=False)),

    (inf, inf, create_tag(is_zero=False, is_inf=True)),
    (inf, noninf, create_tag(is_zero=False, is_inf=True)),

    (noninf, noninf, create_tag(is_zero=False, is_inf=False)),
  ]

  for A, B, expected in test_cases:
    result = A.add_update(B)
    assert_tags_equal(result, expected)

def test_mat_mul_update():
  # Create all combinations
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)
  noninf = create_tag(is_zero=False, is_inf=False)

  test_cases = [
    # (A, B, expected_result)
    (zero, zero, create_tag(is_zero=True, is_inf=False)),
    (zero, nonzero, create_tag(is_zero=True, is_inf=False)),
    # (zero, inf, None),  # Undefined case
    (zero, noninf, create_tag(is_zero=True, is_inf=False)),

    (nonzero, zero, create_tag(is_zero=True, is_inf=False)),
    (nonzero, nonzero, create_tag(is_zero=False, is_inf=False)),
    (nonzero, inf, create_tag(is_zero=False, is_inf=True)),
    (nonzero, noninf, create_tag(is_zero=False, is_inf=False)),

    # (inf, zero, None),  # Undefined case
    (inf, nonzero, create_tag(is_zero=False, is_inf=True)),
    (inf, inf, create_tag(is_zero=False, is_inf=True)),
    (inf, noninf, create_tag(is_zero=False, is_inf=True)),

    (noninf, zero, create_tag(is_zero=True, is_inf=False)),
    (noninf, nonzero, create_tag(is_zero=False, is_inf=False)),
    (noninf, inf, create_tag(is_zero=False, is_inf=True)),
    (noninf, noninf, create_tag(is_zero=False, is_inf=False)),
  ]

  for A, B, expected in test_cases:
    result = A.mat_mul_update(B)
    assert_tags_equal(result, expected)

def test_solve_update():
  # Create all combinations
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)
  noninf = create_tag(is_zero=False, is_inf=False)

  test_cases = [
    # (A, B, expected_result)
    # (zero, zero, None),  # Undefined case
    (zero, nonzero, create_tag(is_zero=False, is_inf=True)),
    (zero, inf, create_tag(is_zero=False, is_inf=True)),
    (zero, noninf, create_tag(is_zero=False, is_inf=True)),

    (nonzero, zero, create_tag(is_zero=True, is_inf=False)),
    (nonzero, nonzero, create_tag(is_zero=False, is_inf=False)),
    (nonzero, inf, create_tag(is_zero=False, is_inf=True)),
    (nonzero, noninf, create_tag(is_zero=False, is_inf=False)),

    (inf, zero, create_tag(is_zero=True, is_inf=False)),
    (inf, nonzero, create_tag(is_zero=True, is_inf=False)),
    # (inf, inf, None),  # Undefined case
    (inf, noninf, create_tag(is_zero=True, is_inf=False)),

    (noninf, zero, create_tag(is_zero=True, is_inf=False)),
    (noninf, nonzero, create_tag(is_zero=False, is_inf=False)),
    (noninf, inf, create_tag(is_zero=False, is_inf=True)),
    (noninf, noninf, create_tag(is_zero=False, is_inf=False)),
  ]

  for A, B, expected in test_cases:
    result = A.solve_update(B)
    assert_tags_equal(result, expected)

def test_inverse_update():
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)

  # Test inverse of zero -> inf
  result = zero.inverse_update()
  expected = create_tag(is_zero=False, is_inf=True)
  assert_tags_equal(result, expected)

  # Test inverse of nonzero -> nonzero
  result = nonzero.inverse_update()
  expected = create_tag(is_zero=False, is_inf=False)
  assert_tags_equal(result, expected)

  # Test inverse of inf -> zero
  result = inf.inverse_update()
  expected = create_tag(is_zero=True, is_inf=False)
  assert_tags_equal(result, expected)

def test_transpose_update():
  # Tags should remain unchanged after transpose
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)

  assert_tags_equal(zero.transpose_update(), zero)
  assert_tags_equal(nonzero.transpose_update(), nonzero)
  assert_tags_equal(inf.transpose_update(), inf)

def test_scalar_mul_update():
  # Tags should remain unchanged after scalar multiplication
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)

  assert_tags_equal(zero.scalar_mul_update(), zero)
  assert_tags_equal(nonzero.scalar_mul_update(), nonzero)
  assert_tags_equal(inf.scalar_mul_update(), inf)

def test_exp_update():
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)

  # Test exp of zero -> nonzero (exp(0) = 1)
  result = zero.exp_update()
  expected = create_tag(is_zero=False, is_inf=False)
  assert_tags_equal(result, expected)

  # Test exp of nonzero -> nonzero
  result = nonzero.exp_update()
  expected = create_tag(is_zero=False, is_inf=False)
  assert_tags_equal(result, expected)

  # Test exp of inf -> inf
  result = inf.exp_update()
  expected = create_tag(is_zero=False, is_inf=True)
  assert_tags_equal(result, expected)

def test_cholesky_update():
  zero = create_tag(is_zero=True, is_inf=False)
  nonzero = create_tag(is_zero=False, is_inf=False)
  inf = create_tag(is_zero=False, is_inf=True)

  # Cholesky decomposition should preserve tags
  assert_tags_equal(zero.cholesky_update(), zero)
  assert_tags_equal(nonzero.cholesky_update(), nonzero)
  assert_tags_equal(inf.cholesky_update(), inf)

def test_is_zero_property():
  zero = Tags(is_nonzero=np.array(False), is_inf=np.array(False))
  nonzero = Tags(is_nonzero=np.array(True), is_inf=np.array(False))

  assert zero.is_zero == np.array(True)
  assert nonzero.is_zero == np.array(False)

def test_tags_constants():
  # Test that TAGS constants have correct values
  assert TAGS.zero_tags.is_nonzero == np.array(False)
  assert TAGS.zero_tags.is_inf == np.array(False)

  assert TAGS.inf_tags.is_nonzero == np.array(True)
  assert TAGS.inf_tags.is_inf == np.array(True)
