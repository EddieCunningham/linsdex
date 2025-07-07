import jax
import jax.numpy as jnp
from plum import dispatch
from jaxtyping import Array, Float

from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.quadratic_form import QuadraticForm

__all__ = ['vdot']

@dispatch
def vdot(a, b):
  """Computes the vector dot product. Falls back to jnp.vdot by default."""
  return jnp.vdot(a, b)

@dispatch
def vdot(a: LinearFunctional, b: LinearFunctional) -> QuadraticForm:
  """Computes the dot product of two LinearFunctionals.

  (A1*x + b1)^T (A2*x + b2) = x^T*A1^T*A2*x + (A1^T*b2 + A2^T*b1)^T*x + b1^T*b2
  This is represented as a QuadraticForm: 0.5*x^T*A*x + b^T*x + c
  """
  A = a.A.T @ b.A + b.A.T @ a.A
  b_vec = a.A.T @ b.b + b.A.T @ a.b
  c_scalar = jnp.vdot(a.b, b.b)
  return QuadraticForm(A, b_vec, c_scalar)

@dispatch
def vdot(a: LinearFunctional, b: Float[Array, 'D']) -> QuadraticForm:
  """Computes the dot product of a LinearFunctional and a vector."""
  A = a.A.zeros_like(a.A)
  b_vec = a.A.T @ b
  c_scalar = jnp.vdot(a.b, b)
  return QuadraticForm(A, b_vec, c_scalar)

@dispatch
def vdot(a: Float[Array, 'D'], b: LinearFunctional) -> QuadraticForm:
  """Computes the dot product of a vector and a LinearFunctional."""
  return vdot(b, a)