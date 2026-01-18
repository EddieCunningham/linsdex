import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import lineax as lx
import abc
import warnings
import jax.tree_util as jtu
from plum import dispatch
import linsdex.util as util
from linsdex.matrix.tags import Tags, TAGS
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.matrix.matrix_base import AbstractSquareMatrix

def matrices_equal(A: Union[AbstractSquareMatrix, Float[Array, 'M N']], B: Union[AbstractSquareMatrix, Float[Array, 'M N']]):
  if isinstance(A, AbstractSquareMatrix):
    Amat = A.as_matrix()
  else:
    Amat = A

  if isinstance(B, AbstractSquareMatrix):
    Bmat = B.as_matrix()
  else:
    Bmat = B

  # Replace any nans with something random (but that stays the same)
  key = random.PRNGKey(12380746)
  if jnp.any(jnp.isnan(Amat)) or jnp.any(jnp.isinf(Amat)):
    Amat = random.normal(key, Amat.shape)
  if jnp.any(jnp.isnan(Bmat)) or jnp.any(jnp.isinf(Bmat)):
    Bmat = random.normal(key, Bmat.shape)

  return jnp.allclose(Amat, Bmat)

def matrix_tests(key, A, B):

  A_dense = A.as_matrix()
  B_dense = B.as_matrix()

  # Check transpose
  if matrices_equal(A.T, A_dense.T) == False:
    raise ValueError(f"Transpose test failed.  Expected {A.T}, got {A_dense.T}")

  # Check addition
  C = A + B
  C_dense = A_dense + B_dense
  if matrices_equal(C, C_dense) == False:
    raise ValueError(f"Addition test failed.  Expected {C}, got {C_dense}")

  # # Check addition with a scalar
  # C = A + 1.0
  # C_dense = A_dense + 1.0
  # if matrices_equal(C, C_dense) == False:
  #   raise ValueError(f"Addition test failed.  Expected {C}, got {C_dense}")

  # Check matrix multiplication
  C = A@B.T
  C_dense = A_dense@B_dense.T
  if matrices_equal(C, C_dense) == False:
    raise ValueError(f"Matrix multiplication test failed.  Expected {C}, got {C_dense}")

  # Check matrix vector products
  x = random.normal(key, (A.shape[1],))
  y = A@x
  y_dense = A_dense@x
  if matrices_equal(y, y_dense) == False:
    raise ValueError(f"Matrix vector product test failed.  Expected {y}, got {y_dense}")

  # Check scalar multiplication
  C = 2.0*A
  C_dense = 2.0*A_dense
  if matrices_equal(C, C_dense) == False:
    raise ValueError(f"Scalar multiplication test failed.  Expected {C}, got {C_dense}")

  if A.shape[0] == A.shape[1]:
    # Check the inverse
    A_inv = A.get_inverse()

    if A.is_inf:
      assert jnp.all(A_inv.is_zero)
    elif A.is_zero:
      assert jnp.all(A_inv.is_inf)
    else:
      A_inv_dense = jnp.linalg.inv(A_dense)
      if matrices_equal(A_inv, A_inv_dense) == False:
        raise ValueError(f"Matrix inverse test failed.  Expected {A_inv}, got {A_inv_dense}")

    # Check solve
    x = random.normal(key, (A.shape[1],))
    y = A.solve(x)
    if A.is_inf:
      pass
    elif A.is_zero:
      pass # Don't check this
    else:
      y_dense = A_inv_dense@x
      if matrices_equal(y, y_dense) == False:
        raise ValueError(f"Matrix solve test failed.  Expected {y}, got {y_dense}")

  # Check the cholesky decomposition
  J = A@A.T
  J_chol = J.get_cholesky()._force_fix_tags()
  J_dense = J.as_matrix()
  J_chol_dense = jnp.linalg.cholesky(J_dense)
  if J.is_zero:
    J_chol_dense = jnp.zeros_like(J_chol_dense)
  if matrices_equal(J_chol, J_chol_dense) == False:
    raise ValueError(f"Cholesky decomposition test failed.  Expected {J_chol}, got {J_chol_dense}")

  # Check the log determinant
  log_det = J.get_log_det()
  log_det_dense = jnp.linalg.slogdet(J_dense)[1]
  if matrices_equal(log_det, log_det_dense) == False:
    raise ValueError(f"Log determinant test failed.  Expected {log_det}, got {log_det_dense}")

  # Check the trace
  trace = J.get_trace()
  trace_dense = jnp.trace(J_dense, axis1=-1, axis2=-2)
  if matrices_equal(trace, trace_dense) == False:
    raise ValueError(f"Trace test failed.  Expected {trace}, got {trace_dense}")

  # Check the SVD
  (U, s, V) = J.get_svd()
  U_dense, s_dense, V_dense = jnp.linalg.svd(J_dense)

  # SVD is not unique due to potential permutations, sign flips, and ordering
  # Instead of direct comparison, verify reconstruction and orthogonality

  if A.is_inf:
    return

  # Verify singular values match (regardless of order)
  s_values = jnp.diag(s.as_matrix())
  if not jnp.allclose(jnp.sort(jnp.abs(s_values)), jnp.sort(jnp.abs(s_dense)), atol=1e-5):
    raise ValueError(f"SVD test failed: singular values don't match")

  # Verify matrices are orthogonal (U*U^T = I, V*V^T = I)
  U_mat = U.as_matrix()
  V_mat = V.as_matrix()
  if not jnp.allclose(U_mat @ U_mat.T, jnp.eye(U_mat.shape[0]), atol=1e-5):
    raise ValueError(f"SVD test failed: U is not orthogonal")
  if not jnp.allclose(V_mat @ V_mat.T, jnp.eye(V_mat.shape[0]), atol=1e-5):
    raise ValueError(f"SVD test failed: V is not orthogonal")

  # Verify reconstruction: A = U*S*V^T
  reconstruction = U_mat @ jnp.diag(s_values) @ V_mat.T
  if not jnp.allclose(reconstruction, J_dense, atol=1e-5):
    raise ValueError(f"SVD test failed: reconstruction doesn't match original matrix")

def performance_tests(A, B):
  # Basic operations
  C1 = A + B
  C1 = jtu.tree_map(lambda x: x.block_until_ready(), C1)
  C2 = C1 - B
  C2 = jtu.tree_map(lambda x: x.block_until_ready(), C2)
  C3 = 2.0 * C2
  C3 = jtu.tree_map(lambda x: x.block_until_ready(), C3)
  C4 = C3 / 2.0
  C4 = jtu.tree_map(lambda x: x.block_until_ready(), C4)
  C5 = C4 @ B
  C5 = jtu.tree_map(lambda x: x.block_until_ready(), C5)
  C6 = C5.T
  C6 = jtu.tree_map(lambda x: x.block_until_ready(), C6)

  # Single matrix operations
  C7 = C6.get_inverse()
  C7 = jtu.tree_map(lambda x: x.block_until_ready(), C7)
  C8 = C7.get_cholesky()
  C8 = jtu.tree_map(lambda x: x.block_until_ready(), C8)

  # Matrix-vector operations
  x = jnp.ones(A.shape[1])
  y = C8 @ x
  z = C8.solve(x).reshape(-1, 1) @ y.reshape(1, -1)  # Outer product to get matrix
  z = jtu.tree_map(lambda x: x.block_until_ready(), z)  # Force computation to complete

  # Get log determinant (scalar) and convert back to matrix
  log_det = C8.get_log_det()
  log_det = jtu.tree_map(lambda x: x.block_until_ready(), log_det)