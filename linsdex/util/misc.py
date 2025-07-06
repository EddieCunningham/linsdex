import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import jax.tree_util as jtu
from jax._src.util import curry
from plum import dispatch, ModuleType
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
# from linsdex.potential.gaussian.dist import NaturalGaussian, StandardGaussian, MixedGaussian

def psd_check(J: "AbstractSquareMatrix"):
  # J = eqx.error_if(J, jnp.any(jnp.linalg.eigvalsh(J.as_matrix()) < 0), "Matrix must be positive definite")
  # J = eqx.error_if(J, jnp.any(J.as_matrix() - J.T.as_matrix() != 0), "Matrix must be symmetric for real!")
  # J = eqx.error_if(J, ~J.tags.is_symmetric, "Matrix must be symmetric")
  return J

def inverse_check(A: "AbstractSquareMatrix"):
  # A = eqx.error_if(A, A.tags.is_zero, "Cannot invert a zero matrix")
  return A

################################################################################################################

def matrix_sqrt(mat: Float[Array, 'D D']) -> Float[Array, 'D D']:
  eigvals, eigvecs = jnp.linalg.eigh(mat)
  return eigvecs@jnp.diag(jnp.sqrt(eigvals))@eigvecs.T

def empirical_dist(xts: Float[Array, 'N D']) -> 'AbstractSquareMatrix':
  from linsdex.potential.gaussian.dist import StandardGaussian
  from linsdex.matrix import DenseMatrix
  from linsdex.matrix.tags import TAGS
  mu = xts.mean(axis=0)
  cov = jnp.einsum('bi,bj->ij', xts - mu, xts - mu)/xts.shape[0] + 1e-10*jnp.eye(xts.shape[1])
  cov = DenseMatrix(cov, tags=TAGS.no_tags)
  return StandardGaussian(mu, cov)

def w2_distance(gaussian1: 'AbstractSquareMatrix',
                gaussian2: 'AbstractSquareMatrix') -> Scalar:
  """Compute the Wasserstein-2 distance between two Gaussians"""
  try:
    gaussian1 = gaussian1.to_std()
  except:
    pass
  try:
    gaussian2 = gaussian2.to_std()
  except:
    pass
  cov1 = gaussian1.Sigma.as_matrix()
  cov2 = gaussian2.Sigma.as_matrix()
  mu1 = gaussian1.mu
  mu2 = gaussian2.mu

  cov1_sqrt = matrix_sqrt(cov1)

  term = cov1_sqrt@cov2@cov1_sqrt
  term = term + jnp.eye(cov1.shape[0])*1e-10
  cov_term_sqrt = matrix_sqrt(term)

  return jnp.sum((mu1 - mu2)**2) + jnp.trace(cov1 + cov2 - 2*cov_term_sqrt)

################################################################################################################

def where(cond: Bool, true: PyTree, false: PyTree) -> Any:
  # return jax.lax.cond(cond, lambda: true, lambda: false)
  return jtu.tree_map(lambda x, y: jnp.where(cond, x, y), true, false)

################################################################################################################

def fill_array(buffer: Float[Array, 'T D'], i: Any, value: Union[Float[Array, 'D'], Float[Array, 'K D']]):
  return jtu.tree_map(lambda t, elt: t.at[i].set(elt), buffer, value)

################################################################################################################
from plum import dispatch

@dispatch
def _to_matrix(A: Float[Array, 'D'], symmetric: bool = False):
  from linsdex.matrix import DiagonalMatrix, ParametricSymmetricDiagonalMatrix
  from linsdex.matrix.tags import TAGS
  if symmetric:
    return ParametricSymmetricDiagonalMatrix(A)
  else:
    return DiagonalMatrix(A, tags=TAGS.no_tags)

@dispatch
def _to_matrix(A: Float[Array, 'D D'], symmetric: bool = False):
  from linsdex.matrix import DenseMatrix, ParametricSymmetricDenseMatrix
  from linsdex.matrix.tags import TAGS
  if symmetric:
    return ParametricSymmetricDenseMatrix(A)
  else:
    return DenseMatrix(A, tags=TAGS.no_tags)

@dispatch
def _to_matrix(A: Float[Array, '2 2 D'], symmetric: bool = False):
  from linsdex.matrix import Diagonal2x2BlockMatrix, ParametricSymmetricDiagonal2x2BlockMatrix
  from linsdex.matrix.tags import TAGS
  if symmetric:
    return ParametricSymmetricDiagonal2x2BlockMatrix(A)
  else:
    return Diagonal2x2BlockMatrix(A, tags=TAGS.no_tags)

@dispatch
def _to_matrix(A: Float[Array, '3 3 D'], symmetric: bool = False):
  from linsdex.matrix import Diagonal3x3BlockMatrix, ParametricSymmetricDiagonal3x3BlockMatrix
  from linsdex.matrix.tags import TAGS
  if symmetric:
    return ParametricSymmetricDiagonal3x3BlockMatrix(A)
  else:
    return Diagonal3x3BlockMatrix(A, tags=TAGS.no_tags)

def to_matrix(A: Union[Float[Array, 'D'],
                       Float[Array, 'D D'],
                       Float[Array, '2 2 D'],
                       Float[Array, '3 3 D']], symmetric: bool = False) -> 'AbstractSquareMatrix':
  return _to_matrix(A, symmetric)

################################################################################################################

def get_times_to_interleave_for_upsample(ts: Float[Array, 'N'],
                                         n_points_to_add_inbetween: int) -> Float[Array, 'N * (n_points_to_add_inbetween + 1)']:
  """Get the times to interleave for upsampling a time series"""
  # Construct n_points_to_add_inbetween points in between each point
  assert ts.ndim == 1
  dts = jnp.diff(ts)
  dts = jnp.concatenate([dts, dts[-1:]])
  offsets = dts[:,None]*jnp.arange(1, (n_points_to_add_inbetween + 2))/(n_points_to_add_inbetween + 1)
  new_times = ts[:,None] + dts[:,None]*offsets
  return new_times[...,:-1].ravel()

################################################################################################################

def tree_shapes(tree: PyTree) -> Tuple[Tuple[int]]:
  """Get the shapes of a tree"""
  return jtu.tree_map(lambda x: x.shape, tree)

