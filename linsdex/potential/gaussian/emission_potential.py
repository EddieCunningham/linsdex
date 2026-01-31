import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, PyTree, Scalar, Bool
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import warnings
import abc
from linsdex.potential.gaussian.dist import MixedGaussian
from plum import dispatch

def make_emission_potential(H: Float[Array, 'Dy Dx'],
                            v: Float[Array, 'Dy'],
                            R: Union[Float[Array, 'Dy Dy'], Float[Array, 'Dy']],
                            y: Float[Array, 'Dy'],
                            mask: Optional[Bool[Array, 'Dy']] = None) -> MixedGaussian:
  r"""Make a node potential that corresponds to the emission distribution
  \phi(x) \propto N(y|Hx + v, R)

  **Arguments**
  - `H`: Emission matrix
  - `v`: Emission vector
  - `R`: Emission covariance matrix
  - `y`: Observed value
  - `mask`: Mask for the observed value

  **Returns**
  A mixed Gaussian that represents the emission potential
  """

  assert y.ndim == 1
  using_mask = True
  if mask is None:
    mask = jnp.ones(y.shape[-1], dtype=bool)
    using_mask = False

  H = H*mask[:,None]

  y_ = v - y
  y_ = jnp.where(mask, y_, 0.0)

  if R.ndim == 2:
    Rinv = jnp.linalg.inv(R)
    Rinv = Rinv*mask[:, None]
    Rinv = Rinv*mask[None, :]
    J = H.T@Rinv@H

    logZ = 0.5*jnp.vdot(Rinv@y_, y_)
    logZ += 0.5*jnp.linalg.slogdet(R)[1]
    if using_mask:
      warnings.warn("The normalizing constant of an emission potential with full covariance cannot be masked correctly")
  elif R.ndim == 1:
    Rinv = 1/R
    Rinv = Rinv*mask
    J = H.T@(H*Rinv[:,None])
  else:
    raise ValueError(f"R must be a 1D or 2D array, got {R.ndim}D array")

  # HTR_inv = H.T@Rinv
  # HTR_inv = R.T.solve(A).T
  # h = HTR_inv@y_
  mu = H.T@y_
  logZ += 0.5*mask.sum()*jnp.log(2*jnp.pi)

  return MixedGaussian(mu, J, logZ)