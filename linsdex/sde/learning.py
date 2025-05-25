import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Annotated
import einops
import equinox as eqx
from abc import ABC, abstractmethod
import diffrax
from jaxtyping import Array, PRNGKeyArray, Float, Scalar
from jax._src.util import curry
import abc
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix, TAGS
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
import linsdex.util as util
import warnings
import lineax as lx
from linsdex.potential.gaussian.transition import *
from linsdex.potential.gaussian.dist import StandardGaussian, NaturalGaussian
from linsdex.matrix.matrix_with_inverse import MatrixWithInverse
from linsdex.sde.sde_base import AbstractLinearTimeInvariantSDE, AbstractLinearSDE
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.block.block_3x3 import Block3x3Matrix
from linsdex.sde.sde_examples import LinearTimeInvariantSDE

__all__ = ['max_likelihood_ltisde']

################################################################################################################

def max_likelihood_ltisde(xts: Float[Array, 'B N D'], dt: Scalar) -> LinearTimeInvariantSDE:
  """Compute the maximum likelihood parameters for the SDE given a batch of data that is sampled uniformly
  in time with step size dt."""
  if xts.ndim == 2:
    xts = xts[None]
  assert xts.ndim == 3

  # Compute the covariance of the data
  xt_xtT = jnp.einsum('bni,bnj->ij', xts[:,1:], xts[:,1:])
  xt_xtm1T = jnp.einsum('bni,bnj->ij', xts[:,1:], xts[:,:-1])
  A = xt_xtm1T@jnp.linalg.inv(xt_xtT)

  xts_diff = xts[:,1:] - jnp.einsum('ij,bnj->bni', A, xts[:,:-1])
  Sigma = jnp.einsum('bni,bnj->ij', xts_diff, xts_diff) / xts.shape[1]

  # Compute the block matrix
  AinvT = jnp.linalg.inv(A).T
  upper_left = A
  upper_right = Sigma@AinvT
  lower_left = jnp.zeros_like(upper_right)
  lower_right = AinvT
  block_matrix = jnp.block([[upper_left, upper_right], [lower_left, lower_right]])

  # Compute the matrix logarithm of the block matrix
  import scipy.linalg
  Psi = scipy.linalg.logm(block_matrix)/dt
  Psi = Psi.real

  D = xts.shape[2]
  F = Psi[:D,:D] # Top left
  LLT = Psi[:D,D:] # Top right
  negF = Psi[D:,D:] # Bottom right

  L = jnp.linalg.cholesky(LLT)

  return LinearTimeInvariantSDE(DenseMatrix(F, tags=TAGS.no_tags), DenseMatrix(L, tags=TAGS.no_tags))

