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
import numpy as np
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap

__all__ = ['Tags', 'TAGS']

class Tags(AbstractBatchableObject):
  """Contains different properties of a matrix.  Knowing these can facilitate more efficient code"""
  is_nonzero: Bool # Use non-zero so that creating a zero matrix can be done with jtu.tree_map(jnp.zeros_like, ...)
  is_inf: Bool

  def __init__(self, is_nonzero: Bool, is_inf: Bool):
    self.is_nonzero = jnp.array(is_nonzero)
    self.is_inf = jnp.array(is_inf)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.is_nonzero.ndim == 0:
      return None
    elif self.is_nonzero.ndim == 1:
      return self.is_nonzero.shape[0]
    else:
      return self.is_nonzero.shape[:-1]

  def __str__(self):
    return f"Tags(is_nonzero={self.is_nonzero}, is_inf={self.is_inf})"

  @property
  def is_zero(self):
    return ~self.is_nonzero

  def add_update(self, update: 'Tags') -> 'Tags':
    """
    Addition (A + B)
        B
        |     =0        ≠0        =∞        ≠∞
    A  |-------------------------------------------------
    =0 | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    ≠0 |  .         (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    =∞ |  .          .         (≠0, =∞)   (≠0, =∞)
    ≠∞ |  .          .          .         (≠0, ≠∞)
    """
    is_nonzero_after_add = self.is_nonzero | update.is_nonzero
    is_inf_after_add = self.is_inf | update.is_inf
    return Tags(is_nonzero_after_add, is_inf_after_add)

  def mat_mul_update(self, update: 'Tags') -> 'Tags':
    """
    Multiplication (A@B)
      B
      |     =0        ≠0        =∞        ≠∞
    A |-------------------------------------------------
    =0 | (=0, ≠∞)   (=0, ≠∞)    ?         (=0, ≠∞)
    ≠0 | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    =∞ |  ?         (≠0, =∞)   (≠0, =∞)   (≠0, =∞)
    ≠∞ | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    """
    is_nonzero_after_mul = self.is_nonzero & update.is_nonzero
    is_inf_after_mul = self.is_inf | update.is_inf
    return Tags(is_nonzero_after_mul, is_inf_after_mul)

  def scalar_mul_update(self) -> 'Tags':
    return self

  def transpose_update(self) -> 'Tags':
    return self

  def solve_update(self, update: 'Tags') -> 'Tags':
    """Solve (A⁻¹ @ B)
        B
        |     =0        ≠0        =∞        ≠∞
    A  |-------------------------------------------------
    =0 | ?          (≠0, =∞)   (≠0, =∞)   (≠0, =∞)
    ≠0 | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    =∞ | (=0, ≠∞)   (=0, ≠∞)    ?         (=0, ≠∞)
    ≠∞ | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    """
    # When A is zero (non-invertible), result should be infinite for nonzero B
    zero_case_inf = self.is_zero & update.is_nonzero

    is_nonzero_after_solve = ~self.is_inf & update.is_nonzero
    is_inf_after_solve = (update.is_inf & ~self.is_inf) | zero_case_inf

    return Tags(is_nonzero_after_solve, is_inf_after_solve)

  def inverse_update(self) -> 'Tags':
    is_nonzero_after_invert = ~self.is_inf
    is_inf_after_invert = self.is_zero
    return Tags(is_nonzero_after_invert, is_inf_after_invert)

  def cholesky_update(self) -> 'Tags':
    is_nonzero_after_cholesky = self.is_nonzero
    is_inf_after_cholesky = self.is_inf
    return Tags(is_nonzero_after_cholesky, is_inf_after_cholesky)

  def exp_update(self) -> 'Tags':
    is_nonzero_after_exp = jnp.ones_like(self.is_nonzero)
    is_inf_after_exp = self.is_inf
    return Tags(is_nonzero_after_exp, is_inf_after_exp)

class TAGS:
  zero_tags = Tags(is_nonzero=np.array(False), is_inf=np.array(False))
  inf_tags = Tags(is_nonzero=np.array(True), is_inf=np.array(True))
  no_tags = Tags(is_nonzero=np.array(True), is_inf=np.array(False))
