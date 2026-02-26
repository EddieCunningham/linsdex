import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Dict, overload, Literal
import einops
import equinox as eqx
import abc
from jaxtyping import Array, PRNGKeyArray, PyTree
import jax.tree_util as jtu
import os
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool
from linsdex import AbstractBatchableObject, auto_vmap
from linsdex import TimeSeries, InterleavedTimes
from linsdex.ssm.simple_encoder import AbstractEncoder
from linsdex.ssm.simple_decoder import AbstractDecoder
from linsdex import AbstractLinearSDE, ConditionedLinearSDE, InterleavedTimes
import linsdex.util as util
from linsdex.crf import Messages, CRF
from linsdex.nn.nn_models.nn_abstract import AbstractNeuralNet

################################################################################################################

class AbstractGenerativeModel(AbstractBatchableObject, abc.ABC):
  """Abstract class for generative models."""
  nn: eqx.AbstractVar[AbstractNeuralNet]

  @abc.abstractmethod
  def loss_fn(
    self,
    key: PRNGKeyArray,
    data: PyTree,
    condition_info: Optional[PyTree] = None,
    debug: Optional[bool] = False
  ) -> Tuple[Scalar, Dict[str, Scalar]]:
    """Compute the loss function from the series.

    Arguments:
      yts: The series to compute the loss function from.  This must NOT be upsampled!
      key: The key to use for the random number generator.
      debug: Whether to enter pdb after computing the loss function for debugging.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def sample(
    self,
    key: PRNGKeyArray,
    condition_info: Optional[PyTree] = None,
    debug: Optional[bool] = False,
    **kwargs
  ) -> Array:
    """Returns a latent series that is sampled from the model.

    Arguments:
      key: The key to use for the random number generator.
      yts: The observed series to sample from.
      debug: Whether to enter pdb after sampling for debugging.
                            the latent encoding of the input time series.
    """
    raise NotImplementedError
