from typing import Optional
import equinox as eqx
import abc
from jaxtyping import Array, Float, Scalar, PyTree
from linsdex import AbstractBatchableObject
from linsdex import TimeSeries

################################################################################################################

class AbstractHyperParams(eqx.Module, abc.ABC):
  """Data structure that contains the hyperparameters for a model."""
  pass

################################################################################################################

class AbstractNeuralNet(AbstractBatchableObject, abc.ABC):

  hypers: eqx.AbstractVar[AbstractHyperParams]

  @abc.abstractmethod
  def __call__(self,
               x: PyTree,
               condition_info: Optional[PyTree] = None) -> PyTree:
    """Apply the neural network to the x.

    Arguments:
      x: The input to the neural network.
      condition_info: The information to condition on.

    Returns:
      output: The output of the neural network.
    """
    pass

class AbstractTimeDependentNeuralNet(AbstractNeuralNet, abc.ABC):
  """An abstract class for time-dependent neural networks."""

  hypers: eqx.AbstractVar[AbstractHyperParams]

  @abc.abstractmethod
  def __call__(self,
               t: Scalar,
               x: PyTree,
               condition_info: Optional[PyTree] = None) -> PyTree:
    """Apply the time-dependent neural network to the x.

    Arguments:
      t: The time step.
      x: The input to the neural network.
      condition_info: The information to condition on.

    Returns:
      output: The output of the neural network.
    """
    pass

################################################################################################################

class AbstractSeq2SeqModel(AbstractNeuralNet, abc.ABC):

  hypers: eqx.AbstractVar[AbstractHyperParams]

  @abc.abstractmethod
  def create_context(self, condition_info: PyTree) -> PyTree:
    """Create a representation of the condition information for the decoder network.

    Arguments:
      condition_info: The information to condition on.

    Returns:
      context: The context for the decoder network
    """
    pass

  @abc.abstractmethod
  def __call__(self,
               series: TimeSeries,
               condition_info: Optional[PyTree] = None, # Only optional if context is provided
               context: Optional[PyTree] = None) -> TimeSeries:
    """Apply the encoder-decoder model to the series.  We must still pass the
    series to the model even if context is provided because the decoder
    network takes as input the series that is prepended with part of the
    end of series.

    Arguments:
      series: The series to encode.
      condition_info: The information to condition on.
      context: Optional context for the decoder network.  This is so that
      the encoder network can be reused for multiple calls to the decoder network.

    Returns:
      series: The encoded series.
    """
    pass

class AbstractTimeDependentSeq2SeqModel(AbstractTimeDependentNeuralNet, abc.ABC):
  """An abstract class for time-dependent encoder-decoder models."""

  hypers: eqx.AbstractVar[AbstractHyperParams]

  @abc.abstractmethod
  def __call__(self,
               t: Scalar,
               series: TimeSeries,
               condition_info: Optional[PyTree] = None, # Only optional if context is provided
               context: Optional[PyTree] = None) -> TimeSeries:
    """Apply the encoder-decoder model to the series.  We must still pass the
    series to the model even if context is provided because the decoder
    network takes as input the series that is prepended with part of the
    end of series.  This is so that the encoder network can be reused for
    multiple calls to the decoder network.

    Arguments:
      t: The time step.
      series: The series to encode.
      condition_info: The information to condition on.
      context: Optional context for the decoder network.  This is so that
      the encoder network can be reused for multiple calls to the decoder network.

    Returns:
      series: The encoded series.
    """
    pass