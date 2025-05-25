import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Literal
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree, Int
import abc
import jax.tree_util as jtu
import jax.random as random
from plum import dispatch
import inspect
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap

__all__ = [
    'AbstractPotential',
    'AbstractTransition',
    'Interval',
    'JointPotential'
]

################################################################################################################

class AbstractPotential(AbstractBatchableObject, abc.ABC):
  """Abstract base class for potential functions.  These are basically exponential family distributions.

  A potential function represents unnormalized probability distributions over
  variables in a graphical model. Potentials can be combined through addition,
  representing the product of the corresponding distributions in log-space.

  The AbstractPotential class defines the core interface for all potentials
  in the library, supporting operations like evaluation, combination,
  sampling, and computing normalizing constants.

  Potentials form the building blocks of nodes in Conditional Random Fields
  and are used to represent distributions over variables.
  """

  @abc.abstractmethod
  def __call__(self, x: PyTree) -> Scalar:
    """Evaluate the (log) potential at a given value of x.

    This method computes the unnormalized log-probability of a value x.

    Args:
      x: A value at which to evaluate the potential

    Returns:
      A scalar value representing the log-potential at x
    """
    pass

  def __add__(self, other: 'AbstractPotential') -> 'AbstractPotential':
    """Add two potentials, representing multiplication of distributions.

    Args:
      other: Another potential to add to this one

    Returns:
      A new potential representing the sum (in log-space) of this
      potential and the other
    """
    pass

  @abc.abstractmethod
  def normalizing_constant(self) -> Scalar:
    """Compute the log of the partition function for the potential.

    The partition function (or normalizing constant) converts the
    unnormalized potential into a proper probability distribution.

    Returns:
      A scalar value representing the log of the normalizing constant
    """
    pass

  @abc.abstractmethod
  def log_prob(self, x: PyTree) -> Scalar:
    """Evaluate the log probability of x under the normalized distribution.

    This computes log(p(x)) = potential(x) - normalizing_constant().

    Args:
      x: A PyTree representing the value to evaluate

    Returns:
      A scalar value representing the log probability of x
    """
    pass

  @abc.abstractmethod
  def sample(self, key: PRNGKeyArray) -> PyTree:
    """Sample from the distribution represented by this potential.

    Args:
      key: A PRNG key for random number generation

    Returns:
      A PyTree sample from the distribution
    """
    pass

################################################################################################################

class AbstractTransition(AbstractBatchableObject, abc.ABC):
  """Abstract base class for transition potentials between variables.

  A transition potential represents a conditional distribution p(y|x) between
  two variables in a graphical model. In the context of Conditional Random Fields,
  transitions represent the dependencies between adjacent nodes in the sequence.

  This class provides the interface for all transition potentials, including
  operations for message passing, marginalization, conditioning, and chaining
  transitions together. These operations form the basis of inference algorithms
  in graphical models.
  """

  @abc.abstractmethod
  def __call__(self, y: PyTree, x: PyTree) -> Scalar:
    """Evaluate the log potential of a transition from x to y.

    Args:
      y: The target variable
      x: The source variable

    Returns:
      A scalar representing the log of the transition potential
    """
    pass

  @abc.abstractmethod
  def swap_variables(self) -> 'AbstractTransition':
    """Create a new transition with x and y variables swapped.

    Returns:
      A new AbstractTransition representing p(x|y) instead of p(y|x)
    """
    pass

  def unnormalized_update_y(self, potential: AbstractPotential) -> 'AbstractTransition':
    """Incorporate a potential over y into the transition potential.

    This version returns only the transition part of the result.

    Args:
      potential: A potential function over the y variable

    Returns:
      A modified transition incorporating the potential
    """
    return self.update_y(potential, True) # No keyword arguments because of dispatch

  @abc.abstractmethod
  def marginalize_out_y(self) -> AbstractPotential:
    """Marginalize out y from the transition potential.

    Computes ∫ p(y|x) dy as a potential over x.

    Returns:
      A potential over x representing the marginalization
    """
    pass

  @abc.abstractmethod
  def update_y(self, potential: AbstractPotential, only_return_transition: bool = False) -> Union['JointPotential', 'AbstractTransition']:
    """Incorporate a potential over y into the transition.

    This operation combines the transition p(y|x) with a potential on y,
    resulting in either a joint potential p(y,x) or a modified transition.

    Args:
      potential: A potential function over the y variable
      only_return_transition: If True, return only the transition part

    Returns:
      Either a JointPotential or just the modified transition
    """
    pass

  def update_and_marginalize_out_y(self, potential: AbstractPotential) -> AbstractPotential:
    """Update with a y potential and then marginalize out y.

    This is a common operation in message passing algorithms that
    computes ∫ p(y|x) ψ(y) dy as a potential over x.

    Args:
      potential: A potential function over the y variable

    Returns:
      A potential over x after updating and marginalizing
    """
    out1 = self.update_y(potential)
    return out1.marginalize_out_y()

  def update_and_marginalize_out_x(self, potential: AbstractPotential) -> AbstractPotential:
    """Update with an x potential and then marginalize out x.

    Computes ∫ p(y|x) ψ(x) dx as a potential over y.

    Args:
      potential: A potential function over the x variable

    Returns:
      A potential over y after updating and marginalizing
    """
    return self.swap_variables().update_and_marginalize_out_y(potential)

  @abc.abstractmethod
  def condition_on_x(self, x: PyTree) -> AbstractPotential:
    """Condition the transition on a specific value of x.

    Args:
      x: The value to condition on

    Returns:
      A potential over y representing p(y|x=x₀)
    """
    pass

  def condition_on_y(self, y: PyTree) -> AbstractPotential:
    """Condition the transition on a specific value of y.

    Args:
      y: The value to condition on

    Returns:
      A potential over x representing p(x|y=y₀)
    """
    return self.swap_variables().condition_on_x(y)

  @abc.abstractmethod
  def chain(self, other: 'AbstractTransition') -> 'AbstractTransition':
    """Chain two transitions together.

    Creates a new transition that represents applying this transition
    followed by the other transition. This operation is critical for
    message passing in graphical models.

    Args:
      other: Another transition to chain with this one

    Returns:
      A new transition representing the chained operation
    """
    pass

  @auto_vmap
  def zero_message_like(self, potential: AbstractPotential) -> AbstractPotential:
    """Initialize an uninformative (zero) message of the appropriate type.

    Creates a zero-information potential that is compatible with the
    provided potential in terms of type and structure.

    Args:
      potential: A template potential to match type with

    Returns:
      A zero-information potential of the same type
    """
    out_type = self.update_y(potential).prior
    return out_type.total_uncertainty_like(out_type)

################################################################################################################

class Interval(AbstractBatchableObject):
  """Represents an interval or segment with start, end, and length attributes.

  This class is used to represent time intervals or segments in the sequence
  for bookkeeping during operations like marginalization and chaining.

  Attributes:
    start: The starting index of the interval
    end: The ending index of the interval
    length: The length of the interval
  """
  start: Int
  end: Int
  length: Int

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    """Get the batch dimensions of this interval.

    Returns:
      Batch size information extracted from the start index dimensions

    Raises:
      ValueError: If start has an invalid number of dimensions
    """
    if self.start.ndim > 1:
      return self.start.shape[:-1]
    elif self.start.ndim == 1:
      return self.start.shape[0]
    elif self.start.ndim == 0:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.start.ndim}")

  def chain(self, other: 'Interval') -> 'Interval':
    """Chain two intervals to create a new interval.

    Creates a new interval that spans from this interval's start
    to the other interval's end.

    Args:
      other: Another interval to chain with this one

    Returns:
      A new interval spanning both intervals
    """
    return Interval(self.start, other.end, self.length + other.length)

class JointPotential(AbstractPotential):
  """Represents a joint distribution over two variables p(y, x).

  A JointPotential combines a transition potential p(y|x) with a prior
  potential p(x) to form a joint distribution p(y,x) = p(y|x)p(x).

  This class is critical for message passing algorithms as it represents
  the combination of node and transition potentials in a graphical model.
  It supports marginalization operations that are fundamental to inference.

  Attributes:
    transition: The conditional distribution p(y|x)
    prior: The potential over x (p(x))
    interval: Bookkeeping information about the interval this potential covers
  """

  transition: AbstractTransition
  prior: AbstractPotential
  interval: Interval

  def __init__(self, transition: AbstractTransition, prior: AbstractPotential):
    """Initialize a joint potential from a transition and prior.

    Args:
      transition: A conditional distribution p(y|x)
      prior: A potential over x
    """
    self.transition = transition
    self.prior = prior
    self.interval = Interval(jnp.array(0), jnp.array(1), jnp.array(1))

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    """Get the batch dimensions of this joint potential.

    Returns:
      The batch size from the prior potential
    """
    return self.prior.batch_size

  def __call__(self, y: PyTree, x: PyTree) -> Scalar:
    """Evaluate the log of the joint potential at (y, x).

    Computes log p(y,x) = log p(y|x) + log p(x).

    Args:
      y: The y value to evaluate
      x: The x value to evaluate

    Returns:
      A scalar representing the log joint potential
    """
    return self.transition(y, x) + self.prior(x)

  def log_prob(self, y: PyTree, x: PyTree) -> Scalar:
    """Evaluate the log probability of (y, x) under the joint distribution.

    Computes the normalized log probability log(p(y,x)).

    Args:
      y: The y value to evaluate
      x: The x value to evaluate

    Returns:
      A scalar representing the log probability
    """
    return self.prior.log_prob(x) + self.transition.condition_on_x(x).log_prob(y)

  def normalizing_constant(self) -> Scalar:
    """Compute the log normalizing constant of the joint distribution.

    Returns:
      The log of the partition function
    """
    return self.prior.normalizing_constant()

  def sample(self, key: PRNGKeyArray) -> Tuple[PyTree, PyTree]:
    """Sample from the joint distribution p(y,x).

    Samples x ~ p(x) followed by y ~ p(y|x).

    Args:
      key: A PRNG key for random number generation

    Returns:
      A tuple (y, x) sampled from the joint distribution
    """
    k1, k2 = random.split(key)
    x = self.prior.sample(k1)
    y = self.transition.condition_on_x(x).sample(k2)
    return y, x

  def marginalize_out_y(self) -> AbstractPotential:
    """Marginalize out y from the joint potential.

    Computes p(x) = ∫ p(y,x) dy.

    Returns:
      A potential over x representing the marginal distribution
    """
    marginal = self.transition.marginalize_out_y()
    # This order ensures that the return type is self.prior's type!
    out = self.prior + marginal
    return out

  def marginalize_out_x(self) -> AbstractPotential:
    """Marginalize out x from the joint potential.

    Computes p(y) = ∫ p(y,x) dx.

    Returns:
      A potential over y representing the marginal distribution
    """
    swapped = self.transition.swap_variables()
    new_prior = swapped.update_y(self.prior)
    return new_prior.marginalize_out_y()

  @auto_vmap
  def update_y(self, potential: AbstractPotential) -> 'JointPotential':
    """Incorporate a potential over y into the joint potential.

    Creates a new joint potential that includes an additional
    factor ψ(y), resulting in p'(y,x) ∝ p(y,x)ψ(y).

    Args:
      potential: A potential function over y

    Returns:
      A new joint potential with the y potential included
    """
    updated_joint = self.transition.update_y(potential)
    return JointPotential(updated_joint.transition, updated_joint.prior + self.prior)

  @auto_vmap
  def update_x(self, potential: AbstractPotential) -> 'JointPotential':
    """Incorporate a potential over x into the joint potential.

    Creates a new joint potential that includes an additional
    factor ψ(x), resulting in p'(y,x) ∝ p(y,x)ψ(x).

    Args:
      potential: A potential function over x

    Returns:
      A new joint potential with the x potential included
    """
    return JointPotential(self.transition, self.prior + potential)

  @auto_vmap
  def condition_on_x(self, x: PyTree) -> AbstractPotential:
    """Condition the joint distribution on a specific value of x.

    Args:
      x: The value to condition on

    Returns:
      A potential over y representing p(y|x=x₀)
    """
    return self.transition.condition_on_x(x)

  @auto_vmap
  def chain(self, other: 'JointPotential') -> 'JointPotential':
    """Chain two joint potentials together.

    This operation is fundamental for parallel message passing algorithms,
    allowing the combination of potentials across segments of the model.

    Args:
      other: Another joint potential to chain with this one

    Returns:
      A new joint potential representing the chained potential
    """
    updated_self = self.update_y(other.prior)
    new_transition = updated_self.transition.chain(other.transition)
    new_joint = JointPotential(new_transition, updated_self.prior)
    # Update interval for bookkeeping
    new_interval = self.interval.chain(other.interval)
    return eqx.tree_at(lambda x: x.interval, new_joint, new_interval)

################################################################################################################
