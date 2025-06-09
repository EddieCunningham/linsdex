import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree, Int
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject, auto_vmap
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.util.parallel_scan import parallel_segmented_scan, parallel_scan, segmented_scan
from jax._src.util import curry
import linsdex.util as util
import warnings

__all__ = ['CRF', 'Messages']

class Messages(AbstractBatchableObject):
  """Container for forward and backward messages used in CRF inference.

  This class provides a convenient way to store and manage message passing
  results, allowing for efficient reuse of pre-computed messages.

  Attributes:
    fwd: Forward messages (can be None if not yet computed)
    bwd: Backward messages (can be None if not yet computed)
  """
  fwd: Union[AbstractPotential, None]
  bwd: Union[AbstractPotential, None]

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    """Get the batch size of the messages.

    Returns:
      Batch size of the messages, or None if both fwd and bwd are None.

    Raises:
      ValueError: If one message is None but the other is not.
    """
    if self.fwd is None and self.bwd is None:
      return None
    elif self.fwd is None:
      return self.bwd.batch_size
    elif self.bwd is None:
      return self.fwd.batch_size
    else:
      raise ValueError("Both fwd and bwd must be None or both must be non-None")

  def set_fwd(self, fwd: AbstractPotential):
    """Create a new Messages object with updated forward messages.

    Args:
      fwd: New forward messages

    Returns:
      New Messages instance with updated forward messages
    """
    return Messages(fwd, self.bwd)

  def set_bwd(self, bwd: AbstractPotential):
    """Create a new Messages object with updated backward messages.

    Args:
      bwd: New backward messages

    Returns:
      New Messages instance with updated backward messages
    """
    return Messages(self.fwd, bwd)

  @classmethod
  def from_messages(
    cls,
    messages: Union['Messages', None],
    crf: 'CRF',
    need_fwd: bool = False,
    need_bwd: bool = False
  ):
    """Create or update Messages instance, computing any needed messages.

    This factory method ensures that all required messages (forward and/or
    backward) are available, computing them if necessary.

    Args:
      messages: Existing Messages object or None
      crf: CRF to compute messages for (if needed)
      need_fwd: Whether forward messages are needed
      need_bwd: Whether backward messages are needed

    Returns:
      Messages object with required messages populated
    """
    fwd, bwd = None, None
    if messages is None:
      if need_fwd:
        fwd = crf.get_forward_messages()
      if need_bwd:
        bwd = crf.get_backward_messages()
      return cls(fwd, bwd)
    else:
      if need_fwd:
        if messages.fwd is None:
          fwd = crf.get_forward_messages()
        else:
          fwd = messages.fwd
      if need_bwd:
        if messages.bwd is None:
          bwd = crf.get_backward_messages()
        else:
          bwd = messages.bwd
      return cls(fwd, bwd)

################################################################################################################

def pytree_has_nan(x: PyTree) -> bool:
  return not jtu.tree_all(jtu.tree_map(lambda x: (~jnp.isnan(x)).all(), x))

def debug_crf(crf: 'CRF'):
  # Start with the last node's potential
  message = crf.base_transitions[-1].update_and_marginalize_out_y(crf.node_potentials[-1])
  if pytree_has_nan(message):
    import pdb; pdb.set_trace()
  zero_message = message.total_uncertainty_like(message)
  messages = [zero_message, message]

  # Loop backwards through remaining nodes
  for i in range(len(crf)-2, 0, -1):
    updated_message = message + crf.node_potentials[i]
    if pytree_has_nan(updated_message):
      import pdb; pdb.set_trace()
    message = crf.base_transitions[i-1].update_and_marginalize_out_y(updated_message)
    if pytree_has_nan(message):
      import pdb; pdb.set_trace()
    messages.append(message)

  # Reverse the list and stack all messages together
  messages = messages[::-1]
  messages = jtu.tree_map(lambda *xs: jnp.stack(xs), *messages)
  return messages

class CRF(AbstractPotential):
  """Conditional Random Field (CRF) implementation for sequential latent variable models.

  A CRF represents a probabilistic graphical model with node potentials at each timestep
  and transition potentials between adjacent timesteps. This implementation supports
  various forms of message passing and inference operations.

  Attributes:
    node_potentials: Potential functions for individual nodes in the CRF
    base_transitions: Transition functions between adjacent nodes
    parallel: Whether to use parallel message passing algorithms
    max_unroll_length: Maximum length for sequential unrolling instead of scan
  """

  node_potentials: AbstractPotential
  base_transitions: AbstractTransition

  parallel: bool = eqx.field(static=True)
  max_unroll_length: Optional[int] = 3

  def __init__(
    self,
    node_potentials: AbstractPotential,
    base_transitions: AbstractTransition,
    parallel: Optional[bool] = None,
    max_unroll_length: Optional[int] = 3
  ):
    """Initialize a Conditional Random Field.

    Args:
      node_potentials: Potential functions for each node in the CRF
      base_transitions: Transition functions between adjacent nodes
      parallel: Whether to use parallel message passing (defaults to True on GPU)
      max_unroll_length: Maximum length for which to use sequential unrolling

    Raises:
      ValueError: If CRF has fewer than 2 nodes
      AssertionError: If node_potentials and base_transitions have inconsistent sizes
    """
    assert node_potentials.batch_size == base_transitions.batch_size + 1
    self.node_potentials = node_potentials
    self.base_transitions = base_transitions
    if parallel is None:
      parallel = jax.devices()[0].platform == 'gpu'
    self.parallel = parallel
    self.max_unroll_length = max_unroll_length

    if len(self) <= 1:
      raise ValueError("CRF must have at least 2 nodes")

  @classmethod
  def total_certainty_like(cls, x: Float[Array, 'D'], other: 'AbstractPotential') -> 'AbstractPotential':
    raise NotImplementedError

  @classmethod
  def total_uncertainty_like(cls, other: 'AbstractPotential') -> 'AbstractPotential':
    raise NotImplementedError

  def __getitem__(self, idx: Any):
    """Get a slice of the CRF.  We need to ensure that the size of the base transitions is
    always one less than that of the node potentials
    """
    # Get the slice for node_potentials
    node_potentials = self.node_potentials[idx]

    # Create appropriate slice for transitions
    if isinstance(idx, slice):
      # For slice objects, adjust the stop index if provided
      start = idx.start if idx.start is not None else 0
      stop = idx.stop if idx.stop is not None else len(self)
      step = idx.step if idx.step is not None else 1
      # Transitions slice should be one less than nodes
      trans_slice = slice(start, stop-1, step)
      base_transitions = self.base_transitions[trans_slice]
    elif isinstance(idx, int):
      raise ValueError("Cannot take a single index of a CRF")
    else:
      # raise NotImplementedError(f"Unsupported index type: {type(idx)}")
      # For other types (lists, boolean masks, etc.),
      # we need to ensure transitions are one less
      node_len = node_potentials.batch_size
      if node_len < 2:
        raise ValueError("Cannot take a slice of a CRF with fewer than 2 nodes")
      trans_idx = idx[:-1] if node_len > 1 else idx
      base_transitions = self.base_transitions[trans_idx]

    # Create new CRF with sliced components
    return CRF(
      node_potentials=node_potentials,
      base_transitions=base_transitions,
      parallel=self.parallel,
      max_unroll_length=self.max_unroll_length
    )

  def reverse(self):
    """Create a new CRF with nodes and transitions in reverse order.

    This method creates a new CRF where the sequence order is reversed.
    This is particularly useful for computing forward messages by reusing
    backward message passing logic.

    Returns:
      A new CRF with reversed ordering of nodes and transitions.
    """
    # Reverse the node potentials and base transitions
    reversed_node_potentials = jtu.tree_map(lambda x: x[::-1], self.node_potentials)
    reversed_base_transitions = jtu.tree_map(lambda x: x[::-1], self.base_transitions)
    reversed_base_transitions = eqx.filter_vmap(lambda x: x.swap_variables())(reversed_base_transitions)
    reversed = CRF(reversed_node_potentials, reversed_base_transitions, parallel=self.parallel)
    return reversed

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    node_batch_size = self.node_potentials.batch_size
    if isinstance(node_batch_size, Iterable):
      return node_batch_size[-2]
    elif isinstance(node_batch_size, int):
      return None
    else:
      raise ValueError(f"Invalid batch size: {node_batch_size}")

  def __len__(self):
    node_batch_size = self.node_potentials.batch_size
    if isinstance(node_batch_size, int):
      return node_batch_size
    elif isinstance(node_batch_size, Iterable):
      return node_batch_size[-1]
    else:
      raise ValueError(f"Invalid batch size: {node_batch_size}")

  def to_prior_and_chain(self, *, messages: Optional[Messages] = None) -> Tuple[AbstractPotential, AbstractTransition]:
    """Convert the CRF to its canonical form consisting of a prior and transitions.

    Transforms the CRF into the canonical representation of a single prior distribution
    for the first node and a set of transition functions between nodes.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      A tuple of (prior, transitions) where:
        - prior: Potential representing the marginal of the first node
        - transitions: Transition functions between adjacent nodes incorporating
          the original transitions and node potentials
    """
    messages = Messages.from_messages(messages, self, need_bwd=True)
    bwd = messages.bwd
    prior = bwd[0] + self.node_potentials[0]
    transitions = self.get_transitions(messages=messages)
    return prior, transitions

  def to_canonical_form(self, *, messages: Optional[Messages] = None) -> 'CRF':
    """Convert the CRF to its canonical form with a prior and transitions.

    Creates a new CRF in canonical form, where all node potentials except the first
    are replaced with uninformative potentials, and transitions incorporate the
    information from the original node potentials.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      A new CRF in canonical form with only a prior at the first node and transitions.
    """
    # Get the prior and transitions between the remaining nodes
    messages = Messages.from_messages(messages, self, need_bwd=True)
    bwd = messages.bwd
    prior = bwd[0] + self.node_potentials[0]

    # Create new node potentials that are no-ops everywhere except at the first node
    zero = prior.total_uncertainty_like(prior)
    def make_new_potential(_):
      return zero
    new_node_potentials = jax.vmap(make_new_potential)(self.node_potentials)
    new_node_potentials = util.fill_array(new_node_potentials, jnp.array([0]), prior)

    # Get the transitions between the remaining nodes
    transitions = self.get_transitions(messages=messages)

    # Create the new CRF
    return CRF(new_node_potentials, transitions, parallel=self.parallel, max_unroll_length=self.max_unroll_length)

  def get_marginal_log_likelihood(self, *, messages: Optional[Messages] = None) -> Scalar:
    """Calculate the log likelihood of the marginal distribution.

    Computes the log of the integral over all possible configurations in the CRF.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      The log likelihood as a scalar value.
    """
    messages = Messages.from_messages(messages, self, need_bwd=True)
    first_message = messages.bwd[0]
    return (first_message + self.node_potentials[0]).integrate()

  def normalizing_constant(self, *, messages: Optional[Messages] = None) -> Scalar:
    """Get the normalizing constant (partition function) for the distribution.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      The log of the normalizing constant as a scalar value.
    """
    return self.get_marginal_log_likelihood(messages=messages)

  @auto_vmap
  def __call__(self, xs: PyTree, *, messages: Optional[Messages] = None) -> Scalar:
    """Compute the normalized log probability of a sequence of values.

    Args:
      xs: Sequence of values to evaluate
      messages: Pre-computed messages for efficiency

    Returns:
      Log probability of the given sequence
    """
    return self.log_prob(xs, messages=messages)

  def get_marginals(self, *, messages: Optional[Messages] = None) -> AbstractPotential:
    """Calculate the marginal distributions for each node in the CRF.

    Computes p(x_i) for each node i by combining forward and backward messages
    with node potentials.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      Potential representing the marginal distribution at each node.
    """
    messages = Messages.from_messages(messages, self, need_fwd=True, need_bwd=True)
    fwd, bwd = messages.fwd, messages.bwd
    return fwd + bwd + self.node_potentials

  def sample(self, key: PRNGKeyArray, *, messages: Optional[Messages] = None) -> PyTree:
    """Generate samples from the CRF distribution.

    Samples a sequence of values from the distribution represented by the CRF.
    For Gaussian transitions with parallel=True, uses an optimized parallel sampling
    algorithm. Otherwise, uses sequential forward sampling.

    Args:
      key: Random key for generating samples
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      A sequence of samples, one for each node in the CRF.
    """
    # Get the messages
    messages = Messages.from_messages(messages, self, need_bwd=True)
    bwd = messages.bwd

    # Get the transitions
    smoothed_transitions = self.get_transitions(messages=messages)

    keys = random.split(key, bwd.batch_size)

    # Sample the first point
    first_message = bwd[0] + self.node_potentials[0]
    x0 = first_message.sample(keys[0])

    if isinstance(smoothed_transitions, GaussianTransition) and self.parallel:
      # If Gaussian chain, then we can do parallel sampling
      from linsdex.potential.gaussian.transition import gaussian_chain_parallel_sample
      return gaussian_chain_parallel_sample(smoothed_transitions, x0, keys[1:])

    else:
      # Otherwise, do sequential sampling
      def forward_sampling(carry, inputs):
        xt = carry
        transition, key = inputs

        xtp1 = transition.condition_on_x(xt).sample(key)
        return xtp1, xtp1

      # Run the sequential sampling
      xT, xts = jax.lax.scan(forward_sampling, x0, (smoothed_transitions, keys[1:]))

      # Concatenate the first point
      return jtu.tree_map(lambda x0, x: jnp.concatenate([x0[None], x]), x0, xts)

  def get_transitions(self, *, messages: Optional[Messages] = None) -> AbstractTransition:
    """Get the smoothed transitions for the CRF incorporating backward messages.

    Computes the transition distributions p(x_{t+1}|x_t) that take into account
    both the original transitions and the information from future observations
    via backward messages.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      Smoothed transition potentials between adjacent nodes.
    """
    messages = Messages.from_messages(messages, self, need_bwd=True)
    bwd = messages.bwd
    bwd_node = bwd + self.node_potentials # p(y_{k:N}|x_k)

    def update_transition(bwd_node_kp1, transition) -> AbstractTransition:
      return transition.unnormalized_update_y(bwd_node_kp1) # Only returns the transition

    return jax.vmap(update_transition)(bwd_node[1:], self.base_transitions)

  def get_joints(self, *, messages: Optional[Messages] = None) -> AbstractPotential:
    """Calculate the joint distributions between consecutive pairs of nodes.

    Computes p(x_{k+1},x_k) for each adjacent pair of nodes by combining
    both forward and backward messages with transition potentials.

    Args:
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      Joint potential representing the pairwise distributions between adjacent nodes.
    """
    messages = Messages.from_messages(messages, self, need_fwd=True, need_bwd=True)
    bwd = messages.bwd
    bwd_node = bwd + self.node_potentials # p(y_{k:N}|x_k)

    def update_transition(bwd_node_kp1, transition) -> JointPotential:
      return transition.update_y(bwd_node_kp1) # We need the right normalizing constant

    transitions = jax.vmap(update_transition)(bwd_node[1:], self.base_transitions)

    # Need to update the transitions with the forward node potentials
    def update_transition_x(fwd_node_k, joint) -> JointPotential:
      return joint.update_x(fwd_node_k)

    fwd_node = messages.fwd + self.node_potentials
    joints = jax.vmap(update_transition_x)(fwd_node[:-1], transitions)
    return joints

  def get_forward_messages(self) -> AbstractPotential:
    """Calculate forward messages (alpha values) for inference in the CRF.

    Computes the forward messages by reversing the CRF, computing backward messages,
    and then reversing the result again. This approach leverages the symmetry between
    forward and backward message passing.

    Returns:
      Forward messages for each node in the CRF.
    """
    return self.reverse().get_backward_messages()[::-1]

  def get_backward_messages(self) -> AbstractPotential:
    """Calculate the backward messages (beta values) for inference in the CRF.

    Computes p(y_{k:N}|x_k) using message passing. Uses one of three implementations:
    1. Direct unrolled loop for short sequences (length <= max_unroll_length)
    2. Parallel message passing if parallel=True
    3. Sequential message passing otherwise

    The unrolled approach can be more efficient for short sequences, especially
    in matching applications.

    Returns:
      Backward messages for each node in the CRF.
    """
    if len(self) <= self.max_unroll_length:
      # Start with the last node's potential
      message = self.base_transitions[-1].update_and_marginalize_out_y(self.node_potentials[-1])
      zero_message = message.total_uncertainty_like(message)
      messages = [zero_message, message]

      # Loop backwards through remaining nodes
      for i in range(len(self)-2, 0, -1):
        updated_message = message + self.node_potentials[i]
        message = self.base_transitions[i-1].update_and_marginalize_out_y(updated_message)
        messages.append(message)

      # Reverse the list and stack all messages together
      messages = messages[::-1]
      messages = jtu.tree_map(lambda *xs: jnp.stack(xs), *messages)
      return messages

    if self.parallel:
      return self.parallel_bwd_messages()
    else:
      return self.sequential_bwd_messages()

  def sequential_bwd_messages(self) -> AbstractPotential:
    """Calculate backward messages using sequential scanning approach.

    Implements the backward pass of message passing using jax.lax.scan,
    which processes nodes sequentially from the end to the beginning.

    Returns:
      Backward messages for each node in the CRF.
    """
    def backward_step(carry, inputs):
      message = carry
      node_potential, transition_potential = inputs

      updated_potential = node_potential + message
      new_message = transition_potential.update_and_marginalize_out_y(updated_potential)
      return new_message, new_message

    zero_message = self.base_transitions[0].zero_message_like(self.node_potentials[0])

    # Perform message passing
    _, messages = jax.lax.scan(
        backward_step,
        zero_message,
        (self.node_potentials[1:], self.base_transitions),
        reverse=True
    )

    # Pad with a zero message to the end
    messages = jtu.tree_map(lambda x0, x: jnp.concatenate([x, x0[None]]), zero_message, messages)
    return messages

  def parallel_fwd_messages(self, return_joints: bool = False) -> AbstractPotential:
    """Calculate forward messages using parallel scanning approach.

    Similar to get_forward_messages(), but specifically uses the parallel
    implementation of message passing.

    Args:
      return_joints: If True, returns the joint potentials instead of marginalizing.

    Returns:
      Forward messages for each node in the CRF, or joint potentials if return_joints=True.
    """
    return self.reverse().parallel_bwd_messages(return_joints=return_joints)[::-1]

  def parallel_bwd_messages(self, return_joints: bool = False) -> AbstractPotential:
    """Calculate backward messages using parallel scanning approach.

    Implements the backward pass of message passing using parallel scan,
    which can be more efficient on parallel hardware like GPUs.

    This computes p(x_N | x_k, y_{k+1:N}). The backward messages can be
    evaluated by marginalizing out x.

    Args:
      return_joints: If True, returns the joint potentials instead of marginalizing.

    Returns:
      Backward messages for each node in the CRF, or joint potentials if return_joints=True.
    """
    # Construct the transition joints that will be used in the parallel scan
    def update_transition(node_potential, transition):
      transition_joint = JointPotential(transition, zero_message)
      return transition_joint.update_y(node_potential)

    zero_message = self.base_transitions[0].zero_message_like(self.node_potentials[0])
    joints = jax.vmap(update_transition)(self.node_potentials[1:], self.base_transitions)

    # Perform the parallel scan
    def operator(left: AbstractTransition, right: AbstractTransition) -> AbstractTransition:
      return left.chain(right)

    # Must run in reverse. operator is not commutative, so reverse=False gives wrong answer
    merged_joints = parallel_scan(operator, joints, reverse=True)

    if return_joints:
      return merged_joints

    messages = merged_joints.marginalize_out_y()
    messages = jtu.tree_map(lambda x0, x: jnp.concatenate([x, x0[None]]), zero_message, messages)
    return messages

  @auto_vmap
  def log_prob(self, xs: PyTree, *, messages: Optional[Messages] = None) -> Scalar:
    """Calculate the log probability of a sequence under the CRF distribution.

    Computes log p(x_1, x_2, ..., x_T) for the given sequence using the chain rule:
    log p(x_1) + sum_{t=1}^{T-1} log p(x_{t+1}|x_t)

    Args:
      xs: Sequence of values to evaluate
      messages: Pre-computed messages for efficiency. If None, they will be computed.

    Returns:
      Log probability of the sequence as a scalar.

    Raises:
      AssertionError: If sequence length doesn't match CRF length.
    """
    assert xs.shape[0] == len(self), "xs must have the same number of points as nodes in the CRF"

    # Get the messages
    messages = Messages.from_messages(messages, self, need_bwd=True)
    bwd = messages.bwd
    bwd_node = bwd + self.node_potentials

    # Get the transitions
    smoothed_transitions = self.get_transitions(messages=messages)

    def log_likelihood(xt, xtp1, transition):
      pxtp1 = transition.condition_on_x(xt)
      return pxtp1.log_prob(xtp1)

    # First node probability
    log_p0 = bwd_node[0].log_prob(xs[0])

    # Transition probabilities for remaining nodes
    log_likelihoods = jax.vmap(log_likelihood)(xs[:-1], xs[1:], smoothed_transitions)

    return log_p0 + log_likelihoods.sum()

  def marginalize(self, keep_indices: Int[Array, 'K']) -> 'CRF':
    """Marginalize the CRF by keeping only the nodes at the specified indices.

    This performs marginalization by first converting to canonical form and then
    using segmented scan operations to efficiently marginalize out the unneeded nodes.

    Args:
      keep_indices: Indices of the nodes to keep in the marginalized CRF.

    Returns:
      A new CRF containing only the specified nodes with correct marginal distributions.

    Notes:
      - This implementation adds a dummy node at the beginning to handle the case
        of marginalizing out the first node.
      - There are open TODOs regarding potentially more efficient implementations
        and ensuring correctness of the log normalizer terms.
    """
    ##################################
    # Add a dummy node at the beginning so that we don't need a special case for if we want to
    # marginalize out the first node.
    ##################################
    # Add in a zero node and transition at the beginning
    zero = self.node_potentials[0].total_uncertainty_like(self.node_potentials[0])
    noop = self.base_transitions[0].no_op_like(self.base_transitions[0])

    potentials = jax.vmap(lambda i: zero)(jnp.arange(len(self) + 1))
    base_transitions = jax.vmap(lambda i: noop)(jnp.arange(len(self)))

    # Add in the zero node and transition at the beginning
    potentials = util.fill_array(potentials, slice(1, None), self.node_potentials)
    base_transitions = util.fill_array(base_transitions, slice(1, None), self.base_transitions)
    extended_crf = CRF(potentials, base_transitions, parallel=self.parallel, max_unroll_length=self.max_unroll_length)

    # Also add one to all of the keep_indices.  This is because we added a zero node at the beginning
    keep_indices = keep_indices + 1

    ##################################
    # Perform marginalization by chaining
    ##################################
    # Turn this into a canonical form.  The canonical form has only a prior over the
    # first node and transitions between the remaining nodes
    crf = extended_crf.to_canonical_form()
    potentials = crf.node_potentials
    base_transitions = crf.base_transitions
    zero = potentials[0].total_uncertainty_like(potentials[0])

    # Construct the transition joints that will be used in the parallel scan
    def update_transition(node_potential, transition):
      return JointPotential(transition, node_potential)
    joints = jax.vmap(update_transition)(potentials[:-1], base_transitions)

    # Merge the joints together
    def operator(left: JointPotential, right: JointPotential) -> JointPotential:
      return left.chain(right)

    # Get a mask that will tell us when to reset the scan
    reset_mask = (jnp.arange(len(crf) - 1)[:,None] == keep_indices[None,:]).sum(axis=-1)
    if self.parallel:
      merged_joints = parallel_segmented_scan(operator, joints, reset_mask, reverse=False)
    else:
      merged_joints = segmented_scan(operator, joints, reset_mask, reverse=False)

    filtered_joints = merged_joints[keep_indices - 1]
    marginal_transitions = filtered_joints.transition

    # The new node potentials are the prior of the merged joints, with 0 at the end
    marginal_node_potentials = jax.vmap(lambda i: zero)(jnp.arange(marginal_transitions.batch_size + 1))
    marginal_node_potentials = util.fill_array(marginal_node_potentials, 0, potentials[0])

    # Get the marginal CRF
    marginal_crf = CRF(marginal_node_potentials, marginal_transitions, parallel=self.parallel, max_unroll_length=self.max_unroll_length)

    ##################################
    # Finally, we need to marginalize out the first node
    ##################################
    new_prior = JointPotential(marginal_crf.base_transitions[0], marginal_crf.node_potentials[0]).marginalize_out_x()
    new_node_potentials = util.fill_array(marginal_crf.node_potentials[1:], 0, new_prior)
    new_transitions = marginal_crf.base_transitions[1:]
    out = CRF(new_node_potentials, new_transitions, parallel=self.parallel, max_unroll_length=self.max_unroll_length)

    warnings.warn('Not sure if the log normalizers in the CRF are correct after crf.marginalize!')
    return out

  def marginalize_and_make_prior_and_chain(self, keep_indices: Int[Array, 'K']) -> Tuple[AbstractPotential, AbstractTransition]:
    """Marginalize the CRF and return the resulting prior and transition chain.

    This is a convenience method that calls marginalize() and then extracts
    the prior and transition chain from the resulting CRF.

    Args:
      keep_indices: Indices of the nodes to keep in the marginalized CRF.

    Returns:
      A tuple containing (prior, chain) where:
        - prior: The potential for the first node in the marginalized CRF
        - chain: The transition functions for the marginalized CRF
    """
    crf = self.marginalize(keep_indices)
    prior = crf.node_potentials[0]
    chain = crf.base_transitions
    return prior, chain

################################################################################################################

from linsdex.potential.gaussian.dist import *
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.matrix import DenseMatrix, TAGS

def ssm_to_gaussian(crf: CRF) -> NaturalGaussian:
  transition_nat = crf.base_transitions.to_nat().swap_variables()

  if isinstance(crf.node_potentials, StandardGaussian):
    node_potentials_nat = crf.node_potentials.to_nat()
  else:
    node_potentials_nat = crf.node_potentials

  N = len(crf)
  D = crf.base_transitions.A.shape[1]

  # Upper diagonal of the tridiagonal block matrix
  upper_diagonal = transition_nat.J12.as_matrix()
  diagonal = jnp.zeros((N, D, D))
  diagonal = diagonal.at[:-1].add(transition_nat.J11.as_matrix())
  diagonal = diagonal.at[1:].add(transition_nat.J22.as_matrix())
  diagonal = diagonal + node_potentials_nat.J.as_matrix()

  # Construct the tridiagonal block matrix
  N, D = upper_diagonal.shape[0] + 1, upper_diagonal.shape[1]
  J = jnp.zeros((N * D, N * D))

  # Fill the diagonal blocks
  for i in range(N):
    J = J.at[i*D:(i+1)*D, i*D:(i+1)*D].set(diagonal[i])

  # Fill the upper diagonal blocks
  for i in range(N - 1):
    J = J.at[i*D:(i+1)*D, (i+1)*D:(i+2)*D].set(upper_diagonal[i])

  # Fill the lower diagonal blocks (transpose of upper diagonal)
  for i in range(N - 1):
    J = J.at[(i+1)*D:(i+2)*D, i*D:(i+1)*D].set(upper_diagonal[i].T)

  J = DenseMatrix(J, tags=TAGS.no_tags)

  h = jnp.zeros((N, D))
  h = h.at[:-1].add(transition_nat.h1)
  h = h.at[1:].add(transition_nat.h2)
  h = h + node_potentials_nat.h
  h = einops.rearrange(h, 'N D -> (N D)')

  logZ = crf.node_potentials.logZ.sum() + crf.base_transitions.logZ.sum()

  return NaturalGaussian(J, h, logZ)

def brute_force_marginals(crf: CRF) -> AbstractPotential:
  """Compute the marginals by brute force"""
  dist = ssm_to_gaussian(crf)
  x_dim = crf.base_transitions.A.shape[1]

  marginals = []
  for i in range(len(crf)):
    joint = dist.to_joint(dim=x_dim) # P(y,x) where y represents :x_dim and x represents :x_dim, which is the remaining nodes
    marginal = joint.marginalize_out_x()
    marginals.append(marginal)

    # For the next iteration
    dist = joint.marginalize_out_y()

  # Stack all of the marginals
  marginals = jtu.tree_map(lambda *xs: jnp.array(xs), *marginals)
  return marginals

def brute_force_joints(crf: CRF) -> AbstractPotential:
  """Compute the joints by brute force"""
  dist = ssm_to_gaussian(crf)
  x_dim = crf.base_transitions.A.shape[1]

  joints = []
  for i in range(len(crf) - 1):
    joint = dist.to_joint(dim=2*x_dim)
    marginal = joint.marginalize_out_x()
    import pdb; pdb.set_trace()
    joints.append(marginal)

    # For the next iteration
    joint = dist.to_joint(dim=x_dim) # P(y,x) where y represents :x_dim and x represents :x_dim, which is the remaining nodes
    marginal = joint.marginalize_out_x()
    dist = joint.marginalize_out_y()

  # Stack all of the joints
  joints = jtu.tree_map(lambda *xs: jnp.array(xs), *joints)
  return joints

################################################################################################################
################################################################################################################
################################################################################################################

def crf_tests(crf: CRF):
  if isinstance(crf.node_potentials, StandardGaussian):
    node_potentials_std = crf.node_potentials
    node_potentials_nat = crf.node_potentials.to_nat()
    node_potentials_mix = crf.node_potentials.to_mixed()
  elif isinstance(crf.node_potentials, MixedGaussian):
    node_potentials_mix = crf.node_potentials
    node_potentials_nat = crf.node_potentials.to_nat()
    node_potentials_std = crf.node_potentials.to_std()
  else:
    node_potentials_nat = crf.node_potentials
    node_potentials_std = crf.node_potentials.to_std()
    node_potentials_mix = crf.node_potentials.to_mixed()

  crf_nat = CRF(node_potentials_nat, crf.base_transitions, parallel=crf.parallel)
  crf_std = CRF(node_potentials_std, crf.base_transitions, parallel=crf.parallel)
  crf_mix = CRF(node_potentials_mix, crf.base_transitions, parallel=crf.parallel)

  #############################################
  # Check that the CRF has stable gradients
  #############################################
  def log_prob(crf, xts):
    return crf.log_prob(xts)
  xts = crf_nat.sample(key)
  log_prob_grad = eqx.filter_grad(log_prob)(crf_nat, xts + 10)

  #############################################
  # Check that the zeros and infs work fine
  #############################################
  # See if zero potentials work the same in each parametrization
  zero_nat = node_potentials_nat[0].total_uncertainty_like(node_potentials_nat[0])
  zero_std = node_potentials_std[0].total_uncertainty_like(node_potentials_std[0])
  zero_mix = node_potentials_mix[0].total_uncertainty_like(node_potentials_mix[0])

  # Try adding zero to a potential
  potential_nat = zero_nat + node_potentials_nat[-1]
  potential_std = zero_std + node_potentials_std[-1]
  potential_mix = zero_mix + node_potentials_mix[-1]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat.to_std(), potential_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat.to_std(), potential_mix.to_std()))

  potential_nat = node_potentials_nat[-1] + zero_nat
  potential_std = node_potentials_std[-1] + zero_std
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat.to_std(), potential_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat.to_std(), potential_mix.to_std()))

  # Try updating y using a zero potential
  transition_nat = transitions[-1].update_y(zero_nat)
  transition_std = transitions[-1].update_y(zero_std)
  transition_mix = transitions[-1].update_y(zero_mix)
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, transition_nat.transition, transition_std.transition))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, transition_nat.transition, transition_mix.transition))

  potential_deterministic = potential_std.make_deterministic()
  out = transitions[-1].update_y(potential_deterministic)

  #############################################
  # Check the functionality of the CRF
  #############################################
  # Try updating y using a message passing update
  transition_nat = transitions[-1].update_y(potential_nat)
  transition_std = transitions[-1].update_y(potential_std)
  transition_mix = transitions[-1].update_y(potential_mix)
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, transition_nat.transition, transition_std.transition))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, transition_nat.transition, transition_mix.transition))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, transition_nat.prior.to_std(), transition_std.prior))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, transition_nat.prior.to_mixed(), transition_mix.prior))

  # Check that update and marginalize out y is the same
  potential_nat2 = transitions[-1].update_and_marginalize_out_y(potential_nat)
  potential_std2 = transitions[-1].update_and_marginalize_out_y(potential_std)
  potential_mix2 = transitions[-1].update_and_marginalize_out_y(potential_mix)
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat2.to_std(), potential_std2))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat2.to_std(), potential_mix2.to_std()))

  # Check that the backward messages are the same
  bwd_nat = crf_nat.get_backward_messages()[:-1]
  bwd_std = crf_std.get_backward_messages()[:-1]
  bwd_mix = crf_mix.get_backward_messages()[:-1]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_mix.to_std()))

  # Check that the parallel backward messages are the same
  bwd_nat = crf_nat.get_backward_messages()[:-1]
  bwd_std = crf_std.get_backward_messages()[:-1]
  bwd_mix = crf_mix.get_backward_messages()[:-1]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_mix.to_std()))

  # Check that the backward messages are the same in the sequential and parallel versions
  bwd_nat = crf_nat.get_backward_messages()[:-1]
  bwd_std = crf_std.get_backward_messages()[:-1]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_mix.to_std()))

  # Check that the forward messages are the same
  fwd_nat = crf_nat.get_forward_messages()[1:]
  fwd_std = crf_std.get_forward_messages()[1:]
  fwd_mix = crf_mix.get_forward_messages()[1:]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_mix.to_std()))

  # Check that the parallel forward messages are the same
  fwd_nat = crf_nat.parallel_fwd_messages()[1:]
  fwd_std = crf_std.parallel_fwd_messages()[1:]
  fwd_mix = crf_mix.parallel_fwd_messages()[1:]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_mix.to_std()))

  # Check that the forward messages are the same in the sequential and parallel versions
  fwd_nat = crf_nat.get_forward_messages()[1:]
  fwd_std = crf_std.get_forward_messages()[1:]
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_std))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_mix.to_std()))

  #############################################
  # Check that the CRF has the correct marginals
  #############################################
  true_marginals = brute_force_marginals(crf_nat)
  marginals_nat = crf_nat.get_marginals()
  marginals_std = crf_std.get_marginals()
  marginals_mix = crf_mix.get_marginals()

  assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_nat, true_marginals))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_std.to_nat(), true_marginals))
  assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_mix.to_nat(), true_marginals))

  #############################################
  # Check that the CRF has the correct joints
  #############################################
  x_dim = crf_nat.base_transitions.A.shape[1]
  joints_nat = crf_nat.get_joints()
  joints_std = crf_std.get_joints()
  joints_mix = crf_mix.get_joints()

  def check_joint(joint):
    marginals_check1 = joint.marginalize_out_y().to_nat()
    marginals_check2 = joint.marginalize_out_x().to_nat()
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_check1, true_marginals[:-1]))
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_check2, true_marginals[1:]))

  check_joint(joints_nat)
  check_joint(joints_std)
  check_joint(joints_mix)

  #############################################
  # Check that the CRF has the correct log prob
  #############################################
  crf_gaussian = ssm_to_gaussian(crf_nat)
  xts = crf_nat.sample(key)
  log_prob1 = crf_gaussian.log_prob(xts.ravel())
  log_prob2 = crf_nat.log_prob(xts)
  log_prob3 = crf_std.log_prob(xts)
  log_prob4 = crf_mix.log_prob(xts)
  assert jnp.allclose(log_prob1, log_prob2)
  assert jnp.allclose(log_prob1, log_prob3)
  assert jnp.allclose(log_prob1, log_prob4)


  print("All tests passed!")

################################################################################################################

def compare_trees(A: PyTree, B: PyTree, atol: float = 1e-4):
  params, static = eqx.partition(A, eqx.is_inexact_array)
  params_B, static_B = eqx.partition(B, eqx.is_inexact_array)
  return jtu.tree_all(jtu.tree_map(partial(jnp.allclose, atol=atol), params, params_B))

def edge_case_test(crf: CRF):
  # Check to see that the CRF can handle edge cases with fully certain/uncertain potentials

  # Compare what happens when we turn one of the potentials into a deterministic one
  last_potential = crf.node_potentials[-1]
  last_potential = eqx.tree_at(lambda x: x.mu, last_potential, last_potential.mu*0 + 3.0)

  d_last_potential = last_potential.make_deterministic() # deterministic
  ad_last_potential = eqx.tree_at(lambda x: x.J, last_potential, last_potential.J*10000000) # almost deterministic

  d_potentials = jtu.tree_map(lambda xs, x: xs.at[-1].set(x), crf.node_potentials, d_last_potential)
  ad_potentials = jtu.tree_map(lambda xs, x: xs.at[-1].set(x), crf.node_potentials, ad_last_potential)

  d_crf = CRF(d_potentials, crf.base_transitions)
  ad_crf = CRF(ad_potentials, crf.base_transitions)

  # Get the backward messages for each
  d_bwd = d_crf.get_backward_messages()
  ad_bwd = ad_crf.get_backward_messages()
  assert jnp.allclose(d_bwd.mu, ad_bwd.mu)

  # Get the smoothed transitions for each
  d_smoothed = d_crf.get_transitions()
  ad_smoothed = ad_crf.get_transitions()
  assert compare_trees(d_smoothed, ad_smoothed, atol=1e-4)

  # Get samples from each
  d_samples = d_crf.sample(key)
  ad_samples = ad_crf.sample(key)
  assert jnp.allclose(d_samples[:-1], ad_samples[:-1], atol=1e-4)

  # Get the marginals for each
  d_marginals = d_crf.get_marginals()
  ad_marginals = ad_crf.get_marginals()
  assert jnp.allclose(d_marginals.mu[:-1], ad_marginals.mu[:-1], atol=1e-4)

################################################################################################################

def marginalize_test(crf: CRF):

  def test_prune(keep_indices):
    pruned_crf = crf.marginalize(keep_indices)

    marginals_true = crf.get_marginals()
    marginals_true = jtu.tree_map(lambda x: x[keep_indices], marginals_true)

    marginals_check = pruned_crf.get_marginals()
    try:
      assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.J, marginals_check.J))
    except:
      assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.Sigma, marginals_check.Sigma))

    try:
      assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.mu, marginals_check.mu))
    except:
      assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.h, marginals_check.h))

  test_prune(jnp.array([1, 3, 4]))
  test_prune(jnp.array([2, 4, 5]))
  test_prune(jnp.array([0, 3, 4]))

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt

  # turn on x64
  jax.config.update('jax_enable_x64', True)

  # Test the SSM
  key = random.PRNGKey(0)
  N = 6
  x_dim = 2
  y_dim = 2
  ts = jnp.linspace(0, 1, N)
  sigma = 1e-4

  #############################################
  #############################################
  #############################################

  # Create the parameters of the model
  W = random.normal(key, (y_dim,))
  yts = jnp.cos(2*jnp.pi*ts[:,None]*W[None,:])

  #############################################
  #############################################
  #############################################

  def create_transition(key):
    k1, k2 = random.split(key, 2)
    A = random.normal(k1, (x_dim, x_dim))
    A, _ = jnp.linalg.qr(A)
    Sigma = random.normal(k2, (x_dim, x_dim))
    Sigma = Sigma@Sigma.T
    _A = DenseMatrix(A, tags=TAGS.no_tags)
    _Sigma = DenseMatrix(Sigma, tags=TAGS.no_tags)
    _u = jnp.zeros(x_dim)
    return GaussianTransition(_A, _u, _Sigma)

  keys = random.split(key, N - 1)
  transitions = jax.vmap(create_transition)(keys)

  # R = DenseMatrix(R, tags=TAGS.no_tags)
  # prior = StandardGaussian(mu0, DenseMatrix(Sigma0, tags=TAGS.no_tags)).to_nat()
  def create_node_potential(i, y):
    H = jnp.eye(x_dim)[:y_dim]
    R = jnp.eye(y_dim)*1e-2
    _H = DenseMatrix(H, tags=TAGS.no_tags)
    _R = DenseMatrix(R, tags=TAGS.no_tags)
    joint = GaussianTransition(_H, jnp.zeros_like(y), _R)
    potential = joint.condition_on_y(y)
    return potential.to_mixed()

  ts = jnp.linspace(0, 1, N)
  keys = random.split(key, N)
  node_potentials = jax.vmap(create_node_potential)(jnp.arange(N), yts)

  #############################################
  # Create the CRF
  #############################################
  node_potentials_std = node_potentials
  crf = CRF(node_potentials_std, transitions, parallel=False)
  crf_tests(crf)

  marginalize_test(crf)

  crf2 = crf[:2]
  crf3 = crf[:3]
  crf4 = crf[:4]

  # Test the CRF
  edge_case_test(crf4)
  crf_tests(crf4)

  crf = CRF(node_potentials_std, transitions, parallel=True)
  crf_tests(crf)


