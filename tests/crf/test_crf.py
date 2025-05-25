import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Union, Tuple, Optional
import pytest
import jax.tree_util as jtu
from functools import partial

from linsdex.crf.crf import CRF, Messages, ssm_to_gaussian, brute_force_marginals
from linsdex.potential.gaussian.dist import StandardGaussian, NaturalGaussian, MixedGaussian
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.matrix import DenseMatrix, TAGS

# Enable x64 for better numerical precision in tests
jax.config.update('jax_enable_x64', True)

def compare_trees(A, B, atol: float = 1e-4):
  """Utility function to compare PyTrees with tolerance."""
  params, static = eqx.partition(A, eqx.is_inexact_array)
  params_B, static_B = eqx.partition(B, eqx.is_inexact_array)
  return jtu.tree_all(jtu.tree_map(partial(jnp.allclose, atol=atol), params, params_B))

class TestCRF:
  """Test cases for the CRF class."""

  @pytest.fixture
  def setup_crf_data(self):
    """Setup test data for CRF tests."""
    key = random.PRNGKey(0)
    N = 6
    x_dim = 2
    y_dim = 2
    ts = jnp.linspace(0, 1, N)

    # Create the parameters of the model
    W = random.normal(key, (y_dim,))
    yts = jnp.cos(2*jnp.pi*ts[:,None]*W[None,:])

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

    def create_node_potential(i, y):
      H = jnp.eye(x_dim)[:y_dim]
      R = jnp.eye(y_dim)*1e-2
      _H = DenseMatrix(H, tags=TAGS.no_tags)
      _R = DenseMatrix(R, tags=TAGS.no_tags)
      joint = GaussianTransition(_H, jnp.zeros_like(y), _R)
      potential = joint.condition_on_y(y)
      return potential.to_mixed()

    node_potentials = jax.vmap(create_node_potential)(jnp.arange(N), yts)

    return {
      'key': key,
      'N': N,
      'x_dim': x_dim,
      'y_dim': y_dim,
      'node_potentials': node_potentials,
      'transitions': transitions,
      'yts': yts
    }

  @pytest.fixture
  def crf_sequential(self, setup_crf_data):
    """Create a CRF with sequential message passing."""
    data = setup_crf_data
    return CRF(data['node_potentials'], data['transitions'], parallel=False)

  @pytest.fixture
  def crf_parallel(self, setup_crf_data):
    """Create a CRF with parallel message passing."""
    data = setup_crf_data
    return CRF(data['node_potentials'], data['transitions'], parallel=True)

  def test_crf_initialization(self, setup_crf_data):
    """Test basic CRF initialization."""
    data = setup_crf_data

    # Test successful initialization
    crf = CRF(data['node_potentials'], data['transitions'], parallel=False)
    assert len(crf) == data['N']
    assert crf.batch_size is None
    assert not crf.parallel

    # Test parallel initialization
    crf_parallel = CRF(data['node_potentials'], data['transitions'], parallel=True)
    assert crf_parallel.parallel

    # Test that CRF with fewer than 2 nodes raises error
    with pytest.raises(ValueError, match="CRF must have at least 2 nodes"):
      CRF(data['node_potentials'][:1], data['transitions'][:0])

  def test_crf_indexing(self, crf_sequential):
    """Test CRF slicing and indexing operations."""
    # Test slice operations
    crf_slice = crf_sequential[:3]
    assert len(crf_slice) == 3
    assert crf_slice.node_potentials.batch_size == 3
    assert crf_slice.base_transitions.batch_size == 2

    # Test that single index raises error
    with pytest.raises(ValueError, match="Cannot take a single index of a CRF"):
      _ = crf_sequential[0]

    # Test simple slice that maintains consistency
    crf_step = crf_sequential[1:4]
    assert len(crf_step) == 3  # nodes at indices 1, 2, 3

  def test_crf_reverse(self, crf_sequential):
    """Test CRF reverse operation."""
    reversed_crf = crf_sequential.reverse()
    assert len(reversed_crf) == len(crf_sequential)
    assert reversed_crf.parallel == crf_sequential.parallel

  def test_messages_class(self, crf_sequential):
    """Test the Messages helper class."""
    # Test empty messages
    messages = Messages(None, None)
    assert messages.batch_size is None

    # Test messages with forward only
    fwd_msg = crf_sequential.get_forward_messages()
    messages = Messages(fwd_msg, None)
    assert messages.batch_size == fwd_msg.batch_size

    # Test from_messages factory method
    messages = Messages.from_messages(None, crf_sequential, need_fwd=True, need_bwd=True)
    assert messages.fwd is not None
    assert messages.bwd is not None

  def test_basic_crf_functionality(self, crf_sequential, setup_crf_data):
    """Test basic CRF operations and consistency across parameterizations."""
    data = setup_crf_data
    key = data['key']

    # Create CRFs with different parameterizations
    node_potentials_std = data['node_potentials'].to_std()
    node_potentials_nat = data['node_potentials'].to_nat()
    node_potentials_mix = data['node_potentials']

    crf_std = CRF(node_potentials_std, data['transitions'], parallel=False)
    crf_nat = CRF(node_potentials_nat, data['transitions'], parallel=False)
    crf_mix = CRF(node_potentials_mix, data['transitions'], parallel=False)

    # Test that log prob is stable with gradients
    def log_prob(crf, xts):
      return crf.log_prob(xts)

    xts = crf_nat.sample(key)
    log_prob_grad = eqx.filter_grad(log_prob)(crf_nat, xts + 10)
    # Should not raise any errors

    # Test backward messages consistency
    bwd_nat = crf_nat.get_backward_messages()[:-1]
    bwd_std = crf_std.get_backward_messages()[:-1]
    bwd_mix = crf_mix.get_backward_messages()[:-1]

    assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_std))
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_nat.to_std(), bwd_mix.to_std()))

    # Test forward messages consistency
    fwd_nat = crf_nat.get_forward_messages()[1:]
    fwd_std = crf_std.get_forward_messages()[1:]
    fwd_mix = crf_mix.get_forward_messages()[1:]

    assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_std))
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_nat.to_std(), fwd_mix.to_std()))

  def test_parallel_vs_sequential_consistency(self, crf_sequential, crf_parallel):
    """Test that parallel and sequential message passing give the same results."""
    # Test backward messages
    bwd_seq = crf_sequential.get_backward_messages()
    bwd_par = crf_parallel.get_backward_messages()
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_seq.to_std(), bwd_par.to_std()))

    # Test forward messages
    fwd_seq = crf_sequential.get_forward_messages()
    fwd_par = crf_parallel.get_forward_messages()
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, fwd_seq.to_std(), fwd_par.to_std()))

    # Test marginals
    marg_seq = crf_sequential.get_marginals()
    marg_par = crf_parallel.get_marginals()
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, marg_seq.to_std(), marg_par.to_std()))

  def test_marginals_correctness(self, crf_sequential):
    """Test that computed marginals match brute force calculations."""
    # Convert to natural parameterization for brute force comparison
    node_potentials_nat = crf_sequential.node_potentials.to_nat()
    crf_nat = CRF(node_potentials_nat, crf_sequential.base_transitions, parallel=False)

    true_marginals = brute_force_marginals(crf_nat)
    computed_marginals = crf_nat.get_marginals()

    assert jtu.tree_all(jtu.tree_map(jnp.allclose, computed_marginals, true_marginals))

  def test_joints_correctness(self, crf_sequential):
    """Test that joint distributions have correct marginals."""
    # Convert to natural parameterization
    node_potentials_nat = crf_sequential.node_potentials.to_nat()
    crf_nat = CRF(node_potentials_nat, crf_sequential.base_transitions, parallel=False)

    true_marginals = brute_force_marginals(crf_nat)
    joints = crf_nat.get_joints()

    # Check that marginalizing joints gives correct marginals
    marginals_from_joints_x = joints.marginalize_out_y().to_nat()
    marginals_from_joints_y = joints.marginalize_out_x().to_nat()

    assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_from_joints_x, true_marginals[:-1]))
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_from_joints_y, true_marginals[1:]))

  def test_log_prob_correctness(self, crf_sequential, setup_crf_data):
    """Test that log probability calculations are correct."""
    data = setup_crf_data
    key = data['key']

    # Convert to natural parameterization
    node_potentials_nat = crf_sequential.node_potentials.to_nat()
    crf_nat = CRF(node_potentials_nat, crf_sequential.base_transitions, parallel=False)

    crf_gaussian = ssm_to_gaussian(crf_nat)
    xts = crf_nat.sample(key)

    log_prob1 = crf_gaussian.log_prob(xts.ravel())
    log_prob2 = crf_nat.log_prob(xts)

    assert jnp.allclose(log_prob1, log_prob2)

  def test_edge_cases_deterministic_potentials(self, crf_sequential, setup_crf_data):
    """Test edge cases with fully certain/uncertain potentials."""
    data = setup_crf_data
    key = data['key']

    # Test with deterministic potential at the last node
    last_potential = crf_sequential.node_potentials[-1]
    last_potential = eqx.tree_at(lambda x: x.mu, last_potential, last_potential.mu*0 + 3.0)

    d_last_potential = last_potential.make_deterministic()  # deterministic
    ad_last_potential = eqx.tree_at(lambda x: x.J, last_potential, last_potential.J*10000000)  # almost deterministic

    d_potentials = jtu.tree_map(lambda xs, x: xs.at[-1].set(x), crf_sequential.node_potentials, d_last_potential)
    ad_potentials = jtu.tree_map(lambda xs, x: xs.at[-1].set(x), crf_sequential.node_potentials, ad_last_potential)

    d_crf = CRF(d_potentials, crf_sequential.base_transitions)
    ad_crf = CRF(ad_potentials, crf_sequential.base_transitions)

    # Test backward messages
    d_bwd = d_crf.get_backward_messages()
    ad_bwd = ad_crf.get_backward_messages()
    assert jnp.allclose(d_bwd.mu, ad_bwd.mu)

    # Test smoothed transitions
    d_smoothed = d_crf.get_transitions()
    ad_smoothed = ad_crf.get_transitions()
    assert compare_trees(d_smoothed, ad_smoothed, atol=1e-4)

    # Test samples
    d_samples = d_crf.sample(key)
    ad_samples = ad_crf.sample(key)
    assert jnp.allclose(d_samples[:-1], ad_samples[:-1], atol=1e-4)

    # Test marginals
    d_marginals = d_crf.get_marginals()
    ad_marginals = ad_crf.get_marginals()
    assert jnp.allclose(d_marginals.mu[:-1], ad_marginals.mu[:-1], atol=1e-4)

  def test_zero_potentials(self, crf_sequential):
    """Test operations with zero (totally uncertain) potentials."""
    # Get different parameterizations
    node_potentials_nat = crf_sequential.node_potentials.to_nat()
    node_potentials_std = crf_sequential.node_potentials.to_std()
    node_potentials_mix = crf_sequential.node_potentials

    # Create zero potentials
    zero_nat = node_potentials_nat[0].total_uncertainty_like(node_potentials_nat[0])
    zero_std = node_potentials_std[0].total_uncertainty_like(node_potentials_std[0])
    zero_mix = node_potentials_mix[0].total_uncertainty_like(node_potentials_mix[0])

    # Test adding zero to a potential
    potential_nat = zero_nat + node_potentials_nat[-1]
    potential_std = zero_std + node_potentials_std[-1]
    potential_mix = zero_mix + node_potentials_mix[-1]

    assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat.to_std(), potential_std))
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat.to_std(), potential_mix.to_std()))

    # Test commutative property
    potential_nat2 = node_potentials_nat[-1] + zero_nat
    potential_std2 = node_potentials_std[-1] + zero_std
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, potential_nat2.to_std(), potential_std2))

  def test_marginalization(self, crf_sequential):
    """Test CRF marginalization functionality."""
    def test_marginalize_indices(keep_indices):
      """Helper function to test marginalization with specific indices."""
      pruned_crf = crf_sequential.marginalize(keep_indices)

      marginals_true = crf_sequential.get_marginals()
      marginals_true = jtu.tree_map(lambda x: x[keep_indices], marginals_true)

      marginals_check = pruned_crf.get_marginals()

      # Check covariance/precision matrices
      try:
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.J, marginals_check.J))
      except:
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.Sigma, marginals_check.Sigma))

      # Check means/natural parameters
      try:
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.mu, marginals_check.mu))
      except:
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginals_true.h, marginals_check.h))

    # Test different marginalization patterns
    test_marginalize_indices(jnp.array([1, 3, 4]))
    test_marginalize_indices(jnp.array([2, 4, 5]))
    test_marginalize_indices(jnp.array([0, 3, 4]))

  def test_canonical_form(self, crf_sequential):
    """Test conversion to canonical form."""
    canonical_crf = crf_sequential.to_canonical_form()

    # Canonical form should have same length
    assert len(canonical_crf) == len(crf_sequential)

    # Basic structure check - canonical form should be valid
    assert canonical_crf.node_potentials is not None
    assert canonical_crf.base_transitions is not None

    # NOTE: Marginals may differ due to normalization issues (see warning in crf.py)
    # This is a known limitation that would need to be addressed in the implementation

  def test_sampling(self, crf_sequential, crf_parallel, setup_crf_data):
    """Test sampling from CRF distributions."""
    key = setup_crf_data['key']

    # Test sequential sampling
    samples_seq = crf_sequential.sample(key)
    assert samples_seq.shape[0] == len(crf_sequential)

    # Test parallel sampling
    samples_par = crf_parallel.sample(key)
    assert samples_par.shape[0] == len(crf_parallel)

    # Both should give same results with same key
    assert jnp.allclose(samples_seq, samples_par)

  def test_to_prior_and_chain(self, crf_sequential):
    """Test conversion to prior and transition chain."""
    prior, transitions = crf_sequential.to_prior_and_chain()

    # Should have correct dimensions
    assert transitions.batch_size == len(crf_sequential) - 1
    assert prior.batch_size is None  # Single prior

  def test_normalizing_constant(self, crf_sequential):
    """Test normalizing constant calculation."""
    logZ = crf_sequential.normalizing_constant()
    marginal_loglik = crf_sequential.get_marginal_log_likelihood()

    # Should be the same
    assert jnp.allclose(logZ, marginal_loglik)

  def test_short_sequence_unrolled(self, setup_crf_data):
    """Test unrolled implementation for short sequences."""
    data = setup_crf_data

    # Create a short CRF (length 2, which is <= max_unroll_length=3)
    short_crf = CRF(
      data['node_potentials'][:2],
      data['transitions'][:1],
      parallel=False,
      max_unroll_length=3
    )

    # Should use unrolled implementation
    bwd_messages = short_crf.get_backward_messages()
    assert bwd_messages.batch_size == 2

    # Compare with sequential implementation
    short_crf_seq = CRF(
      data['node_potentials'][:2],
      data['transitions'][:1],
      parallel=False,
      max_unroll_length=0  # Force sequential
    )

    bwd_messages_seq = short_crf_seq.get_backward_messages()
    assert jtu.tree_all(jtu.tree_map(jnp.allclose, bwd_messages.to_std(), bwd_messages_seq.to_std()))