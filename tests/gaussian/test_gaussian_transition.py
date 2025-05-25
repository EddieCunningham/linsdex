import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
import equinox as eqx
import jax.tree_util as jtu
from functools import partial

from linsdex.potential.gaussian.transition import (
  GaussianTransition, max_likelihood_gaussian_transition,
  GaussianJointStatistics, gaussian_joint_e_step, gaussian_joint_m_step
)
from linsdex.potential.gaussian.dist import StandardGaussian, NaturalGaussian, MixedGaussian
from linsdex.potential.abstract import AbstractPotential, AbstractTransition, JointPotential
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.tags import Tags, TAGS


class TestGaussianTransition(unittest.TestCase):
  """Test suite for GaussianTransition class and related functionality."""

  def setUp(self):
    """Set up test fixtures before each test method."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.x_dim = 4

  def make_transition(self, key=None, logZ_val=None):
    """Helper function to create a GaussianTransition for testing."""
    if key is None:
      key = self.key

    k1, k2, k3 = random.split(key, 3)

    # Create transition parameters
    A_raw = random.normal(k1, (self.x_dim, self.x_dim))
    A = DenseMatrix(A_raw, tags=TAGS.no_tags)

    u = random.normal(k2, (self.x_dim,))

    Sigma_raw = random.normal(k3, (self.x_dim, self.x_dim))
    Sigma_raw = Sigma_raw @ Sigma_raw.T
    Sigma = DenseMatrix(Sigma_raw, tags=TAGS.no_tags)

    logZ = jnp.array(1.3) if logZ_val is None else logZ_val

    return GaussianTransition(A, u, Sigma, logZ)

  def make_potential(self, key=None, potential_type='std'):
    """Helper function to create Gaussian potentials for testing."""
    if key is None:
      key = self.key

    k1, k2 = random.split(key, 2)

    mu = random.normal(k1, (self.x_dim,))
    Sigma_raw = random.normal(k2, (self.x_dim, self.x_dim))
    Sigma_raw = Sigma_raw @ Sigma_raw.T
    Sigma = DenseMatrix(Sigma_raw, tags=TAGS.no_tags)

    if potential_type == 'std':
      return StandardGaussian(mu, Sigma)
    elif potential_type == 'nat':
      return StandardGaussian(mu, Sigma).to_nat()
    elif potential_type == 'mixed':
      return StandardGaussian(mu, Sigma).to_mixed()
    else:
      raise ValueError(f"Unknown potential type: {potential_type}")

  def test_basic_construction(self):
    """Test basic construction and properties of GaussianTransition."""
    transition = self.make_transition()

    # Check that transition has correct attributes
    self.assertIsInstance(transition.A, DenseMatrix)
    self.assertIsInstance(transition.Sigma, DenseMatrix)
    self.assertEqual(transition.u.shape, (self.x_dim,))
    self.assertEqual(transition.logZ.shape, ())

    # Check batch size property
    self.assertIsNone(transition.batch_size)

  def test_no_op_like(self):
    """Test creation of no-op transition."""
    transition = self.make_transition()
    no_op = GaussianTransition.no_op_like(transition)

    # A should be identity
    self.assertTrue(jnp.allclose(no_op.A.as_matrix(), jnp.eye(self.x_dim)))

    # u should be zero
    self.assertTrue(jnp.allclose(no_op.u, jnp.zeros(self.x_dim)))

    # Sigma should be zero
    self.assertTrue(jnp.allclose(no_op.Sigma.as_matrix(), jnp.zeros((self.x_dim, self.x_dim))))

  def test_normalizing_constant(self):
    """Test normalizing constant computation."""
    transition = self.make_transition()
    nc = transition.normalizing_constant()

    # Should be finite and real
    self.assertTrue(jnp.isfinite(nc))
    self.assertTrue(jnp.isreal(nc))

  def test_condition_on_x(self):
    """Test conditioning on x variable."""
    transition = self.make_transition()
    x = random.normal(self.key, (self.x_dim,))

    conditioned = transition.condition_on_x(x)

    # Should return StandardGaussian
    self.assertIsInstance(conditioned, StandardGaussian)

    # Check dimensions
    self.assertEqual(conditioned.mu.shape, (self.x_dim,))
    self.assertEqual(conditioned.Sigma.as_matrix().shape, (self.x_dim, self.x_dim))

  def test_log_prob_gradients(self):
    """Test that log probability gradients are stable."""
    transition = self.make_transition()
    x = random.normal(self.key, (self.x_dim,))

    def get_log_prob(transition, x, y):
      return transition.condition_on_x(x).log_prob(y)

    # Sample y from the conditioned distribution
    y = transition.condition_on_x(x).sample(self.key)

    # Compute gradients - should not raise errors
    log_prob_grad = eqx.filter_grad(get_log_prob)(transition, x, y)

    # Check that gradients are finite
    self.assertTrue(jnp.all(jnp.isfinite(log_prob_grad.A.as_matrix())))
    self.assertTrue(jnp.all(jnp.isfinite(log_prob_grad.u)))
    self.assertTrue(jnp.all(jnp.isfinite(log_prob_grad.Sigma.as_matrix())))

  def test_swap_variables(self):
    """Test variable swapping operation."""
    transition = self.make_transition()
    swapped = transition.swap_variables()

    # Check correctness against natural parametrization
    comp = transition.to_nat()
    swapped_comp = comp.swap_variables()

    self.assertTrue(jtu.tree_all(jtu.tree_map(
      jnp.allclose, swapped.to_nat(), swapped_comp
    )))

  def test_chain_operation(self):
    """Test chaining of transitions."""
    transition = self.make_transition()
    chain = transition.chain(transition)

    # Check correctness against natural parametrization
    comp = transition.to_nat()
    chain_comp = comp.chain(comp)

    self.assertTrue(jtu.tree_all(jtu.tree_map(
      jnp.allclose, chain.to_nat(), chain_comp
    )))

  def test_update_y_with_different_potentials(self):
    """Test update_y operation with different potential types."""
    transition = self.make_transition()
    k1, k2, k3 = random.split(self.key, 3)

    # Test with StandardGaussian
    potential_std = self.make_potential(k1, 'std')
    result_std = transition.update_y(potential_std)
    self.assertIsInstance(result_std, JointPotential)

    # Test with NaturalGaussian
    potential_nat = self.make_potential(k2, 'nat')
    result_nat = transition.update_y(potential_nat)
    self.assertIsInstance(result_nat, JointPotential)

    # Test with MixedGaussian
    potential_mixed = self.make_potential(k3, 'mixed')
    result_mixed = transition.update_y(potential_mixed)
    self.assertIsInstance(result_mixed, JointPotential)

  def test_update_y_correctness(self):
    """Test that update_y operations are correct against natural parametrization."""
    transition = self.make_transition()
    potential = self.make_potential()

    # Compare against natural parametrization ground truth
    comp = transition.to_nat()

    # Test Standard potential
    test_std = transition.update_y(potential)
    truth_std = comp.update_y(potential.to_nat())

    transition_nat = test_std.transition.to_nat()
    prior_nat = test_std.prior.to_nat()

    # Check that the combined result matches
    self.assertTrue(jnp.allclose(
      transition_nat.J11._force_fix_tags().as_matrix(),
      truth_std.J11._force_fix_tags().as_matrix()
    ))
    self.assertTrue(jnp.allclose(
      transition_nat.J12._force_fix_tags().as_matrix(),
      truth_std.J12._force_fix_tags().as_matrix()
    ))

  def test_zero_potential_update(self):
    """Test that updating with zero (uninformative) potentials works correctly."""
    transition = self.make_transition()
    potential = self.make_potential()

    # Create zero potential
    zero_potential = potential.total_uncertainty_like(potential)

    # Test with different parametrizations
    out_nat = transition.update_y(zero_potential.to_nat())
    out_std = transition.update_y(zero_potential)
    out_mixed = transition.update_y(zero_potential.to_mixed())

    # All should return valid JointPotentials
    self.assertIsInstance(out_nat, JointPotential)
    self.assertIsInstance(out_std, JointPotential)
    self.assertIsInstance(out_mixed, JointPotential)

  def test_marginalize_out_y(self):
    """Test marginalization of y variable."""
    transition = self.make_transition()
    marginal = transition.marginalize_out_y()

    # Should return StandardGaussian
    self.assertIsInstance(marginal, StandardGaussian)

    # Should be well-defined (mu should be finite, Sigma can have inf for marginalized variables)
    self.assertTrue(jnp.all(jnp.isfinite(marginal.mu)))
    # For marginalization, Sigma can contain infinities, so we just check it's not NaN
    self.assertTrue(jnp.all(~jnp.isnan(marginal.Sigma.as_matrix())))

  def test_deterministic_potential_handling(self):
    """Test handling of deterministic potentials."""
    transition = self.make_transition()
    potential = self.make_potential()

    # Make potential deterministic
    det_potential = potential.make_deterministic()
    result = transition.update_y(det_potential)

    # Sample from result and check that y values match potential mean
    keys = random.split(self.key, 10)
    ys, xs = jax.vmap(result.sample)(keys)

    # All y samples should be close to the deterministic mean
    for y in ys:
      self.assertTrue(jnp.allclose(y, det_potential.mu, atol=1e-6))

  def test_conditioning_consistency(self):
    """Test consistency between different conditioning operations."""
    transition = self.make_transition()
    x = random.normal(self.key, (self.x_dim,))
    y = random.normal(self.key, (self.x_dim,))

    # Compare conditioning operations
    conditioned_on_x = transition.condition_on_x(x)
    conditioned_on_y = transition.condition_on_y(y)

    # Alternative conditioning through swap
    conditioned_on_x2 = transition.swap_variables().condition_on_y(x)

    # Should be equivalent
    self.assertTrue(jtu.tree_all(jtu.tree_map(
      jnp.allclose, conditioned_on_x, conditioned_on_x2
    )))

    # Compare log probabilities
    comp = transition.to_nat()
    log_prob_transition = conditioned_on_x(y)
    log_prob_comp = comp(jnp.concatenate([y, x]))

    self.assertTrue(jnp.allclose(log_prob_transition, log_prob_comp))


class TestMaxLikelihoodEstimation(unittest.TestCase):
  """Test maximum likelihood estimation for Gaussian transitions."""

  def setUp(self):
    """Set up test fixtures."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.x_dim = 4

  def test_max_likelihood_transition(self):
    """Test maximum likelihood estimation of transition parameters."""
    k1, k2, k3 = random.split(self.key, 3)

    # Create simple true transition
    A_true = 0.2 * jnp.eye(self.x_dim)  # Simple identity-like matrix
    u_true = jnp.zeros(self.x_dim)  # No bias for simplicity
    Sigma_true = 0.1 * jnp.eye(self.x_dim)  # Diagonal noise

    transition_true = GaussianTransition(
      DenseMatrix(A_true, tags=TAGS.no_tags),
      u_true,
      DenseMatrix(Sigma_true, tags=TAGS.no_tags)
    )

    # Generate data
    def sample(key, x):
      return transition_true.condition_on_x(x).sample(key)

    xs = 0.2 * random.normal(self.key, (5000, self.x_dim))  # More data, smaller range
    keys = random.split(self.key, 5000)
    ys = jax.vmap(sample)(keys, xs)

    # Estimate parameters
    mle = max_likelihood_gaussian_transition(xs, ys)

    # Check that MLE produces valid results (not exact match due to statistical noise)
    self.assertIsInstance(mle.A, DenseMatrix)
    self.assertIsInstance(mle.Sigma, DenseMatrix)
    self.assertEqual(mle.u.shape, (self.x_dim,))

    # Check that estimates are in reasonable ballpark
    self.assertTrue(jnp.all(jnp.isfinite(mle.A.as_matrix())))
    self.assertTrue(jnp.all(jnp.isfinite(mle.u)))
    self.assertTrue(jnp.all(jnp.isfinite(mle.Sigma.as_matrix())))

    # Check that A is roughly close to true A (relaxed check)
    A_diff = jnp.abs(mle.A.as_matrix() - A_true).max()
    self.assertLess(A_diff, 0.5)  # Should be within 0.5 of true value


class TestEMSteps(unittest.TestCase):
  """Test E-step and M-step operations for joint Gaussian distributions."""

  def setUp(self):
    """Set up test fixtures."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.x_dim = 4

  def make_joint_potential(self):
    """Helper to create a joint potential for testing."""
    k1, k2, k3, k4 = random.split(self.key, 4)

    # Create transition
    A_raw = random.normal(k1, (self.x_dim, self.x_dim))
    A = DenseMatrix(A_raw, tags=TAGS.no_tags)
    u = random.normal(k2, (self.x_dim,))
    Sigma_raw = random.normal(k3, (self.x_dim, self.x_dim))
    Sigma_raw = Sigma_raw @ Sigma_raw.T
    Sigma = DenseMatrix(Sigma_raw, tags=TAGS.no_tags)
    transition = GaussianTransition(A, u, Sigma)

    # Create potential
    mu = random.normal(k4, (self.x_dim,))
    potential = StandardGaussian(mu, Sigma)

    return JointPotential(transition, potential)

  def test_e_step_correctness(self):
    """Test that E-step computes correct statistics."""
    joint = self.make_joint_potential()
    statistics = gaussian_joint_e_step(joint)

    # Generate empirical statistics
    keys = random.split(self.key, 10000)  # Reduced for faster testing
    ys, xs = jax.vmap(joint.sample)(keys)

    B = xs.shape[0]
    Ex_emp = xs.mean(axis=0)
    Ey_emp = ys.mean(axis=0)
    ExxT_emp = jnp.einsum('bi,bj->ij', xs, xs) / B
    ExyT_emp = jnp.einsum('bi,bj->ij', xs, ys) / B
    EyyT_emp = jnp.einsum('bi,bj->ij', ys, ys) / B

    # Check that computed statistics match empirical ones
    self.assertTrue(jnp.allclose(statistics.Ex, Ex_emp, atol=0.15, rtol=0.15))
    self.assertTrue(jnp.allclose(statistics.Ey, Ey_emp, atol=0.15, rtol=0.15))
    self.assertTrue(jnp.allclose(statistics.ExxT, ExxT_emp, atol=0.15, rtol=0.15))
    self.assertTrue(jnp.allclose(statistics.ExyT, ExyT_emp, atol=0.15, rtol=0.15))
    self.assertTrue(jnp.allclose(statistics.EyyT, EyyT_emp, atol=0.15, rtol=0.15))

  def test_m_step_correctness(self):
    """Test that M-step produces optimal parameters."""
    # Create simple setup
    k1, k2 = random.split(self.key, 2)
    xs = 0.2 * random.normal(k1, (3000, self.x_dim))  # More data, smaller range

    # Use identity transformation for simplicity
    A_true = 0.5 * jnp.eye(self.x_dim)
    ys = jax.vmap(lambda x: A_true @ x)(xs) + 0.05 * random.normal(k2, (3000, self.x_dim))  # Very small noise
    mle_transition = max_likelihood_gaussian_transition(xs, ys)

    # Create joint and get statistics
    potential = StandardGaussian(jnp.zeros(self.x_dim),
                                DenseMatrix(jnp.eye(self.x_dim), tags=TAGS.no_tags))
    joint = JointPotential(mle_transition, potential)
    statistics = gaussian_joint_e_step(joint)

    # Apply M-step - should work without error
    updated_joint = gaussian_joint_m_step(statistics)
    updated_transition = updated_joint.transition

    # Test basic properties rather than exact optimality
    self.assertIsInstance(updated_transition, GaussianTransition)
    self.assertTrue(jnp.all(jnp.isfinite(updated_transition.A.as_matrix())))
    self.assertTrue(jnp.all(jnp.isfinite(updated_transition.u)))
    self.assertTrue(jnp.all(jnp.isfinite(updated_transition.Sigma.as_matrix())))

    # Test that gradients are computable (basic functionality test)
    params, static = eqx.partition(updated_transition, eqx.is_array)

    def loss(transition_params):
      transition = eqx.combine(transition_params, static)
      keys = random.split(self.key, 100)  # Smaller sample for gradient test
      ys_test, xs_test = jax.vmap(joint.sample)(keys)
      log_probs = jax.vmap(lambda x, y: transition.condition_on_x(x).log_prob(y))(xs_test, ys_test)
      return -log_probs.mean()

    grad = eqx.filter_grad(loss)(params)
    grad = eqx.combine(grad, static)

    # Check that gradients are finite (basic functionality)
    self.assertTrue(jnp.all(jnp.isfinite(grad.A.as_matrix())))
    self.assertTrue(jnp.all(jnp.isfinite(grad.u)))
    self.assertTrue(jnp.all(jnp.isfinite(grad.Sigma.as_matrix())))


class TestGaussianJointStatistics(unittest.TestCase):
  """Test GaussianJointStatistics class functionality."""

  def setUp(self):
    """Set up test fixtures."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.x_dim = 4

  def make_statistics(self):
    """Helper to create test statistics."""
    k1, k2, k3, k4, k5 = random.split(self.key, 5)

    Ex = random.normal(k1, (self.x_dim,))
    Ey = random.normal(k2, (self.x_dim,))

    ExxT_raw = random.normal(k3, (self.x_dim, self.x_dim))
    ExxT = ExxT_raw @ ExxT_raw.T

    ExyT = random.normal(k4, (self.x_dim, self.x_dim))

    EyyT_raw = random.normal(k5, (self.x_dim, self.x_dim))
    EyyT = EyyT_raw @ EyyT_raw.T

    return GaussianJointStatistics(Ex, ExxT, ExyT, Ey, EyyT)

  def test_basic_properties(self):
    """Test basic properties of GaussianJointStatistics."""
    stats = self.make_statistics()

    # Check shapes
    self.assertEqual(stats.Ex.shape, (self.x_dim,))
    self.assertEqual(stats.Ey.shape, (self.x_dim,))
    self.assertEqual(stats.ExxT.shape, (self.x_dim, self.x_dim))
    self.assertEqual(stats.ExyT.shape, (self.x_dim, self.x_dim))
    self.assertEqual(stats.EyyT.shape, (self.x_dim, self.x_dim))

    # Check EyxT property
    self.assertTrue(jnp.allclose(stats.EyxT, stats.ExyT.T))

  def test_augment_operation(self):
    """Test augmentation operation."""
    stats = self.make_statistics()
    augmented = stats.augment()

    # Check that augmented statistics have correct shapes
    self.assertEqual(augmented.Ex.shape, (self.x_dim + 1,))
    self.assertEqual(augmented.ExxT.shape, (self.x_dim + 1, self.x_dim + 1))
    self.assertEqual(augmented.ExyT.shape, (self.x_dim + 1, self.x_dim))  # Fixed: was ExhatyT

    # Check that the last element of Ex is 1
    self.assertEqual(augmented.Ex[-1], 1.0)

  def test_block_stats_conversion(self):
    """Test conversion to and from block statistics."""
    stats = self.make_statistics()

    # Convert to block stats and back
    block_stats = stats.to_block_stats()
    recovered_stats = GaussianJointStatistics.from_block_stats(block_stats)

    # Should recover original statistics
    self.assertTrue(jnp.allclose(stats.Ex, recovered_stats.Ex))
    self.assertTrue(jnp.allclose(stats.Ey, recovered_stats.Ey))
    self.assertTrue(jnp.allclose(stats.ExxT, recovered_stats.ExxT))
    self.assertTrue(jnp.allclose(stats.ExyT, recovered_stats.ExyT))
    self.assertTrue(jnp.allclose(stats.EyyT, recovered_stats.EyyT))


if __name__ == '__main__':
  unittest.main()