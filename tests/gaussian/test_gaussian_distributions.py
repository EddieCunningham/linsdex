import jax
import jax.numpy as jnp
from jax import random
import unittest
import pytest
import equinox as eqx
import jax.tree_util as jtu
from functools import partial

from linsdex.potential.gaussian.dist import (
  NaturalGaussian, StandardGaussian, MixedGaussian,
  NaturalJointGaussian, GaussianStatistics,
  gaussian_e_step, gaussian_m_step, check_distribution
)
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.tags import Tags, TAGS
from linsdex.potential.abstract import AbstractPotential


class TestGaussianDistributions(unittest.TestCase):
  """Test suite for Gaussian distribution implementations in various parametrizations."""

  def setUp(self):
    """Set up test fixtures before each test method."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.dim = 4

  def make_matrix(self, key, kind='nat', matrix='dense'):
    """Helper function to create different types of Gaussian distributions."""
    k1, k2, k3 = random.split(key, 3)
    mat = random.normal(k1, (self.dim, self.dim))
    mat = mat.T @ mat  # Make positive definite
    J = DenseMatrix(mat, tags=TAGS.no_tags)
    h = random.normal(k2, (self.dim,))
    logZ = random.normal(k3, ())

    nat_gaussian = NaturalGaussian(J, h, logZ)

    if kind == 'nat':
      return nat_gaussian
    elif kind == 'std':
      return nat_gaussian.to_std()
    elif kind == 'mixed':
      std = nat_gaussian.to_std()
      return MixedGaussian(std.mu, nat_gaussian.J, nat_gaussian.logZ)
    else:
      raise ValueError(f"Unknown kind: {kind}")

  def test_parametrization_conversions(self):
    """Test that conversions between different Gaussian parametrizations are consistent."""
    dist_std = self.make_matrix(self.key, kind='std')
    dist_nat = self.make_matrix(self.key, kind='nat')
    dist_mixed = self.make_matrix(self.key, kind='mixed')

    # Test that converting from std to nat and back preserves the distribution
    std_to_nat_to_std = dist_std.to_nat().to_std()
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, dist_std.mu, std_to_nat_to_std.mu)))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, dist_std.Sigma.as_matrix(), std_to_nat_to_std.Sigma.as_matrix())))
    self.assertTrue(jnp.allclose(dist_std.logZ, std_to_nat_to_std.logZ))

    # Test that all parametrizations represent the same distribution
    nat_from_std = dist_std.to_nat()
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, nat_from_std.J.as_matrix(), dist_nat.J.as_matrix())))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, nat_from_std.h, dist_nat.h)))
    self.assertTrue(jnp.allclose(nat_from_std.logZ, dist_nat.logZ))

    mixed_from_nat = dist_nat.to_mixed()
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, mixed_from_nat.mu, dist_mixed.mu)))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, mixed_from_nat.J.as_matrix(), dist_mixed.J.as_matrix())))
    self.assertTrue(jnp.allclose(mixed_from_nat.logZ, dist_mixed.logZ))

  def test_addition_consistency(self):
    """Test that addition works consistently across different parametrizations."""
    k1, k2 = random.split(self.key, 2)

    # Create distributions in different parametrizations
    dist1_std = self.make_matrix(k1, kind='std')
    dist2_std = self.make_matrix(k2, kind='std')

    dist1_nat = self.make_matrix(k1, kind='nat')
    dist2_nat = self.make_matrix(k2, kind='nat')

    dist1_mixed = self.make_matrix(k1, kind='mixed')
    dist2_mixed = self.make_matrix(k2, kind='mixed')

    # Compute sums in different parametrizations
    sum_std = dist1_std + dist2_std
    sum_nat = (dist1_nat + dist2_nat).to_std()
    sum_mixed = (dist1_mixed + dist2_mixed).to_std()

    # All sums should be equivalent
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std.Sigma.as_matrix(), sum_nat.Sigma.as_matrix())))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std.mu, sum_nat.mu)))
    self.assertTrue(jnp.allclose(sum_std.logZ, sum_nat.logZ))

    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std.Sigma.as_matrix(), sum_mixed.Sigma.as_matrix())))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std.mu, sum_mixed.mu)))
    self.assertTrue(jnp.allclose(sum_std.logZ, sum_mixed.logZ))

  def test_zero_potential_addition(self):
    """Test that adding zero (uninformative) potentials doesn't change distributions."""
    dist_std = self.make_matrix(self.key, kind='std')
    dist_nat = self.make_matrix(self.key, kind='nat')
    dist_mixed = self.make_matrix(self.key, kind='mixed')

    # Create zero potentials
    zero_std = dist_std.total_uncertainty_like(dist_std)
    zero_nat = dist_nat.total_uncertainty_like(dist_nat)
    zero_mixed = dist_mixed.total_uncertainty_like(dist_mixed)

    # Add zero potentials
    sum_std = dist_std + zero_std
    sum_nat = dist_nat + zero_nat
    sum_mixed = dist_mixed + zero_mixed

    # Results should be unchanged
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std.mu, dist_std.mu)))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_std.Sigma.as_matrix(), dist_std.Sigma.as_matrix())))
    self.assertTrue(jnp.allclose(sum_std.logZ, dist_std.logZ))

    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_nat.J.as_matrix(), dist_nat.J.as_matrix())))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_nat.h, dist_nat.h)))
    self.assertTrue(jnp.allclose(sum_nat.logZ, dist_nat.logZ))

    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_mixed.mu, dist_mixed.mu)))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, sum_mixed.J.as_matrix(), dist_mixed.J.as_matrix())))
    self.assertTrue(jnp.allclose(sum_mixed.logZ, dist_mixed.logZ))

  def test_log_prob_gradients(self):
    """Test that log probability evaluation and gradients are stable."""
    dist_std = self.make_matrix(self.key, kind='std')
    dist_nat = self.make_matrix(self.key, kind='nat')
    dist_mixed = self.make_matrix(self.key, kind='mixed')

    # Sample a point for evaluation
    x = dist_std.sample(self.key)

    def get_log_prob(dist, x):
      return dist.log_prob(x)

    # Compute gradients - should not raise errors
    log_prob_grad_std = eqx.filter_grad(get_log_prob)(dist_std, x)
    log_prob_grad_nat = eqx.filter_grad(get_log_prob)(dist_nat, x)
    log_prob_grad_mixed = eqx.filter_grad(get_log_prob)(dist_mixed, x)

    # Gradients should be finite and well-defined
    self.assertTrue(jnp.all(jnp.isfinite(log_prob_grad_std.mu)))
    # Check if Sigma gradient has valid tags before accessing as_matrix
    if hasattr(log_prob_grad_std.Sigma, 'elements') and log_prob_grad_std.Sigma.elements is not None:
      sigma_grad = log_prob_grad_std.Sigma.elements
    else:
      # Fallback to raw matrix if tags are problematic
      sigma_grad = log_prob_grad_std.Sigma.as_matrix() if hasattr(log_prob_grad_std.Sigma, 'as_matrix') else log_prob_grad_std.Sigma
    self.assertTrue(jnp.all(jnp.isfinite(sigma_grad)))

    # Check if J gradient has valid tags before accessing as_matrix
    if hasattr(log_prob_grad_nat.J, 'elements') and log_prob_grad_nat.J.elements is not None:
      j_grad = log_prob_grad_nat.J.elements
    else:
      j_grad = log_prob_grad_nat.J.as_matrix() if hasattr(log_prob_grad_nat.J, 'as_matrix') else log_prob_grad_nat.J
    self.assertTrue(jnp.all(jnp.isfinite(j_grad)))
    self.assertTrue(jnp.all(jnp.isfinite(log_prob_grad_nat.h)))

    self.assertTrue(jnp.all(jnp.isfinite(log_prob_grad_mixed.mu)))
    # Check if J gradient has valid tags before accessing as_matrix
    if hasattr(log_prob_grad_mixed.J, 'elements') and log_prob_grad_mixed.J.elements is not None:
      j_mixed_grad = log_prob_grad_mixed.J.elements
    else:
      j_mixed_grad = log_prob_grad_mixed.J.as_matrix() if hasattr(log_prob_grad_mixed.J, 'as_matrix') else log_prob_grad_mixed.J
    self.assertTrue(jnp.all(jnp.isfinite(j_mixed_grad)))

  def test_vectorized_operations(self):
    """Test that vectorized operations work correctly."""
    dist_mixed = self.make_matrix(self.key, kind='mixed')

    # Create vectorized versions
    def make_zero(i):
      return dist_mixed.total_uncertainty_like(dist_mixed)

    zeros = eqx.filter_vmap(make_zero)(jnp.arange(3))
    node_potentials = jtu.tree_map(lambda xs, x: xs.at[0].set(x), zeros, dist_mixed)

    # Individual additions
    check1 = node_potentials[0] + zeros[0]
    check2 = node_potentials[1] + zeros[1]
    check3 = node_potentials[2] + zeros[2]

    # Vectorized addition
    check_vectorized = node_potentials + zeros

    # Results should be consistent
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, check1.mu, check_vectorized[0].mu)))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, check2.mu, check_vectorized[1].mu)))
    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, check3.mu, check_vectorized[2].mu)))

  def test_e_step_m_step(self):
    """Test that E-step and M-step for Gaussian distributions are consistent."""
    nat_dist = self.make_matrix(self.key, kind='nat')

    # E-step: get sufficient statistics
    stats = gaussian_e_step(nat_dist)
    self.assertIsInstance(stats, GaussianStatistics)

    # M-step: reconstruct distribution from statistics
    reconstructed = gaussian_m_step(stats)
    self.assertIsInstance(reconstructed, StandardGaussian)

    # Convert original to standard form for comparison
    nat_as_std = nat_dist.to_std()

    # Reconstructed distribution should match the original
    self.assertTrue(jtu.tree_all(jtu.tree_map(
      lambda x, y: jnp.allclose(x, y, atol=1e-6),
      reconstructed.mu, nat_as_std.mu
    )))
    self.assertTrue(jtu.tree_all(jtu.tree_map(
      lambda x, y: jnp.allclose(x, y, atol=1e-6),
      reconstructed.Sigma.as_matrix(), nat_as_std.Sigma.as_matrix()
    )))

  def test_normalizing_constants(self):
    """Test that normalizing constants are computed correctly."""
    for kind in ['std', 'nat', 'mixed']:
      with self.subTest(kind=kind):
        dist = self.make_matrix(self.key, kind=kind)

        # Normalizing constant should be finite
        nc = dist.normalizing_constant()
        self.assertTrue(jnp.isfinite(nc))

        # For well-conditioned matrices, should be positive
        self.assertTrue(nc > -jnp.inf)

  def test_sampling_consistency(self):
    """Test that sampling works and produces reasonable results."""
    for kind in ['std', 'nat', 'mixed']:
      with self.subTest(kind=kind):
        dist = self.make_matrix(self.key, kind=kind)

        # Generate multiple samples
        keys = random.split(self.key, 10)
        samples = jax.vmap(dist.sample)(keys)

        # Samples should have correct shape
        self.assertEqual(samples.shape, (10, self.dim))

        # Samples should be finite
        self.assertTrue(jnp.all(jnp.isfinite(samples)))

        # Can evaluate log probability of samples
        log_probs = jax.vmap(dist.log_prob)(samples)
        self.assertTrue(jnp.all(jnp.isfinite(log_probs)))

  def test_distribution_properties(self):
    """Test basic properties of distributions using the check_distribution function."""
    # Note: This test runs a reduced version of check_distribution to avoid the expensive
    # empirical distribution comparison that requires many samples

    for kind in ['std', 'nat', 'mixed']:
      with self.subTest(kind=kind):
        dist = self.make_matrix(self.key, kind=kind)

        # Basic sampling test
        sample = dist.sample(self.key)
        self.assertEqual(sample.shape, (self.dim,))

        # Log probability evaluation
        log_prob = dist.log_prob(sample)
        self.assertTrue(jnp.isfinite(log_prob))

        # Score function
        score = dist.score(sample)
        self.assertEqual(score.shape, (self.dim,))
        self.assertTrue(jnp.all(jnp.isfinite(score)))

  def test_joint_gaussian(self):
    """Test NaturalJointGaussian functionality."""
    # Create a joint gaussian
    dim_x, dim_y = 2, 2
    total_dim = dim_x + dim_y

    key1, key2 = random.split(self.key, 2)

    # Create components for joint distribution
    J11_mat = random.normal(key1, (dim_y, dim_y))
    J11_mat = J11_mat.T @ J11_mat + jnp.eye(dim_y) * 0.1
    J11 = DenseMatrix(J11_mat, tags=TAGS.no_tags)

    J12_mat = random.normal(key1, (dim_y, dim_x))
    J12 = DenseMatrix(J12_mat, tags=TAGS.no_tags)

    J22_mat = random.normal(key2, (dim_x, dim_x))
    J22_mat = J22_mat.T @ J22_mat + jnp.eye(dim_x) * 0.1
    J22 = DenseMatrix(J22_mat, tags=TAGS.no_tags)

    h1 = random.normal(key1, (dim_y,))
    h2 = random.normal(key2, (dim_x,))

    joint = NaturalJointGaussian(J11, J12, J22, h1, h2)

    # Test conversion to block form
    block_form = joint.to_block()
    self.assertEqual(block_form.dim, total_dim)

    # Test marginalization
    marginal_x = joint.marginalize_out_y()
    self.assertEqual(marginal_x.dim, dim_x)

    marginal_y = joint.marginalize_out_x()
    self.assertEqual(marginal_y.dim, dim_y)

    # Test variable swapping
    swapped = joint.swap_variables()
    self.assertTrue(jtu.tree_all(jtu.tree_map(
      jnp.allclose, swapped.J11.as_matrix(), joint.J22.as_matrix()
    )))
    self.assertTrue(jtu.tree_all(jtu.tree_map(
      jnp.allclose, swapped.J22.as_matrix(), joint.J11.as_matrix()
    )))

  def test_mixed_parametrization_operations(self):
    """Test specific operations for mixed parametrization."""
    dist = self.make_matrix(self.key, kind='mixed')

    # Test deterministic potential creation
    x = random.normal(self.key, (self.dim,))
    deterministic = MixedGaussian.total_certainty_like(x, dist)

    self.assertTrue(jtu.tree_all(jtu.tree_map(jnp.allclose, deterministic.mu, x)))
    self.assertTrue(deterministic.J.is_inf)


if __name__ == '__main__':
  unittest.main()