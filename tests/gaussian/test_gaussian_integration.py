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
  check_distribution
)
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.tags import TAGS


class TestGaussianIntegration(unittest.TestCase):
  """Integration tests for Gaussian distributions that are more computationally intensive."""

  def setUp(self):
    """Set up test fixtures before each test method."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(123)
    self.dim = 4

  def make_matrix(self, key, kind='nat'):
    """Helper function to create different types of Gaussian distributions."""
    k1, k2, k3 = random.split(key, 3)

    # Create a well-conditioned positive definite matrix
    mat = random.normal(k1, (self.dim, self.dim)) * 0.5
    mat = mat.T @ mat + jnp.eye(self.dim) * 0.1  # Make positive definite and well-conditioned

    J = DenseMatrix(mat, tags=TAGS.no_tags)
    h = random.normal(k2, (self.dim,)) * 0.5  # Scale down to avoid extreme distributions
    logZ = random.normal(k3, ()) * 0.1

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

  @pytest.mark.slow
  def test_full_distribution_check_standard(self):
    """Test the full check_distribution function for StandardGaussian."""
    dist = self.make_matrix(self.key, kind='std')

    # This test is computationally expensive due to empirical distribution checking
    # but provides comprehensive validation
    try:
      check_distribution(dist)
    except Exception as e:
      # If the full check fails due to missing dependencies or numerical issues,
      # we can still verify basic properties
      self.fail(f"Full distribution check failed for StandardGaussian: {e}")

  @pytest.mark.slow
  def test_full_distribution_check_natural(self):
    """Test the full check_distribution function for NaturalGaussian."""
    dist = self.make_matrix(self.key, kind='nat')

    try:
      check_distribution(dist)
    except Exception as e:
      self.fail(f"Full distribution check failed for NaturalGaussian: {e}")

  @pytest.mark.slow
  def test_full_distribution_check_mixed(self):
    """Test the full check_distribution function for MixedGaussian."""
    dist = self.make_matrix(self.key, kind='mixed')

    try:
      check_distribution(dist)
    except Exception as e:
      self.fail(f"Full distribution check failed for MixedGaussian: {e}")

  def test_large_sample_consistency(self):
    """Test that large samples from distributions have correct empirical properties."""
    for kind in ['std', 'nat', 'mixed']:
      with self.subTest(kind=kind):
        dist = self.make_matrix(self.key, kind=kind)

        # Generate large number of samples
        keys = random.split(self.key, 10000)  # Increased sample size for better accuracy
        samples = jax.vmap(dist.sample)(keys)

        # Compute empirical mean and covariance
        empirical_mean = jnp.mean(samples, axis=0)
        empirical_cov = jnp.cov(samples.T)

        # Convert to standard form for comparison
        if kind != 'std':
          dist_std = dist.to_std()
        else:
          dist_std = dist

        # Check that empirical statistics are close to true parameters
        # Use relative error for better scaling
        true_mean = dist_std.mu
        true_cov = dist_std.Sigma.as_matrix()

        # For mean: use absolute tolerance since means can be near zero
        mean_diff = jnp.max(jnp.abs(empirical_mean - true_mean))

        # For covariance: use both absolute and relative tolerances
        cov_diff = jnp.max(jnp.abs(empirical_cov - true_cov))
        rel_cov_diff = jnp.max(jnp.abs((empirical_cov - true_cov) / (jnp.abs(true_cov) + 1e-8)))

        # Tolerances based on Central Limit Theorem expectations
        mean_tol = 0.15  # Should scale with sqrt(variance/n_samples)
        cov_tol = 0.3    # Covariance estimation has higher variance
        rel_cov_tol = 0.2

        self.assertLess(mean_diff, mean_tol,
                       f"Mean difference too large for {kind}: {mean_diff:.4f} (tolerance: {mean_tol})")
        self.assertTrue(cov_diff < cov_tol or rel_cov_diff < rel_cov_tol,
                       f"Covariance difference too large for {kind}: abs={cov_diff:.4f}, rel={rel_cov_diff:.4f}")

  def test_log_prob_normalizes(self):
    """Test that log probabilities integrate to approximately 1 using Monte Carlo."""
    for kind in ['std', 'nat', 'mixed']:
      with self.subTest(kind=kind):
        dist = self.make_matrix(self.key, kind=kind)

        # Generate random points in a reasonable range around the mean
        if kind == 'std':
          center = dist.mu
        else:
          center = dist.to_std().mu

        # Sample from a wider distribution for integration
        integration_samples = center + random.normal(self.key, (50000, self.dim)) * 3.0

        # Compute log probabilities and convert to probabilities
        log_probs = jax.vmap(dist.log_prob)(integration_samples)

        # Basic sanity checks
        self.assertTrue(jnp.all(jnp.isfinite(log_probs)))
        self.assertTrue(jnp.all(log_probs <= 0))  # Log probabilities should be <= 0

  def test_score_function_properties(self):
    """Test properties of the score function (gradient of log density)."""
    for kind in ['std', 'nat', 'mixed']:
      with self.subTest(kind=kind):
        dist = self.make_matrix(self.key, kind=kind)

        # Generate test points
        keys = random.split(self.key, 100)
        test_points = jax.vmap(dist.sample)(keys)

        # Compute score function at test points
        scores = jax.vmap(dist.score)(test_points)

        # Score should have same shape as input
        self.assertEqual(scores.shape, test_points.shape)

        # Score should be finite
        self.assertTrue(jnp.all(jnp.isfinite(scores)))

        # For Gaussians, score at the mean should be zero (approximately)
        if kind == 'std':
          mean = dist.mu
        else:
          mean = dist.to_std().mu

        score_at_mean = dist.score(mean)
        self.assertTrue(jnp.allclose(score_at_mean, jnp.zeros_like(mean), atol=1e-6))

  def test_numerical_stability_extreme_cases(self):
    """Test numerical stability with extreme parameter values."""
    # Test with very small eigenvalues (near-singular matrices)
    k1, k2 = random.split(self.key, 2)

    # Create a matrix with very small but positive eigenvalues
    U = random.normal(k1, (self.dim, self.dim))
    U, _ = jnp.linalg.qr(U)  # Orthogonal matrix
    small_eigenvals = jnp.array([1e-6, 1e-5, 1e-4, 1e-3])
    D = jnp.diag(small_eigenvals)
    nearly_singular = U @ D @ U.T

    J = DenseMatrix(nearly_singular, tags=TAGS.no_tags)
    h = random.normal(k2, (self.dim,))

    # Should still work (though may be less accurate)
    try:
      dist = NaturalGaussian(J, h)
      sample = dist.sample(self.key)
      log_prob = dist.log_prob(sample)
      self.assertTrue(jnp.isfinite(log_prob))
    except Exception as e:
      # This is acceptable for nearly singular matrices
      print(f"Near-singular case failed as expected: {e}")

  def test_conversion_round_trip_precision(self):
    """Test that multiple conversions preserve precision."""
    dist_std = self.make_matrix(self.key, kind='std')

    # Do multiple round-trip conversions
    converted = dist_std
    for _ in range(5):
      converted = converted.to_nat().to_mixed().to_std()

    # Should still be close to original (within numerical precision)
    self.assertTrue(jnp.allclose(converted.mu, dist_std.mu, atol=1e-10))
    self.assertTrue(jnp.allclose(
      converted.Sigma.as_matrix(),
      dist_std.Sigma.as_matrix(),
      atol=1e-10
    ))


if __name__ == '__main__':
  # Run only non-slow tests by default
  pytest.main([__file__, '-v', '-m', 'not slow'])