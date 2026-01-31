import jax
import jax.numpy as jnp
from jax import random
import unittest
import equinox as eqx
import jax.tree_util as jtu

from linsdex.potential.gaussian.transition import GaussianTransition, functional_potential_to_transition
from linsdex.potential.gaussian.dist import StandardGaussian, NaturalGaussian, MixedGaussian
from linsdex.linear_functional.linear_functional import LinearFunctional, resolve_linear_functional
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.tags import TAGS

class TestFunctionalPotential(unittest.TestCase):
  """Test suite for functional_potential_to_transition functionality."""

  def setUp(self):
    """Set up test fixtures."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.dim = 4
    self.x_dim = 4

    # Create components for LinearFunctional: f(x) = Ax + b
    k1, k2, k3 = random.split(self.key, 3)
    self.A_mat = random.normal(k1, (self.dim, self.x_dim))
    self.A = DenseMatrix(self.A_mat, tags=TAGS.no_tags)
    self.b = random.normal(k2, (self.dim,))
    self.lf = LinearFunctional(self.A, self.b)

    # Create covariance and logZ
    Sigma_raw = random.normal(k3, (self.dim, self.dim))
    self.Sigma = DenseMatrix(Sigma_raw @ Sigma_raw.T, tags=TAGS.no_tags)
    self.logZ = jnp.array(1.5)

    # Latent variable for verification
    self.x = random.normal(random.split(k3)[0], (self.x_dim,))

  def test_standard_functional_potential(self):
    """Test conversion from StandardGaussian with LinearFunctional mean."""
    potential = StandardGaussian(self.lf, self.Sigma, self.logZ)
    transition = functional_potential_to_transition(potential)

    # Verify attributes
    self.assertIsInstance(transition, GaussianTransition)
    self.assertTrue(jnp.allclose(transition.A.as_matrix(), self.A.as_matrix()))
    self.assertTrue(jnp.allclose(transition.u, self.b))
    self.assertTrue(jnp.allclose(transition.Sigma.as_matrix(), self.Sigma.as_matrix()))
    self.assertTrue(jnp.allclose(transition.logZ, self.logZ))

    # Verify conditioning consistency: transition(x) should match potential resolved at x
    conditioned = transition.condition_on_x(self.x)
    resolved_potential = StandardGaussian(self.lf(self.x), self.Sigma, self.logZ)

    self.assertTrue(jnp.allclose(conditioned.mu, resolved_potential.mu))
    self.assertTrue(jnp.allclose(conditioned.Sigma.as_matrix(), resolved_potential.Sigma.as_matrix()))
    # Note: logZ in StandardGaussian might be updated during construction if not provided,
    # but here we provide it. However, GaussianTransition.condition_on_x adds its own logZ term.
    # Let's check log_prob consistency instead.
    y = random.normal(random.split(self.key)[0], (self.dim,))
    self.assertTrue(jnp.allclose(conditioned.log_prob(y), resolved_potential.log_prob(y)))

  def test_natural_functional_potential(self):
    """Test conversion from NaturalGaussian with LinearFunctional h."""
    # potential = exp(-0.5 * y^T J y + y^T h - logZ)
    # where h = lf = A_h x + b_h
    # to_std() converts this to StandardGaussian(mu=J^-1 h, Sigma=J^-1, ...)
    # mu = J^-1 (A_h x + b_h) = (J^-1 A_h) x + (J^-1 b_h)
    potential = NaturalGaussian(self.Sigma.get_inverse(), self.lf, self.logZ)
    transition = functional_potential_to_transition(potential)

    # Expected parameters for the transition
    J_inv = self.Sigma
    expected_A = J_inv @ self.A
    expected_u = J_inv @ self.b

    self.assertTrue(jnp.allclose(transition.A.as_matrix(), expected_A.as_matrix()))
    self.assertTrue(jnp.allclose(transition.u, expected_u))
    self.assertTrue(jnp.allclose(transition.Sigma.as_matrix(), self.Sigma.as_matrix()))

    # Verify conditioning
    conditioned = transition.condition_on_x(self.x)
    resolved_potential = NaturalGaussian(self.Sigma.get_inverse(), self.lf(self.x), self.logZ).to_std()

    y = random.normal(random.split(self.key)[0], (self.dim,))
    self.assertTrue(jnp.allclose(conditioned.log_prob(y), resolved_potential.log_prob(y)))

  def test_mixed_functional_potential(self):
    """Test conversion from MixedGaussian with LinearFunctional mu."""
    potential = MixedGaussian(self.lf, self.Sigma.get_inverse(), self.logZ)
    transition = functional_potential_to_transition(potential)

    self.assertTrue(jnp.allclose(transition.A.as_matrix(), self.A.as_matrix()))
    self.assertTrue(jnp.allclose(transition.u, self.b))
    self.assertTrue(jnp.allclose(transition.Sigma.as_matrix(), self.Sigma.as_matrix()))

    # Verify conditioning
    conditioned = transition.condition_on_x(self.x)
    resolved_potential = MixedGaussian(self.lf(self.x), self.Sigma.get_inverse(), self.logZ).to_std()

    y = random.normal(random.split(self.key)[0], (self.dim,))
    self.assertTrue(jnp.allclose(conditioned.log_prob(y), resolved_potential.log_prob(y)))

  def test_batched_functional_potential(self):
    """Test conversion with batched potentials."""
    batch_size = 3
    keys = random.split(self.key, batch_size)

    def make_potential(k):
      k1, k2, k3 = random.split(k, 3)
      A = DenseMatrix(random.normal(k1, (self.dim, self.x_dim)))
      b = random.normal(k2, (self.dim,))
      lf = LinearFunctional(A, b)
      Sigma_raw = random.normal(k3, (self.dim, self.dim))
      Sigma = DenseMatrix(Sigma_raw @ Sigma_raw.T)
      return StandardGaussian(lf, Sigma)

    potentials = jax.vmap(make_potential)(keys)

    # Apply functional_potential_to_transition which should be auto-vmapped if it's decorated or handled by JAX
    # It is NOT decorated with @auto_vmap in transition.py, but StandardGaussian.to_std IS auto-vmapped.
    # However, functional_potential_to_transition itself is just a function.
    # If the input 'potentials' is a batched StandardGaussian, potentials.to_std() will work because of @auto_vmap on to_std.
    # Let's check if functional_potential_to_transition works directly on batched objects.

    transitions = functional_potential_to_transition(potentials)

    self.assertEqual(transitions.batch_size, batch_size)
    self.assertEqual(transitions.A.shape, (batch_size, self.dim, self.x_dim))
    self.assertEqual(transitions.u.shape, (batch_size, self.dim))
    self.assertEqual(transitions.Sigma.shape, (batch_size, self.dim, self.dim))

if __name__ == '__main__':
  unittest.main()
