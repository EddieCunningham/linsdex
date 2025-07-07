import jax
import jax.numpy as jnp
import jax.random as random
import unittest
import jax.tree_util as jtu

from linsdex.potential.gaussian.dist import NaturalGaussian, StandardGaussian, MixedGaussian
from linsdex.matrix.dense import DenseMatrix
from linsdex.linear_functional.linear_functional import LinearFunctional, resolve_linear_functional
from linsdex.linear_functional.quadratic_form import resolve_quadratic_form, QuadraticForm
from linsdex.linear_functional.functional_ops import vdot
from linsdex.potential.gaussian.config import USE_CHOLESKY_SAMPLING

class TestFunctionalGaussian(unittest.TestCase):
  def setUp(self):
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.dim = 4
    self.x_dim = self.dim  # Dimension of the latent variable for the functional

    # Create components for the gaussian
    self.J = DenseMatrix(random.normal(self.key, (self.dim, self.dim)))
    self.logZ = random.normal(self.key, ())

    # Create components for the linear functional
    self.A = DenseMatrix(random.normal(self.key, (self.dim, self.x_dim)))
    self.b = random.normal(self.key, (self.dim,))
    self.h_lf = LinearFunctional(self.A, self.b)

    # Latent variable
    self.x = random.normal(self.key, (self.x_dim,))

    # The resolved, standard h vector
    self.h_vec = self.h_lf(self.x)

    # Create the two versions of the NaturalGaussian
    self.ng_functional = NaturalGaussian(self.J, self.h_lf, self.logZ)
    self.ng_standard = NaturalGaussian(self.J, self.h_vec, self.logZ)

    # Create StandardGaussian instances
    self.sg_functional = StandardGaussian(self.h_lf, self.J, self.logZ)
    self.sg_standard = StandardGaussian(self.h_vec, self.J, self.logZ)

    # Create MixedGaussian instances
    self.mg_functional = MixedGaussian(self.h_lf, self.J, self.logZ)
    self.mg_standard = MixedGaussian(self.h_vec, self.J, self.logZ)

  def test_dim(self):
    self.assertEqual(self.ng_functional.dim, self.dim)
    self.assertEqual(self.ng_standard.dim, self.dim)

  def test_addition(self):
    # Add two functional gaussians
    ng_sum_functional = self.ng_functional + self.ng_functional

    # Resolve the result
    resolved_h = resolve_linear_functional(ng_sum_functional.h, self.x)

    # Add two standard gaussians
    ng_sum_standard = self.ng_standard + self.ng_standard

    self.assertTrue(jnp.allclose(resolved_h, ng_sum_standard.h))
    self.assertTrue(jnp.allclose(ng_sum_functional.logZ, ng_sum_standard.logZ))

  def test_natural_to_nat(self):
    nat_functional = self.ng_functional.to_nat()
    self.assertIs(nat_functional, self.ng_functional)

  def test_natural_to_joint_raises_error(self):
    with self.assertRaises(NotImplementedError):
      self.ng_functional.to_joint(dim=2)

  def test_natural_normalizing_constant(self):
    # Get the quadratic form from the functional version
    nc_qf = self.ng_functional.normalizing_constant()

    # Resolve it with the latent variable
    resolved_nc = resolve_quadratic_form(nc_qf, self.x)

    # Get the scalar value from the standard version
    expected_nc = self.ng_standard.normalizing_constant()

    self.assertTrue(jnp.allclose(resolved_nc, expected_nc))

  def test_natural_call(self):
    y = random.normal(self.key, (self.dim,))

    # Get the quadratic form from the functional version
    call_qf = self.ng_functional(y)

    # Resolve it
    resolved_call = resolve_quadratic_form(call_qf, self.x)

    # Get the scalar value from the standard version
    expected_call = self.ng_standard(y)

    self.assertTrue(jnp.allclose(resolved_call, expected_call))

  def test_natural_score(self):
    y = random.normal(self.key, (self.dim,))

    # Get the linear functional from the score
    score_lf = self.ng_functional.score(y)

    # Resolve it
    resolved_score = resolve_linear_functional(score_lf, self.x)

    # Get the vector from the standard version
    expected_score = self.ng_standard.score(y)

    self.assertTrue(jnp.allclose(resolved_score, expected_score))

  def test_standard_to_nat(self):
    # Convert functional StandardGaussian to NaturalGaussian
    ng_from_sg = self.sg_functional.to_nat()

    # Resolve the h from the new NaturalGaussian
    resolved_h = resolve_linear_functional(ng_from_sg.h, self.x)

    # Get the h from the standard conversion
    expected_h = self.sg_standard.to_nat().h

    self.assertTrue(jnp.allclose(resolved_h, expected_h))

  def test_standard_normalizing_constant(self):
    nc_qf = self.sg_functional.normalizing_constant()
    resolved_nc = resolve_quadratic_form(nc_qf, self.x)
    expected_nc = self.sg_standard.normalizing_constant()
    self.assertTrue(jnp.allclose(resolved_nc, expected_nc))

  def test_standard_call(self):
    y = random.normal(self.key, (self.dim,))
    call_qf = self.sg_functional(y)
    resolved_call = resolve_quadratic_form(call_qf, self.x)
    expected_call = self.sg_standard(y)
    self.assertTrue(jnp.allclose(resolved_call, expected_call))

  def test_standard_score(self):
    y = random.normal(self.key, (self.dim,))
    score_lf = self.sg_functional.score(y)
    resolved_score = resolve_linear_functional(score_lf, self.x)
    expected_score = self.sg_standard.score(y)
    self.assertTrue(jnp.allclose(resolved_score, expected_score))

  def test_standard_log_prob(self):
    y = random.normal(self.key, (self.dim,))
    log_prob_qf = self.sg_functional.log_prob(y)
    resolved_log_prob = resolve_quadratic_form(log_prob_qf, self.x)
    expected_log_prob = self.sg_standard.log_prob(y)
    self.assertTrue(jnp.allclose(resolved_log_prob, expected_log_prob))

  def test_standard_sample(self):
    sample_lf = self.sg_functional.sample(self.key)
    resolved_sample = resolve_linear_functional(sample_lf, self.x)
    # To test the sample, we check if the noise it corresponds to is correct
    noise = self.sg_standard.get_noise(resolved_sample)
    noise_functional = self.sg_functional.get_noise(sample_lf)
    expected_noise = resolve_linear_functional(noise_functional, self.x)

    self.assertTrue(jnp.allclose(noise, expected_noise))

  def test_mixed_to_nat(self):
    ng_from_mg = self.mg_functional.to_nat()
    resolved_h = resolve_linear_functional(ng_from_mg.h, self.x)
    expected_h = self.mg_standard.to_nat().h
    self.assertTrue(jnp.allclose(resolved_h, expected_h))

  def test_mixed_to_std(self):
    sg_from_mg = self.mg_functional.to_std()
    resolved_mu = resolve_linear_functional(sg_from_mg.mu, self.x)
    expected_mu = self.mg_standard.to_std().mu
    self.assertTrue(jnp.allclose(resolved_mu, expected_mu))

  def test_mixed_normalizing_constant(self):
    nc_qf = self.mg_functional.normalizing_constant()
    resolved_nc = resolve_quadratic_form(nc_qf, self.x)
    expected_nc = self.mg_standard.normalizing_constant()
    self.assertTrue(jnp.allclose(resolved_nc, expected_nc))

  def test_mixed_call(self):
    y = random.normal(self.key, (self.dim,))
    call_lf = self.mg_functional(y)
    resolved_call = resolve_linear_functional(call_lf, self.x)
    resolved_call2 = resolve_quadratic_form(resolved_call, self.x)
    expected_call = self.mg_standard(y)
    self.assertTrue(jnp.allclose(resolved_call2, expected_call))

  def test_mixed_log_prob(self):
    y = random.normal(self.key, (self.dim,))
    log_prob_qf = self.mg_functional.log_prob(y)
    resolved_log_prob = resolve_quadratic_form(log_prob_qf, self.x)
    expected_log_prob = self.mg_standard.log_prob(y)
    self.assertTrue(jnp.allclose(resolved_log_prob, expected_log_prob))

  def test_mixed_score(self):
    y = random.normal(self.key, (self.dim,))
    score_lf = self.mg_functional.score(y)
    resolved_score = resolve_linear_functional(score_lf, self.x)
    expected_score = self.mg_standard.score(y)
    self.assertTrue(jnp.allclose(resolved_score, expected_score))

  def test_mixed_sample(self):
    sample_lf = self.mg_functional.sample(self.key)
    resolved_sample = resolve_linear_functional(sample_lf, self.x)
    noise = self.mg_standard.get_noise(resolved_sample)
    expected_noise_functional = self.mg_functional.get_noise(sample_lf)
    expected_noise = resolve_linear_functional(expected_noise_functional, self.x)
    self.assertTrue(jnp.allclose(noise, expected_noise))


if __name__ == '__main__':
  unittest.main()