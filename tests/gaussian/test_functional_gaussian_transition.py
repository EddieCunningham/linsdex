import unittest
import jax
import jax.numpy as jnp
import jax.random as random
from linsdex.potential.gaussian.transition import GaussianTransition, JointPotential
from linsdex.potential.gaussian.dist import NaturalGaussian, StandardGaussian, MixedGaussian
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.functional_ops import resolve_functional
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.tags import TAGS
import jax.tree_util as jtu

class TestFunctionalGaussianTransition(unittest.TestCase):
  def setUp(self):
    self.key = random.PRNGKey(0)
    self.dim_x = 4
    self.dim_y = 4
    self.latent_dim = 4
    k1, k2, k3, k4, k5, k6, k7 = random.split(self.key, 7)

    # 1. Create a base GaussianTransition
    A = random.normal(k1, (self.dim_y, self.dim_x))
    u = random.normal(k2, (self.dim_y,))
    Sigma_raw = random.normal(k3, (self.dim_y, self.dim_y))
    Sigma = Sigma_raw @ Sigma_raw.T + jnp.eye(self.dim_y) * 1e-4
    self.transition = GaussianTransition(
      DenseMatrix(A, tags=TAGS.no_tags),
      u,
      DenseMatrix(Sigma, tags=TAGS.no_tags),
      logZ=jnp.array(0.0)
    )

    # 2. Create the latent variable and functional mean
    self.x = random.normal(k4, (self.latent_dim,))

    # 3. Create functional and standard potentials over y
    J_raw = random.normal(k5, (self.dim_y, self.dim_y))
    J = J_raw @ J_raw.T + jnp.eye(self.dim_y) * 1e-4
    J_mat = DenseMatrix(J, tags=TAGS.no_tags)

    # Create mu and h functionals independently
    mu_A = DenseMatrix(random.normal(k6, (self.dim_y, self.latent_dim)))
    mu_b = random.normal(k7, (self.dim_y,))
    self.mu_lf = LinearFunctional(mu_A, mu_b)
    self.mu_vec = resolve_functional(self.mu_lf, self.x)

    # For the standard case, ensure h and mu are consistent
    h_vec = J_mat @ self.mu_vec
    # For the functional case, h_lf is just a placeholder to create the potential
    # The correctness of the update rules will be checked against the standard case
    self.h_lf = J_mat @ self.mu_lf


    self.sg_functional = StandardGaussian(mu=self.mu_lf, Sigma=J_mat.get_inverse())
    self.sg_standard = StandardGaussian(mu=self.mu_vec, Sigma=J_mat.get_inverse())

    self.ng_functional = NaturalGaussian(h=self.h_lf, J=J_mat)
    self.ng_standard = NaturalGaussian(h=h_vec, J=J_mat)

    self.mg_functional = MixedGaussian(mu=self.mu_lf, J=J_mat)
    self.mg_standard = MixedGaussian(mu=self.mu_vec, J=J_mat)

  def _compare_pytree(self, pytree_func, pytree_std):
    resolved_func = resolve_functional(pytree_func, self.x)
    are_close = jtu.tree_map(jnp.allclose, resolved_func, pytree_std)
    self.assertTrue(jtu.tree_all(are_close))

  def test_update_y_with_standard_functional(self):
    joint_functional = self.transition.update_y(self.sg_functional)
    joint_standard = self.transition.update_y(self.sg_standard)

    self._compare_pytree(joint_functional, joint_standard)

    prior_func = joint_functional.prior
    prior_std = joint_standard.prior
    self.assertTrue(jnp.allclose(prior_func.Sigma.as_matrix(), prior_std.Sigma.as_matrix()))
    resolved_mu = resolve_functional(prior_func.mu, self.x)
    self.assertTrue(jnp.allclose(resolved_mu, prior_std.mu))

  def test_update_y_with_natural_functional(self):
    joint_functional = self.transition.update_y(self.ng_functional)
    joint_standard = self.transition.update_y(self.ng_standard)

    self._compare_pytree(joint_functional, joint_standard)

    prior_func = joint_functional.prior
    prior_std = joint_standard.prior
    self.assertTrue(jnp.allclose(prior_func.J.as_matrix(), prior_std.J.as_matrix()))
    resolved_h = resolve_functional(prior_func.h, self.x)
    self.assertTrue(jnp.allclose(resolved_h, prior_std.h))

  def test_update_y_with_mixed_functional(self):
    joint_functional = self.transition.update_y(self.mg_functional)
    joint_standard = self.transition.update_y(self.mg_standard)

    self._compare_pytree(joint_functional, joint_standard)

    prior_func = joint_functional.prior
    prior_std = joint_standard.prior
    self.assertTrue(jnp.allclose(prior_func.J.as_matrix(), prior_std.J.as_matrix()))
    resolved_mu = resolve_functional(prior_func.mu, self.x)
    self.assertTrue(jnp.allclose(resolved_mu, prior_std.mu))

if __name__ == '__main__':
  unittest.main()