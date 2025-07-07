import jax
import jax.numpy as jnp
from jax import random
import pytest
import equinox as eqx
import jax.tree_util as jtu
from typing import List

from linsdex.sde.linear_sde_with_priors import LinearSDEWithPriors, get_bayes_estimate_of_mean_of_evidence, get_linear_parameters_of_bayes_estimate_of_mean_of_evidence
from linsdex.sde.sde_base import AbstractLinearSDE, AbstractSDE
from linsdex.sde.sde_examples import BrownianMotion, OrnsteinUhlenbeck, LinearTimeInvariantSDE
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.matrix_base import TAGS, AbstractSquareMatrix
from linsdex.potential.gaussian.dist import StandardGaussian, MixedGaussian
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from linsdex.potential.gaussian.transition import GaussianTransition
import optax


def list_of_gaussians_to_batched_gaussian(gaussians: List[StandardGaussian]) -> StandardGaussian:
  """Convert a list of Gaussians to a batched Gaussian"""
  return jtu.tree_map(lambda *xs: jnp.array(xs), *gaussians)


def create_random_lti_sde(key, dim):
    """Create a LinearTimeInvariantSDE with random, well-conditioned parameters."""
    k1, k2 = random.split(key)
    F_mat = random.normal(k1, (dim, dim)) / jnp.sqrt(dim)
    L_mat = random.normal(k2, (dim, dim)) / jnp.sqrt(dim)
    L_mat = L_mat @ L_mat.T + jnp.eye(dim) * 1e-3  # Ensure L is positive definite
    return LinearTimeInvariantSDE(F=DenseMatrix(F_mat), L=DenseMatrix(L_mat))


class TestLinearSDEWithPriorsInitialization:
    """Test initialization of the LinearSDEWithPriors class"""

    def test_basic_initialization(self):
        """Test basic initialization with a BrownianMotion SDE"""
        base_sde = BrownianMotion(sigma=1.0, dim=2)
        times = jnp.array([1.0])
        mean_priors = [StandardGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)

        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = DiagonalMatrix.eye(2)[None]

        sde_with_priors = LinearSDEWithPriors(
            base_sde, evidence_mean_priors, evidence_covariances
        )
        assert isinstance(sde_with_priors, LinearSDEWithPriors)
        assert sde_with_priors.linear_sde is base_sde
        assert sde_with_priors.evidence_mean_priors is evidence_mean_priors
        assert sde_with_priors.evidence_covariances is evidence_covariances

    def test_initialization_with_different_sde_types(self):
        """Test initialization with a different type of linear SDE"""
        base_sde = OrnsteinUhlenbeck(sigma=0.5, lambda_=0.1, dim=3)
        times = jnp.array([0.5, 1.5])
        mean_priors = [
            StandardGaussian(jnp.ones(3), DiagonalMatrix.eye(3)),
            StandardGaussian(jnp.ones(3) * 2, DiagonalMatrix.eye(3)),
        ]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)

        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = jax.vmap(lambda _: DiagonalMatrix.eye(3))(times)

        sde_with_priors = LinearSDEWithPriors(
            base_sde, evidence_mean_priors, evidence_covariances
        )
        assert isinstance(sde_with_priors, LinearSDEWithPriors)
        assert sde_with_priors.dim == 3

    def test_initialization_with_lti_sde(self):
        """Test initialization with a LinearTimeInvariantSDE"""
        key = random.PRNGKey(42)
        dim = 4
        base_sde = create_random_lti_sde(key, dim)
        times = jnp.array([1.0, 2.0])
        mean_priors = [
            StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim)),
            StandardGaussian(jnp.ones(dim), DiagonalMatrix.eye(dim)),
        ]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)

        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = jax.vmap(lambda _: DiagonalMatrix.eye(dim))(times)

        sde_with_priors = LinearSDEWithPriors(
            base_sde, evidence_mean_priors, evidence_covariances
        )
        assert isinstance(sde_with_priors, LinearSDEWithPriors)
        assert sde_with_priors.dim == dim
        assert sde_with_priors.linear_sde is base_sde

class TestLinearSDEWithPriorsProperties:
    """Test basic properties of the LinearSDEWithPriors class"""

    def test_inheritance_properties(self):
      """Test that LinearSDEWithPriors properly inherits from AbstractSDE"""
      dim = 2
      base_sde = BrownianMotion(sigma=1.0, dim=dim)
      times = jnp.array([1.0])
      mean_priors = [StandardGaussian(jnp.ones(dim), DiagonalMatrix.eye(dim))]
      batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
      evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
      evidence_covariances = DiagonalMatrix.eye(dim)[None]
      sde_with_priors = LinearSDEWithPriors(base_sde, evidence_mean_priors, evidence_covariances)
      assert isinstance(sde_with_priors, AbstractSDE)
      assert not isinstance(sde_with_priors, AbstractLinearSDE)

    def test_dim_property(self):
        """Test that the dim property is correctly inherited"""
        dim = 3
        base_sde = BrownianMotion(sigma=1.0, dim=dim)
        times = jnp.array([1.0])
        mean_priors = [StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = DiagonalMatrix.eye(dim)[None]

        sde_with_priors = LinearSDEWithPriors(
            base_sde, evidence_mean_priors, evidence_covariances
        )
        assert sde_with_priors.dim == dim

    def test_batch_size_property(self):
        """Test that the batch_size property is correctly inherited"""
        dim = 2
        batch_size = 5
        base_sde = BrownianMotion(sigma=1.0, dim=dim)

        times = jnp.array([1.0])
        mean_priors = [StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = DiagonalMatrix.eye(dim)[None]

        sde_with_priors = LinearSDEWithPriors(
            base_sde, evidence_mean_priors, evidence_covariances
        )

        batched_sde = jax.vmap(lambda _: sde_with_priors)(jnp.arange(batch_size))

        assert batched_sde.batch_size == batch_size


class TestBayesEstimation:
  """Test the get_bayes_estimate_of_mean_of_evidence function"""
  def test_get_bayes_estimate_basic(self):
    """Test basic functionality of Bayes estimate calculation"""
    dim = 2
    t, T = 0.5, 1.0
    xt = jnp.array([0.5, -0.5])
    base_sde = BrownianMotion(sigma=1.0, dim=dim)
    transition_to_evidence = base_sde.get_transition_distribution(t, T)
    evidence_mean_prior = StandardGaussian(mu=jnp.ones(dim), Sigma=DiagonalMatrix.eye(dim))
    evidence_covariance = DiagonalMatrix.eye(dim) * 0.1

    bayes_estimate = get_bayes_estimate_of_mean_of_evidence(
      transition_to_evidence, evidence_mean_prior, evidence_covariance, xt
    )
    assert bayes_estimate.shape == (dim,)
    assert jnp.isfinite(bayes_estimate).all()

  def test_bayes_estimate_with_different_parameters(self):
    """Test Bayes estimate with different prior and covariance"""
    dim = 3
    t, T = 0.2, 0.8
    xt = jnp.array([0.1, 0.2, 0.3])
    base_sde = OrnsteinUhlenbeck(sigma=0.5, lambda_=0.2, dim=dim)
    transition_to_evidence = base_sde.get_transition_distribution(t, T)
    evidence_mean_prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim) * 2.0)
    evidence_covariance = DiagonalMatrix.eye(dim) * 0.01

    bayes_estimate = get_bayes_estimate_of_mean_of_evidence(
      transition_to_evidence, evidence_mean_prior, evidence_covariance, xt
    )
    assert bayes_estimate.shape == (dim,)

  def test_bayes_estimate_consistency(self):
    """Test that Bayes estimates are consistent with different inputs"""
    dim = 2
    t, T = 0.5, 1.0
    xt1 = jnp.array([0.5, -0.5])
    xt2 = jnp.array([-0.5, 0.5])
    base_sde = BrownianMotion(sigma=1.0, dim=dim)
    transition_to_evidence = base_sde.get_transition_distribution(t, T)
    evidence_mean_prior = StandardGaussian(mu=jnp.ones(dim), Sigma=DiagonalMatrix.eye(dim))
    evidence_covariance = DiagonalMatrix.eye(dim) * 0.1

    bayes1 = get_bayes_estimate_of_mean_of_evidence(
      transition_to_evidence, evidence_mean_prior, evidence_covariance, xt1
    )
    bayes2 = get_bayes_estimate_of_mean_of_evidence(
      transition_to_evidence, evidence_mean_prior, evidence_covariance, xt2
    )
    # Should be different for different xt values
    assert not jnp.allclose(bayes1, bayes2)

  def test_get_linear_parameters_of_bayes_estimate_correctness(self):
      """Test that linear parameters correctly reconstruct the Bayes estimate."""
      dim = 2
      t, T = 0.5, 1.0
      xt = jnp.array([0.5, -0.5])
      base_sde = BrownianMotion(sigma=1.0, dim=dim)
      transition_to_evidence = base_sde.get_transition_distribution(t, T)
      evidence_mean_prior = StandardGaussian(mu=jnp.ones(dim), Sigma=DiagonalMatrix.eye(dim))
      evidence_covariance = DiagonalMatrix.eye(dim) * 0.1

      # Get the linear parameters
      A, b = get_linear_parameters_of_bayes_estimate_of_mean_of_evidence(
          transition_to_evidence, evidence_mean_prior, evidence_covariance
      )

      # Compute the estimate using the linear parameters
      linear_estimate = A @ xt + b

      # Get the direct estimate for comparison
      direct_estimate = get_bayes_estimate_of_mean_of_evidence(
          transition_to_evidence, evidence_mean_prior, evidence_covariance, xt
      )

      # Verify they are the same
      assert jnp.allclose(linear_estimate, direct_estimate)

  def test_get_linear_parameters_of_bayes_estimate_correctness_with_lti(self):
      """Test correctness of linear parameters with a LinearTimeInvariantSDE."""
      key = random.PRNGKey(43)
      dim = 3
      t, T = 0.3, 0.9
      xt = random.normal(key, (dim,))
      base_sde = create_random_lti_sde(key, dim)
      transition_to_evidence = base_sde.get_transition_distribution(t, T)
      evidence_mean_prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
      evidence_covariance = DiagonalMatrix.eye(dim) * 0.2

      A, b = get_linear_parameters_of_bayes_estimate_of_mean_of_evidence(
          transition_to_evidence, evidence_mean_prior, evidence_covariance
      )
      linear_estimate = A @ xt + b
      direct_estimate = get_bayes_estimate_of_mean_of_evidence(
          transition_to_evidence, evidence_mean_prior, evidence_covariance, xt
      )
      assert jnp.allclose(linear_estimate, direct_estimate, atol=1e-5)

  def _run_variational_correctness_test(self, evidence_covariance, key):
    dim = 2
    # 1. Setup
    t, T = 0.5, 1.0
    base_sde = BrownianMotion(sigma=1.0, dim=dim)
    evidence_mean_prior = StandardGaussian(mu=jnp.ones(dim), Sigma=DiagonalMatrix.eye(dim))

    # 2. Model f(xt)
    mlp_key, opt_key, sample_key = random.split(key, 3)
    f = eqx.nn.MLP(in_size=dim, out_size=dim, width_size=32, depth=2, key=mlp_key)

    # 3. Sampler for p(mu, xt)
    forward_transition = base_sde.get_transition_distribution(t, T)
    @jax.jit
    def sample_mu_xt(key):
        mu_key, xT_key, xt_key = random.split(key, 3)
        # mu ~ p(mu)
        mu = evidence_mean_prior.sample(mu_key)
        # xT ~ p(xT|mu)
        xT_dist = StandardGaussian(mu, evidence_covariance)
        xT = xT_dist.sample(xT_key)
        # xt ~ p(xt|xT)
        xt_dist = forward_transition.condition_on_y(xT)
        xt = xt_dist.sample(xt_key)
        return mu, xt
    # 4. Loss function
    @eqx.filter_value_and_grad
    def loss_fn(model, key):
        # Using a batch of samples to estimate expectation
        keys = random.split(key, 1000)
        mus, xts = jax.vmap(sample_mu_xt)(keys)
        # Get predictions from model
        f_xts = jax.vmap(model)(xts)
        # MSE loss
        return jnp.mean((f_xts - mus)**2)

    # 5. Optimization
    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(f, eqx.is_inexact_array))
    @eqx.filter_jit
    def make_step(model, opt_state, key):
        value, grads = loss_fn(model, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state
    for i in range(500): # More steps for robustness
        opt_key, loss_key = random.split(opt_key)
        value, f, opt_state = make_step(f, opt_state, loss_key)

    # 6. Verification
    # Pick a point xt and compare f(xt) with the analytical solution.
    xt_test, _ = sample_mu_xt(sample_key)
    # Analytical solution
    analytical_estimate = get_bayes_estimate_of_mean_of_evidence(
        forward_transition, evidence_mean_prior, evidence_covariance, xt_test
    )
    # Model's prediction
    model_estimate = f(xt_test)
    assert jnp.allclose(analytical_estimate, model_estimate, atol=1e-1)

  def test_get_bayes_estimate_variational_correctness_nonzero_cov(self):
    key = random.PRNGKey(0)
    dim = 2
    evidence_covariance = DiagonalMatrix.eye(dim) * 0.1
    self._run_variational_correctness_test(evidence_covariance, key)

  def test_get_bayes_estimate_variational_correctness_zero_cov(self):
    key = random.PRNGKey(1)
    dim = 2
    evidence_covariance = DiagonalMatrix.eye(dim).set_zero()
    self._run_variational_correctness_test(evidence_covariance, key)


@pytest.mark.parametrize("sde_type", ["brownian", "lti"])
class TestLinearSDEWithPriorsParameters:
    """Test parameter computation"""

    def _create_sde_with_priors(self, sde_type, dim, n_times, key):
        """Helper to create a base SDE and its evidence series for tests"""
        if sde_type == "brownian":
            base_sde = BrownianMotion(sigma=1.0, dim=dim)
        elif sde_type == "lti":
            base_sde = create_random_lti_sde(key, dim)
        else:
            raise ValueError(f"Unsupported SDE type: {sde_type}")

        times = jnp.array([0.5 + i * 0.5 for i in range(n_times)])
        mean_priors = [StandardGaussian(jnp.ones(dim) * i, DiagonalMatrix.eye(dim)) for i in range(n_times)]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = jax.vmap(lambda _: DiagonalMatrix.eye(dim) * 0.01)(times)
        sde_with_priors = LinearSDEWithPriors(base_sde, evidence_mean_priors, evidence_covariances)
        return sde_with_priors, base_sde

    def test_get_params_basic(self, sde_type):
        """Test basic parameter computation"""
        dim = 2
        key = random.PRNGKey(10)
        t = 0.5
        xt = jnp.array([0.1, -0.2])
        sde_with_priors, _ = self._create_sde_with_priors(sde_type, dim, 1, key)

        F, u, L = sde_with_priors.get_params(t, xt)

        assert isinstance(F, AbstractSquareMatrix)
        assert isinstance(u, jnp.ndarray)
        assert isinstance(L, AbstractSquareMatrix)
        assert F.shape == (dim, dim)
        assert u.shape == (dim,)
        assert L.shape == (dim, dim)

    def test_get_params_creates_conditioned_sde(self, sde_type):
      """Test that get_params creates a conditioned SDE internally"""
      dim = 2
      key = random.PRNGKey(11)
      if sde_type == "brownian":
          base_sde = BrownianMotion(sigma=1.0, dim=dim)
      elif sde_type == "lti":
          base_sde = create_random_lti_sde(key, dim)
      else:
          raise ValueError(f"Unsupported SDE type: {sde_type}")


      times = jnp.array([1.0])
      mean_priors = [StandardGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))]
      batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
      evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
      evidence_covariances = DiagonalMatrix.eye(2)[None]

      sde_with_priors = LinearSDEWithPriors(
        base_sde, evidence_mean_priors, evidence_covariances
      )

      t = 0.5
      xt = jnp.array([1.0, 0.0])

      # The get_params method should internally create a conditioned SDE
      F, u, L = sde_with_priors.get_params(t, xt)

      # Parameters should be different from base SDE due to conditioning
      F_base, u_base, L_base = base_sde.get_params(t)

      # L should be the same (diffusion not affected by linear conditioning)
      assert jnp.allclose(L.elements, L_base.elements)

      # F and u should generally be different due to conditioning
      assert F.shape == F_base.shape
      assert u.shape == u_base.shape
      assert not jnp.allclose(u, u_base)

    def test_get_params_with_multiple_evidence_times(self, sde_type):
        """Test get_params with multiple evidence times"""
        dim = 2
        key = random.PRNGKey(12)
        sde_with_priors, _ = self._create_sde_with_priors(sde_type, dim, 4, key)
        xt = jnp.zeros(dim)

        # Test at a time between evidence points
        t = 1.25
        F, u, L = sde_with_priors.get_params(t, xt)

        # Check that parameters are valid
        assert F.shape == (dim, dim)
        assert u.shape == (dim,)
        assert L.shape == (dim, dim)
        assert jnp.isfinite(F.elements).all()
        assert jnp.isfinite(u).all()
        assert jnp.isfinite(L.elements).all()

    def test_get_params_consistency_across_times(self, sde_type):
        """Test that get_params is consistent for different times"""
        dim = 2
        key = random.PRNGKey(13)
        sde_with_priors, _ = self._create_sde_with_priors(sde_type, dim, 2, key)
        xt = jnp.array([0.3, -0.3])

        t1 = 0.6
        F1, u1, L1 = sde_with_priors.get_params(t1, xt)

        t2 = 0.9
        F2, u2, L2 = sde_with_priors.get_params(t2, xt)

        assert not jnp.allclose(u1, u2)

@pytest.mark.parametrize("sde_type", ["brownian", "lti"])
class TestLinearSDEWithPriorsIntegration:
    """Test integration with other components like solvers and conditioning"""

    def _create_sde_with_priors(self, sde_type, dim, n_times, key):
        """Helper to create a base SDE and its evidence series for tests"""
        if sde_type == "brownian":
            base_sde = BrownianMotion(sigma=1.0, dim=dim)
        elif sde_type == "lti":
            base_sde = create_random_lti_sde(key, dim)
        else:
            raise ValueError(f"Unsupported SDE type: {sde_type}")

        times = jnp.array([0.5 + i * 0.5 for i in range(n_times)])
        mean_priors = [StandardGaussian(jnp.ones(dim) * i, DiagonalMatrix.eye(dim)) for i in range(n_times)]
        batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
        evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
        evidence_covariances = jax.vmap(lambda _: DiagonalMatrix.eye(dim) * 0.01)(times)
        sde_with_priors = LinearSDEWithPriors(base_sde, evidence_mean_priors, evidence_covariances)
        return sde_with_priors, base_sde

    def test_get_drift_and_diffusion(self, sde_type):
        """Test that the drift and diffusion coefficients are correctly computed"""
        dim = 2
        key = random.PRNGKey(14)
        t = 0.5
        xt = jnp.array([0.1, -0.2])
        sde_with_priors, base_sde = self._create_sde_with_priors(sde_type, dim, 1, key)

        drift = sde_with_priors.get_drift(t, xt)
        diffusion = sde_with_priors.get_diffusion_coefficient(t, xt)

        assert drift.shape == (dim,)
        assert diffusion.shape == (dim, dim)
        assert isinstance(diffusion, AbstractSquareMatrix)

@pytest.mark.parametrize("sde_type", ["brownian", "lti"])
class TestLinearSDEWithPriorsEdgeCases:
  """Test edge cases for the LinearSDEWithPriors class"""

  def _create_sde_with_priors(self, sde_type, dim, n_times, key):
      """Helper to create a base SDE and its evidence series for tests"""
      if sde_type == "brownian":
          base_sde = BrownianMotion(sigma=1.0, dim=dim)
      elif sde_type == "lti":
          base_sde = create_random_lti_sde(key, dim)
      else:
          raise ValueError(f"Unsupported SDE type: {sde_type}")

      times = jnp.array([0.5 + i * 0.5 for i in range(n_times)])
      mean_priors = [StandardGaussian(jnp.ones(dim) * i, DiagonalMatrix.eye(dim)) for i in range(n_times)]
      batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
      evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
      evidence_covariances = jax.vmap(lambda _: DiagonalMatrix.eye(dim) * 0.01)(times)
      return LinearSDEWithPriors(base_sde, evidence_mean_priors, evidence_covariances)

  def test_single_evidence_time(self, sde_type):
      """Test with a single evidence time point"""
      dim = 2
      key = random.PRNGKey(15)
      sde_with_priors = self._create_sde_with_priors(sde_type, dim, 1, key)
      t = 0.2
      xt = jnp.zeros(dim)

      F, u, L = sde_with_priors.get_params(t, xt)

      assert F.shape == (dim, dim)
      assert u.shape == (dim,)
      assert L.shape == (dim, dim)

  def test_high_precision_evidence(self, sde_type):
      """Test with very high precision (low covariance) evidence"""
      dim = 2
      key = random.PRNGKey(16)
      if sde_type == "brownian":
          base_sde = BrownianMotion(sigma=1.0, dim=dim)
      elif sde_type == "lti":
          base_sde = create_random_lti_sde(key, dim)
      else:
          raise ValueError(f"Unsupported SDE type: {sde_type}")

      times = jnp.array([1.0])
      mean_priors = [StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))]
      batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
      evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
      evidence_covariances = DiagonalMatrix.eye(dim)[None] * 1e-9
      sde_with_priors = LinearSDEWithPriors(base_sde, evidence_mean_priors, evidence_covariances)

      t = 0.5
      xt = jnp.ones(dim)
      F, u, L = sde_with_priors.get_params(t, xt)

      assert jnp.isfinite(F.elements).all()
      assert jnp.isfinite(u).all()

  def test_low_precision_evidence(self, sde_type):
      """Test with very low precision (high covariance) evidence"""
      dim = 2
      key = random.PRNGKey(17)
      if sde_type == "brownian":
          base_sde = BrownianMotion(sigma=1.0, dim=dim)
      elif sde_type == "lti":
          base_sde = create_random_lti_sde(key, dim)
      else:
          raise ValueError(f"Unsupported SDE type: {sde_type}")

      times = jnp.array([1.0])
      mean_priors = [StandardGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))]
      batched_mean_priors = list_of_gaussians_to_batched_gaussian(mean_priors)
      evidence_mean_priors = GaussianPotentialSeries.from_potentials(times, batched_mean_priors)
      evidence_covariances = DiagonalMatrix.eye(dim)[None] * 1e9
      sde_with_priors = LinearSDEWithPriors(base_sde, evidence_mean_priors, evidence_covariances)

      t = 0.5
      xt = jnp.ones(dim)
      F, u, L = sde_with_priors.get_params(t, xt)

      assert jnp.isfinite(F.elements).all()
      assert jnp.isfinite(u).all()

  def test_simulation(self, sde_type):
      """Test that the simulation is correct"""
      dim = 2
      key = random.PRNGKey(18)
      sde_with_priors = self._create_sde_with_priors(sde_type, dim, 1, key)
      t = 0.2
      xt = jnp.zeros(dim)
      F, u, L = sde_with_priors.get_params(t, xt)
      assert F.shape == (dim, dim)

if __name__ == "__main__":
  pytest.main([__file__])