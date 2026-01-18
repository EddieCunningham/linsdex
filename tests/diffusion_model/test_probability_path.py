import pytest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import random
from linsdex.diffusion_model.probability_path import DiffusionModelComponents, ProbabilityPath, probability_path_transition
from linsdex.diffusion_model.probability_path import DiffusionModelConversions, Y1ToBwdMean, Y1ToMarginalMean
from linsdex import OrnsteinUhlenbeck, empirical_dist, w2_distance, LinearTimeInvariantSDE
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.potential.gaussian.dist import StandardGaussian, MixedGaussian
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from jaxtyping import Array, PRNGKeyArray, Float
import jax.tree_util as jtu
from functools import partial
import diffrax
from linsdex.sde.sde_base import AbstractLinearSDE
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.functional_ops import resolve_functional
from linsdex.matrix.matrix_base import AbstractSquareMatrix
import equinox as eqx

@pytest.fixture
def key():
    """JAX PRNGKey fixture."""
    return random.PRNGKey(0)


@pytest.fixture
def diffusion_components():
    """Create a DiffusionModelComponents object for testing."""
    dim = 2
    t0 = 0.0
    t1 = 1.0
    sde = OrnsteinUhlenbeck(dim=dim, sigma=1.0, lambda_=1.0)
    prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
    evidence_cov = DiagonalMatrix(0.1 * jnp.ones(dim))
    return DiffusionModelComponents(
        linear_sde=sde,
        t0=t0,
        x_t0_prior=prior,
        t1=t1,
        evidence_cov=evidence_cov
    )

@pytest.fixture
def lti_sde_diffusion_components(key):
    """Create a DiffusionModelComponents object with a LTI SDE for testing."""
    dim = 2
    t0 = 0.0
    t1 = 1.0
    k1, k2 = random.split(key)
    F_mat = DenseMatrix(random.normal(k1, (dim, dim)))
    L_mat = DenseMatrix(random.normal(k2, (dim, dim)))
    sde = LinearTimeInvariantSDE(F=F_mat, L=L_mat)
    prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
    evidence_cov = DiagonalMatrix(0.1 * jnp.ones(dim))
    return DiffusionModelComponents(
        linear_sde=sde,
        t0=t0,
        x_t0_prior=prior,
        t1=t1,
        evidence_cov=evidence_cov
    )

@pytest.mark.parametrize("components_fixture", ["diffusion_components", "lti_sde_diffusion_components"])
class TestDiffusionHubConversions:
    """Test the conversion functions in diffusion_hub.py for consistency."""

    def test_y1_bwd_message_conversion(self, components_fixture, request, key):
        """Test y1 <-> backward message conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        key, _ = random.split(key)
        dim = diffusion_components.linear_sde.dim
        y1 = random.normal(key, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        bwd_message = conversions.y1_to_bwd_message(y1)
        y1_recon = conversions.bwd_message_to_y1(bwd_message)

        assert jnp.allclose(y1, y1_recon, atol=1e-5)

    def test_y1_marginal_conversion(self, components_fixture, request, key):
        """Test y1 <-> marginal distribution conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        key, _ = random.split(key)
        dim = diffusion_components.linear_sde.dim
        y1 = random.normal(key, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        marginal = conversions.y1_to_marginal(y1)
        y1_recon = conversions.marginal_to_y1(marginal)

        assert jnp.allclose(y1, y1_recon, atol=1e-5)

    def test_bwd_marginal_conversion(self, components_fixture, request, key):
        """Test backward message <-> marginal distribution conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        key, _ = random.split(key)
        dim = diffusion_components.linear_sde.dim
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        y1_to_bwd_mean = Y1ToBwdMean(diffusion_components, t)
        bwd_mean = random.normal(key, (dim,))
        bwd_message = MixedGaussian(mu=bwd_mean, J=y1_to_bwd_mean.precision)

        marginal = conversions.bwd_message_to_marginal(bwd_message)
        bwd_message_recon = conversions.marginal_to_bwd_message(marginal)

        assert jnp.allclose(bwd_message.mu, bwd_message_recon.mu, atol=1e-5)
        assert jnp.allclose(bwd_message.J.as_matrix(), bwd_message_recon.J.as_matrix(), atol=1e-5)

    def test_bwd_drift_conversion(self, components_fixture, request, key):
        """Test backward message <-> drift conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        key, subkey = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(key, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        y1_to_bwd_mean = Y1ToBwdMean(diffusion_components, t)
        bwd_mean = random.normal(subkey, (dim,))
        bwd_message = MixedGaussian(mu=bwd_mean, J=y1_to_bwd_mean.precision)

        drift = conversions.bwd_message_to_drift(bwd_message, xt)
        bwd_message_recon = conversions.drift_to_bwd_message(xt, drift)

        assert jnp.allclose(bwd_message.mu, bwd_message_recon.mu, atol=1e-5)
        assert jnp.allclose(bwd_message.J.as_matrix(), bwd_message_recon.J.as_matrix(), atol=1e-5)

    def test_marginal_score_conversion(self, components_fixture, request, key):
        """Test marginal distribution <-> score conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        key, subkey = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(key, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        y1_to_marginal_mean = Y1ToMarginalMean(diffusion_components, t)
        marginal_mean = random.normal(subkey, (dim,))
        marginal_cov = y1_to_marginal_mean.precision.get_inverse()
        marginal = StandardGaussian(mu=marginal_mean, Sigma=marginal_cov)

        score = conversions.marginal_to_score(marginal, xt)
        marginal_recon = conversions.score_to_marginal(xt, score)

        assert jnp.allclose(marginal.mu, marginal_recon.mu, atol=1e-5)
        assert jnp.allclose(marginal.Sigma.as_matrix(), marginal_recon.Sigma.as_matrix(), atol=1e-5)

    def test_diffusion_path_quantities_properties(self, components_fixture, request, key):
        """Test the properties of ProbabilityPath."""
        diffusion_components = request.getfixturevalue(components_fixture)
        t = 0.5
        quantities = ProbabilityPath(diffusion_components, t)

        # Test beta_precision property
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, quantities.beta_precision.as_matrix(), quantities.functional_beta_t.J.as_matrix()))

        # Test y1_to_marginal_mean property
        y1_to_marginal_mean = quantities.y1_to_marginal_mean
        assert isinstance(y1_to_marginal_mean, LinearFunctional)
        # Compare to direct calculation
        expected_mu = quantities.functional_pt_given_y1.to_mixed().mu
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, y1_to_marginal_mean, expected_mu))

        # Test marginal_precision property
        marginal_precision = quantities.marginal_precision
        expected_precision = quantities.functional_pt_given_y1.Sigma.get_inverse()
        assert jtu.tree_all(jtu.tree_map(jnp.allclose, marginal_precision.as_matrix(), expected_precision.as_matrix()))

    def test_y1_score_conversion(self, components_fixture, request, key):
        """Test y1 -> score conversion matches equivalent paths."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        y1 = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        score_direct = conversions.y1_to_score(xt, y1)

        # Via marginal
        marginal = conversions.y1_to_marginal(y1)
        score_via_marginal = conversions.marginal_to_score(marginal, xt)
        assert jnp.allclose(score_direct, score_via_marginal, atol=1e-5)

        # Note: equality via the flow path is not required in general; tested separately for flow/score inverses.

    def test_y1_drift_conversion(self, components_fixture, request, key):
        """Test y1 <-> drift conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        y1 = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        drift = conversions.y1_to_drift(y1, xt)
        y1_recon = conversions.drift_to_y1(xt, drift)

        assert jnp.allclose(y1, y1_recon, atol=1e-5)

    def test_drift_flow_conversion(self, components_fixture, request, key):
        """Test drift <-> flow conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        drift = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        flow = conversions.drift_to_flow(xt, drift)
        drift_recon = conversions.flow_to_drift(xt, flow)

        assert jnp.allclose(drift, drift_recon, atol=1e-5)

    def test_y1_flow_conversion(self, components_fixture, request, key):
        """Test y1 <-> flow conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        y1 = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        flow = conversions.y1_to_flow(y1, xt)
        y1_recon = conversions.flow_to_y1(xt, flow)

        assert jnp.allclose(y1, y1_recon, atol=1e-5)

    def test_score_flow_conversion(self, components_fixture, request, key):
        """Test score <-> flow conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        score = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        flow = conversions.score_to_flow(xt, score)
        score_recon = conversions.flow_to_score(xt, flow)

        assert jnp.allclose(score, score_recon, atol=1e-5)

    def test_epsilon_bwd_message_conversion(self, components_fixture, request, key):
        """Test epsilon <-> backward message conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        epsilon = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        bwd_message = conversions.epsilon_to_bwd_message(xt, epsilon)
        epsilon_recon = conversions.bwd_message_to_epsilon(xt, bwd_message)

        assert jnp.allclose(epsilon, epsilon_recon, atol=1e-5)

    def test_epsilon_drift_conversion(self, components_fixture, request, key):
        """Test epsilon <-> drift conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        epsilon = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        drift = conversions.epsilon_to_drift(xt, epsilon)
        epsilon_recon = conversions.drift_to_epsilon(xt, drift)

        assert jnp.allclose(epsilon, epsilon_recon, atol=1e-5)

    def test_epsilon_score_conversion(self, components_fixture, request, key):
        """Test epsilon <-> score conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        epsilon = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        score = conversions.epsilon_to_score(xt, epsilon)
        epsilon_recon = conversions.score_to_epsilon(xt, score)

        assert jnp.allclose(epsilon, epsilon_recon, atol=1e-5)

    def test_epsilon_flow_conversion(self, components_fixture, request, key):
        """Test epsilon <-> flow conversion."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        epsilon = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        flow = conversions.epsilon_to_flow(xt, epsilon)
        epsilon_recon = conversions.flow_to_epsilon(xt, flow)

        assert jnp.allclose(epsilon, epsilon_recon, atol=1e-5)

    def test_quantity_covariances(self, components_fixture, request, key):
        """Test analytical covariance matches empirical covariance."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        num_samples = 100000
        epsilons = random.normal(k2, (num_samples, dim))

        # Empirical covariances
        flow_samples = jax.vmap(conversions.epsilon_to_flow, in_axes=(None, 0))(xt, epsilons)
        score_samples = jax.vmap(conversions.epsilon_to_score, in_axes=(None, 0))(xt, epsilons)
        drift_samples = jax.vmap(conversions.epsilon_to_drift, in_axes=(None, 0))(xt, epsilons)

        def get_emp_cov(samples):
            return jnp.cov(samples, rowvar=False)

        emp_flow_cov = get_emp_cov(flow_samples)
        emp_score_cov = get_emp_cov(score_samples)
        emp_drift_cov = get_emp_cov(drift_samples)

        # Analytical covariances
        ana_flow_cov = conversions.get_flow_covariance(xt)
        ana_score_cov = conversions.get_score_covariance(xt)
        ana_drift_cov = conversions.get_drift_covariance(xt)

        assert jnp.allclose(emp_flow_cov, ana_flow_cov, atol=0.2)
        assert jnp.allclose(emp_score_cov, ana_score_cov, atol=0.2)
        assert jnp.allclose(emp_drift_cov, ana_drift_cov, atol=0.2)

    def test_drift_score_consistency(self, components_fixture, request, key):
        """Test that drift_to_score is consistent with other conversions."""
        diffusion_components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = diffusion_components.linear_sde.dim
        xt = random.normal(k1, (dim,))
        drift = random.normal(k2, (dim,))
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        score_from_drift = conversions.drift_to_score(xt, drift)

        bwd_message = conversions.drift_to_bwd_message(xt, drift)
        marginal = conversions.bwd_message_to_marginal(bwd_message)
        score_from_marginal = conversions.marginal_to_score(marginal, xt)

        assert jnp.allclose(score_from_drift, score_from_marginal, atol=1e-5)

    def test_marginal_distribution(self, components_fixture, request, key):
        """Test that the analytical marginal distribution matches the empirical one."""
        components = request.getfixturevalue(components_fixture)
        k1, k2 = random.split(key)
        dim = components.linear_sde.dim
        y1 = random.normal(k1, (dim,))
        t = random.uniform(k2, minval=components.t0 + 1e-2, maxval=components.t1 - 1e-2)
        conversions = DiffusionModelConversions(components, t)

        marginal = conversions.y1_to_marginal(y1)

        # Check that the marginal is correct by comparing to empirical samples
        keys = random.split(key, 1024)
        t0 = components.t0
        t1 = components.t1

        def empirical_sample(key: PRNGKeyArray) -> Float[Array, 'D']:
            k1, k2 = random.split(key)
            x0 = components.x_t0_prior.sample(k1)

            phi0 = StandardGaussian(x0, components.evidence_cov.set_zero())
            phi1 = StandardGaussian(y1, components.evidence_cov)
            times = jnp.array([t0, t1])
            evidence = jtu.tree_map(lambda *xs: jnp.array(xs), phi0, phi1)
            evidence = GaussianPotentialSeries(times, evidence)
            cond_sde: ConditionedLinearSDE = components.linear_sde.condition_on(evidence)

            # sample returns a TimeSeries, we want the values at time t
            xt = cond_sde.sample(k2, jnp.array([t]))
            return xt.values[0]

        xts = jax.vmap(empirical_sample)(keys)
        pxt_empirical: StandardGaussian = empirical_dist(xts)

        dist = w2_distance(marginal, pxt_empirical)
        assert dist < 0.2

    def test_probability_path_transition(self, components_fixture, request, key):
        """Test that the probability path transition is correct by checking the Chapman-Kolmogorov property."""
        components_t = request.getfixturevalue(components_fixture)

        if components_fixture == "diffusion_components":
            other_fixture = "lti_sde_diffusion_components"
        else:
            other_fixture = "diffusion_components"
        components_s = request.getfixturevalue(other_fixture)

        dim = components_t.linear_sde.dim

        k1, k2 = random.split(key)
        # Make sure s < t
        s = random.uniform(k1, minval=components_t.t0 + 1e-2, maxval=components_t.t1 - 1e-2)
        t = random.uniform(k2, minval=s, maxval=components_t.t1)

        # Assume a distribution for y1, e.g., N(0, I) to marginalize it out
        y1_mean = jnp.zeros(dim)
        y1_cov = DenseMatrix.eye(dim)

        # Compute p(x_t) by marginalizing out y1
        y1_to_marginal_mean_t = Y1ToMarginalMean(components_t, t)
        At, bt = y1_to_marginal_mean_t.A, y1_to_marginal_mean_t.b
        Sigmat = y1_to_marginal_mean_t.precision.get_inverse()

        marginal_t_mean = At @ y1_mean + bt
        marginal_t_cov = At @ y1_cov @ At.T + Sigmat
        marginal_t = StandardGaussian(marginal_t_mean, marginal_t_cov)

        # Compute p(x_s) by marginalizing out y1
        y1_to_marginal_mean_s = Y1ToMarginalMean(components_s, s)
        As, bs = y1_to_marginal_mean_s.A, y1_to_marginal_mean_s.b
        Sigmas = y1_to_marginal_mean_s.precision.get_inverse()

        marginal_s_mean = As @ y1_mean + bs
        marginal_s_cov = As @ y1_cov @ As.T + Sigmas
        marginal_s = StandardGaussian(marginal_s_mean, marginal_s_cov)

        # Get the transition p(x_t | x_s)
        transition: GaussianTransition = probability_path_transition(components_t, components_s, t, s)

        # Compute p(x_t) using p(x_s) and p(x_t | x_s)
        marginal_t_from_s = transition.update_and_marginalize_out_x(marginal_s)

        # Compare the two distributions for p(x_t)
        dist = w2_distance(marginal_t, marginal_t_from_s)
        assert dist < 1e-5


def test_transition_precomputation(key):
  from linsdex.crf import CRF
  from linsdex.crf.continuous_crf import DiscretizeResult
  from linsdex.series.interleave_times import InterleavedTimes
  from linsdex.util.parallel_scan import parallel_scan
  dim = 2
  t0, t1 = 0.0, 1.0
  k1, k2, k3, k4, k5, k6, k7, k8 = random.split(key, 8)

  F_mat = DiagonalMatrix(random.normal(k1, (dim,)))
  L_mat = DiagonalMatrix(random.normal(k2, (dim,)))
  sde = LinearTimeInvariantSDE(F=F_mat, L=L_mat)

  prior_mu = random.normal(k3, (dim,))
  prior_sigma = DiagonalMatrix(jnp.exp(random.normal(k4, (dim,))))
  prior = StandardGaussian(mu=prior_mu, Sigma=prior_sigma)

  evidence_cov = DiagonalMatrix.zeros(dim)

  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov
  )

  ProbabilityPath(components, t0)

  n_times = 4
  times = jnp.linspace(t0, t1, n_times)

  # Get the quantities at each time step so that we can get p(x_t | y_1)
  def get_quantities(t) -> ProbabilityPath:
    return ProbabilityPath(components, t)

  functional_quantities: ProbabilityPath = eqx.filter_vmap(get_quantities)(times)
  functional_pts_given_y1: StandardGaussian = functional_quantities.functional_pt_given_y1

  from linsdex.potential.gaussian.transition import functional_potential_to_transition
  transitions: GaussianTransition = eqx.filter_vmap(functional_potential_to_transition)(functional_pts_given_y1)
  reversed_transitions: GaussianTransition = transitions.swap_variables()

  n_samples = 1024
  keys = random.split(key, (n_samples, n_times))
  functional_samples = eqx.filter_vmap(functional_pts_given_y1.sample)(keys)

  # Take a y1 to fill the samples with
  y1 = random.normal(k6, (dim,))

  samples: Float[Array, 'D'] = functional_samples(y1)

  assert samples.shape == (n_samples, n_times, dim)

  # Compute Wasserstein-2 distance between empirical distributions and the prior
  def compute_w2(samples_at_t):
    emp_dist = empirical_dist(samples_at_t)
    return emp_dist, w2_distance(emp_dist, prior)

  empirical_dists, w2_distances = jax.vmap(compute_w2, in_axes=1)(samples)
  print(f"W2 distances to prior: {w2_distances}")

  # Check that at t=0, the W2 distance to the prior is very small
  assert w2_distances[0] < 1e-2



def test_auto_vmap_unbatched_args(key):
  """Specifically test that auto_vmap handles unbatched arguments correctly.
  This would have caught the original failure where auto_vmap was too aggressive."""
  dim = 2
  # Create a batched LinearFunctional (batch size 10)
  A = DiagonalMatrix(jnp.ones((10, dim)))
  b = jnp.zeros((10, dim))
  lf = LinearFunctional(A, b)

  # Unbatched input
  x = jnp.ones(dim)

  # This direct call should work and handle the batching of 'lf' automatically
  out: Float[Array, '10 D'] = lf(x)
  assert out.shape == (10, dim)
  assert jnp.allclose(out, 1.0)


def test_diffusion_marginal_at_t0_matches_prior(key):
  """Verify that the marginal distribution at t=t0 matches the prior.
  This would have caught the numerical instability and logical error in ProbabilityPath."""
  dim = 2
  t0, t1 = 0.0, 1.0
  sde = OrnsteinUhlenbeck(dim=dim, sigma=1.0, lambda_=1.0)
  prior = StandardGaussian(mu=jnp.ones(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix.zeros(dim)

  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov
  )

  # Get quantities exactly at t=t0
  quantities = ProbabilityPath(components, t0)
  p_xt0_given_y1_functional = quantities.functional_pt_given_y1

  # Resolve the functional with an arbitrary y1
  y1 = jnp.zeros(dim)
  p_xt0_given_y1 = resolve_functional(p_xt0_given_y1_functional, y1)

  # At t=t0, if we haven't incorporated any y1 yet, it should exactly match the prior
  # Note: ProbabilityPath incorporates y1 via the bridge formula.
  # If we want to check that it matches the prior at t=0, we need to ensure the formula is stable.
  dist = w2_distance(p_xt0_given_y1, prior)
  assert dist < 1e-10


def create_random_linear_functional(key, dim):
    k1, k2 = random.split(key)
    A = DenseMatrix(random.normal(k1, (dim, dim)))
    b = random.normal(k2, (dim,))
    return LinearFunctional(A, b)


@pytest.mark.parametrize("components_fixture", ["diffusion_components", "lti_sde_diffusion_components"])
class TestDiffusionFunctionalConversions:
    """Test that conversion functions work correctly with LinearFunctional inputs."""

    def test_functional_consistency(self, components_fixture, request, key):
        diffusion_components = request.getfixturevalue(components_fixture)
        dim = diffusion_components.linear_sde.dim
        t = 0.5
        conversions = DiffusionModelConversions(diffusion_components, t)

        k1, k2, k3, k4, k5 = random.split(key, 5)
        y1_func = create_random_linear_functional(k1, dim)
        xt_func = create_random_linear_functional(k2, dim)
        eps_func = create_random_linear_functional(k3, dim)
        drift_func = create_random_linear_functional(k4, dim)
        score_func = create_random_linear_functional(k5, dim)

        x_val = random.normal(random.split(k1)[0], (dim,))

        def check_conversion(func, input_func, *args, **kwargs):
            # Evaluate with LinearFunctional
            out_func = func(input_func, *args, **kwargs)
            # Resolve the result
            resolved_out = resolve_functional(out_func, x_val)

            # Evaluate with resolved input
            resolved_input = input_func(x_val)
            direct_out = func(resolved_input, *args, **kwargs)

            # Compare
            if isinstance(resolved_out, (StandardGaussian, MixedGaussian)):
                resolved_out_mu = resolve_functional(resolved_out.mu, x_val) if hasattr(resolved_out.mu, '__call__') else resolved_out.mu
                direct_out_mu = resolve_functional(direct_out.mu, x_val) if hasattr(direct_out.mu, '__call__') else direct_out.mu
                assert jnp.allclose(resolved_out_mu, direct_out_mu, atol=1e-5)
                # Also check J or Sigma
                if isinstance(resolved_out, MixedGaussian):
                    assert jnp.allclose(resolved_out.J.as_matrix(), direct_out.J.as_matrix(), atol=1e-5)
                else:
                    assert jnp.allclose(resolved_out.Sigma.as_matrix(), direct_out.Sigma.as_matrix(), atol=1e-5)
            else:
                assert jnp.allclose(resolved_out, direct_out, atol=1e-5)

        # y1_to_bwd_message
        check_conversion(conversions.y1_to_bwd_message, y1_func)

        # y1_to_marginal
        check_conversion(conversions.y1_to_marginal, y1_func)

        # y1_to_drift
        check_conversion(lambda y, x: conversions.y1_to_drift(y, x), y1_func, xt_func(x_val))
        check_conversion(lambda x, y: conversions.y1_to_drift(y, x), xt_func, y1_func(x_val))

        # epsilon_to_bwd_message
        check_conversion(lambda e, x: conversions.epsilon_to_bwd_message(x, e), eps_func, xt_func(x_val))

        # epsilon_to_drift
        check_conversion(lambda e, x: conversions.epsilon_to_drift(x, e), eps_func, xt_func(x_val))

        # epsilon_to_score
        check_conversion(lambda e, x: conversions.epsilon_to_score(x, e), eps_func, xt_func(x_val))

        # epsilon_to_flow
        check_conversion(lambda e, x: conversions.epsilon_to_flow(x, e), eps_func, xt_func(x_val))

        # score_to_marginal
        check_conversion(lambda s, x: conversions.score_to_marginal(x, s), score_func, xt_func(x_val))

        # score_to_flow
        check_conversion(lambda s, x: conversions.score_to_flow(x, s), score_func, xt_func(x_val))

        # drift_to_bwd_message
        check_conversion(lambda d, x: conversions.drift_to_bwd_message(x, d), drift_func, xt_func(x_val))

        # drift_to_epsilon
        check_conversion(lambda d, x: conversions.drift_to_epsilon(x, d), drift_func, xt_func(x_val))

        # drift_to_score
        check_conversion(lambda d, x: conversions.drift_to_score(x, d), drift_func, xt_func(x_val))

        # drift_to_flow
        check_conversion(lambda d, x: conversions.drift_to_flow(x, d), drift_func, xt_func(x_val))

        # flow_to_drift
        check_conversion(lambda f, x: conversions.flow_to_drift(x, f), drift_func, xt_func(x_val)) # Reuse drift_func as flow

        # flow_to_y1
        check_conversion(lambda f, x: conversions.flow_to_y1(x, f), drift_func, xt_func(x_val))

        # flow_to_score
        check_conversion(lambda f, x: conversions.flow_to_score(x, f), drift_func, xt_func(x_val))
