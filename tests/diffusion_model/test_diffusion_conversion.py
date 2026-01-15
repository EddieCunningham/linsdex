import pytest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import random
from linsdex.diffusion_model.diffusion_conversion import DiffusionModelComponents, probability_path_transition
from linsdex.diffusion_model.diffusion_conversion import DiffusionModelConversions, Y1ToBwdMean, Y1ToMarginalMean
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

# def test_conversion_identities_with_spline(key):
#     """
#     Tests the self-consistency of the conversion formulas using a synthetic ground truth
#     defined by a spline-based Gaussian probability path.
#     """
#     assert False, "This test is not working yet"
#     dim = 2
#     t0, t1 = 0.0, 3.0
#     k1, k2, k3, k4, k5 = random.split(key, 5)

#     # 1. Create a synthetic ground-truth path p(xt) = N(mu(t), Sigma(t)) using simple polynomials
#     # Mean polynomial: mu(t) = c3*t^3 + c2*t^2 + c1*t + c0
#     mean_coeffs = random.normal(k1, (4, dim))
#     def mu_fn(t):
#       return mean_coeffs[0] * t**3 + mean_coeffs[1] * t**2 + mean_coeffs[2] * t + mean_coeffs[3]

#     # Covariance polynomial (diagonal): Sigma_diag(t) = exp(c1*t + c0)
#     log_diag_coeffs = random.normal(k2, (2, dim)) * 0.1
#     def cov_diag_fn(t):
#       return jnp.exp(log_diag_coeffs[0] * t + log_diag_coeffs[1])

#     def get_dist(t):
#       mu = mu_fn(t)
#       cov_diag = cov_diag_fn(t)
#       return StandardGaussian(mu, DiagonalMatrix(cov_diag))

#     # 2. Derive ground-truth quantities from the path
#     def get_true_flow(t, xt):
#       pt = get_dist(t)
#       noise = pt.get_noise(xt)
#       def sample_fn(s):
#         dist = get_dist(s)
#         return dist._sample(noise)

#       _, flow = jax.jvp(sample_fn, (t,), (jnp.ones_like(t),))
#       return flow

#     def get_true_score(t, xt):
#       return get_dist(t).score(xt)


#     # 3. Define a simple SDE to link drift, score, and flow
#     F = DenseMatrix(random.normal(k3, (dim, dim)) * 0.5)
#     L = DenseMatrix(random.normal(k4, (dim, dim)) * 0.5)
#     sde = LinearTimeInvariantSDE(F, L)
#     LLT = L @ L.T
#     components = DiffusionModelComponents(
#         linear_sde=sde, t0=t0, t1=t1,
#         x_t0_prior=get_dist(t0),
#         evidence_cov=DiagonalMatrix.eye(dim).set_zero()
#     )

#     # Get ground truth values
#     t = random.uniform(k5, minval=t0 + 0.1, maxval=t1 - 0.1)
#     xt = get_dist(t).sample(key)
#     true_flow = get_true_flow(t, xt)
#     true_score = get_true_score(t, xt)
#     true_drift = true_flow + 0.5*LLT @ true_score

#     import pdb; pdb.set_trace()
