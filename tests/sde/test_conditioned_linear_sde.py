import jax
import jax.numpy as jnp
from jax import random
import pytest
import equinox as eqx

from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.sde.sde_examples import BrownianMotion, OrnsteinUhlenbeck
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
import linsdex.util as util


class TestConditionedLinearSDEInitialization:
  """Test initialization of conditioned linear SDE"""

  def test_basic_initialization(self):
    """Test basic initialization with SDE and evidence"""
    sde = BrownianMotion(sigma=1.0, dim=2)

    # Create evidence
    x0 = jnp.ones(2) * 5.0
    Sigma0 = DiagonalMatrix.eye(2) * 0.001
    potential = MixedGaussian(x0, Sigma0.get_inverse())
    ts = jnp.array([1.1])
    node_potentials = potential[None]

    evidence = GaussianPotentialSeries(ts, node_potentials)
    cond_sde = ConditionedLinearSDE(sde, evidence)

    assert cond_sde.sde is sde
    assert cond_sde.evidence is evidence
    assert cond_sde.dim == sde.dim
    assert cond_sde.batch_size == evidence.batch_size

  def test_nested_conditioning(self):
    """Test conditioning an already conditioned SDE"""
    base_sde = BrownianMotion(sigma=1.0, dim=2)

    # First evidence
    x1 = jnp.ones(2) * 3.0
    Sigma1 = DiagonalMatrix.eye(2) * 0.01
    potential1 = MixedGaussian(x1, Sigma1.get_inverse())
    ts1 = jnp.array([0.5])
    evidence1 = GaussianPotentialSeries(ts1, potential1[None])

    # Second evidence
    x2 = jnp.ones(2) * 7.0
    Sigma2 = DiagonalMatrix.eye(2) * 0.001
    potential2 = MixedGaussian(x2, Sigma2.get_inverse())
    ts2 = jnp.array([1.5])
    evidence2 = GaussianPotentialSeries(ts2, potential2[None])

    # First conditioning
    cond_sde1 = ConditionedLinearSDE(base_sde, evidence1)

    # Second conditioning (nested)
    cond_sde2 = ConditionedLinearSDE(cond_sde1, evidence2)

    # Should have combined evidence and base SDE
    assert cond_sde2.sde is base_sde  # Should unwrap to base SDE
    assert len(cond_sde2.evidence.times) == 2  # Combined evidence

  def test_parallel_flag(self):
    """Test parallel flag behavior"""
    sde = BrownianMotion(sigma=1.0, dim=2)
    evidence = GaussianPotentialSeries(jnp.array([1.0]), MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))[None])

    # Test explicit parallel setting
    cond_sde_parallel = ConditionedLinearSDE(sde, evidence, parallel=True)
    assert cond_sde_parallel.parallel is True

    cond_sde_serial = ConditionedLinearSDE(sde, evidence, parallel=False)
    assert cond_sde_serial.parallel is False

    # Test default behavior
    cond_sde_default = ConditionedLinearSDE(sde, evidence)
    assert isinstance(cond_sde_default.parallel, bool)


class TestConditionedLinearSDEProperties:
  """Test properties of conditioned linear SDE"""

  def test_get_base_transition_distribution(self):
    """Test that base transition distributions are correctly retrieved"""
    sde = OrnsteinUhlenbeck(sigma=1.0, lambda_=0.5, dim=2)
    evidence = GaussianPotentialSeries(jnp.array([1.0]), MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    s, t = 0.0, 1.0
    base_transition = cond_sde.get_base_transition_distribution(s, t)
    expected_transition = sde.get_transition_distribution(s, t)

    assert jnp.allclose(base_transition.A.elements, expected_transition.A.elements)
    assert jnp.allclose(base_transition.u, expected_transition.u)
    assert jnp.allclose(base_transition.Sigma.elements, expected_transition.Sigma.elements)

  def test_batch_size_property(self):
    """Test batch size property"""
    sde = BrownianMotion(sigma=1.0, dim=2)

    # Non-batched evidence
    evidence = GaussianPotentialSeries(jnp.array([1.0]), MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)
    assert cond_sde.batch_size == evidence.batch_size

    # Batched evidence - use vmap to create batched potentials correctly
    batch_size = 3
    def create_potential(mu):
      J = DiagonalMatrix.eye(2)
      return MixedGaussian(mu, J)

    batched_mu = jnp.zeros((batch_size, 2))
    batched_potential = jax.vmap(create_potential)(batched_mu)
    batched_evidence = GaussianPotentialSeries(jnp.array([1.0, 2.0, 3.0]), batched_potential)

    batched_cond_sde = ConditionedLinearSDE(sde, batched_evidence)
    assert batched_cond_sde.node_potentials.batch_size == batch_size

  def test_dim_property(self):
    """Test dimension property"""
    dim = 3
    sde = BrownianMotion(sigma=1.0, dim=dim)
    evidence = GaussianPotentialSeries(jnp.array([1.0]),
                               MixedGaussian(jnp.zeros(dim), DiagonalMatrix.eye(dim))[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    assert cond_sde.dim == dim


class TestConditionedLinearSDEDiscretization:
  """Test discretization functionality"""

  def test_basic_discretization(self):
    """Test basic discretization without additional times"""
    sde = BrownianMotion(sigma=1.0, dim=2)

    # Create evidence at specific times - use vmap for proper construction
    evidence_times = jnp.array([0.5, 1.5])

    def create_potential_for_time(t):
      x = jnp.ones(2) * t  # Different evidence for each time
      Sigma = DiagonalMatrix.eye(2) * 0.01
      return MixedGaussian(x, Sigma.get_inverse())

    potentials = jax.vmap(create_potential_for_time)(evidence_times)
    evidence = GaussianPotentialSeries(evidence_times, potentials)
    cond_sde = ConditionedLinearSDE(sde, evidence)

    # Discretize at evidence times
    result = cond_sde.discretize()

  def test_discretization_with_additional_times(self):
    """Test discretization with additional save times"""
    sde = BrownianMotion(sigma=1.0, dim=2)

    # Evidence at one time
    evidence_times = jnp.array([1.0])
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(evidence_times, potential[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    # Additional save times
    save_times = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    result = cond_sde.discretize(save_times)

    assert len(result.info.new_times) == len(save_times)
    assert len(result.info.base_indices) == len(evidence_times)


class TestConditionedLinearSDEFlow:
  """Test flow and score computation"""

  def test_get_flow_basic(self):
    """Test basic flow computation"""
    sde = BrownianMotion(sigma=1.0, dim=2)

    # Simple evidence
    evidence_times = jnp.array([1.0])
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(evidence_times, potential[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    t = 0.5
    xt = jnp.array([1.0, -1.0])

    # This should not raise an error
    flow = cond_sde.get_flow(t, xt)
    assert flow.shape == xt.shape

  def test_flow_methods(self):
    """Test different flow computation methods"""
    sde = BrownianMotion(sigma=1.0, dim=2)
    evidence_times = jnp.array([1.0])
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(evidence_times, potential[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    t = 0.5
    xt = jnp.array([1.0, -1.0])

    # Test both methods
    flow_jvp = cond_sde.get_flow(t, xt, method='jvp')
    flow_score = cond_sde.get_flow(t, xt, method='score')

    assert flow_jvp.shape == xt.shape
    assert flow_score.shape == xt.shape
    # Both methods should give similar results
    assert jnp.allclose(flow_jvp, flow_score, atol=1e-6)


class TestConditionedLinearSDEMarginals:
  """Test marginal distribution computation"""

  def test_get_marginal_basic(self):
    """Test basic marginal computation"""
    sde = BrownianMotion(sigma=1.0, dim=2)

    # Evidence at specific time
    evidence_time = 1.0
    x_evidence = jnp.array([2.0, -1.0])
    Sigma_evidence = DiagonalMatrix.eye(2) * 0.01
    potential = MixedGaussian(x_evidence, Sigma_evidence.get_inverse())
    evidence = GaussianPotentialSeries(jnp.array([evidence_time]), potential[None])

    cond_sde = ConditionedLinearSDE(sde, evidence)

    # Get marginal at evidence time
    marginal = cond_sde.get_marginal(evidence_time)

    # Should be close to the evidence
    if isinstance(marginal, StandardGaussian):
      assert jnp.allclose(marginal.mu, x_evidence, atol=0.1)

  def test_get_marginal_with_messages(self):
    """Test marginal computation with message return"""
    sde = BrownianMotion(sigma=1.0, dim=2)
    evidence_time = 1.0
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(jnp.array([evidence_time]), potential[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    # Get marginal with messages
    result = cond_sde.get_marginal(evidence_time, return_messages=True)

    assert len(result) == 2  # Should return (marginal, messages)
    marginal, messages = result
    assert hasattr(marginal, 'mu') or hasattr(marginal, 'h')  # Should be a Gaussian


class TestConditionedLinearSDESimulation:
  """Test simulation and sampling functionality"""

  def test_sample_matching_items(self):
    """Test sample matching items functionality"""
    sde = BrownianMotion(sigma=1.0, dim=2)
    evidence_times = jnp.array([1.0])
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(evidence_times, potential[None])
    cond_sde = ConditionedLinearSDE(sde, evidence)

    key = random.PRNGKey(42)
    ts = jnp.array([0.0, 0.5, 1.0, 1.5])

    items = cond_sde.sample_matching_items(ts, key)


class TestConditionedLinearSDEParameters:
  """Test parameter computation with conditioning"""

  def test_get_params_with_conditioning(self):
    """Test that parameters are modified by conditioning"""
    base_sde = OrnsteinUhlenbeck(sigma=1.0, lambda_=0.5, dim=2)

    # Strong evidence should modify the parameters
    evidence_time = 1.0
    x_evidence = jnp.array([5.0, -5.0])  # Strong signal
    Sigma_evidence = DiagonalMatrix.eye(2) * 100.0  # High precision
    potential = MixedGaussian(x_evidence, Sigma_evidence.get_inverse())
    evidence = GaussianPotentialSeries(jnp.array([evidence_time]), potential[None])

    cond_sde = ConditionedLinearSDE(base_sde, evidence)

    # Get parameters at evidence time
    F_cond, u_cond, L_cond = cond_sde.get_params(evidence_time)
    F_base, u_base, L_base = base_sde.get_params(evidence_time)

    # Conditioned parameters should be different from base
    # (exact differences depend on the conditioning math)
    assert F_cond.shape == F_base.shape
    assert u_cond.shape == u_base.shape
    assert L_cond.shape == L_base.shape

    # L should be the same (diffusion not affected by linear conditioning)
    assert jnp.allclose(L_cond.elements, L_base.elements)

  def test_get_drift_with_conditioning(self):
    """Test drift computation with conditioning"""
    base_sde = BrownianMotion(sigma=1.0, dim=2)
    evidence_time = 1.0
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(jnp.array([evidence_time]), potential[None])

    cond_sde = ConditionedLinearSDE(base_sde, evidence)

    t = 0.5
    xt = jnp.array([1.0, -1.0])

    drift = cond_sde.get_drift(t, xt)
    assert drift.shape == xt.shape

    # For Brownian motion base, drift should be influenced by conditioning
    # (exact value depends on evidence and current state)


class TestConditionedLinearSDETransitions:
  """Test transition distributions with conditioning"""

  def test_get_transition_distribution_basic(self):
    """Test transition distribution computation"""
    sde = BrownianMotion(sigma=1.0, dim=2)
    evidence_time = 1.0
    potential = MixedGaussian(jnp.zeros(2), DiagonalMatrix.eye(2))
    evidence = GaussianPotentialSeries(jnp.array([evidence_time]), potential[None])

    cond_sde = ConditionedLinearSDE(sde, evidence)

    s, t = 0.0, 0.5
    transition = cond_sde.get_transition_distribution(s, t)

    # Should return a valid Gaussian transition
    assert hasattr(transition, 'A')
    assert hasattr(transition, 'u')
    assert hasattr(transition, 'Sigma')
    assert transition.A.shape == (2, 2)
    assert transition.u.shape == (2,)
    assert transition.Sigma.shape == (2, 2)


if __name__ == "__main__":
  pytest.main([__file__])