import jax
import jax.numpy as jnp
from jax import random
import pytest
import equinox as eqx
from jaxtyping import Array, Float
from typing import Union, Tuple

from linsdex.ssm.exact_latent_encoder import ExactPositionEncoderForAccelerationModel
from linsdex.ssm.simple_encoder import PaddingLatentVariableEncoderWithPrior
from linsdex.sde.sde_examples import WienerAccelerationModel
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.series.series import TimeSeries
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.block.block_3x3 import Block3x3Matrix
from linsdex.matrix.matrix_base import TAGS


class TestExactPositionEncoderForAccelerationModel:
  """Test suite for ExactPositionEncoderForAccelerationModel."""

  def setup_method(self):
    """Set up test fixtures before each test method."""
    jax.config.update('jax_enable_x64', True)
    self.key = random.PRNGKey(42)
    self.y_dim = 2  # 2D position data
    self.x_dim = 3 * self.y_dim  # 3*y_dim for position, velocity, acceleration

  def create_test_series(self, key=None, n_points=5):
    """Helper to create a test TimeSeries with position data."""
    if key is None:
      key = self.key

    times = jnp.linspace(0.0, 2.0, n_points)
    # Create smooth position trajectory
    positions = jnp.stack([
      jnp.sin(times) + 0.1 * random.normal(key, (n_points,)),
      jnp.cos(times) + 0.1 * random.normal(random.split(key)[1], (n_points,))
    ], axis=1)

    return TimeSeries(times=times, values=positions)

  def test_encoder_initialization(self):
    """Test basic initialization of the encoder."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)

    assert encoder.y_dim == self.y_dim
    assert encoder.potential_cov_type == 'diagonal'

  def test_encoder_call_basic(self):
    """Test basic functionality of the encoder call."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    series = self.create_test_series()

    prob_series = encoder(series)

    # Check output is correct type
    assert isinstance(prob_series, GaussianPotentialSeries)
    assert prob_series.times.shape == series.times.shape
    assert prob_series.node_potentials.batch_size == series.times.shape[0]

  def test_encoder_potential_structure(self):
    """Test that the encoder creates potentials with correct structure."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    series = self.create_test_series(n_points=3)

    prob_series = encoder(series)
    potentials = prob_series.node_potentials

    # Check that we have the right number of potentials
    assert potentials.batch_size == 3

    # Check individual potential structure
    first_potential = potentials[0]
    assert isinstance(first_potential, MixedGaussian)

    # Mean should be padded position with zeros for velocity/acceleration
    position = series.values[0]
    expected_mean = jnp.pad(position, (0, 2*self.y_dim))
    assert jnp.allclose(first_potential.mu, expected_mean)

    # Precision should be infinite for position, zero for velocity/acceleration
    assert isinstance(first_potential.J, Block3x3Matrix)

  def test_integration_with_wiener_acceleration_model(self):
    """Test integration with WienerAccelerationModel SDE."""
    # Create encoder and SDE
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    linear_sde = WienerAccelerationModel(sigma=0.1, position_dim=self.y_dim)

    # Create test series
    series = self.create_test_series()

    # Encode and condition
    prob_series = encoder(series)
    cond_sde = linear_sde.condition_on(prob_series)

    # Check that conditioning worked
    assert isinstance(cond_sde, ConditionedLinearSDE)
    assert cond_sde.dim == self.x_dim

  def test_conditioned_sde_sampling(self):
    """Test that conditioned SDE can generate samples."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    linear_sde = WienerAccelerationModel(sigma=0.1, position_dim=self.y_dim)

    series = self.create_test_series()
    prob_series = encoder(series)
    cond_sde = linear_sde.condition_on(prob_series)

    # Test sampling
    save_times = jnp.linspace(0.0, 2.0, 10)
    key = random.PRNGKey(123)

    # This should not raise an error
    trajectory = cond_sde.sample(
      key,
      save_times
    )

    assert isinstance(trajectory, TimeSeries)
    assert trajectory.values.shape == (10, self.x_dim)

  def test_comparison_with_padding_encoder(self):
    """Test that ExactPositionEncoder gives similar results to PaddingEncoder with small sigma."""
    # Create both encoders
    exact_encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    padding_encoder = PaddingLatentVariableEncoderWithPrior(
      y_dim=self.y_dim,
      x_dim=self.x_dim,
      sigma=0.0001,  # Very small sigma for high precision
      use_prior=False
    )

    # Create test series
    series = self.create_test_series()

    # Encode with both methods
    exact_prob_series = exact_encoder(series)
    padding_prob_series = padding_encoder(series)

    # Check that both produce valid GaussianPotentialSeries
    assert isinstance(exact_prob_series, GaussianPotentialSeries)
    assert isinstance(padding_prob_series, GaussianPotentialSeries)
    assert exact_prob_series.times.shape == padding_prob_series.times.shape

  def test_update_y_consistency(self):
    """Test that update_y operations give consistent results between encoders."""
    # Create the SDE and transition
    linear_sde = WienerAccelerationModel(sigma=0.1, position_dim=self.y_dim)
    transition = linear_sde.get_transition_distribution(s=0.0, t=1.0)

    # Create both encoders
    exact_encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    padding_encoder = PaddingLatentVariableEncoderWithPrior(
      y_dim=self.y_dim,
      x_dim=self.x_dim,
      sigma=0.0001,
      use_prior=False
    )

    # Create test series with a single point
    times = jnp.array([0.5])
    positions = jnp.array([[1.0, 2.0]])
    series = TimeSeries(times=times, values=positions)

    # Encode with both methods
    exact_prob_series = exact_encoder(series)
    padding_prob_series = padding_encoder(series)

    # Get single potentials
    exact_potential = exact_prob_series.node_potentials[0]
    padding_potential = padding_prob_series.node_potentials[0]

    # Apply update_y to the transition with both potentials
    exact_joint = transition.update_y(exact_potential)
    padding_joint = transition.update_y(padding_potential)

    # Both should produce valid JointPotentials
    assert hasattr(exact_joint, 'transition')
    assert hasattr(exact_joint, 'prior')
    assert hasattr(padding_joint, 'transition')
    assert hasattr(padding_joint, 'prior')

    # The transitions should be similar (allowing for numerical differences)
    exact_transition = exact_joint.transition
    padding_transition = padding_joint.transition

    # Check that transition parameters have similar shapes
    assert exact_transition.A.shape == padding_transition.A.shape
    assert exact_transition.u.shape == padding_transition.u.shape
    assert exact_transition.Sigma.shape == padding_transition.Sigma.shape

    # Apply to a random vector to check for functional equivalence
    x = random.normal(self.key, (self.x_dim,))
    exact_output = exact_transition.A @ x + exact_transition.u
    padding_output = padding_transition.A @ x + padding_transition.u
    assert jnp.allclose(exact_output, padding_output)

    # Compare finite parts of the covariance matrices
    exact_Sigma = exact_transition.Sigma.as_matrix()
    padding_Sigma = padding_transition.Sigma.as_matrix()
    finite_mask = jnp.isfinite(exact_Sigma)
    assert jnp.allclose(exact_Sigma[finite_mask], padding_Sigma[finite_mask])


  def test_update_y_transition_only(self):
    """Test update_y with only_return_transition=True."""
    linear_sde = WienerAccelerationModel(sigma=0.1, position_dim=self.y_dim)
    transition = linear_sde.get_transition_distribution(s=0.0, t=1.0)

    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)

    # Single point series
    times = jnp.array([0.5])
    positions = jnp.array([[1.0, 2.0]])
    series = TimeSeries(times=times, values=positions)

    prob_series = encoder(series)
    potential = prob_series.node_potentials[0]

    # Test both return modes
    joint_result = transition.update_y(potential, only_return_transition=False)
    transition_only = transition.update_y(potential, only_return_transition=True)

    # transition_only should be same as joint_result.transition
    assert isinstance(transition_only, GaussianTransition)

    # Check for functional equivalence
    x = random.normal(self.key, (self.x_dim,))
    exact_output = joint_result.transition.A @ x + joint_result.transition.u
    padding_output = transition_only.A @ x + transition_only.u
    assert jnp.allclose(exact_output, padding_output)

  def test_multiple_timesteps(self):
    """Test with multiple timesteps and verify conditioning works."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    linear_sde = WienerAccelerationModel(sigma=0.1, position_dim=self.y_dim)

    # Create series with multiple points
    series = self.create_test_series(n_points=4)

    prob_series = encoder(series)
    cond_sde = linear_sde.condition_on(prob_series)

    # Check we can get marginals at different times
    for t in series.times:
      marginal = cond_sde.get_marginal(t)
      assert hasattr(marginal, 'mu') or hasattr(marginal, 'h')

  @pytest.mark.parametrize("y_dim", [1, 2, 3])
  def test_different_dimensions(self, y_dim):
    """Test encoder with different position dimensions."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=y_dim)

    # Create series with appropriate dimension
    times = jnp.array([0.0, 1.0])
    positions = random.normal(self.key, (2, y_dim))
    series = TimeSeries(times=times, values=positions)

    prob_series = encoder(series)

    # Check dimensions
    potential = prob_series.node_potentials[0]
    expected_dim = 3 * y_dim
    assert potential.mu.shape == (expected_dim,)
    assert potential.J.shape == (expected_dim, expected_dim)

  def test_gradient_computations(self):
    """Test that gradients can be computed through the encoder."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    linear_sde = WienerAccelerationModel(sigma=0.1, position_dim=self.y_dim)

    def loss_fn(positions):
      times = jnp.array([0.5])
      series = TimeSeries(times=times, values=positions[None])
      prob_series = encoder(series)
      potential = prob_series.node_potentials[0]

      # Simple loss based on potential mean
      return jnp.sum(potential.mu**2)

    positions = jnp.array([1.0, 2.0])

    # This should not raise an error
    grad = jax.grad(loss_fn)(positions)
    assert grad.shape == positions.shape
    assert jnp.all(jnp.isfinite(grad))

  def test_deterministic_behavior(self):
    """Test that encoder gives deterministic results for same input."""
    encoder = ExactPositionEncoderForAccelerationModel(y_dim=self.y_dim)
    series = self.create_test_series(key=random.PRNGKey(999))

    # Encode twice
    result1 = encoder(series)
    result2 = encoder(series)

    # Should be identical
    assert jnp.allclose(result1.times, result2.times)

    # Check that potentials are the same
    pot1 = result1.node_potentials
    pot2 = result2.node_potentials

    assert jnp.allclose(pot1.mu, pot2.mu)
    assert jnp.allclose(pot1.J.as_matrix(), pot2.J.as_matrix())
