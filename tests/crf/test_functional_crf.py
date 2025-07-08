import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import pytest
import jax.tree_util as jtu
from functools import partial

from linsdex.crf.crf import CRF
from linsdex.potential.gaussian.dist import MixedGaussian
from linsdex.potential.gaussian.transition import GaussianTransition
from linsdex.matrix import DenseMatrix, TAGS
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.functional_ops import resolve_functional

# Enable x64 for better numerical precision in tests
jax.config.update('jax_enable_x64', True)

def compare_trees(A, B, atol: float = 1e-6):
  """Utility function to compare PyTrees with tolerance."""
  params_A, static_A = eqx.partition(A, eqx.is_inexact_array)
  params_B, static_B = eqx.partition(B, eqx.is_inexact_array)
  return jtu.tree_all(jtu.tree_map(partial(jnp.allclose, atol=atol), params_A, params_B))

class TestFunctionalCRF:
  """Test cases for a CRF with functional node potentials."""

  @pytest.fixture
  def setup_data(self):
    """Setup test data for functional CRF tests."""
    key = random.PRNGKey(0)
    N = 5
    x_dim = 4
    y_dim = 4
    latent_dim = 4
    ts = jnp.linspace(0, 1, N)

    k1, k2, k3 = random.split(key, 3)

    # The latent variable that the functionals will depend on
    latent_z = random.normal(k1, (latent_dim,))

    # Create the y-values that the functional will resolve to
    W = random.normal(k2, (y_dim,))
    yts = jnp.cos(2 * jnp.pi * ts[:, None] * W[None, :])

    # Create base transitions for the CRF
    def create_transition(key):
      k1, k2 = random.split(key)
      A = random.normal(k1, (x_dim, x_dim))
      A, _ = jnp.linalg.qr(A)
      Sigma = random.normal(k2, (x_dim, x_dim))
      Sigma = Sigma @ Sigma.T
      return GaussianTransition(
          DenseMatrix(A, tags=TAGS.no_tags),
          jnp.zeros(x_dim),
          DenseMatrix(Sigma, tags=TAGS.no_tags)
      )

    keys = random.split(k3, N - 1)
    transitions = jax.vmap(create_transition)(keys)

    # Create standard and functional node potentials
    def create_potentials(y):
      H = jnp.eye(x_dim, M=y_dim)
      R = jnp.eye(y_dim) * 1e-2
      joint = GaussianTransition(
          DenseMatrix(H, tags=TAGS.no_tags),
          jnp.zeros_like(y),
          DenseMatrix(R, tags=TAGS.no_tags)
      )

      # Standard potential from a concrete vector y
      potential_std = joint.condition_on_y(y).to_mixed()

      # Functional potential from a LinearFunctional that resolves to y
      # Here, we use a simple functional that ignores z and just returns y
      y_lf_A = jnp.zeros((y_dim, latent_dim))
      y_lf = LinearFunctional(DenseMatrix(y_lf_A), y)
      potential_func = joint.condition_on_y(y_lf).to_mixed()

      return potential_std, potential_func

    node_potentials_std, node_potentials_func = jax.vmap(create_potentials)(yts)

    blah = jax.vmap(resolve_functional, in_axes=(0, None))(node_potentials_func, latent_z)
    compare_trees(blah, node_potentials_std)

    return {
      'key': key,
      'latent_z': latent_z,
      'transitions': transitions,
      'node_potentials_std': node_potentials_std,
      'node_potentials_func': node_potentials_func,
    }

  @pytest.fixture
  def crf_standard(self, setup_data):
    """Create a standard CRF for baseline comparison."""
    return CRF(setup_data['node_potentials_std'], setup_data['transitions'])

  @pytest.fixture
  def crf_functional(self, setup_data):
    """Create a CRF with functional node potentials."""
    return CRF(setup_data['node_potentials_func'], setup_data['transitions'])

  def test_marginals_consistency(self, crf_standard, crf_functional, setup_data):
    """Test if functional marginals resolve to standard marginals."""
    latent_z = setup_data['latent_z']

    marginals_std = crf_standard.get_marginals()
    marginals_func = crf_functional.get_marginals()

    resolved_marginals = jax.vmap(resolve_functional, in_axes=(0, None))(marginals_func, latent_z)

    assert jtu.tree_all(jtu.tree_map(jnp.allclose, resolved_marginals, marginals_std))

  def test_joints_consistency(self, crf_standard, crf_functional, setup_data):
    """Test if functional joints resolve to standard joints."""
    latent_z = setup_data['latent_z']

    joints_std = crf_standard.get_joints()
    joints_func = crf_functional.get_joints()

    resolved_joints = jax.vmap(resolve_functional, in_axes=(0, None))(joints_func, latent_z)

    # Compare the resolved functional tree with the standard one
    assert compare_trees(resolved_joints, joints_std)

  @pytest.mark.xfail(reason="log_prob with functional mean returns a QuadraticForm, which is not handled by the test.")
  def test_log_prob_consistency(self, crf_standard, crf_functional, setup_data):
    """Test if functional log_prob resolves to standard log_prob."""
    key = setup_data['key']
    latent_z = setup_data['latent_z']

    # Sample from the standard CRF to get a valid trajectory
    xts = crf_standard.sample(key)

    log_prob_std = crf_standard.log_prob(xts)

    # The log_prob of the functional CRF will itself be a functional
    log_prob_func = crf_functional.log_prob(xts)
    resolved_log_prob = jax.vmap(resolve_functional, in_axes=(0, None))(log_prob_func, latent_z)

    assert jnp.allclose(resolved_log_prob, log_prob_std)

  def test_sampling_consistency(self, crf_standard, crf_functional, setup_data):
    """Test if sampling from a functional CRF is consistent."""
    key = setup_data['key']
    latent_z = setup_data['latent_z']

    # Sampling from functional CRF will produce a trajectory of functionals
    samples_func = crf_functional.sample(key)
    resolved_samples = jax.vmap(resolve_functional, in_axes=(0, None))(samples_func, latent_z)

    # For the same key, the standard CRF should produce the same samples
    samples_std = crf_standard.sample(key)

    assert jnp.allclose(resolved_samples, samples_std)