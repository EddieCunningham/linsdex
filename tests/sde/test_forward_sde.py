import jax
import jax.numpy as jnp
from jax import random
import pytest

from linsdex.sde.forward_sde import ForwardSDE
from linsdex.sde.sde_examples import LinearTimeInvariantSDE, BrownianMotion, OrnsteinUhlenbeck
from linsdex.potential.gaussian.dist import StandardGaussian
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.util.misc import w2_distance


@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("sde_type", ["LTI", "BM", "OU"])
@pytest.mark.parametrize("prior_scale", [0.5])
class TestForwardSDE:
  def test_forward_sde_marginal(self, key, dim, sde_type, prior_scale):
    """
    Tests that the marginal distribution of the ForwardSDE at t1, conditioned on y0 at t0,
    is close to the provided prior distribution at t1.
    """
    k1, k2, k3 = random.split(key, 3)

    # Base SDE
    if sde_type == "LTI":
        F = DenseMatrix(random.normal(k1, (dim, dim)))
        L = DenseMatrix(random.normal(k2, (dim, dim)))
        base_sde = LinearTimeInvariantSDE(F, L)
    elif sde_type == "BM":
        base_sde = BrownianMotion(sigma=1.0, dim=dim)
    elif sde_type == "OU":
        base_sde = OrnsteinUhlenbeck(sigma=1.0, lambda_=0.5, dim=dim)


    # Endpoints
    t0 = 0.0
    T = 1.0
    y0 = random.normal(k3, (dim,))

    # Prior at T
    prior_mean = random.normal(k1, (dim,))
    prior_covariance = DiagonalMatrix.eye(dim) * prior_scale
    yT_prior = StandardGaussian(prior_mean, prior_covariance)

    # Forward SDE
    forward_sde = ForwardSDE(base_sde, t0, y0, T, yT_prior)

    # Get marginal at T
    transition = forward_sde.get_transition_distribution(t0 + 1e-6, T - 1e-6)
    endpoint_marginal = transition.condition_on_x(y0)

    # Compare with prior
    dist = w2_distance(endpoint_marginal, yT_prior)

    assert dist < 1e-2


@pytest.fixture
def key():
    return random.PRNGKey(0)