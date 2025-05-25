import jax
import jax.numpy as jnp
from jax import random
import pytest
import equinox as eqx

from linsdex.sde.sde_examples import (
  LinearTimeInvariantSDE,
  BrownianMotion,
  OrnsteinUhlenbeck,
  VariancePreserving,
  WienerVelocityModel,
  CriticallyDampedLangevinDynamics,
  TOLD,
  StochasticHarmonicOscillator
)
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Literal
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.dense import DenseMatrix
from linsdex.matrix.block.block_2x2 import Block2x2Matrix
from linsdex.matrix.block.block_3x3 import Block3x3Matrix
from linsdex.matrix.matrix_base import TAGS
import linsdex.util as util
from linsdex.sde.conditioned_linear_sde import ConditionedLinearSDE
from linsdex.sde.sde_base import AbstractSDE
from linsdex.potential.abstract import AbstractPotential
from linsdex.potential.gaussian.dist import MixedGaussian, StandardGaussian, NaturalGaussian
from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
import jax.tree_util as jtu

def check_sde_marginal_distribution(sde: ConditionedLinearSDE):
  key = random.PRNGKey(0)

  # Test that samples from the ODE and CRF have the same distribution
  keys = random.split(key, 10000)
  N = 3
  save_times = random.uniform(key, (N,), minval=0.0, maxval=2/N).cumsum()

  # Get the marginals from the CRF
  crf_result = sde.discretize(save_times)
  crf, info = crf_result.crf, crf_result.info
  marginals = crf.get_marginals()

  # Sample from the CRF
  samples_crf = jax.vmap(crf.sample)(keys)

  # Sample from the probability flow ODE
  def pflow_sample(key):
    x0 = marginals[0].sample(key)
    return sde.simulate_probability_flow(x0, save_times)
  samples_pflow = jax.vmap(pflow_sample)(keys)

  # Look at the W2 distance between the marginals
  marginals_crf = jax.vmap(util.empirical_dist, in_axes=1, out_axes=0)(samples_crf)
  marginals_pflow = jax.vmap(util.empirical_dist, in_axes=1, out_axes=0)(samples_pflow)
  marginals_pflow = info.filter_new_times(marginals_pflow)

  w2_dist = jax.vmap(util.w2_distance)(marginals_crf, marginals_pflow)
  print(f'W2 distance between CRF and ODE: {w2_dist}')
  assert jnp.all(w2_dist < 1e-2)


# SDE factory functions
def create_lti_sde_dense():
  key = random.PRNGKey(0)
  dim = 3
  F, L = random.normal(key, (2, dim, dim))*0.1
  F, L = DenseMatrix(F, tags=TAGS.no_tags), DenseMatrix(L, tags=TAGS.no_tags)
  return LinearTimeInvariantSDE(F=F, L=L)

def create_lti_sde_diagonal():
  key = random.PRNGKey(0)
  dim = 3
  F, L = random.normal(key, (2, dim))*0.1
  F, L = DiagonalMatrix(F, tags=TAGS.no_tags), DiagonalMatrix(L, tags=TAGS.no_tags)
  return LinearTimeInvariantSDE(F=F, L=L)

def create_lti_sde_block_2x2():
  key = random.PRNGKey(0)
  dim = 4  # Total dimension (even number for 2x2 blocks)
  half_dim = dim // 2  # Each block will be 2x2

  # Generate random matrices F and L
  k1, k2 = random.split(key, 2)
  F_full = random.normal(k1, (dim, dim)) * 0.1
  L_full = random.normal(k2, (dim, dim)) * 0.1

  # Split F into 4 blocks
  F_A = F_full[:half_dim, :half_dim]
  F_B = F_full[:half_dim, half_dim:]
  F_C = F_full[half_dim:, :half_dim]
  F_D = F_full[half_dim:, half_dim:]

  # Split L into 4 blocks
  L_A = L_full[:half_dim, :half_dim]
  L_B = L_full[:half_dim, half_dim:]
  L_C = L_full[half_dim:, :half_dim]
  L_D = L_full[half_dim:, half_dim:]

  # Create DenseMatrix instances for F
  F_A_mat = DenseMatrix(F_A, tags=TAGS.no_tags)
  F_B_mat = DenseMatrix(F_B, tags=TAGS.no_tags)
  F_C_mat = DenseMatrix(F_C, tags=TAGS.no_tags)
  F_D_mat = DenseMatrix(F_D, tags=TAGS.no_tags)

  # Create DenseMatrix instances for L
  L_A_mat = DenseMatrix(L_A, tags=TAGS.no_tags)
  L_B_mat = DenseMatrix(L_B, tags=TAGS.no_tags)
  L_C_mat = DenseMatrix(L_C, tags=TAGS.no_tags)
  L_D_mat = DenseMatrix(L_D, tags=TAGS.no_tags)

  # Create Block2x2Matrix instances
  F = Block2x2Matrix.from_blocks(F_A_mat, F_B_mat, F_C_mat, F_D_mat)
  L = Block2x2Matrix.from_blocks(L_A_mat, L_B_mat, L_C_mat, L_D_mat)

  return LinearTimeInvariantSDE(F=F, L=L)

def create_lti_sde_block_3x3():
  key = random.PRNGKey(0)
  dim = 6  # Total dimension (divisible by 3 for 3x3 blocks)
  third_dim = dim // 3  # Each block will be 2x2

  # Generate random matrices F and L
  k1, k2 = random.split(key, 2)
  F_full = random.normal(k1, (dim, dim)) * 0.1
  L_full = random.normal(k2, (dim, dim)) * 0.1

  # Split F into 9 blocks
  F_blocks = []
  L_blocks = []
  for i in range(3):
    for j in range(3):
      F_block = F_full[i*third_dim:(i+1)*third_dim, j*third_dim:(j+1)*third_dim]
      L_block = L_full[i*third_dim:(i+1)*third_dim, j*third_dim:(j+1)*third_dim]

      F_blocks.append(DenseMatrix(F_block, tags=TAGS.no_tags))
      L_blocks.append(DenseMatrix(L_block, tags=TAGS.no_tags))

  # Create Block3x3Matrix instances
  F = Block3x3Matrix.from_blocks(*F_blocks)
  L = Block3x3Matrix.from_blocks(*L_blocks)

  return LinearTimeInvariantSDE(F=F, L=L)

# Test configuration data
SDE_FACTORIES = [
  ("lti_dense", create_lti_sde_dense),
  ("lti_diagonal", create_lti_sde_diagonal),
  ("lti_block_2x2", create_lti_sde_block_2x2),
  ("lti_block_3x3", create_lti_sde_block_3x3),
]

PARAMETERIZATIONS = ['standard', 'mixed', 'natural']

MATRIX_TYPES = ['dense', 'diagonal']

POTENTIAL_CONFIGS = [
  # Length 1
  ("single_potential", lambda p, z, i: [p]),
  ("single_zero", lambda p, z, i: [z]),
  ("single_inf", lambda p, z, i: [i]),

  # Length 2
  ("potential_zero", lambda p, z, i: [p, z]),
  ("potential_inf", lambda p, z, i: [p, i]),
  ("zero_potential", lambda p, z, i: [z, p]),
  ("zero_inf", lambda p, z, i: [z, i]),
  ("inf_potential", lambda p, z, i: [i, p]),
  ("inf_zero", lambda p, z, i: [i, z]),

  # Length 3
  ("potential_zero_inf", lambda p, z, i: [p, z, i]),
  ("potential_inf_zero", lambda p, z, i: [p, i, z]),
  ("zero_potential_inf", lambda p, z, i: [z, p, i]),
  ("zero_inf_potential", lambda p, z, i: [z, i, p]),
  ("inf_potential_zero", lambda p, z, i: [i, p, z]),
  ("inf_zero_potential", lambda p, z, i: [i, z, p]),
]


def create_potentials(sde: AbstractSDE, parameterization: str, matrix_type: str):
  """Create the three types of potentials for a given SDE and parameterization."""
  vec = jnp.ones(sde.dim)
  mat_data = jnp.eye(sde.dim)

  if matrix_type == 'dense':
    mat = DenseMatrix(mat_data, tags=TAGS.no_tags)
  elif matrix_type == 'diagonal':
    mat = DiagonalMatrix(jnp.diag(mat_data), tags=TAGS.no_tags)
  else:
    raise ValueError(f'Invalid matrix_type: {matrix_type}')

  if parameterization == 'standard':
    potential = StandardGaussian(mu=vec, Sigma=mat)
  elif parameterization == 'mixed':
    potential = MixedGaussian(mu=vec, J=mat)
  elif parameterization == 'natural':
    potential = NaturalGaussian(J=mat, h=vec)
  else:
    raise ValueError(f'Invalid parameterization: {parameterization}')

  zero_potential = potential.total_uncertainty_like(potential)

  if isinstance(potential, NaturalGaussian):
    inf_potential = potential # NaturalGaussian doesn't have a total certainty like
  else:
    inf_potential = potential.total_certainty_like(vec, potential)

  return potential, zero_potential, inf_potential


@pytest.mark.parametrize("sde_name,sde_factory", SDE_FACTORIES)
@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
@pytest.mark.parametrize("config_name,config_factory", POTENTIAL_CONFIGS)
def test_sde_configuration(sde_name, sde_factory, parameterization, matrix_type, config_name, config_factory):
  """Test a specific SDE with a specific parameterization, matrix type, and potential configuration."""
  # Create the SDE
  sde = sde_factory()

  # Create the potentials for this parameterization and matrix type
  potential, zero_potential, inf_potential = create_potentials(sde, parameterization, matrix_type)

  # Create the specific configuration
  config = config_factory(potential, zero_potential, inf_potential)

  # Build the potential series
  potentials = jtu.tree_map(lambda *xs: jnp.array(xs), *config)
  ts = jnp.linspace(0.0, 1.0, potentials.batch_size)
  potential_series = GaussianPotentialSeries(ts, potentials)

  # Condition the SDE and test
  conditioned_sde = sde.condition_on(potential_series)

  # Draw samples from the conditioned SDE
  key = random.PRNGKey(0)

  # Test that samples from the ODE and CRF have the same distribution
  keys = random.split(key, 10000)
  N = 3
  save_times = random.uniform(key, (N,), minval=0.0, maxval=2/N).cumsum()

  # Get the marginals from the CRF
  crf_result = conditioned_sde.discretize(save_times)
  crf, info = crf_result.crf, crf_result.info
  marginals = crf.get_marginals()


  # check_sde_marginal_distribution(conditioned_sde)


if __name__ == "__main__":
  pytest.main([__file__])