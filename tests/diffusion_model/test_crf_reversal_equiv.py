import jax
import jax.numpy as jnp
from jax import random
import pytest
from linsdex.diffusion_model.probability_path import DiffusionModelComponents, ProbabilityPathSlice
from linsdex.diffusion_model.memoryless import MemorylessForwardSDE
from linsdex.crf.crf import CRF
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.potential.gaussian.dist import StandardGaussian
from linsdex.potential.gaussian.transition import functional_potential_to_transition
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex import OrnsteinUhlenbeck
from linsdex.util import misc as util
from linsdex.util.parallel_scan import parallel_scan

def old_implementation(components, times):
  times = jnp.concatenate([times, jnp.array([components.t1])])
  reversed_times = (components.t1 - times)[::-1]
  forward_memoryless_sde = MemorylessForwardSDE(components)

  def get_transition(t_start, t_end):
    return forward_memoryless_sde.get_transition_distribution(t_start, t_end)
  p_xtm1_given_xt = jax.vmap(get_transition)(reversed_times[:-1], reversed_times[1:])

  functional_y1 = LinearFunctional.identity(components.linear_sde.dim)
  functional_evidence = StandardGaussian(functional_y1, components.evidence_cov)
  def make_uncertain_potential(_):
    return StandardGaussian.total_uncertainty_like(functional_evidence)
  node_potentials = jax.vmap(make_uncertain_potential)(reversed_times)
  node_potentials = util.fill_array(node_potentials, 0, functional_evidence)

  functional_reversed_crf = CRF(node_potentials, p_xtm1_given_xt)
  functional_transitions = functional_reversed_crf.get_transitions()
  def op(left, right):
    return left.chain(right)
  chained_reversed_transitions = parallel_scan(op, functional_transitions, reverse=False)
  functional_p_xt_given_y1 = chained_reversed_transitions[::-1].condition_on_y(functional_y1)
  functional_p_y1_given_xt = chained_reversed_transitions.swap_variables()[::-1].condition_on_x(functional_y1)

  p_xt_given_y1 = functional_potential_to_transition(functional_p_xt_given_y1).swap_variables()
  p_y1_given_xt = functional_potential_to_transition(functional_p_y1_given_xt)
  return p_xt_given_y1, p_y1_given_xt

def current_implementation(components, times):
  times = jnp.concatenate([times, jnp.array([components.t1])])
  reversed_times = (components.t1 - times)[::-1]
  forward_memoryless_sde = MemorylessForwardSDE(components)

  def get_transition(t_start, t_end):
    return forward_memoryless_sde.get_transition_distribution(t_start, t_end)
  p_xtm1_given_xt = jax.vmap(get_transition)(reversed_times[:-1], reversed_times[1:])

  functional_y1 = LinearFunctional.identity(components.linear_sde.dim)
  functional_evidence = StandardGaussian(functional_y1, components.evidence_cov)
  def make_uncertain_potential(_):
    return StandardGaussian.total_uncertainty_like(functional_evidence)
  node_potentials = jax.vmap(make_uncertain_potential)(reversed_times)
  node_potentials = util.fill_array(node_potentials, 0, functional_evidence)

  functional_crf = CRF(node_potentials, p_xtm1_given_xt).reverse()
  functional_transitions = functional_crf.base_transitions
  def op(left, right):
    return left.chain(right)
  chained_reversed_transitions = parallel_scan(op, functional_transitions, reverse=True)
  functional_p_xt_given_y1 = chained_reversed_transitions.swap_variables().condition_on_y(functional_y1)
  functional_p_y1_given_xt = chained_reversed_transitions.condition_on_x(functional_y1)

  p_xt_given_y1 = functional_potential_to_transition(functional_p_xt_given_y1).swap_variables()
  p_y1_given_xt = functional_potential_to_transition(functional_p_y1_given_xt)
  return p_xt_given_y1, p_y1_given_xt

def test_crf_reversal_equivalence():
  dim = 2
  t0, t1 = 0.0, 1.0
  sde = OrnsteinUhlenbeck(dim=dim, sigma=1.0, lambda_=1.0)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(0.1 * jnp.ones(dim))

  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  times = jnp.linspace(0.2, 0.8, 4)

  p_xt_given_y1_old, p_y1_given_xt_old = old_implementation(components, times)
  p_xt_given_y1_curr, p_y1_given_xt_curr = current_implementation(components, times)

  # Compare p_xt_given_y1
  assert jnp.allclose(p_xt_given_y1_old.A.as_matrix(), p_xt_given_y1_curr.A.as_matrix())
  assert jnp.allclose(p_xt_given_y1_old.u, p_xt_given_y1_curr.u)
  assert jnp.allclose(p_xt_given_y1_old.Sigma.as_matrix(), p_xt_given_y1_curr.Sigma.as_matrix())

  # Compare p_y1_given_xt
  assert jnp.allclose(p_y1_given_xt_old.A.as_matrix(), p_y1_given_xt_curr.A.as_matrix())
  assert jnp.allclose(p_y1_given_xt_old.u, p_y1_given_xt_curr.u)
  assert jnp.allclose(p_y1_given_xt_old.Sigma.as_matrix(), p_y1_given_xt_curr.Sigma.as_matrix())

if __name__ == "__main__":
  test_crf_reversal_equivalence()
