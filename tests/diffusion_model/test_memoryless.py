import pytest
import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as jtu
import diffrax

from linsdex.linear_functional.functional_ops import resolve_functional
from linsdex.matrix.matrix_base import AbstractSquareMatrix
from linsdex.series.plot import plot_multiple_series
jax.config.update("jax_enable_x64", True)


from jaxtyping import PRNGKeyArray, Float, Array

from linsdex import OrnsteinUhlenbeck, GaussianTransition, BrownianMotion
from linsdex.potential.gaussian.dist import StandardGaussian
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex import empirical_dist, w2_distance

from linsdex.diffusion_model.probability_path import DiffusionModelComponents, get_probability_path
from linsdex.diffusion_model.probability_path import ProbabilityPathSlice, BwdMeanToMarginalMean
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex import AbstractLinearSDE, ConditionedLinearSDE, LinearTimeInvariantSDE
from linsdex.diffusion_model.memoryless import (
  MemorylessForwardSDE,
    sample_memoryless_trajectory,
    get_memoryless_projection_adjoint_path,
    MemorylessFullPath,
)
from linsdex.potential.gaussian.transition import functional_potential_to_transition
import equinox as eqx

class ZeroDriftMemorylessLinearSDE(AbstractLinearSDE):
  """
  A special Linear SDE designed to cancel out the drift in a memoryless projection.
  Specifically, it overrides get_params to provide a drift matrix Ft and bias ut
  that exactly cancel the 'memoryless' drift corrections computed by MemorylessForwardSDE.

  Theoretical Note:
  A zero-drift reverse-time SDE behaves like Brownian motion starting from the terminal state x1.
  Since Brownian motion has finite variance at any finite time, the state xt will always
  be correlated with x1. True independence (p(x0, x1) = p(x0)p(x1)) requires a non-zero
  'forgetting' drift that allows the process to completely lose information about its
  starting point. Consequently, while we can cancel the drift to obtain a zero-drift
  reverse process, this process will NOT be memoryless in the sense of independence.
  """
  base_sde: AbstractLinearSDE
  t0: float
  t1: float
  prior: StandardGaussian
  evidence_cov: DiagonalMatrix
  dim: int

  def __init__(self, base_sde, t0, t1, prior, evidence_cov):
    self.base_sde = base_sde
    self.t0 = t0
    self.t1 = t1
    self.prior = prior
    self.evidence_cov = evidence_cov
    self.dim = base_sde.dim

  @property
  def batch_size(self):
    return self.base_sde.batch_size

  def get_diffusion_coefficient(self, t, xt):
    return self.base_sde.get_diffusion_coefficient(t, xt)

  def get_transition_distribution(self, t0_, t1_):
    return self.base_sde.get_transition_distribution(t0_, t1_)

  def get_params(self, t):
    components_proxy = DiffusionModelComponents(
      linear_sde=self.base_sde,
      t0=self.t0,
      x_t0_prior=self.prior,
      t1=self.t1,
      evidence_cov=self.evidence_cov,
    )
    quantities = ProbabilityPathSlice(components_proxy, t)
    Jb = quantities.beta_precision
    mm2bm = BwdMeanToMarginalMean(components_proxy, t, _quantities=quantities).get_inverse()
    A_m2b, b_m2b = mm2bm.A, mm2bm.b
    I = A_m2b.set_eye()
    L = self.get_diffusion_coefficient(t, ())
    S = L @ L.T @ Jb
    Ft = S @ (I - A_m2b)
    ut = - S @ b_m2b
    return Ft, ut, L

@pytest.mark.parametrize("dim", [2])
def test_memoryless_sampling_simulation_matches_discretization(dim: int):
  key = random.PRNGKey(0)
  k_model, k_x1, k_ts, k_mc = random.split(key, 4)

  # Build simple components with OU base SDE
  t0 = 0.0
  t1 = 1.0
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

  # Choose forward times and a terminal state x1
  # Use two evaluation times strictly inside (t0, t1)
  ts = jnp.linspace(0.001, 0.999, 100) # We need to use a dense grid so that the simulation approach works
  x1 = random.normal(k_x1, (dim,))

  save_indices = jnp.array([20, 60])

  # Monte Carlo draw count
  n = 2048
  keys = random.split(k_mc, n)

  def draw_pair_discretization(key: PRNGKeyArray) -> Float[Array, "2dim"]:
    series = sample_memoryless_trajectory(components, x1, ts, key, method="discretization")
    return series.values[save_indices].reshape(-1)

  def draw_pair_simulation(key: PRNGKeyArray) -> Float[Array, "2dim"]:
    series = sample_memoryless_trajectory(components, x1, ts, key, method="simulation")
    return series.values[save_indices].reshape(-1)

  pairs_disc = jax.vmap(draw_pair_discretization)(keys)
  pairs_sim = jax.vmap(draw_pair_simulation)(keys)

  # Compare empirical joint distributions over (X_{t1}, X_{t2}) via Wasserstein-2
  p_emp_disc = empirical_dist(pairs_disc)
  p_emp_sim = empirical_dist(pairs_sim)
  dist = w2_distance(p_emp_disc, p_emp_sim)

  # Tolerance: simulation should closely match discretization
  assert dist < 0.15


@pytest.mark.parametrize("dim", [2, 4])
def test_memoryless_prior_and_independence(dim: int):
  key = random.PRNGKey(1)
  k_model, k_x1, k_mc = random.split(key, 3)

  # Components (OU base SDE) and prior
  t0 = 0.0
  t1 = 1.0
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

  # Draw terminal states X1 from an arbitrary distribution (broader Gaussian)
  n = 4096
  x1s = random.normal(k_x1, (n, dim)) * 1.7

  # Forward grid: sample close to t0 to approximate X_{t0}
  # Avoid endpoints to keep transitions well-conditioned
  ts = jnp.array([t0 + 1e-5])

  keys = random.split(k_mc, n)

  def draw_x0(key: PRNGKeyArray, x1: Float[Array, "D"]) -> Float[Array, "D"]:
    series = sample_memoryless_trajectory(components, x1, ts, key, method="discretization")
    return series.values[0]

  x0s = jax.vmap(draw_x0)(keys, x1s)

  # 1) Marginal at start should match the prior
  p_emp_x0 = empirical_dist(x0s)
  w2 = w2_distance(prior, p_emp_x0)
  assert w2 < 0.2

  # 2) Independence between X0 and X1: low cross-correlation
  x0_centered = x0s - jnp.mean(x0s, axis=0, keepdims=True)
  x1_centered = x1s - jnp.mean(x1s, axis=0, keepdims=True)
  cov = (x0_centered.T @ x1_centered) / (x0s.shape[0] - 1)
  std0 = jnp.std(x0s, axis=0) + 1e-8
  std1 = jnp.std(x1s, axis=0) + 1e-8
  corr = cov / (std0[:, None] * std1[None, :])
  max_abs_corr = jnp.max(jnp.abs(corr))
  assert max_abs_corr < 0.1

  # 3) Do this analytically
  forward_linear_sde = MemorylessForwardSDE(components)
  transition: GaussianTransition = forward_linear_sde.get_transition_distribution(0.0, t1 - t0 - 1e-3)
  A, u, Sigma = transition.A, transition.u, transition.Sigma
  # Extract raw arrays if wrapped
  A_mat = A.as_matrix() if hasattr(A, "as_matrix") else A
  Sigma_mat = Sigma.as_matrix() if hasattr(Sigma, "as_matrix") else Sigma
  prior_Sigma_mat = prior.Sigma.as_matrix() if hasattr(prior.Sigma, "as_matrix") else prior.Sigma

  # Transition independence: A ≈ 0, u ≈ 0
  assert jnp.linalg.norm(A_mat) < 1e-2
  assert jnp.linalg.norm(u) < 1e-2

  # Terminal marginal at start equals prior: Sigma ≈ prior Sigma
  assert jnp.allclose(Sigma_mat, prior_Sigma_mat, atol=1e-2)


def test_sample_includes_terminal_pair():
  key = random.PRNGKey(42)
  dim = 3
  t0 = 0.0
  t1 = 1.0
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

  # Build ts strictly inside (t0, t1) so t1 is not present
  ts = jnp.linspace(0.1, 0.9, 10)
  x1 = random.normal(key, (dim,))

  series_disc = sample_memoryless_trajectory(components, x1, ts, key, method="discretization")
  series_sim = sample_memoryless_trajectory(components, x1, ts, key, method="simulation")

  # Last time/value should be exactly (t1, x1)
  assert jnp.allclose(series_disc.times[-1], t1)
  assert jnp.allclose(series_sim.times[-1], t1)
  assert jnp.allclose(series_disc.values[-1], x1)
  assert jnp.allclose(series_sim.values[-1], x1)


@pytest.mark.parametrize("dim", [2])
def test_memoryless_forward_sde_brownian(dim: int):
  # Model components with Brownian base SDE
  t0 = 0.0
  t1 = 1.0
  sigma = 0.5
  sde = BrownianMotion(dim=dim, sigma=sigma)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(0.000001 * jnp.ones(dim))
  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  # Choose a forward time t and compute canceling drift overrides Ft, ut
  t = 0.5
  quantities = ProbabilityPathSlice(components, t)
  Jb = quantities.beta_precision
  # Use the mapping that appears in get_params: marginal_mean_to_bwd_mean = (BwdMeanToMarginalMean)^{-1}
  mm2bm = BwdMeanToMarginalMean(components, t, _quantities=quantities).get_inverse()
  A_m2b, b_m2b = mm2bm.A, mm2bm.b
  I = A_m2b.set_eye()
  L = components.linear_sde.get_diffusion_coefficient(t, ())
  S = L @ L.T @ Jb
  Ft = S @ (I - A_m2b)
  ut = - S @ b_m2b

  class DriftOverridingSDE(AbstractLinearSDE):
    base_sde: AbstractLinearSDE
    F_override: any
    u_override: any
    L_override: any
    dim: int

    def __init__(self, base_sde, F_override, u_override, L_override):
      self.base_sde = base_sde
      self.F_override = F_override
      self.u_override = u_override
      self.L_override = L_override
      self.dim = base_sde.dim

    @property
    def batch_size(self):
      return self.base_sde.batch_size

    def get_params(self, _t):
      return self.F_override, self.u_override, self.L_override

    def get_diffusion_coefficient(self, _t, _xt):
      return self.L_override

    def get_transition_distribution(self, t0_, t1_):
      # Delegate transitions to the base SDE
      return self.base_sde.get_transition_distribution(t0_, t1_)

  overridden_sde = DriftOverridingSDE(sde, Ft, ut, L)
  components_overridden = DiffusionModelComponents(
    linear_sde=overridden_sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  forward_linear_sde_overridden = MemorylessForwardSDE(components_overridden)
  s = t1 - t
  A_s, b_s, L_s = forward_linear_sde_overridden.get_params(s)

  # Check Kt == 0 and lt == 0 by asserting A_s == 0 and b_s == 0
  A = A_s.as_matrix() if hasattr(A_s, "as_matrix") else A_s
  b = b_s
  assert jnp.allclose(A, jnp.zeros_like(A), atol=1e-5)
  assert jnp.allclose(b, jnp.zeros_like(b), atol=1e-5)


@pytest.mark.parametrize("dim", [2])
def test_memoryless_sampling_equivalence_zero_drift(dim: int):
  key = random.PRNGKey(0)
  t0 = 0.0
  t1 = 1.0
  sigma = 0.5
  base_sde = BrownianMotion(dim=dim, sigma=sigma)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(0.05 * jnp.ones(dim))
  zero_drift_sde = ZeroDriftMemorylessLinearSDE(base_sde, t0, t1, prior, evidence_cov)
  components = DiffusionModelComponents(
    linear_sde=zero_drift_sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )
  k_x1, k_mc = random.split(key)
  ts = jnp.linspace(0.1, 0.9, 64)
  x1 = random.normal(k_x1, (dim,))
  save_indices = jnp.array([16, 48])
  n = 512
  keys = random.split(k_mc, n)

  def draw_pair_discretization(k):
    series = sample_memoryless_trajectory(components, x1, ts, k, method="discretization")
    return series.values[save_indices].reshape(-1)

  def draw_pair_simulation(k):
    series = sample_memoryless_trajectory(components, x1, ts, k, method="simulation")
    return series.values[save_indices].reshape(-1)

  pairs_sim = jax.vmap(draw_pair_simulation)(keys)
  pairs_disc = jax.vmap(draw_pair_discretization)(keys)
  dist = w2_distance(empirical_dist(pairs_disc), empirical_dist(pairs_sim))
  assert dist < 0.15


@pytest.mark.parametrize("dim", [2])
def test_memoryless_prior_and_independence_zero_drift(dim: int):
  key = random.PRNGKey(1)
  t0 = 0.0
  t1 = 1.0
  sigma = 0.5
  base_sde = BrownianMotion(dim=dim, sigma=sigma)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(0.05 * jnp.ones(dim))
  zero_drift_sde = ZeroDriftMemorylessLinearSDE(base_sde, t0, t1, prior, evidence_cov)
  components = DiffusionModelComponents(
    linear_sde=zero_drift_sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  n = 4096*4
  k_x1, k_mc = random.split(key)
  x1s = random.normal(k_x1, (n, dim)) * 1.2
  ts = jnp.array([t0 + 1e-6])
  keys = random.split(k_mc, n)

  def draw_x0(k, x1):
    series = sample_memoryless_trajectory(components, x1, ts, k, method="discretization")
    return series.values[0]

  x0s = jax.vmap(draw_x0)(keys, x1s)

  x0_centered = x0s - jnp.mean(x0s, axis=0, keepdims=True)
  x1_centered = x1s - jnp.mean(x1s, axis=0, keepdims=True)
  cov = (x0_centered.T @ x1_centered) / (x0s.shape[0] - 1)
  std0 = jnp.std(x0s, axis=0) + 1e-8
  std1 = jnp.std(x1s, axis=0) + 1e-8
  corr = cov / (std0[:, None] * std1[None, :])
  max_abs_corr = jnp.max(jnp.abs(corr))
  # NOTE: We do NOT expect independence here because the drift has been canceled.
  # A zero-drift process (Brownian motion) retains correlation with its starting point.
  # The fact that max_abs_corr is non-zero (e.g., ~0.2-0.4) is theoretically expected.


@pytest.mark.parametrize("dim", [2])
def test_sample_includes_terminal_pair_zero_drift(dim: int):
  key = random.PRNGKey(42)
  t0 = 0.0
  t1 = 1.0
  sigma = 0.5
  base_sde = BrownianMotion(dim=dim, sigma=sigma)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(0.05 * jnp.ones(dim))
  zero_drift_sde = ZeroDriftMemorylessLinearSDE(base_sde, t0, t1, prior, evidence_cov)
  components = DiffusionModelComponents(
    linear_sde=zero_drift_sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  ts = jnp.linspace(0.1, 0.9, 10)
  x1 = random.normal(key, (dim,))
  series_disc = sample_memoryless_trajectory(components, x1, ts, key, method="discretization")
  series_sim = sample_memoryless_trajectory(components, x1, ts, key, method="simulation")
  assert jnp.allclose(series_disc.times[-1], t1)
  assert jnp.allclose(series_sim.times[-1], t1)
  assert jnp.allclose(series_disc.values[-1], x1)
  assert jnp.allclose(series_sim.values[-1], x1)


@pytest.mark.parametrize("dim", [2])
def test_global_zero_drift_memoryless_sde(dim: int):
  # Base SDE for transitions
  t0 = 0.0
  t1 = 1.0
  sigma = 0.5
  base_sde = BrownianMotion(dim=dim, sigma=sigma)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(0.01 * jnp.ones(dim))

  zero_drift_sde = ZeroDriftMemorylessLinearSDE(base_sde, t0, t1, prior, evidence_cov)
  components_zd = DiffusionModelComponents(
    linear_sde=zero_drift_sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  forward_linear_sde = MemorylessForwardSDE(components_zd)
  # Check several times across the interval
  ts = jnp.linspace(t0 + 0.05, t1 - 0.05, 5)
  for t in ts:
    s = t1 - t
    A_s, b_s, _ = forward_linear_sde.get_params(s)
    A = A_s.as_matrix() if hasattr(A_s, "as_matrix") else A_s
    b = b_s
    assert jnp.allclose(A, jnp.zeros_like(A), atol=1e-5)
    assert jnp.allclose(b, jnp.zeros_like(b), atol=1e-5)


@pytest.mark.parametrize("dim", [2, 4])
def test_memoryless_reversal(dim: int):
  from linsdex.crf import CRF
  from linsdex.crf.continuous_crf import DiscretizeResult
  from linsdex.series.interleave_times import InterleavedTimes
  from linsdex.util.parallel_scan import parallel_scan
  key = random.PRNGKey(1)
  t0, t1 = 0.0, 1.0
  k1, k2, k3, k4, k5, k6, k7, k8 = random.split(key, 8)

  F_mat = DiagonalMatrix(random.normal(k1, (dim,)))
  L_mat = DiagonalMatrix(random.normal(k2, (dim,)))
  sde = LinearTimeInvariantSDE(F=F_mat, L=L_mat)

  prior_mu = jnp.zeros((dim,))
  prior_sigma = DiagonalMatrix.eye(dim)
  prior = StandardGaussian(mu=prior_mu, Sigma=prior_sigma)

  evidence_cov = DiagonalMatrix.eye(dim) * 1e-4

  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov
  )

  n_times = 5
  times = jnp.linspace(t0, t1, n_times)

  # Get the quantities at each time step so that we can get p(x_t | y_1)
  def get_quantities(t) -> ProbabilityPathSlice:
    return ProbabilityPathSlice(components, t)

  functional_quantities: ProbabilityPathSlice = eqx.filter_vmap(get_quantities)(jnp.clip(times, 1e-5, 1 - 1e-5))
  functional_pts_given_y1: StandardGaussian = functional_quantities.functional_pt_given_y1

  from linsdex.potential.gaussian.transition import functional_potential_to_transition
  transitions: GaussianTransition = eqx.filter_vmap(functional_potential_to_transition)(functional_pts_given_y1)
  reversed_transitions: GaussianTransition = transitions.swap_variables()

  n_samples = 1024
  keys = random.split(key, (n_samples, n_times))
  functional_xts = eqx.filter_vmap(functional_pts_given_y1.sample)(keys)

  # Take a y1 to fill the samples with
  y1 = random.normal(k6, (dim,))

  xts: Float[Array, 'D'] = functional_xts(y1)

  ########################################
  # Compute the forward memoryless SDE and make
  # sure that it has the same marginal distribution as the original SDE
  ########################################
  forward_memoryless_sde = MemorylessForwardSDE(components)

  cond_sde: ConditionedLinearSDE = forward_memoryless_sde.condition_on_starting_point(0.0, y1)

  # Discretize in reverse time s = 1 - t
  # We want to evaluate at some points in (0, 1)
  s_times = jnp.linspace(0.001, 0.999, n_times)
  result: DiscretizeResult = cond_sde.discretize(s_times)
  crf: CRF = result.crf
  mem_transitions: GaussianTransition = crf.get_transitions()

  def op(left: GaussianTransition, right: GaussianTransition) -> GaussianTransition:
    return left.chain(right)

  # These are p(x_t | y_1) in reverse time s = 1 - t
  mem_cum_transitions: GaussianTransition = parallel_scan(op, mem_transitions)[::-1]
  mem_cum_transitions_reverse: GaussianTransition = mem_cum_transitions.swap_variables()

  # mem_transitions[0] is p(x_{s_1} | x_{s=0}) if s_times[0] is s_1
  # Actually, discretize(s_times) with InterleavedTimes([0.0]) will have all_ts = [0.0, s_times[0], ...]
  # So mem_transitions[0] is p(x_{s_times[0]} | x_{s=0})
  # and mem_transitions[1:] are p(x_{s_times[i]} | x_{s_times[i-1]})

  # We want to compare mem_transitions[1:] with analytical bridge transitions
  # mem_transitions[i] for i >= 1 is p(x_{s_i} | x_{s_{i-1}})
  # which is p(x_{t_i} | x_{t_{i-1}}) where t = 1 - s.
  # Since s_i > s_{i-1}, t_i < t_{i-1}.
  # So it's p(x_{smaller_t} | x_{larger_t}).

  # Compute analytical bridge transitions p(x_{t_i} | x_{t_{i-1}})
  def get_bridge_transition(s_val, t_val):
    from linsdex.diffusion_model.probability_path import probability_path_transition
    # probability_path_transition(comp_t, comp_s, t, s) returns p(x_t | x_s)
    # We want p(x_{1-t_val} | x_{1-s_val}) where t_val > s_val (so 1-t_val < 1-s_val)
    t_smaller = 1.0 - t_val
    t_larger = 1.0 - s_val
    return probability_path_transition(components, components, t_smaller, t_larger)

  s_vals = s_times[:-1]
  t_vals = s_times[1:]
  analytical_transitions = eqx.filter_vmap(get_bridge_transition)(s_vals, t_vals)

  # Check that the transitions are the same
  # Skip the first one because it's from s=0 to s_times[0]
  mem_transitions_to_compare = jtu.tree_map(lambda x: x[1:], mem_transitions)

  assert jnp.allclose(mem_transitions_to_compare.A.as_matrix(), analytical_transitions.A.as_matrix(), atol=1e-5)
  assert jnp.allclose(mem_transitions_to_compare.u, analytical_transitions.u, atol=1e-5)
  assert jnp.allclose(mem_transitions_to_compare.Sigma.as_matrix(), analytical_transitions.Sigma.as_matrix(), atol=1e-5)

  assert xts.shape == (n_samples, n_times, dim)

  # Compute Wasserstein-2 distance between empirical distributions and the prior
  def compute_w2(samples_at_t):
    emp_dist = empirical_dist(samples_at_t)
    return emp_dist, w2_distance(emp_dist, prior)

  empirical_dists, w2_distances = jax.vmap(compute_w2, in_axes=1)(xts)
  print(f"W2 distances to prior: {w2_distances}")

  # Check that at t=0, the W2 distance to the prior is very small
  assert w2_distances[0] < 1e-2



@pytest.mark.parametrize("dim", [2, 4])
def test_memoryless_components(dim: int):
  from linsdex.crf import CRF
  from linsdex.crf.continuous_crf import DiscretizeResult
  from linsdex.series.interleave_times import InterleavedTimes
  from linsdex.util.parallel_scan import parallel_scan
  from linsdex.potential.gaussian.dist import MixedGaussian
  from typing import Tuple
  from linsdex.potential.gaussian.transition import functional_potential_to_transition
  from linsdex.matrix.matrix_base import mat_mul
  from functools import partial

  key = random.PRNGKey(1)
  t0, t1 = 0.0, 1.0
  k1, k2, k3, k4, k5, k6, k7, k8 = random.split(key, 8)

  F_mat = DiagonalMatrix(random.normal(k1, (dim,)))#.set_eye()
  L_mat = DiagonalMatrix(random.normal(k2, (dim,)))#.set_eye()
  sde = LinearTimeInvariantSDE(F=F_mat, L=L_mat)

  prior_mu = jnp.zeros((dim,)) + 4
  prior_sigma = DiagonalMatrix.eye(dim)*2
  prior = StandardGaussian(mu=prior_mu, Sigma=prior_sigma)

  evidence_cov = DiagonalMatrix.eye(dim) * 1e-4

  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov
  )

  n_times = 50
  s_times = jnp.linspace(0.1, 0.9, n_times)
  t_times = 1.0 - s_times

  # Get the quantities at each time step so that we can get p(x_t | y_1)
  def get_quantities(t) -> ProbabilityPathSlice:
    return ProbabilityPathSlice(components, t)

  ########################################
  # Compute the base SDE samples
  ########################################
  functional_quantities: ProbabilityPathSlice = eqx.filter_vmap(get_quantities)(t_times)
  functional_pts_given_y1: StandardGaussian = functional_quantities.functional_pt_given_y1

  n_samples = 1024
  keys = random.split(key, (n_samples, n_times))
  epsilon = random.normal(k7, (n_samples, n_times, dim))
  functional_xts: LinearFunctional = eqx.filter_vmap(functional_pts_given_y1._sample)(epsilon)
  functional_epsilon: LinearFunctional = eqx.filter_vmap(functional_pts_given_y1.get_noise)(functional_xts) # A = 0 and b = epsilon!

  ########################################
  # Compute the forward memoryless SDE and make
  # sure that it has the same marginal distribution as the original SDE
  ########################################
  forward_memoryless_sde = MemorylessForwardSDE(components)

  functional_y1 = LinearFunctional.identity(dim)
  cond_sde: ConditionedLinearSDE = forward_memoryless_sde.condition_on_starting_point(0.0, functional_y1)

  # Discretize in reverse time s = 1 - t
  # We want to evaluate at some points in (0, 1)
  # s_times is already defined

  # Get the diffusion coefficients at each time step
  Ls: AbstractSquareMatrix = eqx.filter_vmap(forward_memoryless_sde.get_diffusion_coefficient, in_axes=(0, None))(s_times, jnp.zeros((dim,)))

  result: DiscretizeResult = cond_sde.discretize(s_times)
  crf: CRF = result.crf
  mem_transitions: GaussianTransition = crf.get_transitions()


  def op(left: GaussianTransition, right: GaussianTransition) -> GaussianTransition:
    return left.chain(right)

  # These are p(x_t | y_1)
  mem_cum_transitions: GaussianTransition = parallel_scan(op, mem_transitions)
  mem_cum_transitions_reverse: GaussianTransition = mem_cum_transitions.swap_variables()

  # Compute the inverse variance of the loss
  adjoint_transitions: AbstractSquareMatrix = mem_cum_transitions_reverse.A
  # The regression target is L.T @ A.T @ energy_score
  # Its covariance is (L.T @ A.T) @ I @ (A @ L) = L.T @ A.T @ A @ L
  loss_cov = Ls.T@adjoint_transitions.T@adjoint_transitions@Ls
  loss_time_scale = 1/loss_cov.get_trace()

  # Sample xt from p(x_t | y_1)
  pxt_given_y1: StandardGaussian = mem_cum_transitions.condition_on_x(functional_y1)
  functional_xts_mem: LinearFunctional = eqx.filter_vmap(pxt_given_y1._sample)(epsilon)
  def functional_xts_to_xts_mem(functional_xts_mem, functional_epsilon):
    return functional_xts_mem(functional_epsilon)
  # mem_coord_change: LinearFunctional = eqx.filter_vmap(functional_xts_to_xts_mem)(functional_xts_mem, functional_epsilon)

  # Sample xt_mem from p(x_t | y_1) and sample the energy scores
  y1 = random.normal(k6, (n_samples, dim))
  def apply_xts_mem(functional_xts_mem, y1):
    return functional_xts_mem(y1)
  xts_mem = eqx.filter_vmap(apply_xts_mem)(functional_xts_mem, y1)
  energy_scores = random.normal(k8, (n_samples, dim))

  @partial(eqx.filter_vmap, in_axes=(None, 0, None))
  @partial(eqx.filter_vmap, in_axes=(0, None, 0))
  def get_rhs_term(L, energy_score, adjoint_transition):
    # return adjoint_transition.T@energy_score
    return L.T@adjoint_transition.T@energy_score

  rhs_terms = get_rhs_term(Ls, energy_scores, adjoint_transitions)

  empirical_dists: StandardGaussian = eqx.filter_vmap(empirical_dist, in_axes=1)(rhs_terms)

  # Check that the empirical distribution matches the analytical loss covariance
  def check_covariance(emp_dist, analytical_cov):
    emp_sigma = emp_dist.Sigma.as_matrix()
    anal_sigma = analytical_cov.as_matrix()
    return jnp.linalg.norm(emp_sigma - anal_sigma) / jnp.linalg.norm(anal_sigma)

  errors = jax.vmap(check_covariance)(empirical_dists, loss_cov)
  assert jnp.all(errors < 0.1) # Relative error less than 10% with 1024 samples


@pytest.mark.parametrize("dim", [2])
def test_memoryless_projection_adjoint_path(dim: int):
  key = random.PRNGKey(0)

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

  # Use stable times
  times = jnp.linspace(0.2, 0.8, 4)

  # 1. Get transitions from get_memoryless_projection_adjoint_path
  # This returns transitions p(y_1 | x_t) for each t in times
  full_path = get_memoryless_projection_adjoint_path(components, times)
  p_y1_given_xt: GaussianTransition = full_path.p_y1_given_xt

  # 2. Get reference transitions from MemorylessForwardSDE directly
  forward_memoryless_sde = MemorylessForwardSDE(components)

  def get_ref_transition(t):
    s = components.t1 - t
    # p(x_s | s=0) in s-time is p(x_t | x_1)
    return forward_memoryless_sde.get_transition_distribution(0.0, s)

  # phi_ref[i] is p(x_{times[i]} | x_1)
  phi_ref = jax.vmap(get_ref_transition)(times)

  # Chain with evidence to get p(y_1 | x_t)
  dim = components.linear_sde.dim
  I = DiagonalMatrix.eye(dim)
  zero = jnp.zeros(dim)
  p_y1_given_x1 = GaussianTransition(I, zero, components.evidence_cov)

  # p_y1_given_xt_ref[i] is p(y_1 | x_{times[i]})
  p_y1_given_xt_ref = phi_ref.swap_variables().chain(p_y1_given_x1)

  # Check A matrices
  A_adj = p_y1_given_xt.A.as_matrix()
  A_ref_swapped = p_y1_given_xt_ref.A.as_matrix()

  assert A_adj.shape == A_ref_swapped.shape
  assert jnp.allclose(A_adj, A_ref_swapped, atol=1e-5)

  # 3. Test p_xt_given_y1 via swap_variables()
  # This corresponds to the marginal of the memoryless process conditioned on y1
  p_xt_given_y1 = p_y1_given_xt.swap_variables()

  # Check against phi_ref (chained with evidence)
  A_xt_given_y1 = p_xt_given_y1.A.as_matrix()
  A_ref_y1 = p_y1_given_xt_ref.swap_variables().A.as_matrix()

  assert jnp.allclose(A_xt_given_y1, A_ref_y1, atol=1e-5)
  assert jnp.allclose(p_xt_given_y1.u, p_y1_given_xt_ref.swap_variables().u, atol=1e-5)
  assert jnp.allclose(p_xt_given_y1.Sigma.as_matrix(), p_y1_given_xt_ref.swap_variables().Sigma.as_matrix(), atol=1e-5)

  # Also check memoryless property of the reference
  # p(x_t | x_1) as t -> 0 should have A -> 0
  t_small = 0.01
  p_x0_given_x1 = get_ref_transition(t_small)
  assert jnp.linalg.norm(p_x0_given_x1.A.as_matrix()) < 0.1


@pytest.mark.parametrize("dim", [2])
def test_lean_adjoint_corollary(dim: int):
  from linsdex.diffusion_model.probability_path import DiffusionModelConversions
  key = random.PRNGKey(42)
  k1, k2, k3 = random.split(key, 3)

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

  memoryless_sde = MemorylessForwardSDE(components)

  # Terminal cost gradient a1
  a1 = random.normal(k1, (dim,))

  # Numerical solution of lean adjoint ODE: da/dt = -F_t^T a
  @diffrax.ODETerm
  def adjoint_ode(t, a, args):
    s = t1 - t
    Ft_val = memoryless_sde.get_params(s)[0].as_matrix()
    return Ft_val.T @ a # Sign flip because of the reverse time

  # Solve backwards from t=1 to t=0.1
  # Use Heun for better accuracy and decreasing ts for backwards integration
  eval_ts = jnp.linspace(0.8, 0.2, 4)
  sol = diffrax.diffeqsolve(
    adjoint_ode,
    diffrax.Heun(),
    t0=1.0,
    t1=0.1,
    dt0=-0.001,
    y0=a1,
    saveat=diffrax.SaveAt(ts=eval_ts),
  )

  # Analytical solution: a_t = A_{1|t}^T a1
  eval_ts_inc = eval_ts[::-1]
  full_path = get_memoryless_projection_adjoint_path(components, eval_ts_inc)
  p_y1_given_xt = full_path.p_y1_given_xt

  # p_y1_given_xt corresponds to [0.2, 0.4, 0.6, 0.8]
  # We want indices for [0.8, 0.6, 0.4, 0.2] which are [3, 2, 1, 0]
  A_1t = p_y1_given_xt.A.as_matrix()[::-1]

  analytical_a = jax.vmap(lambda A: A.T @ a1)(A_1t)

  # Verify numerical matches analytical
  assert jnp.allclose(sol.ys, analytical_a, atol=1e-3)


@pytest.mark.parametrize("dim", [2])
def test_memoryless_marginal_consistency(dim: int):
  key = random.PRNGKey(42)
  k1, k2, k3 = random.split(key, 3)

  t0, t1 = 0.0, 1.0
  times = jnp.linspace(0.1, 0.9, 5)
  sde = OrnsteinUhlenbeck(dim=dim, sigma=1.0, lambda_=1.0)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(1e-6 * jnp.ones(dim))

  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )
  probability_path = get_probability_path(components, times)
  functional_pt_given_y1: StandardGaussian = probability_path.functional_pt_given_y1

  full_path = get_memoryless_projection_adjoint_path(components, times)
  qt_given_y1: GaussianTransition = full_path.p_y1_given_xt

  y1 = random.normal(k2, (dim,))
  qt: StandardGaussian = qt_given_y1.condition_on_y(y1)
  pt: StandardGaussian = resolve_functional(functional_pt_given_y1, y1)

  assert jnp.allclose(pt.mu, qt.mu, atol=1e-6)
  assert jnp.allclose(pt.Sigma.as_matrix(), qt.Sigma.as_matrix(), atol=1e-6)
  assert pt.batch_size == qt.batch_size


@pytest.mark.parametrize("dim", [2])
def test_memoryless_sampling_full_path_matches_sde(dim: int):
  key = random.PRNGKey(42)
  k_y1, k_mc = random.split(key, 2)

  t0, t1 = 0.0, 1.0
  sde = OrnsteinUhlenbeck(dim=dim, sigma=1.0, lambda_=1.0)
  prior = StandardGaussian(mu=jnp.zeros(dim), Sigma=DiagonalMatrix.eye(dim))
  evidence_cov = DiagonalMatrix(1e-5 * jnp.ones(dim))
  components = DiffusionModelComponents(
    linear_sde=sde,
    t0=t0,
    x_t0_prior=prior,
    t1=t1,
    evidence_cov=evidence_cov,
  )

  times = jnp.linspace(0.001, 0.999, 1000)
  n_samples = 8

  # 1. Samples from MemorylessFullPath
  full_path = get_memoryless_projection_adjoint_path(components, times)

  # Sample y1s from some distribution
  y1s = random.normal(k_y1, (n_samples, dim))*0.0

  def sample_full_path(key, y1):
    return full_path.sample(key, y1)

  keys = random.split(k_mc, n_samples)
  series_full_path = jax.vmap(sample_full_path)(keys, y1s)

  # 2. Samples from sample_memoryless_trajectory
  def sample_sde(key, y1):
    return sample_memoryless_trajectory(components, y1, times, key, method="discretization")

  series_sde = jax.vmap(sample_sde)(keys, y1s)

  # 3. Compare empirical marginals at each time point
  assert jnp.allclose(series_full_path.times[0], series_sde.times[0])

  def compute_w2_at_time(v_full, v_sde):
    dist_full = empirical_dist(v_full)
    dist_sde = empirical_dist(v_sde)
    return w2_distance(dist_full, dist_sde)

  # vmap over time dimension (axis 1)
  w2_distances = jax.vmap(compute_w2_at_time, in_axes=1)(series_full_path.values, series_sde.values)

  # The distances should be small
  assert jnp.all(w2_distances < 0.1)


@pytest.mark.parametrize("dim", [2])
def test_memoryless_full_path_diffusion_coefficients(dim: int):
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

  times = jnp.linspace(0.2, 0.8, 5)
  full_path = get_memoryless_projection_adjoint_path(components, times)

  # reversed_times = (t1 - full_path.times)[::-1]
  reversed_times = (t1 - full_path.times)[::-1]

  forward_memoryless_sde = MemorylessForwardSDE(components)

  def get_ref_diffusion(s):
    return forward_memoryless_sde.get_params(s)[2]

  ref_diffusions = jax.vmap(get_ref_diffusion)(reversed_times)

  # Compare matrices
  assert jnp.allclose(full_path.diffusion_coefficients.as_matrix(), ref_diffusions.as_matrix(), atol=1e-5)


@pytest.mark.parametrize("dim", [2])
def test_memoryless_full_path_base_drifts(dim: int):
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

  times = jnp.linspace(0.2, 0.8, 5)
  full_path = get_memoryless_projection_adjoint_path(components, times)

  # reversed_times corresponds to (t1 - times_aug)[::-1]
  # where times_aug = [times, t1]
  # times_aug = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
  # t1 - times_aug = [0.8, 0.65, 0.5, 0.35, 0.2, 0.0]
  # reversed_times = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]
  reversed_times = (t1 - full_path.times)[::-1]

  forward_memoryless_sde = MemorylessForwardSDE(components)

  def get_ref_drift_functional(s):
    F, u, _ = forward_memoryless_sde.get_params(s)
    # The precomputed base_drifts are -drift, so LinearFunctional(-F, -u)
    return LinearFunctional(-F, -u)

  ref_drifts = jax.vmap(get_ref_drift_functional)(reversed_times)

  # Compare A matrices and b vectors
  assert jnp.allclose(full_path.base_drifts.A.as_matrix(), ref_drifts.A.as_matrix(), atol=1e-5)
  assert jnp.allclose(full_path.base_drifts.b, ref_drifts.b, atol=1e-5)

