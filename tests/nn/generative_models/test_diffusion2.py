import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Optional, Tuple, Dict, Any, Annotated
import equinox as eqx

from linsdex import TimeSeries
from linsdex.nn.generative_models.diffusion import (
  DiffusionModel,
  DiffusionModelConfig,
  ImprovedDiffusionTimeSeriesModel,
  ImprovedDiffusionTimeSeriesModelConfig,
)


class DummyDatasetConfigVector(eqx.Module):
  name: str = "dummy_vector"
  unique_name: str = "dummy_vector_unique"
  n_features: int = 4
  train_proportion: float = 0.7
  val_proportion: float = 0.15
  test_proportion: float = 0.15


class DummyDatasetConfigSeries(eqx.Module):
  name: str = "dummy_series"
  unique_name: str = "dummy_series_unique"
  n_features: int = 2
  train_proportion: float = 0.7
  val_proportion: float = 0.15
  test_proportion: float = 0.15
  seq_length: int = 20
  observation_noise: float = 0.1
  process_noise: float = 0.1
  time_scale_mult: float = 1.0
  condition_length: int = 8


def make_vector_config(matching_target: str = "drift"):
  dataset = DummyDatasetConfigVector()

  model = DiffusionModelConfig(
    name="diffusion",
    sde_type="ornstein_uhlenbeck",
    nn_type="time_dependent_resnet",
    matching_target=matching_target,
    use_multisample_matching=False,
    process_noise=0.1,
    lambda_=0.1,
    # NN hyperparameters for time_dependent_resnet
    working_size=8,
    hidden_size=8,
    n_blocks=2,
    embedding_size=8,
    out_features=8,
  )

  return model, dataset, 0


def make_series_config(matching_target: str = "flow"):
  dataset = DummyDatasetConfigSeries()

  model = ImprovedDiffusionTimeSeriesModelConfig(
    name="improved_diffusion_time_series",
    sde_type="ornstein_uhlenbeck",
    nn_type="time_dependent_ssm",
    matching_target=matching_target,
    use_multisample_matching=False,
    process_noise=0.1,
    lambda_=0.1,
    # Optional SSM knobs (defaults are provided in get_nn if omitted)
    d_model=dataset.n_features,
    ssm_size=32,
    blocks=2,
    n_layers=2,
    cond_size=None,
  )

  return model, dataset, 0


class TestDiffusionModelVector:
  def test_init_and_loss_sample_logprob(self):
    # Loss on a batched vector (use drift target here)
    m_cfg, d_cfg, seed = make_vector_config(matching_target="drift")
    model = DiffusionModel(m_cfg, d_cfg, seed)

    assert model is not None
    assert model.components is not None

    key = random.PRNGKey(0)
    B, D = 3, d_cfg.n_features
    data = random.normal(key, (B, D))

    loss, aux = model.loss_fn(key, data)
    assert jnp.isfinite(loss)
    assert isinstance(aux, dict)
    assert "matching_loss" in aux

    # For ODE sampling, use a model trained to predict the flow
    m_cfg_ode, d_cfg_ode, seed_ode = make_vector_config(matching_target="flow")
    model_ode = DiffusionModel(m_cfg_ode, d_cfg_ode, seed_ode)
    x_ode = model_ode.sample(key, method='ode')
    assert x_ode.shape == (D,)
    assert jnp.all(jnp.isfinite(x_ode))

    # For SDE sampling, use a model trained to predict the drift
    m_cfg_sde, d_cfg_sde, seed_sde = make_vector_config(matching_target="drift")
    model_sde = DiffusionModel(m_cfg_sde, d_cfg_sde, seed_sde)
    x_sde = model_sde.sample(key, method='sde')
    assert x_sde.shape == (D,)
    assert jnp.all(jnp.isfinite(x_sde))

    lp = model_ode.log_prob(x_ode)
    assert jnp.isfinite(lp)

  def test_context_path_matches_condition_info_path(self):
    # Build two equivalent runs: one passing raw condition via condition_info,
    # one passing precomputed context via context kwarg.
    m_cfg, d_cfg, seed = make_vector_config(matching_target="drift")
    model = DiffusionModel(m_cfg, d_cfg, seed)

    key = random.PRNGKey(7)
    B, D = 2, d_cfg.n_features
    data = random.normal(key, (B, D))

    # Fabricate a small condition vector and use both paths
    cond = random.normal(key, (3,))
    # Use model.create_context (currently a passthrough) to compute context
    ctx = model.create_context(cond)

    # For loss, condition_info must be batched to match data batch dimension
    cond_batched = jnp.broadcast_to(cond, (B, cond.shape[-1]))
    loss_ci, _ = model.loss_fn(key, data, condition_info=cond_batched)
    loss_ctx, _ = model.loss_fn(key, data, context=ctx)
    assert jnp.isfinite(loss_ci) and jnp.isfinite(loss_ctx)
    # Require identical outputs under context created from condition_info
    assert jnp.allclose(loss_ci, loss_ctx, rtol=1e-6, atol=1e-6)

    # Sampling API parity for both ODE(flow) and SDE(drift)
    m_cfg_ode, d_cfg_ode, seed_ode = make_vector_config(matching_target="flow")
    model_ode = DiffusionModel(m_cfg_ode, d_cfg_ode, seed_ode)
    x1_ci = model_ode.sample(key, condition_info=cond, method='ode')
    x1_ctx = model_ode.sample(key, context=model_ode.create_context(cond), method='ode')
    assert x1_ci.shape == x1_ctx.shape == (D,)
    assert jnp.all(jnp.isfinite(x1_ci)) and jnp.all(jnp.isfinite(x1_ctx))
    assert jnp.allclose(x1_ci, x1_ctx, rtol=1e-6, atol=1e-6)

    m_cfg_sde, d_cfg_sde, seed_sde = make_vector_config(matching_target="drift")
    model_sde = DiffusionModel(m_cfg_sde, d_cfg_sde, seed_sde)
    x1_ci = model_sde.sample(key, condition_info=cond, method='sde')
    x1_ctx = model_sde.sample(key, context=model_sde.create_context(cond), method='sde')
    assert x1_ci.shape == x1_ctx.shape == (D,)
    assert jnp.all(jnp.isfinite(x1_ci)) and jnp.all(jnp.isfinite(x1_ctx))
    assert jnp.allclose(x1_ci, x1_ctx, rtol=1e-6, atol=1e-6)

    # Log-prob parity
    lp_ci = model_ode.log_prob(x1_ci, condition_info=cond)
    lp_ctx = model_ode.log_prob(x1_ci, context=model_ode.create_context(cond))
    assert jnp.isfinite(lp_ci) and jnp.isfinite(lp_ctx)
    assert jnp.allclose(lp_ci, lp_ctx, rtol=1e-6, atol=1e-6)

  def test_drift_fn_accepts_context(self):
    m_cfg, d_cfg, seed = make_vector_config(matching_target="drift")
    model = DiffusionModel(m_cfg, d_cfg, seed)
    t = jnp.array(0.5)
    xt = jnp.zeros((d_cfg.n_features,))
    cond = jnp.ones((3,))
    ctx = model.create_context(cond)
    d1 = model.drift_fn(t, xt, condition_info=cond)
    d2 = model.drift_fn(t, xt, context=ctx)
    assert d1.shape == d2.shape == xt.shape
    assert jnp.all(jnp.isfinite(d1)) and jnp.all(jnp.isfinite(d2))


class TestImprovedDiffusionTimeSeriesSSM:
  def test_init_and_loss_sample_logprob(self):
    # Loss expects a batched TimeSeries; use drift or flow—either is fine
    m_cfg, d_cfg, seed = make_series_config(matching_target="flow")
    model = ImprovedDiffusionTimeSeriesModel(m_cfg, d_cfg, seed)

    assert model is not None
    assert model.components is not None

    # Build a small sinusoid series
    T = d_cfg.seq_length
    times = jnp.linspace(0, 2 * jnp.pi, T)
    values = jnp.stack([
      jnp.sin(times),
      jnp.cos(times)
    ], axis=-1)
    series = TimeSeries(times, values)
    # Batch the series for the loss function
    series_batched = TimeSeries(times[None, :], values[None, :, :])

    key = random.PRNGKey(42)
    loss, aux = model.loss_fn(key, series_batched)
    assert jnp.isfinite(loss)
    assert isinstance(aux, dict)
    assert "matching_loss" in aux

    # For ODE sampling, use a model targeting the flow
    m_cfg_ode, d_cfg_ode, seed_ode = make_series_config(matching_target="flow")
    model_ode = ImprovedDiffusionTimeSeriesModel(m_cfg_ode, d_cfg_ode, seed_ode)
    samp = model_ode.sample(key, times=times)
    assert isinstance(samp, TimeSeries)
    assert samp.times.shape == times.shape
    assert samp.values.shape == values.shape
    assert jnp.all(jnp.isfinite(samp.values))

    # For SDE sampling, use a model targeting the drift
    m_cfg_sde, d_cfg_sde, seed_sde = make_series_config(matching_target="drift")
    model_sde = ImprovedDiffusionTimeSeriesModel(m_cfg_sde, d_cfg_sde, seed_sde)
    samp_sde = model_sde.sample(key, times=times, method='sde')
    assert isinstance(samp_sde, TimeSeries)
    assert samp_sde.values.shape == values.shape

    lp = model_ode.log_prob(series)
    assert jnp.isfinite(lp)

  def test_with_conditioning(self):
    # Model with explicit cond_size for SSM
    dataset = DummyDatasetConfigSeries()

    model_flow = ImprovedDiffusionTimeSeriesModelConfig(
      name="improved_diffusion_time_series",
      sde_type="ornstein_uhlenbeck",
      nn_type="time_dependent_ssm",
      matching_target="flow",
      use_multisample_matching=False,
      process_noise=0.1,
      lambda_=0.1,
      d_model=dataset.n_features,
      ssm_size=32,
      blocks=2,
      n_layers=2,
      cond_size=3,
    )
    model_ode = ImprovedDiffusionTimeSeriesModel(model_flow, dataset, 0)

    model_drift = ImprovedDiffusionTimeSeriesModelConfig(
      name="improved_diffusion_time_series",
      sde_type="ornstein_uhlenbeck",
      nn_type="time_dependent_ssm",
      matching_target="drift",
      use_multisample_matching=False,
      process_noise=0.1,
      lambda_=0.1,
      d_model=dataset.n_features,
      ssm_size=32,
      blocks=2,
      n_layers=2,
      cond_size=3,
    )
    model_sde = ImprovedDiffusionTimeSeriesModel(model_drift, dataset, 0)

    # Build times and series
    T = dataset.seq_length
    times = jnp.linspace(0, 2 * jnp.pi, T)
    values = jnp.stack([jnp.sin(times), jnp.cos(times)], axis=-1)
    series = TimeSeries(times, values)
    series_batched = TimeSeries(times[None, :], values[None, :, :])

    key = random.PRNGKey(321)
    cond_unbatched = TimeSeries(times, random.normal(key, (T, 3)))
    cond_batched = TimeSeries(times[None, :], random.normal(key, (1, T, 3)))

    # Loss with conditioning
    loss, aux = model_sde.loss_fn(key, series_batched, condition_info=cond_batched)
    assert jnp.isfinite(loss)
    assert "matching_loss" in aux

    # ODE sampling (flow)
    samp_flow = model_ode.sample(key, times=times, condition_info=cond_unbatched, method='ode')
    assert isinstance(samp_flow, TimeSeries)
    assert jnp.all(jnp.isfinite(samp_flow.values))

    # SDE sampling (drift)
    samp_drift = model_sde.sample(key, times=times, condition_info=cond_unbatched, method='sde')
    assert isinstance(samp_drift, TimeSeries)
    assert jnp.all(jnp.isfinite(samp_drift.values))

    # Log prob with conditioning
    lp = model_ode.log_prob(series, condition_info=cond_unbatched)
    assert jnp.isfinite(lp)

  def test_context_equality_time_series(self):
    # Ensure outputs equal when using condition_info directly vs context from create_context
    dataset = DummyDatasetConfigSeries()

    model_cfg = ImprovedDiffusionTimeSeriesModelConfig(
      name="improved_diffusion_time_series",
      sde_type="ornstein_uhlenbeck",
      nn_type="time_dependent_ssm",
      matching_target="flow",
      use_multisample_matching=False,
      process_noise=0.1,
      lambda_=0.1,
      d_model=dataset.n_features,
      ssm_size=32,
      blocks=2,
      n_layers=2,
      cond_size=3,
    )
    model = ImprovedDiffusionTimeSeriesModel(model_cfg, dataset, 0)

    # Build series and conditioning series (same times)
    T = dataset.seq_length
    times = jnp.linspace(0, 2 * jnp.pi, T)
    values = jnp.stack([jnp.sin(times), jnp.cos(times)], axis=-1)
    series = TimeSeries(times, values)
    cond_series = TimeSeries(times, random.normal(random.PRNGKey(0), (T, 3)))

    # Loss fn equality (batched)
    series_b = TimeSeries(times[None, :], values[None, :, :])
    cond_b = TimeSeries(times[None, :], cond_series.values[None, :, :])
    key = random.PRNGKey(9)

    from linsdex.nn.generative_models.diffusion import TimeSeriesConditionInfo
    cinfo = TimeSeriesConditionInfo(times=times, seq_length=T, extra_conditioning=cond_series)
    ctx = model.create_context(cinfo)
    loss_ci, _ = model.loss_fn(key, series_b, condition_info=cond_b)
    loss_ctx, _ = model.loss_fn(key, series_b, context=ctx)
    assert jnp.allclose(loss_ci, loss_ctx, rtol=1e-6, atol=1e-6)

    # ODE sample equality (flow target)
    samp_ci = model.sample(key, times=times, condition_info=cond_series, method='ode')
    samp_ctx = model.sample(key, times=times, context=ctx, method='ode')
    assert jnp.allclose(samp_ci.values, samp_ctx.values, rtol=1e-6, atol=1e-6)

    # Log prob equality
    lp_ci = model.log_prob(series, condition_info=cond_series)
    lp_ctx = model.log_prob(series, context=ctx)
    assert jnp.allclose(lp_ci, lp_ctx, rtol=1e-6, atol=1e-6)

  def test_time_series_drift_fn_accepts_context(self):
    dataset = DummyDatasetConfigSeries()
    model_cfg = ImprovedDiffusionTimeSeriesModelConfig(
      name="improved_diffusion_time_series",
      sde_type="ornstein_uhlenbeck",
      nn_type="time_dependent_ssm",
      matching_target="drift",
      use_multisample_matching=False,
      process_noise=0.1,
      lambda_=0.1,
      d_model=dataset.n_features,
      ssm_size=32,
      blocks=2,
      n_layers=2,
      cond_size=3,
    )
    model = ImprovedDiffusionTimeSeriesModel(model_cfg, dataset, 0)
    T = dataset.seq_length
    times = jnp.linspace(0, 2 * jnp.pi, T)
    xt = jnp.zeros((T * dataset.n_features,))
    cond_series = TimeSeries(times, jnp.ones((T, 3)))
    t = jnp.array(0.5)
    # Wrap cond into TimeSeriesConditionInfo and test both paths
    from linsdex.nn.generative_models.diffusion import TimeSeriesConditionInfo
    cinfo = TimeSeriesConditionInfo(times=times, seq_length=T, extra_conditioning=cond_series)
    d1 = model.drift_fn(t, xt, condition_info=cinfo)
    ctx = model.create_context(cinfo)
    d2 = model.drift_fn(t, xt, condition_info=cinfo, context=ctx)
    assert d1.shape == d2.shape == xt.shape
    assert jnp.all(jnp.isfinite(d1)) and jnp.all(jnp.isfinite(d2))
