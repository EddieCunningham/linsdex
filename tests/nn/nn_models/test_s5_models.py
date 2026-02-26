import jax
import jax.numpy as jnp
import jax.random as random

from linsdex.nn.nn_models.s5 import (
  S5Args,
  S5Block,
  S5Seq2SeqModel,
  S5SeqHypers,
  StackedS5Blocks,
  StackedS5BlocksHypers,
  TimeDependentS5Seq2SeqModel,
  TimeDependentS5SeqHypers,
)
import equinox as eqx
from typing import Any, Tuple
from linsdex.nn.nn_models.get_nn import get_nn
# from experiment.config_base import ExperimentConfig, EnvironmentConfig, TrainerConfig, AbstractModelConfig, AbstractDatasetConfig
from linsdex import TimeSeries


def small_args(d_model=8):
  return S5Args(
    d_model=d_model,
    ssm_size=8,
    blocks=1,
    C_init="lecun_normal",
    discretization="zoh",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=False,
    clip_eigs=False,
    bidirectional=False,
    step_rescale=1.0,
  )


def test_s5_block_forward_shape_and_residual():
  key = random.PRNGKey(0)
  args = small_args(d_model=8)
  block = S5Block(args, key=key)

  L, H = 10, args.d_model
  x = random.normal(key, (L, H))
  y = block(x)

  assert y.shape == (L, H)
  assert jnp.isfinite(y).all()
  # Residual path should change values in general
  assert not jnp.allclose(y, x)


def test_s5_block_train_mode_flag_no_effect():
  key = random.PRNGKey(1)
  args = small_args(d_model=8)
  block = S5Block(args, key=key)

  L, H = 12, args.d_model
  x = random.normal(key, (L, H))
  y_train = block(x)
  y_eval = block(x)

  assert y_train.shape == (L, H)
  assert jnp.isfinite(y_train).all()
  assert jnp.allclose(y_train, y_eval)


def test_s5_block_with_conditioning_film_modulates():
  key = random.PRNGKey(4)
  args = small_args(d_model=8)
  cond_size = 3
  block = S5Block(args, key=key, cond_size=cond_size)

  L, H = 10, args.d_model
  x = random.normal(key, (L, H))
  y = random.normal(key, (L, cond_size))
  out_cond = block(x, y=y)
  out_no_cond = block(x, y=None)

  assert out_cond.shape == (L, H)
  assert out_no_cond.shape == (L, H)
  assert jnp.isfinite(out_cond).all()
  assert jnp.isfinite(out_no_cond).all()
  # Conditioning should generally change outputs
  assert not jnp.allclose(out_cond, out_no_cond)


def test_s5_seq_model_forward():
  key = random.PRNGKey(2)
  hypers = S5SeqHypers(d_model=8, num_layers=2)
  model = S5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 9
  x_vals = random.normal(key, (L, 6))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)
  y = model(x_ts)

  assert y.shape == (L, 5)
  assert jnp.isfinite(y).all()


def test_s5_seq_model_forward_with_conditioning_passthrough():
  key = random.PRNGKey(5)
  hypers = S5SeqHypers(d_model=8, num_layers=2, cond_size=2)
  model = S5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 9
  x_vals = random.normal(key, (L, 6))
  y_vals = random.normal(key, (L, 2))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)
  y_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=y_vals)
  out_with_cond = model(x_ts, condition_info=y_ts)
  # When cond_size is set, calling without y should raise
  try:
    _ = model(x_ts, condition_info=None)
    assert False, "Expected ValueError when neither condition_info nor context is provided"
  except ValueError:
    pass
  # Compare against an unconditioned model to ensure difference
  hypers_no_cond = S5SeqHypers(d_model=8, num_layers=2)
  model_no_cond = S5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers_no_cond, key=key)
  out_no_cond = model_no_cond(x_ts)

  assert out_with_cond.shape == (L, 5)
  assert out_no_cond.shape == (L, 5)
  assert jnp.isfinite(out_with_cond).all()
  assert jnp.isfinite(out_no_cond).all()
  # Conditioning should alter outputs
  assert not jnp.allclose(out_with_cond, out_no_cond)


def test_s5_seq_model_context_equivalence_and_reuse():
  key = random.PRNGKey(10)
  hypers = S5SeqHypers(d_model=8, num_layers=2, cond_size=3)
  model = S5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 12
  x_vals = random.normal(key, (L, 6))
  y_vals = random.normal(key, (L, 3))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)
  y_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=y_vals)

  # Compute baseline using condition_info
  out_ci = model(x_ts, condition_info=y_ts)

  # Precompute reusable context and call using context-only
  ctx = model.create_context(y_ts)
  out_ctx = model(x_ts, context=ctx)

  assert out_ci.shape == out_ctx.shape == (L, 5)
  assert jnp.isfinite(out_ci).all() and jnp.isfinite(out_ctx).all()
  # Should be identical when using same model and inputs
  assert jnp.allclose(out_ci, out_ctx)


def test_s5_seq_model_context_required_when_cond_enabled():
  key = random.PRNGKey(11)
  hypers = S5SeqHypers(d_model=8, num_layers=2, cond_size=2)
  model = S5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 7
  x_vals = random.normal(key, (L, 6))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)

  try:
    _ = model(x_ts)
    assert False, "Expected ValueError when cond_size is set but neither condition_info nor context provided"
  except ValueError:
    pass


def test_s5_seq_model_context_ignored_when_no_cond():
  key = random.PRNGKey(12)
  hypers = S5SeqHypers(d_model=8, num_layers=2)
  model = S5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 9
  x_vals = random.normal(key, (L, 6))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)

  # No conditioning enabled; context should be ignored and results equal
  out_no_ctx = model(x_ts)
  dummy_ctx = jnp.zeros((L, 1))
  out_with_ctx = model(x_ts, context=dummy_ctx)

  assert out_no_ctx.shape == out_with_ctx.shape == (L, 5)
  assert jnp.allclose(out_no_ctx, out_with_ctx)


class DummyDatasetConfig(eqx.Module):
  name: str = "dummy_ds"
  unique_name: str = "dummy_ds"
  n_features: int = 6
  train_proportion: float = 0.8
  val_proportion: float = 0.1
  test_proportion: float = 0.1


class DummyModelConfig(eqx.Module):
  name: str = "dummy_model"
  unique_name: str = "dummy_model"
  nn_type: str = "ssm"
  d_model: int = 8
  ssm_size: int = 64
  blocks: int = 2
  num_layers: int = 2
  cond_size: int = 3


def make_base_config(nn_type: str) -> Tuple[Any, Any, int]:
  ds = DummyDatasetConfig()
  model = DummyModelConfig()
  model = eqx.tree_at(lambda m: m.nn_type, model, nn_type)
  return model, ds, 0


def test_get_nn_returns_s5_seq2seq_model():
  model_cfg, ds_cfg, seed = make_base_config("ssm")
  model = get_nn(model_cfg, ds_cfg, seed)
  assert isinstance(model, S5Seq2SeqModel)


def test_get_nn_returns_time_dependent_s5_seq2seq_model():
  model_cfg, ds_cfg, seed = make_base_config("time_dependent_ssm")
  model = get_nn(model_cfg, ds_cfg, seed)
  assert isinstance(model, TimeDependentS5Seq2SeqModel)


def test_time_dependent_s5_seq2seq_decoder_only():
  key = random.PRNGKey(21)
  hypers = TimeDependentS5SeqHypers(d_model=8, num_layers=2)
  model = TimeDependentS5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 9
  x_vals = random.normal(key, (L, 6))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)

  s = jnp.array(0.3)
  out = model(s, x_ts)
  assert out.shape == (L, 5)
  assert jnp.isfinite(out).all()

  # Changing s should change outputs
  out2 = model(jnp.array(0.8), x_ts)
  assert not jnp.allclose(out, out2)


def test_time_dependent_s5_seq2seq_with_condition_info_and_context():
  key = random.PRNGKey(22)
  hypers = TimeDependentS5SeqHypers(d_model=8, num_layers=2, cond_size=3)
  model = TimeDependentS5Seq2SeqModel(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 10
  x_vals = random.normal(key, (L, 6))
  y_vals = random.normal(key, (L, 3))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)
  y_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=y_vals)

  s = jnp.array(0.1)
  # Baseline with condition_info
  out_ci = model(s, x_ts, condition_info=y_ts)

  # Context-only should be identical
  # We need to re-create a fresh model with same parameters or reuse the same model.
  # Use the same model to avoid randomness. Create context via its encoder.
  # Build context once and reuse
  ctx = model.create_context(y_ts)
  out_ctx = model(s, x_ts, context=ctx)

  assert out_ci.shape == out_ctx.shape == (L, 5)
  assert jnp.isfinite(out_ci).all() and jnp.isfinite(out_ctx).all()
  assert jnp.allclose(out_ci, out_ctx)

  # With cond enabled, missing both should error
  try:
    _ = model(s, x_ts)
    assert False, "Expected ValueError when cond_size set but neither condition_info nor context provided"
  except ValueError:
    pass


def test_s5_seq_model_forward_multiple_layers():
  key = random.PRNGKey(3)
  hypers = S5SeqHypers(d_model=8, num_layers=3)
  model = S5Seq2SeqModel(input_size=7, output_size=4, hypers=hypers, key=key)

  L = 11
  x_vals = random.normal(key, (L, 7))
  x_ts = TimeSeries(times=jnp.linspace(0, 1, L), values=x_vals)
  y_train = model(x_ts)
  y_eval = model(x_ts)


def test_stacked_s5_blocks_forward_shapes():
  key = random.PRNGKey(7)
  hypers = S5SeqHypers(d_model=8, num_layers=2)
  stack = StackedS5Blocks(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 10
  x = random.normal(key, (L, 6))
  y = stack(x)

  assert y.shape == (L, 5)
  assert jnp.isfinite(y).all()


def test_stacked_s5_blocks_with_conditioning_modulates():
  key = random.PRNGKey(8)
  hypers = StackedS5BlocksHypers(d_model=8, num_layers=2, cond_size=3)
  stack = StackedS5Blocks(input_size=6, output_size=5, hypers=hypers, key=key)

  L = 10
  x = random.normal(key, (L, 6))
  y = random.normal(key, (L, 3))
  out_with = stack(x, y=y)
  out_without = stack(x, y=None)

  assert out_with.shape == (L, 5)
  assert out_without.shape == (L, 5)
  assert jnp.isfinite(out_with).all()
  assert jnp.isfinite(out_without).all()
  assert not jnp.allclose(out_with, out_without)


