import os
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from linsdex import TimeSeries

from linsdex.nn.nn_models.s5 import (
  TimeFeatures,
  S5, S5Args,
  S5Block,
  StackedS5Blocks, StackedS5BlocksHypers,
  TimeDependentS5Seq2SeqModel, TimeDependentS5SeqHypers,
)


def make_timefeatures(out_features=512, key=random.PRNGKey(0)):
  return TimeFeatures(embedding_size=2 * out_features, out_features=out_features, key=key)


def make_s5(d_model=256, ssm_size=128, blocks=2, key=random.PRNGKey(1)):
  args = S5Args(d_model=d_model, ssm_size=ssm_size, blocks=blocks)
  return S5(args, key=key)


def make_s5block(d_model=256, ssm_size=128, blocks=2, key=random.PRNGKey(2), cond_size: Optional[int] = None):
  args = S5Args(d_model=d_model, ssm_size=ssm_size, blocks=blocks)
  return S5Block(args, key=key, cond_size=cond_size)


def make_stack(d_model=256, ssm_size=128, blocks=2, num_layers=2, key=random.PRNGKey(3), cond_size: Optional[int] = None, in_size=14, out_size=14):
  hypers = StackedS5BlocksHypers(
    d_model=d_model,
    ssm_size=ssm_size,
    blocks=blocks,
    num_layers=num_layers,
    cond_size=cond_size,
  )
  return StackedS5Blocks(in_size, out_size, hypers, key=key)


def make_series(L=16, Dx=14, Dy=16, key=random.PRNGKey(4)):
  kx, ky = random.split(key)
  times = jnp.linspace(0.0, 1.0, L)
  x = TimeSeries(times=times, values=random.normal(kx, (L, Dx)))
  y = TimeSeries(times=times, values=random.normal(ky, (L, Dy)))
  return x, y


def profile_components(profile_dir: str = "/tmp/profile_components", L: int = 16, Dx: int = 14, Dy: int = 16, iters: int = 5):
  os.makedirs(profile_dir, exist_ok=True)

  d_model = 16
  cond_feats_size = 8
  tf = make_timefeatures(out_features=cond_feats_size)
  s5 = make_s5(d_model=d_model, ssm_size=128)
  blk = make_s5block(d_model=d_model, ssm_size=128, cond_size=cond_feats_size)
  stack = make_stack(d_model=d_model, ssm_size=128, num_layers=5, cond_size=cond_feats_size, in_size=Dx, out_size=Dx)
  x, y = make_series(L=L, Dx=Dx, Dy=Dy)

  # Prepare jitted wrappers and compiled executables
  @eqx.filter_jit
  def tf_apply(ts):
    return jax.vmap(tf)(ts)

  @eqx.filter_jit
  def s5_apply(vals):
    return s5(vals)

  @eqx.filter_jit
  def blk_apply(vals, cond):
    return blk(vals, y=cond)

  @eqx.filter_jit
  def stack_apply(vals, cond):
    return stack(vals, y=cond)

  # sample inputs
  vals_stack = x.values                      # (L, Dx) for StackedS5Blocks
  vals_s5 = random.normal(random.PRNGKey(42), (L, d_model))  # (L, H) for S5 and S5Block
  cond_feats = jax.vmap(tf)(y.times)         # (L, cond_feats_size)

  tf_exec = tf_apply.lower(y.times).compile()
  s5_exec = s5_apply.lower(vals_s5).compile()
  blk_exec = blk_apply.lower(vals_s5, cond_feats).compile()
  stack_exec = stack_apply.lower(vals_stack, cond_feats).compile()

  # Precompute iteration inputs
  k = random.PRNGKey(5)
  times_list = []
  vals_s5_list = []
  vals_stack_list = []
  cond_list = []
  for _ in range(iters):
    k, kt, kv_s5, kv_stack = random.split(k, 4)
    t_i = x.times + 1e-6 * random.normal(kt, x.times.shape)
    v_s5_i = vals_s5 + 1e-6 * random.normal(kv_s5, vals_s5.shape)
    v_stack_i = vals_stack + 1e-6 * random.normal(kv_stack, vals_stack.shape)
    c_i = jax.vmap(tf)(t_i)
    times_list.append(t_i)
    vals_s5_list.append(v_s5_i)
    vals_stack_list.append(v_stack_i)
    cond_list.append(c_i)

  # Final precompilation
  tf_exec(times_list[0]).block_until_ready()
  s5_exec(vals_s5_list[0]).block_until_ready()
  blk_exec(vals_s5_list[0], cond_list[0]).block_until_ready()
  stack_exec(vals_stack_list[0], cond_list[0]).block_until_ready()

  stamp = time.strftime("%Y%m%d-%H%M%S")
  out_dir = os.path.join(profile_dir, f"trace-{stamp}")
  with jax.profiler.trace(out_dir, create_perfetto_link=True):
    # TimeFeatures
    with jax.profiler.TraceAnnotation("TimeFeatures_apply"):
      for t in times_list:
        tf_exec(t).block_until_ready()

    # S5 layer alone (expects (L, d_model))
    with jax.profiler.TraceAnnotation("S5_apply"):
      for v in vals_s5_list:
        s5_exec(v).block_until_ready()

    # S5Block with FiLM conditioning (expects (L, d_model) and (L, cond))
    with jax.profiler.TraceAnnotation("S5Block_apply"):
      for v, c in zip(vals_s5_list, cond_list):
        blk_exec(v, c).block_until_ready()

    # StackedS5Blocks with FiLM conditioning (input (L, Dx))
    with jax.profiler.TraceAnnotation("StackedS5Blocks_apply"):
      for v, c in zip(vals_stack_list, cond_list):
        stack_exec(v, c).block_until_ready()

  print(f"Perfetto trace written to: {out_dir}")
  print("Perfetto link printed above by JAX profiler.")


if __name__ == "__main__":
  d = os.environ.get("PROFILE_DIR", "/tmp/profile_components")
  profile_components(profile_dir=d)


