import os
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from linsdex import TimeSeries
from linsdex.nn.nn_models.s5 import TimeDependentS5Seq2SeqModel, TimeDependentS5SeqHypers

from debug import *

def make_model(
  *,
  input_size: int = 14,
  output_size: int = 14,
  cond_size: Optional[int] = 16,
  d_model: int = 16,
  ssm_size: int = 16,
  num_layers: int = 4,
  time_feature_size: int = 16,
  key: jax.Array,
) -> TimeDependentS5Seq2SeqModel:
  hypers = TimeDependentS5SeqHypers(
    d_model=d_model,
    ssm_size=ssm_size,
    blocks=2,
    num_layers=num_layers,
    time_feature_size=time_feature_size,
    cond_size=cond_size,
  )
  return TimeDependentS5Seq2SeqModel(
    input_size=input_size,
    output_size=output_size,
    hypers=hypers,
    key=key,
  )


def make_data(L: int, Dx: int, Dy: int, key: jax.Array):
  kx, ky, kt = random.split(key, 3)
  # Use slightly different times each run to avoid exact cache hits
  base_times = jnp.linspace(0.0, 1.0, L)
  jitter = 1e-4 * random.normal(kt, (L,))
  times = base_times + jitter
  series = TimeSeries(times=times, values=random.normal(kx, (L, Dx)))
  cond = TimeSeries(times=times, values=random.normal(ky, (L, Dy)))
  return series, cond


@eqx.filter_jit
def run_no_context(model: TimeDependentS5Seq2SeqModel, s: jax.Array, x: TimeSeries, y: TimeSeries):
  return model(s, x, condition_info=y)


@eqx.filter_jit
def run_with_context(model: TimeDependentS5Seq2SeqModel, s: jax.Array, x: TimeSeries, ctx: jax.Array):
  return model(s, x, context=ctx)


@eqx.filter_jit
def build_context(model: TimeDependentS5Seq2SeqModel, y: TimeSeries):
  return model.create_context(y)





def profile(profile_dir: str = "/tmp/profile_s5", L: int = 16, Dx: int = 14, Dy: int = 16, iters: int = 20):
  os.makedirs(profile_dir, exist_ok=True)
  key = random.PRNGKey(0)
  model = make_model(input_size=Dx, output_size=Dx, cond_size=Dy, key=key)
  series, cond = make_data(L, Dx, Dy, key)
  s = jnp.array(0.5)

  # Warm-up/compile both paths and context creation
  _ = run_no_context(model, s, series, cond).block_until_ready()
  ctx = build_context(model, cond)
  _ = ctx.block_until_ready()
  _ = run_with_context(model, s, series, ctx).block_until_ready()
  # No additional helpers

  # Precompute jittered inputs for iterations (to exclude from traced regions)
  # 1) For create_context: list of jittered cond TimeSeries
  k = random.PRNGKey(111)
  L = cond.values.shape[0]
  cond_list = []
  for _ in range(iters):
    k, k_t, k_v = random.split(k, 3)
    cond_list.append(
      TimeSeries(
        times=cond.times + 1e-6 * random.normal(k_t, (L,)),
        values=cond.values + 1e-6 * random.normal(k_v, cond.values.shape),
      )
    )

  # 2) For no_context: tuples of (s_i, series_i, cond_i)
  k = random.PRNGKey(222)
  noctx_items = []
  for _ in range(iters):
    k, k_t, k_x, k_y, k_s = random.split(k, 5)
    s_i = s + 1e-6 * random.normal(k_s, ())
    series_i = TimeSeries(
      times=series.times + 1e-6 * random.normal(k_t, series.times.shape),
      values=series.values + 1e-6 * random.normal(k_x, series.values.shape),
    )
    cond_i = TimeSeries(
      times=cond.times + 1e-6 * random.normal(k_t, cond.times.shape),
      values=cond.values + 1e-6 * random.normal(k_y, cond.values.shape),
    )
    noctx_items.append((s_i, series_i, cond_i))

  # 3) For with_context: tuples of (s_i, series_i) (ctx fixed)
  k = random.PRNGKey(333)
  wctx_items = []
  for _ in range(iters):
    k, k_t, k_x, k_s = random.split(k, 4)
    s_i = s + 1e-6 * random.normal(k_s, ())
    series_i = TimeSeries(
      times=series.times + 1e-6 * random.normal(k_t, series.times.shape),
      values=series.values + 1e-6 * random.normal(k_x, series.values.shape),
    )
    wctx_items.append((s_i, series_i))

  # 4) Placeholder to keep structure similar (not used)

  # Precompile executables to avoid compile bands inside trace regions
  ctx_exec = build_context.lower(model, cond).compile()
  no_ctx_exec = run_no_context.lower(model, s, series, cond).compile()
  w_ctx_exec = run_with_context.lower(model, s, series, ctx).compile()


  # Start Perfetto trace (always create Perfetto link)
  stamp = time.strftime("%Y%m%d-%H%M%S")
  out_dir = os.path.join(profile_dir, f"trace-{stamp}")
  with jax.profiler.trace(out_dir, create_perfetto_link=True):
    # Context creation cost
    with jax.profiler.TraceAnnotation("s5_create_context"):
      tmp = None
      for cond_i in cond_list:
        tmp = ctx_exec(model, cond_i)
      # Ensure compute completes
      tmp.block_until_ready()

    # No-context path (encoder executed inside the jitted call)
    with jax.profiler.TraceAnnotation("s5_no_context_forward"):
      y = None
      for s_i, series_i, cond_i in noctx_items:
        y = no_ctx_exec(model, s_i, series_i, cond_i)
      y.block_until_ready()

    # Precomputed-context path (encoder done once outside the jitted call)
    with jax.profiler.TraceAnnotation("s5_with_context_forward"):
      y2 = None
      for s_i, series_i in wctx_items:
        y2 = w_ctx_exec(model, s_i, series_i, ctx)
      y2.block_until_ready()

    # Removed: precompute decoder features + fast decode (had no measurable impact)

  print(f"Perfetto trace written to: {out_dir}")
  print("Perfetto link printed above by JAX profiler.")


if __name__ == "__main__":
  # You can override via: PROFILE_DIR=/path python -m tests.models.nn_models.profile_s5
  profile_dir = os.environ.get("PROFILE_DIR", "/tmp/profile_s5")
  profile(profile_dir=profile_dir)


