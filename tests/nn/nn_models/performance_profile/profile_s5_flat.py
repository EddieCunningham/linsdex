import os
import time
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
from linsdex import TimeSeries

from linsdex.nn.nn_models.s5 import TimeDependentS5Seq2SeqModel, TimeDependentS5SeqHypers


def make_model(
  *,
  input_size: int = 14,
  output_size: int = 14,
  cond_size: Optional[int] = 16,
  d_model: int = 16,
  ssm_size: int = 16,
  num_layers: int = 2,
  time_feature_size: int = 8,
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
  times = jnp.linspace(0.0, 1.0, L)
  series = TimeSeries(times=times, values=random.normal(kx, (L, Dx)))
  cond = TimeSeries(times=times, values=random.normal(ky, (L, Dy)))
  return series, cond


def flatten_model(model: TimeDependentS5Seq2SeqModel) -> Tuple[Tuple[jax.Array, ...], object, object]:
  arrays, static = eqx.partition(model, eqx.is_array)
  flat_arrays, treedef = jtu.tree_flatten(arrays)
  return tuple(flat_arrays), static, treedef


@eqx.filter_jit
def build_context_flat(
  flat_model: Tuple[jax.Array, ...],
  *,
  static_model: object,
  treedef_arrays: object,
  cond_times: jax.Array,
  cond_values: jax.Array,
):
  arrays_tree = jtu.tree_unflatten(treedef_arrays, list(flat_model))
  model = eqx.combine(arrays_tree, static_model)
  cond = TimeSeries(times=cond_times, values=cond_values)
  return model.create_context(cond)


@eqx.filter_jit
def run_no_context_flat(
  flat_model: Tuple[jax.Array, ...],
  *,
  static_model: object,
  treedef_arrays: object,
  s: jax.Array,
  series_times: jax.Array,
  series_values: jax.Array,
  cond_times: jax.Array,
  cond_values: jax.Array,
):
  arrays_tree = jtu.tree_unflatten(treedef_arrays, list(flat_model))
  model = eqx.combine(arrays_tree, static_model)
  series = TimeSeries(times=series_times, values=series_values)
  cond = TimeSeries(times=cond_times, values=cond_values)
  return model(s, series, condition_info=cond)


@eqx.filter_jit
def run_with_context_flat(
  flat_model: Tuple[jax.Array, ...],
  *,
  static_model: object,
  treedef_arrays: object,
  s: jax.Array,
  series_times: jax.Array,
  series_values: jax.Array,
  context: jax.Array,
):
  arrays_tree = jtu.tree_unflatten(treedef_arrays, list(flat_model))
  model = eqx.combine(arrays_tree, static_model)
  series = TimeSeries(times=series_times, values=series_values)
  return model(s, series, context=context)


def profile_flat(profile_dir: str = "/tmp/profile_s5_flat", L: int = 16, Dx: int = 14, Dy: int = 16, iters: int = 20):
  os.makedirs(profile_dir, exist_ok=True)
  key = random.PRNGKey(0)
  model = make_model(input_size=Dx, output_size=Dx, cond_size=Dy, key=key)
  series, cond = make_data(L, Dx, Dy, key)
  s = jnp.array(0.5)

  # Flatten once
  flat_model, static_model, treedef_arrays = flatten_model(model)

  # Warm-up compile for the flat functions
  _ = build_context_flat(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                         cond_times=cond.times, cond_values=cond.values).block_until_ready()
  ctx0 = build_context_flat(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                            cond_times=cond.times, cond_values=cond.values)
  _ = run_with_context_flat(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                            s=s, series_times=series.times, series_values=series.values, context=ctx0).block_until_ready()
  _ = run_no_context_flat(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                          s=s, series_times=series.times, series_values=series.values,
                          cond_times=cond.times, cond_values=cond.values).block_until_ready()

  # Precompute varied inputs to avoid cache-only paths and keep them outside trace
  # Context builds
  k = random.PRNGKey(7)
  cond_list = []
  for _ in range(iters):
    k, kt, kv = random.split(k, 3)
    cond_list.append((cond.times + 1e-6 * random.normal(kt, cond.times.shape),
                      cond.values + 1e-6 * random.normal(kv, cond.values.shape)))

  # No-context inputs: (s_i, series_times_i, series_values_i, cond_times_i, cond_values_i)
  k = random.PRNGKey(8)
  noctx_items = []
  for _ in range(iters):
    k, ks, kt, kx, ky = random.split(k, 5)
    s_i = s + 1e-6 * random.normal(ks, ())
    noctx_items.append((
      s_i,
      series.times + 1e-6 * random.normal(kt, series.times.shape),
      series.values + 1e-6 * random.normal(kx, series.values.shape),
      cond.times + 1e-6 * random.normal(kt, cond.times.shape),
      cond.values + 1e-6 * random.normal(ky, cond.values.shape),
    ))

  # With-context inputs: (s_i, series_times_i, series_values_i, context_i)
  ctx_list = []
  for ct, cv in cond_list:
    ctx_i = build_context_flat(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                               cond_times=ct, cond_values=cv)
    ctx_list.append(ctx_i)
  k = random.PRNGKey(9)
  wctx_items = []
  for i in range(iters):
    k, ks, kt, kx = random.split(k, 4)
    s_i = s + 1e-6 * random.normal(ks, ())
    wctx_items.append((
      s_i,
      series.times + 1e-6 * random.normal(kt, series.times.shape),
      series.values + 1e-6 * random.normal(kx, series.values.shape),
      ctx_list[i],
    ))

  # Precompile executables via lower/compile, then call compiled inside trace
  build_ctx_exec = build_context_flat.lower(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                                            cond_times=cond.times, cond_values=cond.values).compile()
  no_ctx_exec = run_no_context_flat.lower(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                                          s=s, series_times=series.times, series_values=series.values,
                                          cond_times=cond.times, cond_values=cond.values).compile()
  w_ctx_exec = run_with_context_flat.lower(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                                           s=s, series_times=series.times, series_values=series.values,
                                           context=ctx0).compile()

  # Perfetto trace with link
  stamp = time.strftime("%Y%m%d-%H%M%S")
  out_dir = os.path.join(profile_dir, f"trace-{stamp}")
  with jax.profiler.trace(out_dir, create_perfetto_link=True):
    # Context creation (flat)
    with jax.profiler.TraceAnnotation("s5_create_context_flat"):
      tmp = None
      for ct, cv in cond_list:
        tmp = build_ctx_exec(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                             cond_times=ct, cond_values=cv)
      tmp.block_until_ready()

    # No-context forward (flat)
    with jax.profiler.TraceAnnotation("s5_no_context_forward_flat"):
      y = None
      for s_i, st_i, sv_i, ct_i, cv_i in noctx_items:
        y = no_ctx_exec(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                        s=s_i, series_times=st_i, series_values=sv_i,
                        cond_times=ct_i, cond_values=cv_i)
      y.block_until_ready()

    # With-context forward (flat)
    with jax.profiler.TraceAnnotation("s5_with_context_forward_flat"):
      y2 = None
      for s_i, st_i, sv_i, ctx_i in wctx_items:
        y2 = w_ctx_exec(flat_model, static_model=static_model, treedef_arrays=treedef_arrays,
                        s=s_i, series_times=st_i, series_values=sv_i, context=ctx_i)
      y2.block_until_ready()

  print(f"Perfetto trace written to: {out_dir}")
  print("Perfetto link printed above by JAX profiler.")


if __name__ == "__main__":
  profile_dir = os.environ.get("PROFILE_DIR", "/tmp/profile_s5_flat")
  profile_flat(profile_dir=profile_dir)


