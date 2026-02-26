import os
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from linsdex import TimeSeries
import jax.tree_util as jtu

from linsdex.nn.nn_models.s5 import TimeDependentS5Seq2SeqModel, TimeDependentS5SeqHypers


def make_model(input_size=14, output_size=14, cond_size=16, key=jax.random.PRNGKey(0)):
  hypers = TimeDependentS5SeqHypers(d_model=16, ssm_size=16, num_layers=2, time_feature_size=8, cond_size=cond_size)
  return TimeDependentS5Seq2SeqModel(input_size=input_size, output_size=output_size, hypers=hypers, key=key)


def make_data(L=16, Dx=14, Dy=16, key=jax.random.PRNGKey(1)):
  kx, ky, kt = random.split(key, 3)
  times = jnp.linspace(0.0, 1.0, L) + 1e-6 * random.normal(kt, (L,))
  series = TimeSeries(times=times, values=random.normal(kx, (L, Dx)))
  cond = TimeSeries(times=times, values=random.normal(ky, (L, Dy)))
  return series, cond


@eqx.filter_jit
def run_no_context(model, s, series, cond):
  return model(s, series, condition_info=cond)


def print_treedef(tag, obj):
  treedef = jtu.tree_structure(obj)
  print(f"[{tag}] treedef: {treedef}")


def print_static_leaves(tag, obj):
  arrays, static = eqx.partition(obj, eqx.is_array)
  static_leaves = jtu.tree_leaves(static, is_leaf=lambda x: not eqx.is_array(x))
  print(f"[{tag}] #static_leaves: {len(static_leaves)} types: {[type(x) for x in static_leaves]}")


def main():
  os.environ.setdefault("JAX_LOG_COMPILES", "1")
  key = random.PRNGKey(0)
  model = make_model(key=key)
  series, cond = make_data(key=key)
  s = jnp.array(0.5)

  # Show signatures
  print_treedef("args", (model, s, series, cond))
  print_static_leaves("args", (model, s, series, cond))

  # Lower/compile once
  lowered = run_no_context.lower(model, s, series, cond)
  print("lowered avals:", lowered.as_text())
  compiled = lowered.compile()
  _ = compiled(model, s, series, cond).block_until_ready()

  # Now perturb inputs minimally across iterations to see if recompiles happen
  k = random.PRNGKey(42)
  for i in range(3):
    k, kx, ky, kt, ks = random.split(k, 5)
    s_i = s + (0.0 if i == 0 else 1e-6 * random.normal(ks, ()))
    series_i = TimeSeries(
      times=series.times + 1e-6 * random.normal(kt, series.times.shape),
      values=series.values + 1e-6 * random.normal(kx, series.values.shape),
    )
    cond_i = TimeSeries(
      times=cond.times + 1e-6 * random.normal(kt, cond.times.shape),
      values=cond.values + 1e-6 * random.normal(ky, cond.values.shape),
    )

    print_treedef(f"iter {i}", (model, s_i, series_i, cond_i))
    print_static_leaves(f"iter {i}", (model, s_i, series_i, cond_i))
    _ = compiled(model, s_i, series_i, cond_i).block_until_ready()


if __name__ == "__main__":
  main()


