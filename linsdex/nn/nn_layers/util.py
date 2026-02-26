import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterator
import jax.lax as lax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, PyTree
import einops
import jax.tree_util as jtu
from linsdex import TimeSeries

__all__ = ['negate_module_gradients',
           'broadcast_to_first_axis',
           'last_axes',
           'get_reduce_axes',
           'index_list',
           'tree_shapes',
           'tree_concat',
           'tree_hstack',
           'tree_array',
           'square_plus',
           'square_sigmoid',
           'square_swish',
           'only_gradient',
           'mean_and_std',
           'mean_and_inverse_std',
           'list_prod',
           'whiten',
           'extract_multiple_batches_from_iterator',
           'ensure_path_exists',
           'conv',
           'unbatch',
           'count_params',
           'svd']

def apply_mask_to_series(series: TimeSeries, mask_length: int) -> TimeSeries:
  """Apply a mask to the series"""
  mask = series.mask.at[mask_length:].set(False)
  values = series.values*mask[:,None]
  return TimeSeries(series.times, values, mask=mask)

################################################################################################################

def negate(pytree):
  return jtu.tree_map(lambda x: -x, pytree)

@eqx.filter_custom_vjp
def negate_module_gradients(module):
  return module

@negate_module_gradients.def_fwd
def fn_fwd(perturbed, module):
  return module, ()

@negate_module_gradients.def_bwd
def fn_bwd(residuals, grad_obj, perturbed, module):
  del residuals, perturbed, module
  return negate(grad_obj)


################################################################################################################

def count_params(module):
  params, static = eqx.partition(module, eqx.is_array)
  flat_params, treedef = jax.tree_util.tree_flatten(params)
  n_params = sum([p.size for p in flat_params])
  return n_params

def unbatch(pytree):
  return jax.tree_util.tree_map(lambda x: x[0], pytree)

def broadcast_to_first_axis(x, ndim):
  if x.ndim == 0:
    return x
  return jnp.expand_dims(x, axis=tuple(range(1, ndim)))

def last_axes(shape):
  return tuple(range(-1, -1 - len(shape), -1))

def get_reduce_axes(axes, ndim, offset=0):
  if isinstance(axes, int):
    axes = (axes,)
  keep_axes = [ax%ndim for ax in axes]
  reduce_axes = tuple([ax + offset for ax in range(ndim) if ax not in keep_axes])
  return reduce_axes

def index_list(shape, axis):
  ndim = len(shape)
  axis = [ax%ndim for ax in axis]
  shapes = [s for i, s in enumerate(shape) if i in axis]
  return tuple(shapes)

################################################################################################################

def tree_shapes(pytree):
  return jax.tree_util.tree_map(lambda x: x.shape, pytree)

def tree_concat(x, y, axis=0):
  if x is None:
    return y
  return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=axis), x, y)

def tree_hstack(x, y):
  if x is None:
    return jax.tree_util.tree_map(lambda x: x[None], y)
  return jax.tree_util.tree_map(lambda a, b: jnp.hstack([a, b]), x, y)

def tree_array(inputs):
  return jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *inputs)

################################################################################################################

def square_plus(x, gamma=0.5):
  # https://arxiv.org/pdf/1901.08431.pdf
  out = 0.5*(x + jnp.sqrt(x**2 + 4*gamma))
  out = jnp.maximum(out, 0.0)
  return out

def square_sigmoid(x, gamma=0.5):
  # Derivative of proximal relu.  Basically sigmoid without saturated gradients.
  return 0.5*(1 + x*jax.lax.rsqrt(x**2 + 4*gamma))

def square_swish(x, gamma=0.5):
  x2 = x**2
  out = 0.5*(x + x2*jax.lax.rsqrt(x2 + 4*gamma))
  return out

def only_gradient(x):
  return x - jax.lax.stop_gradient(x)

def mean_and_std(x, axis=-1, keepdims=False):
  mean = jnp.mean(x, axis=axis, keepdims=keepdims)
  std = jnp.std(x, axis=axis, keepdims=keepdims)
  return mean, std

def mean_and_inverse_std(x, axis=-1, keepdims=False):
  mean = jnp.mean(x, axis=axis, keepdims=keepdims)
  mean_sq = jnp.mean(lax.square(x), axis=axis, keepdims=keepdims)
  var = mean_sq - lax.square(mean)
  inv_std = lax.rsqrt(var + 1e-6)
  return mean, inv_std

def list_prod(x):
  # We might run into JAX tracer issues if we do something like multiply the elements of a shape tuple with jnp
  return np.prod(x)

def whiten(x):
  U, s, VT = jnp.linalg.svd(x, full_matrices=False)
  return jnp.dot(U, VT)

def extract_multiple_batches_from_iterator(it: Iterator,
                                           n_batches: int,
                                           single_batch=False):
  data = [None for _ in range(n_batches)]
  for i in range(n_batches):
    data[i] = next(it)
  out = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *data)
  if single_batch:
    out = jax.tree_util.tree_map(lambda x: einops.rearrange(x, 'n b ... -> (n b) ...'), out)
  return out


import pickle
from pathlib import Path
from typing import Union

def ensure_path_exists(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def conv(w,
         x,
         stride=1,
         padding='SAME'):
  no_batch = False
  if x.ndim == 3:
    no_batch = True
    x = x[None]

  if isinstance(padding, int):
    padding = ((padding, padding), (padding, padding))

  out = jax.lax.conv_general_dilated(x,
                                     w,
                                     window_strides=(stride, stride),
                                     padding=padding,
                                     lhs_dilation=(1, 1),
                                     rhs_dilation=(1, 1),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
  if no_batch:
    out = out[0]
  return out


def svd(A):
  if A.shape[-1] == A.shape[-2]:
    return my_svd(A)
  return jnp.linalg.svd(A)

@jax.custom_jvp
def my_svd(A):
  U, s, VT = jnp.linalg.svd(A)
  V = jnp.einsum("...ji->...ij", VT)
  return U, s, V

@my_svd.defjvp
def my_svd_jvp(primals, tangents):
  A, = primals
  dA, = tangents
  U, s, V = my_svd(A)
  dU, ds, dV = svd_jvp_work(U, s, V, dA)
  return (U, s, V), (dU, ds, dV)

@partial(jnp.vectorize, signature="(n,n),(n),(n,n),(n,n)->(n,n),(n),(n,n)")
def svd_jvp_work(U, s, V, dA):
  dS = jnp.einsum("ij,iu,jv->uv", dA, U, V)
  ds = jnp.diag(dS)

  sdS = s*dS
  dSs = s[:,None]*dS

  s_diff = s[:,None]**2 - s**2 + 1e-5
  N = s.shape[-1]
  one_over_s_diff = jnp.where(jnp.arange(N)[:,None] == jnp.arange(N), 0.0, 1/s_diff)
  u_components = one_over_s_diff*(sdS + sdS.T)
  v_components = one_over_s_diff*(dSs + dSs.T)

  dU = jnp.einsum("uv,iv->iu", u_components, U)
  dV = jnp.einsum("uv,iv->iu", v_components, V)
  return (dU, ds, dV)
