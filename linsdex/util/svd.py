import jax
import jax.numpy as jnp
from functools import partial

__all__ = ["svd"]

def svd(A):
  if A.shape[-1] == A.shape[-2]:
    return my_svd(A)
  else:
    raise NotImplementedError

@jax.custom_jvp
def my_svd(A):
  U, s, VT = jnp.linalg.svd(A)
  V = jnp.einsum("...ji->...ij", VT)
  U = U[...,:s.shape[-1]]
  return U, s, V

@my_svd.defjvp
def my_svd_jvp(primals, tangents):
  A, = primals
  dA, = tangents
  U, s, V = my_svd(A)
  dU, ds, dV = svd_jvp_work(U, s, V, dA)
  return (U, s, V), (dU, ds, dV)

@partial(jnp.vectorize, signature="(n,k),(k),(k,k),(n,k)->(n,k),(k),(k,k)")
def svd_jvp_work(U, s, V, dA):
  dS = jnp.einsum("ij,iu,jv->uv", dA, U, V)
  ds = jnp.diag(dS)

  sdS = jnp.einsum('j,ij->ij', s, dS)
  dSs = jnp.einsum('i,ij->ij', s, dS)

  s_diff = s[:,None]**2 - s**2
  K = s.shape[-1]
  one_over_s_diff = jnp.where(jnp.arange(K)[:,None] == jnp.arange(K), 0.0, 1/s_diff)
  u_components = one_over_s_diff*(sdS + sdS.T)
  v_components = one_over_s_diff*(dSs + dSs.T)

  dU = jnp.einsum("uv,iv->iu", u_components, U)
  dV = jnp.einsum("uv,iv->iu", v_components, V)
  return (dU, ds, dV)

################################################################################################################

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import jax.random as random
  from jax.test_util import check_grads
  jax.config.update('jax_enable_x64', True)

  # Test function that returns a scalar from SVD
  def test_fn(A):
    U, s, V = svd(A)
    return jnp.sum(U) + jnp.sum(s) + jnp.sum(V)

  # Generate random test matrix
  key = random.PRNGKey(0)
  k1, k2 = random.split(key, 2)
  A = random.normal(k1, (4, 2))
  U, s, V = my_svd(A)
  dA = random.normal(k2, A.shape)

  (U, s, V), (dU, ds, dV) = jax.jvp(my_svd, (A,), (dA,))

  # Create 2x2 block matrices from debugger output
  primals = jnp.array([[1497.9131, 0., 461.4255, 0.],
                       [0., 1497.9131, 0., 461.4255],
                       [461.4255, 0., 302.74088, 0.],
                       [0., 461.4255, 0., 302.74088]], dtype=jnp.float32)
  U, s, V = my_svd(primals)


  import pdb; pdb.set_trace()

  # Check gradients using multiple orders of differentiation
  # This will compare analytical gradients against numerical ones
  check_grads(test_fn, (A,), order=2, modes=["fwd", "rev"])

