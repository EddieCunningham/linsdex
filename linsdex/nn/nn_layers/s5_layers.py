from functools import partial
from typing import Literal, Optional, Union, Tuple, Callable, List, Any
import einops
import equinox as eqx
import jax.random as random
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Bool
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh
# Adapted from https://github.com/lindermanlab/S5/tree/main

# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar

def make_HiPPO(N):
  """ Create a HiPPO-LegS matrix.
      From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
      Args:
          N (int32): state size
      Returns:
          N x N HiPPO LegS matrix
  """
  P = jnp.sqrt(1 + 2 * jnp.arange(N))
  A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
  A = jnp.tril(A) - jnp.diag(jnp.arange(N))
  return -A


def make_NPLR_HiPPO(N):
  """
  Makes components needed for NPLR representation of HiPPO-LegS
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
  Args:
    N (int32): state size

  Returns:
    N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

  """
  # Make -HiPPO
  hippo = make_HiPPO(N)

  # Add in a rank 1 term. Makes it Normal.
  P = jnp.sqrt(jnp.arange(N) + 0.5)

  # HiPPO also specifies the B matrix
  B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
  return hippo, P, B

def log_step_initializer(dt_min=0.001, dt_max=0.1):
  """ Initialize the learnable timescale Delta by sampling
      uniformly between dt_min and dt_max.
      Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
      Returns:
        init function
   """
  def init(key, shape):
    """ Init function
        Args:
          key: jax random key
          shape tuple: desired shape
        Returns:
          sampled log_step (float32)
     """
    return random.uniform(key, shape) * (
        jnp.log(dt_max) - jnp.log(dt_min)
    ) + jnp.log(dt_min)

  return init

def init_log_steps(key, input):
  """ Initialize an array of learnable timescale parameters
      Args:
        key: jax random key
        input: tuple containing the array shape H and
              dt_min and dt_max
      Returns:
        initialized array of timescales (float32): (H,)
   """
  H, dt_min, dt_max = input
  log_steps = []
  for i in range(H):
    key, skey = random.split(key)
    log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
    log_steps.append(log_step)

  return jnp.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
  """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
      Args:
        init_fun:  the initialization function to use, e.g. lecun_normal()
        rng:       jax random key to be used with init function.
        shape (tuple): desired shape  (P,H)
        Vinv: (complex64)     the inverse eigenvectors used for initialization
      Returns:
        B_tilde (complex64) of shape (P,H,2)
   """
  B = init_fun(rng, shape)
  VinvB = Vinv @ B
  VinvB_real = VinvB.real
  VinvB_imag = VinvB.imag
  return jnp.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
  """ Sample C with a truncated normal distribution with standard deviation 1.
      Args:
        key: jax random key
        shape (tuple): desired shape, of length 3, (H,P,_)
      Returns:
        sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
   """
  H, P, _ = shape
  Cs = []
  for i in range(H):
    key, skey = random.split(key)
    C = lecun_normal()(skey, shape=(1, P, 2))
    Cs.append(C)
  return jnp.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
  """ Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
    Args:
      init_fun:  the initialization function to use, e.g. lecun_normal()
      rng:       jax random key to be used with init function.
      shape (tuple): desired shape  (H,P)
      V: (complex64)     the eigenvectors used for initialization
    Returns:
      C_tilde (complex64) of shape (H,P,2)
   """
  C_ = init_fun(rng, shape)
  C = C_[..., 0] + 1j * C_[..., 1]
  CV = C @ V
  CV_real = CV.real
  CV_imag = CV.imag
  return jnp.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


def make_DPLR_HiPPO(N):
  """
  Makes components needed for DPLR representation of HiPPO-LegS
   From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
  Note, we will only use the diagonal part
  Args:
    N:

  Returns:
    eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
    eigenvectors V, HiPPO B pre-conjugation

  """
  A, P, B = make_NPLR_HiPPO(N)

  S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

  S_diag = jnp.diagonal(S)
  Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

  # Diagonalize S to V \Lambda V^*
  Lambda_imag, V = eigh(S * -1j)

  P = V.conj().T @ P
  B_orig = B
  B = V.conj().T @ B
  return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig

def discretize_zoh(Lambda, B_tilde, Delta):
  """ Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
      Lambda (complex64): diagonal state matrix              (P,)
      B_tilde (complex64): input matrix                      (P, H)
      Delta (float32): discretization step sizes             (P,)
    Returns:
      discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
  """
  Identity = jnp.ones(Lambda.shape[0])
  Lambda_bar = jnp.exp(Lambda * Delta)
  B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
  return Lambda_bar, B_bar

# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
  """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
      q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
      q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
      new element ( A_out, Bu_out )
  """
  A_i, b_i = q_i
  A_j, b_j = q_j
  return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
  """ Compute the LxH output of discretized SSM given an LxH input.
    Args:
      Lambda_bar (complex64): discretized diagonal state matrix    (P,)
      B_bar      (complex64): discretized input matrix             (P, H)
      C_tilde    (complex64): output matrix                        (H, P)
      input_sequence (float32): input sequence of features         (L, H)
      conj_sym (bool):         whether conjugate symmetry is enforced
      bidirectional (bool):    whether bidirectional setup is used,
                            Note for this case C_tilde will have 2P cols
    Returns:
      ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
  """
  Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0],
                                          Lambda_bar.shape[0]))
  Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

  _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

  if bidirectional:
    _, xs2 = jax.lax.associative_scan(binary_operator,
                                      (Lambda_elements, Bu_elements),
                                      reverse=True)
    xs = jnp.concatenate((xs, xs2), axis=-1)

  if conj_sym:
    return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
  else:
    return jax.vmap(lambda x: (C_tilde @ x).real)(xs)

