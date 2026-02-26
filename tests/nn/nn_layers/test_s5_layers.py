import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

import linsdex.nn.nn_layers.s5_layers as s5l
from linsdex.nn.nn_layers.s5_layers import (
  make_HiPPO,
  make_NPLR_HiPPO,
  make_DPLR_HiPPO,
  log_step_initializer,
  init_log_steps,
  discretize_zoh,
  apply_ssm,
  init_CV,
  init_VinvB,
)


def test_make_dplr_hippo_shapes():
  P = 8
  Lambda, P_lr, B, V, B_orig = make_DPLR_HiPPO(P)
  assert Lambda.shape == (P,)
  assert P_lr.shape == (P,)
  assert B.shape == (P,)
  assert V.shape == (P, P)
  assert B_orig.shape == (P,)


def test_make_hippo_and_nplr_shapes():
  N = 6
  A = make_HiPPO(N)
  assert A.shape == (N, N)
  A2, P_lr, B = make_NPLR_HiPPO(N)
  assert A2.shape == (N, N)
  assert P_lr.shape == (N,)
  assert B.shape == (N,)


def test_discretize_zoh_and_apply_ssm_shapes():
  # dimensions
  L = 5
  H = 4
  P = 6

  # random inputs
  key = random.PRNGKey(0)
  u = random.normal(key, (L, H))

  # simple diagonal dynamics
  key, k1, k2 = random.split(key, 3)
  Lambda = -jnp.abs(random.normal(k1, (P,))) + 1j * random.normal(k1, (P,))
  B_tilde = random.normal(k2, (P, H)) + 1j * random.normal(k2, (P, H))
  Delta = jnp.ones((P,)) * 0.05

  Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, Delta)
  assert Lambda_bar.shape == (P,)
  assert B_bar.shape == (P, H)

  # C_tilde mapping to outputs
  key, k3 = random.split(key)
  C_real = random.normal(k3, (H, P))
  C_imag = random.normal(k3, (H, P))
  C_tilde = C_real + 1j * C_imag

  y = apply_ssm(Lambda_bar, B_bar, C_tilde, u, conj_sym=False, bidirectional=False)
  assert y.shape == (L, H)
  assert jnp.isfinite(y).all()


def test_init_vinvb_and_init_cv_shapes():
  H = 4
  P = 6
  key = random.PRNGKey(1)

  # random unitary-like V and its inverse for shape tests
  # use identity to keep test simple and deterministic
  V = jnp.eye(P) + 0j
  Vinv = jnp.eye(P) + 0j

  # init_VinvB with identity Vinv should yield purely real-imag split of real B
  B_pair = init_VinvB(jax.nn.initializers.lecun_normal(), key, (P, H), Vinv)
  assert B_pair.shape == (P, H, 2)
  # Imaginary part should be finite (zero-mean random but not necessarily zero)
  assert jnp.isfinite(B_pair).all()

  # Deterministic init_fun for init_CV to verify CV = C @ V
  C_real = jnp.arange(H * P, dtype=jnp.float32).reshape(H, P)
  C_imag = jnp.ones((H, P), dtype=jnp.float32)
  C_const = jnp.stack([C_real, C_imag], axis=-1)

  def const_init(_rng, shape):
    assert shape == (H, P, 2)
    return C_const

  C_pair = init_CV(const_init, key, (H, P, 2), V)
  assert C_pair.shape == (H, P, 2)
  C_complex = C_pair[..., 0] + 1j * C_pair[..., 1]
  assert jnp.allclose(C_complex, (C_real + 1j * C_imag) @ V)


def test_log_step_initializer_and_init_log_steps():
  key = random.PRNGKey(0)
  dt_min = 1e-3
  dt_max = 1e-1
  init_fn = log_step_initializer(dt_min, dt_max)
  sample = init_fn(key, (1,))
  assert sample.shape == (1,)
  assert jnp.all(sample >= jnp.log(dt_min)) and jnp.all(sample <= jnp.log(dt_max))

  H = 5
  logs = init_log_steps(key, (H, dt_min, dt_max))
  assert logs.shape == (H, 1)
  assert jnp.all(logs >= jnp.log(dt_min)) and jnp.all(logs <= jnp.log(dt_max))


def test_discretize_zoh_stability():
  # For negative real continuous-time poles, ZOH should give |Lambda_bar| < 1
  P = 4
  a = -jnp.linspace(0.1, 1.0, P)
  Lambda = a + 0j
  H = 3
  B_tilde = jnp.ones((P, H)) + 0j
  Delta = jnp.ones((P,)) * 0.1
  Lambda_bar, _ = discretize_zoh(Lambda, B_tilde, Delta)
  assert jnp.all(jnp.abs(Lambda_bar) < 1.0)


def test_discretize_bilinear_shapes_and_stability():
  # Attach numpy to module to support discretize_bilinear
  s5l.np = np
  P = 5
  H = 3
  key = random.PRNGKey(3)
  Lambda = -jnp.linspace(0.2, 1.0, P) + 0j
  B_tilde = random.normal(key, (P, H)) + 1j * random.normal(key, (P, H))
  Delta = jnp.ones((P,)) * 0.05
  Lambda_bar, B_bar = s5l.discretize_bilinear(Lambda, B_tilde, Delta)
  assert Lambda_bar.shape == (P,)
  assert B_bar.shape == (P, H)
  # Bilinear transform preserves stability: |Lambda_bar| < 1
  assert jnp.all(jnp.abs(Lambda_bar) < 1.0)


def test_binary_operator_simple():
  P = 4
  A_i = jnp.ones((P,)) * 0.5
  A_j = jnp.ones((P,)) * 0.2
  b_i = jnp.ones((P,))
  b_j = jnp.ones((P,)) * 2.0
  A_out, b_out = s5l.binary_operator((A_i, b_i), (A_j, b_j))
  assert jnp.allclose(A_out, 0.1)
  assert jnp.allclose(b_out, 2.2)


def test_apply_ssm_bidirectional_and_conj_sym():
  L, H, P = 6, 3, 4
  key = random.PRNGKey(4)
  u = random.normal(key, (L, H))
  Lambda = -jnp.linspace(0.2, 0.8, P) + 1j * jnp.linspace(0.0, 0.5, P)
  B_tilde = random.normal(key, (P, H)) + 1j * random.normal(key, (P, H))
  Delta = jnp.ones((P,)) * 0.05
  Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, Delta)

  # Non-bidirectional, conj_sym=False
  C_real = random.normal(key, (H, P))
  C_imag = random.normal(key, (H, P))
  C_tilde = C_real + 1j * C_imag
  y = apply_ssm(Lambda_bar, B_bar, C_tilde, u, conj_sym=False, bidirectional=False)
  assert y.shape == (L, H)

  # Bidirectional requires C_tilde with 2P columns
  C2_real = random.normal(key, (H, P))
  C2_imag = random.normal(key, (H, P))
  C_bidir = jnp.concatenate([(C_real + 1j * C_imag), (C2_real + 1j * C2_imag)], axis=-1)
  y2 = apply_ssm(Lambda_bar, B_bar, C_bidir, u, conj_sym=False, bidirectional=True)
  assert y2.shape == (L, H)

  # Conjugate symmetry path
  y3 = apply_ssm(Lambda_bar, B_bar, C_tilde, u, conj_sym=True, bidirectional=False)
  assert y3.shape == (L, H)


