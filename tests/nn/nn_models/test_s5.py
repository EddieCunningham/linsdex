import jax
import jax.numpy as jnp
import jax.random as random

from linsdex.nn.nn_models.s5 import S5Args, S5


def test_s5_initialization_shapes():
  key = random.PRNGKey(0)
  args = S5Args(
    d_model=8,
    ssm_size=8,
    blocks=2,
    C_init="lecun_normal",
    discretization="zoh",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=False,
    clip_eigs=False,
    bidirectional=False,
    step_rescale=1.0,
  )

  model = S5(args, key=key)

  # Check parameter shapes
  P = args.ssm_size
  H = args.d_model
  assert model.B.shape == (P, H, 2)
  assert (model.C.shape[0] == H) and (model.C.shape[-1] == 2)
  assert model.D.shape == (H,)
  assert model.Lambda_re.shape == (P,)
  assert model.Lambda_im.shape == (P,)
  assert model.log_step.shape == (P, 1)


def test_s5_bidirectional_and_conj_sym():
  key = random.PRNGKey(11)
  args = S5Args(
    d_model=6,
    ssm_size=6,
    blocks=3,
    C_init="lecun_normal",
    discretization="zoh",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=False,
    clip_eigs=False,
    bidirectional=True,
    step_rescale=1.0,
  )
  model = S5(args, key=key)
  x = random.normal(key, (4, args.d_model))
  y = model(x)
  assert y.shape == (4, args.d_model)

  # Conjugate symmetry True reduces P internally; ensure callable
  args_cs = S5Args(
    d_model=6,
    ssm_size=8,
    blocks=2,
    C_init="lecun_normal",
    discretization="zoh",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=True,
    clip_eigs=True,
    bidirectional=False,
    step_rescale=1.0,
  )
  model_cs = S5(args_cs, key=key)
  y_cs = model_cs(x)
  assert y_cs.shape == (4, args_cs.d_model)


def test_s5_forward_pass():
  key = random.PRNGKey(1)
  args = S5Args(
    d_model=6,
    ssm_size=6,
    blocks=3,
    C_init="lecun_normal",
    discretization="zoh",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=False,
    clip_eigs=True,
    bidirectional=False,
    step_rescale=1.0,
  )

  model = S5(args, key=key)

  L = 7
  x = random.normal(key, (L, args.d_model))
  y = model(x)

  assert y.shape == (L, args.d_model)
  assert jnp.isfinite(y).all()


def test_s5_discretization_invalid_raises():
  key = random.PRNGKey(3)
  args = S5Args(
    d_model=4,
    ssm_size=4,
    blocks=2,
    C_init="lecun_normal",
    discretization="invalid",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=False,
    clip_eigs=False,
    bidirectional=False,
    step_rescale=1.0,
  )
  model = S5(args, key=key)
  x = random.normal(key, (3, args.d_model))
  try:
    _ = model(x)
    assert False, "Expected NotImplementedError for invalid discretization"
  except NotImplementedError:
    pass


def test_s5_bilinear_discretization():
  key = random.PRNGKey(2)
  args = S5Args(
    d_model=4,
    ssm_size=4,
    blocks=2,
    C_init="lecun_normal",
    discretization="bilinear",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=False,
    clip_eigs=False,
    bidirectional=False,
    step_rescale=1.0,
  )

  model = S5(args, key=key)

  L = 5
  x = random.normal(key, (L, args.d_model))
  y = model(x)

  assert y.shape == (L, args.d_model)
  assert jnp.isfinite(y).all()


