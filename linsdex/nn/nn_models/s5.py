from functools import partial
from typing import Literal, Optional, Union, Tuple, Callable, List, Any, Annotated
import einops
import equinox as eqx
import jax.random as random
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Bool
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh
import jax.nn as jnn
from linsdex import AbstractBatchableObject
from linsdex import TimeSeries
from linsdex.nn.nn_models.nn_abstract import AbstractHyperParams
from linsdex.nn.nn_layers.time_condition import TimeFeatures
from linsdex.nn.nn_layers.s5_layers import (
  make_DPLR_HiPPO,
  discretize_zoh,
  apply_ssm,
  init_CV,
  init_VinvB,
  init_log_steps,
  trunc_standard_normal,
  discretize_bilinear,
)
# Adapted from https://github.com/lindermanlab/S5/tree/main

class S5Args(eqx.Module):
  """Low-level SSM (S5) hyperparameters used to instantiate a single S5 layer.

  Notes
  - `d_model` is the feature dimension processed by the layer.
  - `ssm_size` × `blocks` defines the global state dimension via block-diagonal HiPPO.
  - `conj_sym=True` halves parameters in the eigenbasis and ensures real outputs.
  - `discretization` chooses ZOH or bilinear for continuous→discrete mapping.
  - `dt_min`, `dt_max` bound the learned step sizes.
  """
  d_model: int                           # Model feature dimension (hidden size)
  ssm_size: int = 256                    # State dimension P of the SSM (before conj_sym halving)
  blocks: int = 4                        # Number of SSM blocks to tile (block-diagonal A)
  C_init: str = "trunc_standard_normal"  # How to init C: "lecun_normal" | "trunc_standard_normal" | "complex_normal"
  discretization: str = "zoh"            # Discretization method: "zoh" or "bilinear"
  dt_min: float = 1e-3                   # Minimum timescale (Delta) lower bound for log-step init
  dt_max: float = 1e-1                   # Maximum timescale (Delta) upper bound for log-step init
  conj_sym: bool = True                  # If True, enforce conjugate symmetry (half parameters, doubled features)
  clip_eigs: bool = False                # If True, clamp Re(Lambda) <= -1e-4 at call time for stability
  bidirectional: bool = False            # If True, use bidirectional scan and 2P columns in C
  step_rescale: float = 1.0              # Scalar multiplier on exp(log_step) at call time

class S5(AbstractBatchableObject):
  """Equinox S5 (Simplified State Space) layer.

  Initializes a continuous-time diagonal SSM via HiPPO-LegS (DPLR) and applies
  a discretization ("zoh" or "bilinear") at call time. Forward computation uses
  a parallel scan over length L to produce outputs of shape (L, H).

  Initialization summary:
  - Eigenstructure from `make_DPLR_HiPPO` with `blocks` tiling and optional
    conjugate symmetry (`conj_sym`).
  - B initialized in eigenbasis with `init_VinvB`, C with `init_CV` (or complex
    normal), D with `normal`, and log-steps with `init_log_steps` in [dt_min, dt_max].

  Attributes (shapes use P = args.ssm_size if not conj_sym else args.ssm_size//2,
  and H = args.d_model):
  - `B`: (P, H, 2) real/imag pair for complex input matrix in eigenbasis.
  - `C`: (H, P, 2) or (H, 2P, 2) if `bidirectional=True`.
  - `D`: (H,) feedthrough vector.
  - `Lambda_re`: (P,) real parts of eigenvalues.
  - `Lambda_im`: (P,) imaginary parts of eigenvalues.
  - `log_step`: (P, 1) learnable log timescales.

  Reference: https://github.com/lindermanlab/S5/tree/main
  """
  args: S5Args = eqx.field(static=True)

  B: Float[Array, 'P H 2']
  C: Float[Array, 'H P 2']
  D: Float[Array, 'H']
  Lambda_im: Float[Array, 'P']
  Lambda_re: Float[Array, 'P']
  log_step: Float[Array, 'P 1']

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    if self.D.ndim == 1:
      return None
    elif self.D.ndim == 2:
      return self.D.shape[0]
    elif self.D.ndim > 2:
      return self.D.shape[1:]
    else:
      raise ValueError(f"Invalid dimension for D: {self.D.ndim}")

  def __init__(
    self,
    args: S5Args,
    *,
    key: PRNGKeyArray,
  ):
    self.args = args

    # Determine effective state size per block
    block_size = int(args.ssm_size / args.blocks)
    if args.conj_sym:
      block_size = block_size // 2

    # HiPPO-based initialization for one block
    Lambda_block, _, _B_unused, V_block, _Borig_unused = make_DPLR_HiPPO(block_size)

    # Build block-diagonal Lambda and eigenvectors for all blocks
    Lambda = (Lambda_block[:block_size] * jnp.ones((args.blocks, block_size))).ravel()
    V_single = V_block[:, :block_size]
    V = jax.scipy.linalg.block_diag(*([V_single] * args.blocks))
    Vinv = jax.scipy.linalg.block_diag(*([V_single.conj().T] * args.blocks))

    # Effective P based on conjugate symmetry
    P = args.ssm_size // 2 if args.conj_sym else args.ssm_size
    H = args.d_model

    # Initialize B using Vinv
    key, kB = random.split(key)
    B_init = lecun_normal()
    self.B = init_VinvB(B_init, kB, (P, H), Vinv)

    # Initialize C using V (or directly as complex normal)
    key, kC1 = random.split(key)
    if args.C_init in ["trunc_standard_normal"]:
      C_init = trunc_standard_normal
      C_shape = (H, P, 2)
      if args.bidirectional:
        C1 = init_CV(C_init, kC1, C_shape, V)
        key, kC2 = random.split(key)
        C2 = init_CV(C_init, kC2, C_shape, V)
        self.C = jnp.concatenate((C1, C2), axis=1)
      else:
        self.C = init_CV(C_init, kC1, C_shape, V)
    elif args.C_init in ["lecun_normal"]:
      C_init = lecun_normal()
      C_shape = (H, P, 2)
      if args.bidirectional:
        C1 = init_CV(C_init, kC1, C_shape, V)
        key, kC2 = random.split(key)
        C2 = init_CV(C_init, kC2, C_shape, V)
        self.C = jnp.concatenate((C1, C2), axis=1)
      else:
        self.C = init_CV(C_init, kC1, C_shape, V)
    elif args.C_init in ["complex_normal"]:
      cnorm = normal(stddev=0.5 ** 0.5)
      if args.bidirectional:
        self.C = cnorm(kC1, (H, 2 * P, 2))
      else:
        self.C = cnorm(kC1, (H, P, 2))
    else:
      raise NotImplementedError(f"C_init method {args.C_init} not implemented")

    # Initialize D
    key, kD = random.split(key)
    self.D = normal(stddev=1.0)(kD, (H,))

    # Store Lambda parts
    self.Lambda_re = Lambda.real
    self.Lambda_im = Lambda.imag

    # Initialize learnable discretization timescales
    key, kStep = random.split(key)
    self.log_step = init_log_steps(kStep, (P, args.dt_min, args.dt_max))

  def __call__(self, input_sequence: Float[Array, 'L H']):
    """
    Compute the LxH output of the S5 SSM given an LxH input sequence
    using a parallel scan.
    Args:
          input_sequence (float32): input sequence (L, H)
    Returns:
        output sequence (float32): (L, H)
    """
    if self.args.clip_eigs:
      Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
    else:
      Lambda = self.Lambda_re + 1j * self.Lambda_im

    step = self.args.step_rescale * jnp.exp(self.log_step[:, 0])
    B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

    if self.args.discretization in ["zoh"]:
      Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
    elif self.args.discretization in ["bilinear"]:
      Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
    else:
      raise NotImplementedError(
        f"Discretization method {self.args.discretization} not implemented"
      )

    C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

    ys = apply_ssm(
      Lambda_bar,
      B_bar,
      C_tilde,
      input_sequence,
      self.args.conj_sym,
      self.args.bidirectional,
    )

    # Add feedthrough matrix output Du;
    Du = jax.vmap(lambda u: self.D * u)(input_sequence)

    return ys + Du


class S5Block(AbstractBatchableObject):
  """Residual S5 block: y = x + Dropout(Activation(SSM(LayerNorm(x)))).

  - Uses pre-norm for stability.
  - Dropout removed.

  Conditioning
  - If `cond_size` is provided then FiLM-style conditioning is enabled.
  - At call, pass `y: (L, cond_size)` which is projected to per-feature shift/scale
    and applied before the SSM on normalized inputs.
  """
  norm: eqx.nn.LayerNorm
  ssm: S5
  activation: Callable
  cond_to_film: Optional[eqx.nn.Linear]

  def __init__(
    self,
    args: S5Args,
    *,
    key: PRNGKeyArray,
    activation: Callable = jnn.silu,
    cond_size: Optional[int] = None,
  ):
    k_ssm, k_film = random.split(key)
    self.norm = eqx.nn.LayerNorm(shape=args.d_model)
    self.ssm = S5(args, key=k_ssm)
    self.activation = activation
    self.cond_to_film = None if cond_size is None else eqx.nn.Linear(cond_size, 2 * args.d_model, key=k_film)

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    return self.ssm.batch_size

  def __call__(
    self,
    x: Float[Array, 'L H'],
    *,
    y: Optional[Float[Array, 'L C']] = None,
  ) -> Float[Array, 'L H']:
    # Apply LayerNorm per time step per Equinox's strict shape invariant
    h = jax.vmap(self.norm)(x)
    if (y is not None) and (self.cond_to_film is not None):
      y_act = jax.vmap(self.activation)(y)
      film = jax.vmap(self.cond_to_film)(y_act)
      shift, scale = jnp.split(film, 2, axis=-1)
      h = shift + h*(1.0 + scale)
    h = self.ssm(h)
    h = self.activation(h)
    return x + h

################################################################################################################

class StackedS5BlocksHypers(AbstractHyperParams):
  """Hyperparameters for a stack of S5Blocks (no time-awareness by itself).

  Fields mirror `S5SeqHypers` relevant to stacking S5 blocks. `cond_size` enables
  FiLM in each block (the stack expects a matching `(L, cond_size)` conditioning
  tensor to be provided at call).
  """
  d_model: int
  ssm_size: int = 256
  blocks: int = 4
  C_init: str = "trunc_standard_normal"
  discretization: str = "zoh"
  dt_min: float = 1e-3
  dt_max: float = 1e-1
  conj_sym: bool = True
  clip_eigs: bool = False
  bidirectional: bool = False
  step_rescale: float = 1.0
  num_layers: int = 4
  cond_size: Optional[int] = None


class StackedS5Blocks(AbstractBatchableObject):
  """Linear(in) → [S5Block x num_layers] → Linear(out).

  - Accepts values `x: (L, input_size)` and optional conditioning `y: (L, cond_size)`
    which is forwarded into each block (FiLM).
  - No notion of time by itself; any time features should be precomputed by the caller
    and passed via `y`.
  """
  in_proj: eqx.nn.Linear
  blocks: Annotated[S5Block, 'n_layers']
  out_proj: eqx.nn.Linear
  n_layers: int = eqx.field(static=True)
  hypers: StackedS5BlocksHypers = eqx.field(static=True)

  def __init__(
    self,
    input_size: int,
    output_size: int,
    hypers: StackedS5BlocksHypers,
    *,
    activation: Callable = jnn.silu,
    key: PRNGKeyArray,
  ):
    self.hypers = hypers
    k_in, k_blocks, k_out = random.split(key, 3)
    self.in_proj = eqx.nn.Linear(input_size, hypers.d_model, key=k_in)
    block_keys = random.split(k_blocks, hypers.num_layers)
    s5_args = S5Args(
      d_model=hypers.d_model,
      ssm_size=hypers.ssm_size,
      blocks=hypers.blocks,
      C_init=hypers.C_init,
      discretization=hypers.discretization,
      dt_min=hypers.dt_min,
      dt_max=hypers.dt_max,
      conj_sym=hypers.conj_sym,
      clip_eigs=hypers.clip_eigs,
      bidirectional=hypers.bidirectional,
      step_rescale=hypers.step_rescale,
    )
    def make_block(k):
      return S5Block(
        s5_args,
        key=k,
        activation=activation,
        cond_size=hypers.cond_size,
      )
    self.blocks = eqx.filter_vmap(make_block)(block_keys)
    self.out_proj = eqx.nn.Linear(hypers.d_model, output_size, key=k_out)
    self.n_layers = hypers.num_layers

  def __call__(
    self,
    x: Float[Array, 'L I'],
    *,
    y: Optional[Float[Array, 'L C']] = None,
  ) -> Float[Array, 'L O']:
    """Run stacked S5 blocks over a length-L sequence of features.

    Args
    - x: sequence of input features of shape (L, input_size)
    - y: optional FiLM conditioning of shape (L, cond_size)

    Returns
    - Sequence of output features of shape (L, output_size)
    """
    h = jax.vmap(self.in_proj)(x)
    dynamic, static = eqx.partition(self.blocks, eqx.is_array)
    def f(carry_h, params):
      block = eqx.combine(params, static)
      new_h = block(carry_h, y=y)
      return new_h, None
    h, _ = jax.lax.scan(f, h, dynamic)
    y = jax.vmap(self.out_proj)(h)
    return y

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    if self.in_proj.weight.ndim == 1:
      return None
    elif self.in_proj.weight.ndim == 2:
      return self.in_proj.weight.shape[0]
    elif self.in_proj.weight.ndim > 2:
      return self.in_proj.weight.shape[1:]
    else:
      raise ValueError(f"Invalid dimension for in_proj.weight: {self.in_proj.weight.ndim}")

################################################################################################################

class S5SeqHypers(AbstractHyperParams):
  """High-level sequence model hyperparameters.

  - `cond_size=None` → decoder-only, conditioned on time features of `x.times`.
  - `cond_size=k`   → encoder/decoder: encode `y.values` into (L, k) with its own
    time features from `y.times`; decoder is conditioned on [encoded_y || time(x)].
  """
  d_model: int
  ssm_size: int = 256
  blocks: int = 4
  C_init: str = "trunc_standard_normal"
  discretization: str = "zoh"
  dt_min: float = 1e-3
  dt_max: float = 1e-1
  conj_sym: bool = True
  clip_eigs: bool = False
  bidirectional: bool = False
  step_rescale: float = 1.0
  num_layers: int = 4
  cond_size: Optional[int] = None
  time_feature_size: int = 32

class S5Seq2SeqModel(AbstractBatchableObject):
  """Sequence model built from stacked S5 blocks (encoder/decoder ready).

  Data flow
  - If `hypers.cond_size is None` (decoder-only):
    - Compute time features from `x.times` → `t_x`, and run
      `stack(x.values, y=t_x)`.
  - If `hypers.cond_size=k` (encoder/decoder):
    - Encoder: run `cond_stack(y.values, y=t_y)` where `t_y` are time features of
      `y.times` to obtain `encoded_y: (L, k)`.
    - Decoder: compute `t_x` from `x.times`, then run
      `stack(x.values, y=concat(encoded_y, t_x))`.

  Notes
  - All randomization (e.g., dropout) is removed by design.
  - Shapes are length-major and vmap/scan friendly.
  """
  hypers: S5SeqHypers = eqx.field(static=True)
  time_features: TimeFeatures
  cond_time_features: Optional[TimeFeatures]
  stack: 'StackedS5Blocks'
  cond_stack: Optional['StackedS5Blocks']

  def __init__(
    self,
    input_size: int,
    output_size: int,
    hypers: S5SeqHypers,
    *,
    activation: Callable = jnn.silu,
    key: PRNGKeyArray,
  ):
    self.hypers = hypers
    k_main, k_tf, k_ctf, k_enc = random.split(key, 4)
    # Always create time features for x.times
    self.time_features = TimeFeatures(
      embedding_size=2 * hypers.time_feature_size,
      out_features=hypers.time_feature_size,
      key=k_tf,
    )
    # Encoder/decoder if hypers.cond_size provided
    if self.hypers.cond_size is not None:
      self.cond_stack = StackedS5Blocks(
        input_size=self.hypers.cond_size,
        output_size=self.hypers.cond_size,
        hypers=StackedS5BlocksHypers(
          d_model=hypers.d_model,
          ssm_size=hypers.ssm_size,
          blocks=hypers.blocks,
          C_init=hypers.C_init,
          discretization=hypers.discretization,
          dt_min=hypers.dt_min,
          dt_max=hypers.dt_max,
          conj_sym=hypers.conj_sym,
          clip_eigs=hypers.clip_eigs,
          bidirectional=hypers.bidirectional,
          step_rescale=hypers.step_rescale,
          num_layers=hypers.num_layers,
          cond_size=hypers.time_feature_size,
        ),
        key=k_enc,
        activation=activation,
      )
      self.stack = StackedS5Blocks(
        input_size=input_size,
        output_size=output_size,
        hypers=StackedS5BlocksHypers(
          d_model=hypers.d_model,
          ssm_size=hypers.ssm_size,
          blocks=hypers.blocks,
          C_init=hypers.C_init,
          discretization=hypers.discretization,
          dt_min=hypers.dt_min,
          dt_max=hypers.dt_max,
          conj_sym=hypers.conj_sym,
          clip_eigs=hypers.clip_eigs,
          bidirectional=hypers.bidirectional,
          step_rescale=hypers.step_rescale,
          num_layers=hypers.num_layers,
          cond_size=self.hypers.cond_size + hypers.time_feature_size,
        ),
        key=k_main,
        activation=activation,
      )
      self.cond_time_features = TimeFeatures(
        embedding_size=2 * hypers.time_feature_size,
        out_features=hypers.time_feature_size,
        key=k_ctf,
      )
    else:
      self.cond_stack = None
      self.stack = StackedS5Blocks(
        input_size=input_size,
        output_size=output_size,
        hypers=StackedS5BlocksHypers(
          d_model=hypers.d_model,
          ssm_size=hypers.ssm_size,
          blocks=hypers.blocks,
          C_init=hypers.C_init,
          discretization=hypers.discretization,
          dt_min=hypers.dt_min,
          dt_max=hypers.dt_max,
          conj_sym=hypers.conj_sym,
          clip_eigs=hypers.clip_eigs,
          bidirectional=hypers.bidirectional,
          step_rescale=hypers.step_rescale,
          num_layers=hypers.num_layers,
          cond_size=hypers.time_feature_size,
        ),
        key=k_main,
        activation=activation,
      )
      self.cond_time_features = None

  def create_context(self, condition_info: TimeSeries) -> Float[Array, 'S C']:
    """Create a context for the sequence model.

    Args
    - condition_info: TimeSeries with fields `(times: (L,), values: (L, Dy))`.

    Returns
    - Context of shape (L, C) where C is the context size.
    """
    if self.cond_stack is None or self.cond_time_features is None:
      raise ValueError("Expected cond_stack and cond_time_features to be not None")

    y_vals = condition_info.values
    t_feats = jax.vmap(self.cond_time_features)(condition_info.times)
    encoded_cond = self.cond_stack(y_vals, y=t_feats)
    return encoded_cond

  def __call__(
    self,
    series: TimeSeries,
    *,
    condition_info: Optional[TimeSeries] = None,
    context: Optional[Float[Array, 'S C']] = None,
  ) -> Float[Array, 'L O']:
    """Apply the sequence model on a TimeSeries.

    Args
    - series: TimeSeries with fields `(times: (L,), values: (L, Dx))`.
    - condition_info: Optional TimeSeries with fields `(times: (L,), values: (L, Dy))`.
      Provide this or `context` when `hypers.cond_size is not None`.

    Returns
    - Sequence of output features `(L, Do)`.
    """
    # Validation: if conditioning is enabled, require one of condition_info or context
    if self.hypers.cond_size is not None:
      if condition_info is None and context is None:
        raise ValueError("Provide condition_info or context when cond_size is set")

    # Build decoder time features for series.times
    x_t_feats = jax.vmap(self.time_features)(series.times)

    # Encoder/decoder path
    if self.cond_stack is not None and self.cond_time_features is not None:
      if context is None:
        context = self.create_context(condition_info)
      L = series.values.shape[0]
      assert context.shape == (L, self.hypers.cond_size)

      dec_cond = jnp.concatenate([context, x_t_feats], axis=-1)
      return self.stack(series.values, y=dec_cond)

    # No conditioning
    return self.stack(series.values, y=x_t_feats)

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    return self.stack.batch_size

################################################################################################################

# Alias hypers for the time-dependent variant (identical hyperparameters)
TimeDependentS5SeqHypers = S5SeqHypers


class TimeDependentS5Seq2SeqModel(AbstractBatchableObject):
  """Time-dependent S5 encoder-decoder.

  Extends `S5Seq2SeqModel` by accepting a simulation time scalar `s` at call,
  embedding it into features and concatenating with the decoder time features.

  Decoder FiLM conditioning receives `[context || time_features(x.times) || time_features(s)]`.
  """
  hypers: TimeDependentS5SeqHypers = eqx.field(static=True)
  time_features: TimeFeatures
  cond_time_features: Optional[TimeFeatures]
  stack: 'StackedS5Blocks'
  cond_stack: Optional['StackedS5Blocks']
  simulation_time_features: TimeFeatures

  def __init__(
    self,
    input_size: int,
    output_size: int,
    hypers: TimeDependentS5SeqHypers,
    *,
    activation: Callable = jnn.silu,
    key: PRNGKeyArray,
  ):
    # Keys: main stack, x.time features, y.time features, encoder stack, simulation time features
    k_main, k_tf, k_ctf, k_enc, k_sim = random.split(key, 5)

    # Create decoder time features for x.times
    self.hypers = hypers
    self.time_features = TimeFeatures(
      embedding_size=2 * hypers.time_feature_size,
      out_features=hypers.time_feature_size,
      key=k_tf,
    )
    # Create simulation-time embedding features
    self.simulation_time_features = TimeFeatures(
      embedding_size=2 * hypers.time_feature_size,
      out_features=hypers.time_feature_size,
      key=k_sim,
    )

    # Encoder/decoder if conditioning is enabled
    if self.hypers.cond_size is not None:
      self.cond_stack = StackedS5Blocks(
        input_size=self.hypers.cond_size,
        output_size=self.hypers.cond_size,
        hypers=StackedS5BlocksHypers(
          d_model=hypers.d_model,
          ssm_size=hypers.ssm_size,
          blocks=hypers.blocks,
          C_init=hypers.C_init,
          discretization=hypers.discretization,
          dt_min=hypers.dt_min,
          dt_max=hypers.dt_max,
          conj_sym=hypers.conj_sym,
          clip_eigs=hypers.clip_eigs,
          bidirectional=hypers.bidirectional,
          step_rescale=hypers.step_rescale,
          num_layers=hypers.num_layers,
          cond_size=hypers.time_feature_size,
        ),
        key=k_enc,
        activation=activation,
      )
      # Decoder now expects [context (cond_size) || t_x (time_feature_size) || t_s (time_feature_size)]
      self.stack = StackedS5Blocks(
        input_size=input_size,
        output_size=output_size,
        hypers=StackedS5BlocksHypers(
          d_model=hypers.d_model,
          ssm_size=hypers.ssm_size,
          blocks=hypers.blocks,
          C_init=hypers.C_init,
          discretization=hypers.discretization,
          dt_min=hypers.dt_min,
          dt_max=hypers.dt_max,
          conj_sym=hypers.conj_sym,
          clip_eigs=hypers.clip_eigs,
          bidirectional=hypers.bidirectional,
          step_rescale=hypers.step_rescale,
          num_layers=hypers.num_layers,
          cond_size=self.hypers.cond_size + 2 * hypers.time_feature_size,
        ),
        key=k_main,
        activation=activation,
      )
      self.cond_time_features = TimeFeatures(
        embedding_size=2 * hypers.time_feature_size,
        out_features=hypers.time_feature_size,
        key=k_ctf,
      )
    else:
      self.cond_stack = None
      # Decoder expects [t_x || t_s]
      self.stack = StackedS5Blocks(
        input_size=input_size,
        output_size=output_size,
        hypers=StackedS5BlocksHypers(
          d_model=hypers.d_model,
          ssm_size=hypers.ssm_size,
          blocks=hypers.blocks,
          C_init=hypers.C_init,
          discretization=hypers.discretization,
          dt_min=hypers.dt_min,
          dt_max=hypers.dt_max,
          conj_sym=hypers.conj_sym,
          clip_eigs=hypers.clip_eigs,
          bidirectional=hypers.bidirectional,
          step_rescale=hypers.step_rescale,
          num_layers=hypers.num_layers,
          cond_size=2 * hypers.time_feature_size,
        ),
        key=k_main,
        activation=activation,
      )
      self.cond_time_features = None

  def create_context(self, condition_info: TimeSeries) -> Float[Array, 'S C']:
    """Create a reusable context from condition_info.

    Returns an encoded sequence of shape (L, cond_size) using the encoder stack
    conditioned on time features of condition_info.times.
    """
    if self.cond_stack is None or self.cond_time_features is None:
      raise ValueError("Expected cond_stack and cond_time_features to be not None")
    y_vals = condition_info.values
    t_feats = jax.vmap(self.cond_time_features)(condition_info.times)
    encoded_cond = self.cond_stack(y_vals, y=t_feats)
    return encoded_cond

  def __call__(
    self,
    s: Scalar,
    series: TimeSeries,
    *,
    condition_info: Optional[TimeSeries] = None,
    context: Optional[Float[Array, 'S C']] = None,
  ) -> Float[Array, 'L O']:
    # Validation: if conditioning is enabled, require one of condition_info or context
    if self.hypers.cond_size is not None:
      if condition_info is None and context is None:
        raise ValueError("Provide condition_info or context when cond_size is set")

    # Build decoder time features for series.times and simulation time
    t_x = jax.vmap(self.time_features)(series.times)
    t_s = self.simulation_time_features(s)
    t_s_broadcast = jnp.broadcast_to(t_s[None], (series.values.shape[0], t_s.shape[-1]))
    x_t_feats = jnp.concatenate([t_x, t_s_broadcast], axis=-1)

    # Encoder/decoder path
    if self.cond_stack is not None and self.cond_time_features is not None:
      if context is None:
        context = self.create_context(condition_info)
      dec_cond = jnp.concatenate([context, x_t_feats], axis=-1)
      return self.stack(series.values, y=dec_cond)

    # No conditioning
    return self.stack(series.values, y=x_t_feats)

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    return self.stack.batch_size
