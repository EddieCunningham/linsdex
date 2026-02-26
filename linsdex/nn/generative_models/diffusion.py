import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Annotated, Optional, Mapping, Tuple, List, Union, Any, Callable, Dict, overload, Literal
import einops
import equinox as eqx
import abc
import diffrax
from jaxtyping import Array, PRNGKeyArray
import jax.tree_util as jtu
import os
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
from jax._src.util import curry
from linsdex import AbstractSDE
from linsdex import TimeSeries
from linsdex.ssm.simple_encoder import AbstractEncoder, PaddingLatentVariableEncoderWithPrior, IdentityEncoder
from linsdex import AbstractLinearSDE, MixedGaussian, NaturalGaussian, GaussianTransition, StandardGaussian
from linsdex import AbstractSquareMatrix, DiagonalMatrix
from linsdex import AbstractTransition, AbstractPotential
from linsdex import ConditionedLinearSDE
import linsdex.util as util
# from experiment.config_base import ExperimentConfig
from linsdex import BrownianMotion, WienerVelocityModel, TimeScaledLinearTimeInvariantSDE, OrnsteinUhlenbeck
from linsdex import ODESolverParams, ode_solve, SDESolverParams, sde_sample
import linsdex as lsde
from linsdex import GaussianPotentialSeries
from linsdex.nn.generative_models.prob_model_abstract import AbstractGenerativeModel
from linsdex.nn.nn_models.nn_abstract import AbstractTimeDependentSeq2SeqModel, AbstractTimeDependentNeuralNet
from linsdex.nn.nn_layers.util import apply_mask_to_series
from linsdex.nn.nn_models.get_nn import get_nn
# from ott.geometry import pointcloud
# from ott.problems.linear import linear_problem
# from ott.solvers.linear import sinkhorn
from linsdex.diffusion_model.probability_path import DiffusionModelConversions, DiffusionModelComponents, noise_schedule_drift_correction
import warnings
from linsdex import AbstractBatchableObject
# import traceax as tx
# import lineax as lx

################################################################################################################

class LocalSDETimeSeries(AbstractSDE):
  """Helper class to pass into the ODE solver"""

  nn: AbstractTimeDependentSeq2SeqModel # Our model that contains a transformer to predict the control at all times

  @property
  def batch_size(self) -> int:
    return self.nn.batch_size

  def get_drift(self, t: Scalar,  xt: Float[Array, 'D']) -> Float[Array, 'D']:
    raise NotImplementedError('we\'re doing flow matching')
  def get_diffusion_coefficient(self, t: Scalar, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    raise NotImplementedError('we\'re doing flow matching')
  def get_transition_distribution(self, s: Scalar, t: Scalar) -> AbstractTransition:
    raise NotImplementedError('No closed form transition distribution for neural SDEs.')

  def get_flow(self, s: Scalar, xts: TimeSeries) -> TimeSeries:
    flow_values = self.nn(s, xts)
    return TimeSeries(xts.times*0.0, flow_values)

  def simulate(
    self,
    x0: TimeSeries,
    save_times: Float[Array, 'T']
  ) -> Annotated[TimeSeries, 'T']:
    """Transport the sample x0 from the start time (at save_times[0]) to the end time (at save_times[-1])"""

    # Simulate the probability flow ODE of the neural SDE
    solver_params = ODESolverParams(rtol=1e-3,
                                    atol=1e-5,
                                    solver='dopri5',
                                    adjoint='recursive_checkpoint',
                                    stepsize_controller='pid',
                                    max_steps=2000,
                                    throw=False,
                                    progress_meter=None)

    simulated_trajectory = ode_solve(self,
                                      x0=x0,
                                      save_times=save_times,
                                      params=solver_params)

    return simulated_trajectory

################################################################################################################

def diffusion_time_series_loss(
  model_config: Any,
  dataset_config: Any,
  nn: AbstractTimeDependentSeq2SeqModel,
  linear_sde: AbstractLinearSDE,
  encoder: AbstractEncoder,
  obs_series: TimeSeries,
  key: PRNGKeyArray,
  debug: Optional[bool] = False
) -> Dict[str, Scalar]:
  """Compute the loss for a structured diffusion model"""
  k1, k2 = random.split(key, 2)

  # Flatten the input series
  series_shape = obs_series.values.shape
  x1 = obs_series.values.ravel()
  x0 = random.normal(k1, x1.shape)

  endpoint_times = jnp.array([0.0, 1.0])
  endpoint_values = jnp.concatenate([x0[None], x1[None]], axis=0)
  endpoint_series = TimeSeries(endpoint_times, endpoint_values)

  # Brownian bridge simulation
  encoder = IdentityEncoder(dim=x0.shape[-1])
  prob_series: GaussianPotentialSeries = encoder(endpoint_series)
  cond_sde: ConditionedLinearSDE = linear_sde.condition_on(prob_series)

  s = random.uniform(k2, shape=(), minval=0.001, maxval=0.999)
  items = cond_sde.sample_matching_items(jnp.array([s]), key)

  latent_series = items.xt[1]
  flow_target = items.flow[1] # We are going to do flow matching because it is the fastest at inference time
  flow_target = flow_target.reshape(series_shape)

  # Predict the flow with out model
  xs_series = TimeSeries(obs_series.times, latent_series.reshape(series_shape))
  pred_flow = nn(s, xs_series, obs_series)

  # Compute the flow matching loss
  matching_loss = jnp.mean((pred_flow - flow_target)**2)

  # As a sanity check, check the cosine similarity between the predicted flow_target and the true flow_target
  cos_sim = jnp.mean(jnp.sum(pred_flow*flow_target, axis=-1) / (jnp.linalg.norm(pred_flow, axis=-1)*jnp.linalg.norm(flow_target, axis=-1)))

  losses = dict(flow_matching=matching_loss, cos_sim=cos_sim)

  if debug:
    import pdb; pdb.set_trace()

  return losses

def diffusion_time_series_model_sample(
  model_config: Any,
  dataset_config: Any,
  nn: AbstractTimeDependentSeq2SeqModel,
  obs_series: TimeSeries,
  key: PRNGKeyArray,
  debug: Optional[bool] = False
) -> TimeSeries:

  obs_series = apply_mask_to_series(obs_series, model_config.condition_length)

  # Sample from the prior
  x0_values = random.normal(key, obs_series.values.shape)
  x0 = TimeSeries(obs_series.times, x0_values)

  # Create the context and fill in the nn model
  context = nn.create_context(obs_series)
  eval_seq2seq = eqx.Partial(nn, context=context)

  # Create the local SDE that will perform the transport
  local_sde = LocalSDETimeSeries(eval_seq2seq)
  save_times = jnp.array([0.0, 1.0])
  generated_series: Annotated[TimeSeries, '2'] = local_sde.simulate(x0, save_times)

  if debug:
    import pdb; pdb.set_trace()

  return generated_series[1]


class DiffusionTimeSeriesModelConfig(eqx.Module):
  name: str

  process_noise: float = 0.1
  condition_length: int = 20

  sde_type: Literal['brownian', 'wiener_velocity', 'wiener_acceleration'] = 'brownian'
  nn_type: Literal['time_dependent_gru_rnn', 'time_dependent_transformer', 'time_dependent_wavenet'] = 'time_dependent_gru_rnn'

  # Dynamic fields for NN models (e.g. GRU, Transformer, etc.)
  hidden_size: Optional[int] = None
  n_layers: Optional[int] = None
  intermediate_channels: Optional[int] = None
  cond_in_channels: Optional[int] = None
  in_channels: Optional[int] = None
  out_channels: Optional[int] = None

  # SSM fields
  d_model: Optional[int] = None
  ssm_size: Optional[int] = None
  blocks: Optional[int] = None
  num_layers: Optional[int] = None
  cond_size: Optional[int] = None

class DiffusionTimeSeriesModel(AbstractGenerativeModel):

  nn: AbstractTimeDependentSeq2SeqModel

  condition_length: int = eqx.field(static=True)
  linear_sde: AbstractLinearSDE = eqx.field(static=True)
  encoder: AbstractEncoder = eqx.field(static=True)
  model_config: DiffusionTimeSeriesModelConfig = eqx.field(static=True)
  dataset_config: Any = eqx.field(static=True)

  @property
  def batch_size(self):
    return self.nn.batch_size

  def __init__(self, model_config: Any, dataset_config: Any, random_seed: int):

    # We flatten the data into a single dimension for the SDE
    dim = dataset_config.n_features*dataset_config.seq_length

    if model_config.sde_type == 'brownian':
      sde = BrownianMotion(sigma=model_config.process_noise, dim=dim)
    else:
      raise ValueError(f"Invalid SDE type: {model_config.sde_type}")

    self.linear_sde = TimeScaledLinearTimeInvariantSDE(sde, time_scale=dataset_config.time_scale_mult)

    self.encoder = PaddingLatentVariableEncoderWithPrior(
      y_dim=dataset_config.n_features,
      x_dim=sde.dim,
      sigma=dataset_config.observation_noise,
      use_prior=True
    )

    self.nn = get_nn(model_config, dataset_config, random_seed)
    self.model_config = model_config
    self.dataset_config = dataset_config
    self.condition_length = model_config.condition_length

  def loss_fn(self, key: PRNGKeyArray, yts: TimeSeries, debug: Optional[bool] = False) -> Tuple[Scalar, Dict[str, Scalar]]:
    losses = diffusion_time_series_loss(self.model_config, self.dataset_config, self.nn, self.linear_sde, self.encoder, yts, key, debug)
    return losses['flow_matching'], losses

  def sample(self, key: PRNGKeyArray, yts: TimeSeries, debug: Optional[bool] = False, **kwargs) -> TimeSeries:
    return diffusion_time_series_model_sample(self.model_config, self.dataset_config, self.nn, yts, key, debug=debug)

################################################################################################################
################################################################################################################
################################################################################################################

def _unbatched_bridge_matching_loss(
  model_config: Any,
  debug: Optional[bool],
  data_shape: Tuple[int, ...],
  components: DiffusionModelComponents,
  nn: AbstractTimeDependentNeuralNet,
  y0: Float[Array, 'D'],
  y1: Float[Array, 'D'],
  condition_info: Optional[PyTree],
  key: PRNGKeyArray,
) -> Tuple[Scalar, Scalar]:
  k1, k2 = random.split(key, 2)

  # Create the endpoint series
  t0 = components.t0
  t1 = components.t1

  # Brownian bridge simulation
  phi0 = StandardGaussian(y0, components.evidence_cov.set_zero())
  phi1 = StandardGaussian(y1, components.evidence_cov)
  times = jnp.array([t0, t1])
  evidence = jtu.tree_map(lambda *xs: jnp.array(xs), phi0, phi1)
  evidence = GaussianPotentialSeries(times, evidence)

  cond_sde: ConditionedLinearSDE = components.linear_sde.condition_on(evidence)

  # Sample from the bridge at time t
  eps = 1e-6
  t = random.uniform(k2, shape=(), minval=t0 + eps, maxval=t1 - eps)
  items = cond_sde.sample_matching_items(jnp.array([t]), key)
  xt = items.xt[1]

  # Predict the control with our model
  pred_control = nn(t, xt, condition_info=condition_info)

  # Get the target control depending on what our matching method is
  """
  # This is the code that is used to compute the target control

  def get_items(t, xt, fwd, bwd, pxt):
    vt = self.sde.get_drift(t, xt)
    L = self.sde.get_diffusion_coefficient(t, xt)
    LLT = L@L.T
    noise = pxt.get_noise(xt)
    fwd_score, bwd_score = fwd.score(xt), bwd.score(xt)

    score = fwd_score + bwd_score
    flow = vt + 0.5*LLT@(bwd_score - fwd_score)
    drift = vt + LLT@bwd_score = flow - 0.5*LLT@score
    return FlowItems(t=t,
                      xt=xt,
                      flow=flow,
                      score=score,
                      noise=noise,
                      drift=drift,
                      fwd_score=fwd_score,
                      bwd_score=bwd_score)

  """
  if model_config.matching_target == "flow":
    target_control = items.flow[1]
  elif model_config.matching_target == "drift":
    target_control = items.drift[1]
  elif model_config.matching_target == "y1":
    target_control = y1
  else:
    raise ValueError(f"Invalid matching target: {model_config.matching_target}")

  # Reshape the target control to the original shape
  target_control = target_control.reshape(data_shape)

  # Compute the flow matching loss
  matching_loss = jnp.mean((pred_control - target_control)**2)

  # As a sanity check, check the cosine similarity between the predicted control and the true control
  cos_sim = jnp.mean(jnp.sum(pred_control*target_control, axis=-1) / (jnp.linalg.norm(pred_control, axis=-1)*jnp.linalg.norm(target_control, axis=-1)))

  if debug:
    import pdb; pdb.set_trace()

  return matching_loss, cos_sim

def stochastic_interpolation_loss(
  model_config: Any,
  components: DiffusionModelComponents,
  nn: AbstractTimeDependentNeuralNet,
  data: Union[Float[Array, 'batch_size D'],
              Float[Array, 'batch_size H W C']],
  *,
  condition_info: Optional[PyTree] = None,
  key: PRNGKeyArray,
  debug: Optional[bool] = False
) -> Dict[str, Scalar]:
  k1, k2 = random.split(key, 2)
  assert data.ndim == 2 or data.ndim == 4, 'data must be batched'

  # Sample the endpoints of the bridge
  full_data_shape = data.shape
  data_shape = full_data_shape[1:]

  # Flatten data while preserving batch dimension if present
  if len(full_data_shape) > 2:  # Has spatial dimensions that need flattening
    y1 = einops.rearrange(data, 'b h w c -> b (h w c)')
  else:
    y1 = data  # Already flat or batched flat

  y0 = random.normal(k1, y1.shape)

  # If the data is batched, then optionally change the coupling
  # This is multisample flow matching https://arxiv.org/pdf/2304.14772
  # assert hasattr(model_config, 'use_multisample_matching'), 'use_multisample_matching must be set in the model config'
  if getattr(model_config, 'use_multisample_matching', False):

    # Use optimal transport to re-couple the endpoints of the bridge
    geom = pointcloud.PointCloud(y0, y1)
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot = solver(ot_prob)
    mat = ot.matrix
    logits = jnp.log(mat + 1e-8)

    # Resample the data
    idx = jax.random.categorical(k2, logits, axis=0)
    y0 = y0[idx]

  # Fill in the static arguemnts to the loss function
  filled_diffusion_loss = eqx.Partial(
    _unbatched_bridge_matching_loss,
    model_config,
    debug,
    data_shape,
    components
  )

  condition_vmap = None if condition_info is None else 0
  # condition_vmap = 0 if condition_info is None else 1
  loss_fn = eqx.filter_vmap(filled_diffusion_loss, in_axes=(None, 0, 0, condition_vmap, 0))

  keys = random.split(key, data.shape[0])
  matching_loss, cos_sim = loss_fn(nn, y0, y1, condition_info, keys)

  matching_loss = jnp.mean(matching_loss)
  cos_sim = jnp.mean(cos_sim)

  losses = dict(matching_loss=matching_loss, cos_sim=cos_sim)

  if debug:
    import pdb; pdb.set_trace()

  return losses

################################################################################################################
################################################################################################################

def diffusion_loss(
  model_config: Any,
  components: DiffusionModelComponents,
  nn: AbstractTimeDependentNeuralNet,
  data: Union[Float[Array, 'batch_size D'],
              Float[Array, 'batch_size H W C']],
  *,
  condition_info: Optional[PyTree] = None,
  key: PRNGKeyArray,
  debug: Optional[bool] = False
) -> Dict[str, Scalar]:
  warnings.warn('Diffusion_loss is deprecated.  Use stochastic_interpolation_loss instead.  This has much higher variance than stochastic interpolation.')
  k1, k2 = random.split(key, 2)
  assert getattr(model_config, 'use_multisample_matching', False) == False, 'multisample matching is not supported for diffusion loss where we have integrated out the prior'

  assert data.ndim == 2 or data.ndim == 4, 'data must be batched'

  # Sample the endpoints of the bridge
  full_data_shape = data.shape
  data_shape = full_data_shape[1:]

  # Flatten data while preserving batch dimension if present
  if len(full_data_shape) > 2:  # Has spatial dimensions that need flattening
    y1 = einops.rearrange(data, 'b h w c -> b (h w c)')
  else:
    y1 = data  # Already flat or batched flat

  # Unpack the components
  eps = 1e-6
  t0 = components.t0
  t1 = components.t1
  matching_target = model_config.matching_target

  def unbatched_loss(y1: PyTree, condition_info: Optional[PyTree], key: PRNGKeyArray) -> Tuple[Scalar, Scalar]:
    t = random.uniform(key, shape=(), minval=t0 - eps, maxval=t1 + eps)
    conversions = DiffusionModelConversions(components, t)
    pt = conversions.y1_to_marginal(y1)
    xt = pt.sample(key)

    # Get the target control depending on what our matching method is
    if matching_target == "y1":
      target = y1
    elif matching_target == "drift":
      target = conversions.y1_to_drift(y1, xt)
    elif matching_target == "flow":
      target = conversions.y1_to_flow(y1, xt)
    else:
      raise ValueError(f"Invalid matching target: {model_config.matching_target}")

    # Evaluate the neural network
    prediction = nn(t, xt, condition_info=condition_info)

    # Reshape the target control to the original shape
    target = target.reshape(data_shape)
    matching_loss = jnp.mean((prediction - target)**2)
    cos_sim = jnp.mean(jnp.sum(prediction*target, axis=-1) / (jnp.linalg.norm(prediction, axis=-1)*jnp.linalg.norm(target, axis=-1)))

    return matching_loss, cos_sim

  in_axes = (0, None, 0) if condition_info is None else (0, 0, 0)
  keys = random.split(key, data.shape[0])
  matching_losses, cos_sims = jax.vmap(unbatched_loss, in_axes=in_axes)(y1, condition_info, keys)

  matching_loss = jnp.mean(matching_losses)
  cos_sim = jnp.mean(cos_sims)

  losses = dict(matching_loss=matching_loss, cos_sim=cos_sim)

  if debug:
    import pdb; pdb.set_trace()

  return losses

def probability_flow_ode_sample(
  model_config: Any,
  components: DiffusionModelComponents,
  nn: AbstractTimeDependentNeuralNet,
  condition_info: Optional[PyTree],
  key: PRNGKeyArray,
  solver_params: Optional[ODESolverParams] = None,
  debug: Optional[bool] = False,
) -> Array:
  """Sample from the probability flow ODE"""

  eps = 1e-6
  t0 = components.t0
  t1 = components.t1

  # Sample from the prior
  x0 = components.x_t0_prior.sample(key)

  # Get the probability flow function
  matching_target = model_config.matching_target
  def flow_fn(t, xt):
    nn_output = nn(t, xt, condition_info=condition_info)
    conversions = DiffusionModelConversions(components, t)
    if matching_target == "y1":
      return conversions.y1_to_flow(nn_output, xt)
    elif matching_target == "drift":
      return conversions.drift_to_flow(xt, nn_output)
    elif matching_target == "flow":
      return nn_output

  # Dry run to make sure that things work
  flow_fn(t0, x0)

  # Simulate the probability flow ODE of the neural SDE
  if solver_params is None:
    solver_params = ODESolverParams(rtol=1e-3,
                                    atol=1e-5,
                                    solver='dopri5',
                                    adjoint='recursive_checkpoint',
                                    stepsize_controller='pid',
                                    max_steps=2000,
                                    throw=False,
                                    progress_meter=None)

  simulated_trajectory: TimeSeries = ode_solve(flow_fn,
                                    x0=x0,
                                    save_times=jnp.array([t0 + eps, t1 - eps]),
                                    params=solver_params)

  if debug:
    import pdb; pdb.set_trace()

  return simulated_trajectory[1].values

def neural_sde_sample(
  model_config: Any,
  components: DiffusionModelComponents,
  nn: AbstractTimeDependentNeuralNet,
  condition_info: Optional[PyTree],
  key: PRNGKeyArray,
  noise_schedule: Optional[Callable[[Scalar], AbstractSquareMatrix]] = None,
  save_times: Optional[Array] = None,
  solver_params: Optional[SDESolverParams] = None,
  debug: Optional[bool] = False
) -> Array:
  """Sample from the neural SDE"""

  eps = 0.01
  t0 = components.t0
  t1 = components.t1

  # Sample from the prior
  x0 = components.x_t0_prior.sample(key)

  # Get the neural SDE function
  matching_target = model_config.matching_target
  def drift_fn(t, xt):
    nn_output = nn(t, xt, condition_info=condition_info)
    conversions = DiffusionModelConversions(components, t)
    if matching_target == "y1":
      base_drift = conversions.y1_to_drift(nn_output, xt)
    elif matching_target == "drift":
      base_drift = nn_output
    elif matching_target == "flow":
      base_drift = conversions.flow_to_drift(xt, nn_output)

    drift = noise_schedule_drift_correction(components, t, xt, base_drift, noise_schedule, conversions)
    return drift

  def diffusion_fn(t, xt):
    if noise_schedule is not None:
      return noise_schedule(t, xt)
    else:
      return components.linear_sde.get_diffusion_coefficient(t, xt)

  # Evaluate the drift and diffusion once to check for errors
  drift_fn(0.5*(t0 + t1), x0)
  diffusion_fn(0.5*(t0 + t1), x0)

  # Simulate the neural SDE
  if solver_params is None:
    solver_params = SDESolverParams(solver='shark',
                                    adjoint='recursive_checkpoint',
                                    max_steps=100,
                                    throw=False,
                                    progress_meter=None)

  passed_in_save_times = False
  if save_times is None:
    save_times = jnp.array([t0 + eps, t1 - eps])
  else:
    passed_in_save_times = True
  simulated_trajectory: TimeSeries = sde_sample((drift_fn, diffusion_fn),
                                                x0=x0,
                                                key=key,
                                                save_times=save_times,
                                                params=solver_params)

  if debug:
    import pdb; pdb.set_trace()

  if passed_in_save_times:
    return simulated_trajectory
  else:
    return simulated_trajectory[1].values

################################################################################################################

class LogProbDynamicsState(AbstractBatchableObject):
  xt: Float[Array, 'D']
  log_prob: Scalar
  vf_norm: Scalar
  jac_norm: Scalar

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.log_prob.ndim == 0:
      return None
    elif self.log_prob.ndim == 1:
      return self.log_prob.shape[0]
    elif self.log_prob.ndim > 1:
      return self.log_prob.shape[:-1]
    else:
      raise ValueError(f"Invalid number of dimensions for log_prob: {self.log_prob.ndim}")

def diffusion_model_log_prob(
  model_config: Any,
  components: DiffusionModelComponents,
  nn: AbstractTimeDependentNeuralNet,
  x1: Float[Array, 'D'],
  condition_info: Optional[PyTree],
  key: Optional[PRNGKeyArray] = None,
  debug: Optional[bool] = False,
) -> LogProbDynamicsState:
  """Compute the log probability of x1 under the marginal distribution of
  the diffusion model at time t1.

  If key is provided, then we will use a stochastic trace estimate.
  """

  matching_target = model_config.matching_target

  # Get the probability flow function
  matching_target = model_config.matching_target
  def flow_fn(t, xt):
    nn_output = nn(t, xt, condition_info=condition_info)
    conversions = DiffusionModelConversions(components, t)
    if matching_target == "y1":
      return conversions.y1_to_flow(nn_output, xt)
    elif matching_target == "drift":
      return conversions.drift_to_flow(xt, nn_output)
    elif matching_target == "flow":
      return nn_output

  if key is not None:
    v = random.normal(key, x1.shape)

  @diffrax.ODETerm
  def state_material_derivative(
      t, state: LogProbDynamicsState, args: Any
  ) -> LogProbDynamicsState:
    # Compute the divergence of the flow at xt
    xt = state.xt

    def apply_vf(x, args=None):
      return flow_fn(t, x)

    if key is None:
      # Brute force dlogpx/dt
      xt_flat = xt.ravel()
      eye = jnp.eye(xt_flat.shape[-1])
      xt_shape = xt.shape

      def jvp_flat(xt_flat, dxt_flat):
        xt = xt_flat.reshape(xt_shape)
        dxt = dxt_flat.reshape(xt_shape)
        dxdt, d2dx_dtdx = jax.jvp(apply_vf, (xt,), (dxt,))
        return dxdt, d2dx_dtdx.ravel()

      dxtdt, d2dxt_dtdxt_flat = jax.vmap(jvp_flat, in_axes=(None, 0))(xt_flat, eye)
      dxtdt = dxtdt[0].reshape(xt_shape)
      dlogpxdt = -jnp.trace(d2dxt_dtdxt_flat)
      dtjfndt = jnp.sum(d2dxt_dtdxt_flat**2)

    else:
      dxtdt, dudxv = jax.jvp(apply_vf, (xt,), (v,))
      dlogpxdt = -jnp.sum(dudxv*v)
      dtjfndt = jnp.sum(dudxv**2)

    dvfnormdt = jnp.sum(dxtdt**2)
    return LogProbDynamicsState(dxtdt, dlogpxdt, dvfnormdt, dtjfndt)

  t0 = components.t0 + 1e-6
  t1 = components.t1 - 1e-6

  state_t1 = LogProbDynamicsState(x1, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
  saveat = diffrax.SaveAt(ts=jnp.array([t1, t0]))

  # Get the solver object
  solver = diffrax.Dopri5()
  dt0 = (t1 - t0)/100

  # Solve the SDE
  sol: diffrax.Solution = diffrax.diffeqsolve(
      state_material_derivative,
      solver,
      t1,
      t0,
      dt0=-dt0,  # Simulate backwards in time
      y0=state_t1,
      args=None,
      saveat=saveat,
      adjoint=diffrax.RecursiveCheckpointAdjoint(),
      stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
      max_steps=1000,
      throw=True,
      # progress_meter=diffrax.TqdmProgressMeter())
      progress_meter=diffrax.NoProgressMeter(),
  )

  # Take the last state and flip the signs of sol.ys to account for the backwards simulation
  last_state = jtu.tree_map(lambda x: -x, sol.ys[-1])

  # Add the prior log prob to the log prob of the last state
  prior_log_prob = components.x_t0_prior.log_prob(last_state.xt)
  log_prob = prior_log_prob + last_state.log_prob

  # Insert the log prob into the last state
  last_state = eqx.tree_at(lambda x: x.log_prob, last_state, log_prob)

  if debug:
    import pdb; pdb.set_trace()

  return last_state

################################################################################################################

class DiffusionModelConfig(eqx.Module):
  name: str

  sde_type: Literal['brownian', 'ornstein_uhlenbeck']
  nn_type: Literal['time_dependent_resnet']
  matching_target: Literal['flow', 'drift', 'y1'] # Stable matching targets for training.  Can be reparameterized at test time.
  use_multisample_matching: bool

  process_noise: float = 0.1
  lambda_: float = 1.0

  # ResNet hyperparameters
  working_size: Optional[int] = None
  hidden_size: Optional[int] = None
  n_blocks: Optional[int] = None
  embedding_size: Optional[int] = None
  out_features: Optional[int] = None

class DiffusionModel(AbstractGenerativeModel):
  """Diffusion model that generates data using a unit Gaussian prior"""

  nn: AbstractTimeDependentNeuralNet
  components: DiffusionModelComponents = eqx.field(static=True)
  model_config: DiffusionModelConfig = eqx.field(static=True)
  dataset_config: Any = eqx.field(static=True)

  @property
  def batch_size(self):
    return self.nn.batch_size

  def __init__(self, model_config: Any, dataset_config: Any, random_seed: int):
    """Initialize the diffusion model.

    Args:
      model_config: Model configuration.
      dataset_config: Dataset configuration.
      random_seed: Random seed for initialization.
    """

    # We flatten the data into a single dimension for the SDE
    dim = dataset_config.n_features

    if model_config.sde_type == 'brownian':
      sde = BrownianMotion(sigma=model_config.process_noise, dim=dim)
    elif model_config.sde_type == 'ornstein_uhlenbeck':
      sde = OrnsteinUhlenbeck(sigma=model_config.process_noise, lambda_=model_config.lambda_, dim=dim)
    else:
      raise ValueError(f"Invalid SDE type: {model_config.sde_type}")

    evidence_cov = DiagonalMatrix.eye(dim)*0.001
    t0 = jnp.array(0.0)
    t1 = jnp.array(1.0)
    xt0_prior = StandardGaussian(jnp.zeros(dim), evidence_cov.set_eye())
    self.components = DiffusionModelComponents(sde, t0, xt0_prior, t1, evidence_cov)

    self.nn = get_nn(model_config, dataset_config, random_seed)
    self.model_config = model_config
    self.dataset_config = dataset_config

  def create_context(self, condition_info: Optional[PyTree]) -> Optional[PyTree]:
    """Create a processed context vector from raw conditioning information.

    For time-dependent feedforward nets (e.g., TimeDependentResNet), there is
    no separate context computation; we simply pass through the provided
    `condition_info`. For models that support it, this vector can be provided
    via the `context` kwarg to NN calls.
    """
    if hasattr(self.nn, 'create_context') and callable(getattr(self.nn, 'create_context')):
      return self.nn.create_context(condition_info)
    return condition_info

  def drift_fn(self, t: Scalar, xt: Float[Array, 'D'], condition_info: Optional[PyTree] = None, *, context: Optional[PyTree] = None) -> Float[Array, 'D']:
    """Neural drift field used for SDE simulation.

    Args:
      t: Scalar time in [t0, t1].
      xt: State at time t with shape `(D,)`.
      condition_info: Optional conditioning info forwarded to the NN.

    Returns:
      The drift vector field evaluated at `(t, xt)` with shape `(D,)`.
    """
    matching_target = self.model_config.matching_target
    nn_output = self.nn(t, xt, condition_info=condition_info, context=context)
    conversions = DiffusionModelConversions(self.components, t)
    if matching_target == "y1":
      drift = conversions.y1_to_drift(nn_output, xt)
    elif matching_target == "drift":
      drift = nn_output
    elif matching_target == "flow":
      drift = conversions.flow_to_drift(xt, nn_output)
    else:
      raise ValueError(f"Invalid matching target: {self.model_config.matching_target}")
    return drift

  def diffusion_fn(self, t: Scalar, xt: Float[Array, 'D']) -> AbstractSquareMatrix:
    """Linear SDE diffusion coefficient L(t, x).

    Returns the diffusion matrix from the underlying linear SDE components.
    """
    return self.components.linear_sde.get_diffusion_coefficient(t, xt)

  def loss_fn(
    self,
    key: PRNGKeyArray,
    data: Union[Float[Array, 'batch_size D'],
                Float[Array, 'batch_size H W C']],
    condition_info: Optional[PyTree] = None,
    *,
    context: Optional[PyTree] = None,
    debug: Optional[bool] = False
  ) -> Tuple[Scalar, Dict[str, Scalar]]:
    """Compute training losses via stochastic interpolation/bridge matching.

    Args:
      key: PRNGKey for sampling bridges and noise.
      data: Batched data; either `(B, D)` vectors or `(B, H, W, C)` images.
      condition_info: Optional batched conditioning with leading batch size `B`.
      context: Optional processed context to pass directly into the NN. If set,
        it takes precedence over `condition_info` internally.
      debug: If true, enables additional internal assertions/debugging.

    Returns:
      (loss, metrics_dict) where `loss` is a scalar and `metrics_dict`
      contains auxiliary statistics (e.g., matching loss, cosine similarity).
    """
    # Wrap NN to thread through optional context as KW arg
    def nn_with_context(t, xt, *, condition_info=None):
      # Prefer explicit context when provided; fall back to condition_info.
      effective_cond = context if context is not None else condition_info
      return self.nn(t, xt, condition_info=effective_cond, context=context)

    losses = stochastic_interpolation_loss(
      self.model_config,
      self.components,
      nn_with_context,
      data,
      condition_info=condition_info,
      key=key,
      debug=debug
    )
    return losses['matching_loss'], losses

  def sample(
    self,
    key: PRNGKeyArray,
    condition_info: Optional[PyTree] = None,
    *,
    context: Optional[PyTree] = None,
    debug: Optional[bool] = False,
    method: Literal['ode', 'sde'] = 'ode',
    noise_schedule: Optional[Callable[[Scalar], AbstractSquareMatrix]] = None,
    solver_params: Optional[Union[SDESolverParams, ODESolverParams]] = None,
    **kwargs
  ) -> Array:
    """Generate a single sample x1 from the model.

    Args:
      key: PRNGKey used for prior and SDE sampling.
      condition_info: Optional conditioning info.
      context: Optional processed context (overrides `condition_info`).
      method: 'ode' for probability-flow ODE or 'sde' for forward SDE sampling.
      noise_schedule: Optional custom diffusion schedule for SDE sampling.
      solver_params: Optional ODE/SDE solver parameters.
      debug: If true, enables additional checks during integration.

    Returns:
      Sampled final state with shape `(D,)` (or flattened image).
    """

    def nn_with_context(t, xt, *, condition_info=None):
      effective_cond = context if context is not None else condition_info
      return self.nn(t, xt, condition_info=effective_cond, context=context)

    if method == 'ode':
      return probability_flow_ode_sample(
        self.model_config,
        self.components,
        nn_with_context,
        condition_info,
        key,
        solver_params=solver_params,
        debug=debug)
    elif method == 'sde':
      return neural_sde_sample(
        self.model_config,
        self.components,
        nn_with_context,
        condition_info,
        key,
        noise_schedule=noise_schedule,
        solver_params=solver_params,
        debug=debug
      )
    else:
      raise ValueError(f"Invalid method: {method}")

  def log_prob(
    self,
    data: Union[Float[Array, 'D'], Float[Array, 'H W C']],
    condition_info: Optional[PyTree] = None,
    key: Optional[PRNGKeyArray] = None,
    *,
    context: Optional[PyTree] = None,
    debug: Optional[bool] = False
  ) -> Scalar:
    """Estimate log p(x1) via integrating the probability-flow ODE backward.

    Args:
      data: Observation at t1; either `(D,)` or `(H, W, C)`.
      condition_info: Optional conditioning info.
      key: Optional PRNGKey for Hutchinson trace estimator; if None, uses
        exact divergence (O(D^2)).
      context: Optional processed context (overrides `condition_info`).
      debug: If true, enables additional checks during integration.

    Returns:
      Scalar log-probability estimate for `data` under the model.
    """
    def nn_with_context(t, xt, *, condition_info=None):
      effective_cond = context if context is not None else condition_info
      return self.nn(t, xt, condition_info=effective_cond, context=context)

    return diffusion_model_log_prob(
      self.model_config,
      self.components,
      nn_with_context,
      data,
      condition_info,
      key=key,
      debug=debug
    ).log_prob

################################################################################################################

class TimeSeriesConditionInfo(AbstractBatchableObject):
  times: Float[Array, 'T']
  seq_length: int = eqx.field(static=True)
  extra_conditioning: Optional[PyTree] = None

  @property
  def batch_size(self) -> Union[None, int, Tuple[int, ...]]:
    if self.times.ndim == 1:
      return None
    elif self.times.ndim == 2:
      return self.times.shape[0]
    elif self.times.ndim >= 3:
      return self.times.shape[:-1]
    else:
      raise ValueError(f"Invalid times shape: {self.times.shape}")

class ImprovedDiffusionTimeSeriesModelConfig(eqx.Module):
  name: str
  sde_type: Literal['brownian', 'ornstein_uhlenbeck']
  nn_type: Literal['time_dependent_gru_rnn', 'time_dependent_transformer', 'time_dependent_wavenet', 'time_dependent_ssm']
  matching_target: Literal['flow', 'drift', 'y1']
  use_multisample_matching: bool

  process_noise: float = 0.1
  condition_length: int = 20
  lambda_: float = 1.0

  # Dynamic fields for NN models (e.g. GRU, Transformer, etc.)
  hidden_size: Optional[int] = None
  n_layers: Optional[int] = None
  intermediate_channels: Optional[int] = None
  cond_in_channels: Optional[int] = None
  in_channels: Optional[int] = None
  out_channels: Optional[int] = None

  # SSM fields
  d_model: Optional[int] = None
  ssm_size: Optional[int] = None
  blocks: Optional[int] = None
  num_layers: Optional[int] = None
  cond_size: Optional[int] = None

class ImprovedDiffusionTimeSeriesModel(AbstractGenerativeModel):
  """Diffusion model over flattened time series with seq2seq neural parameterization.

  This variant treats each time series window `(T, D)` as a vector of length
  `T*D` for the SDE, while using a time-dependent sequence-to-sequence NN to
  predict controls (flow/drift/y1) on structured `(T, D)` data. Supports
  optional conditioning and reusable NN contexts.
  """

  seq_to_seq_nn: AbstractTimeDependentSeq2SeqModel
  components: DiffusionModelComponents = eqx.field(static=True)
  model_config: ImprovedDiffusionTimeSeriesModelConfig = eqx.field(static=True)
  dataset_config: Any = eqx.field(static=True)

  @property
  def batch_size(self):
    return self.seq_to_seq_nn.batch_size

  def __init__(self, model_config: Any, dataset_config: Any, random_seed: int):

    # We flatten the data into a single dimension for the SDE
    dim = dataset_config.n_features*dataset_config.seq_length

    if model_config.sde_type == 'brownian':
      sde = BrownianMotion(sigma=model_config.process_noise, dim=dim)
    elif model_config.sde_type == 'ornstein_uhlenbeck':
      sde = OrnsteinUhlenbeck(sigma=model_config.process_noise, lambda_=model_config.lambda_, dim=dim)
    else:
      raise ValueError(f"Invalid SDE type: {model_config.sde_type}")

    evidence_cov = DiagonalMatrix.eye(dim)*0.001
    t0 = jnp.array(0.0)
    t1 = jnp.array(1.0)
    xt0_prior = StandardGaussian(jnp.zeros(dim), evidence_cov.set_eye())
    self.components = DiffusionModelComponents(sde, t0, xt0_prior, t1, evidence_cov)

    # assert config.model.nn_type in ['time_dependent_gru_rnn', 'time_dependent_transformer', 'time_dependent_wavenet']

    self.seq_to_seq_nn = get_nn(model_config, dataset_config, random_seed)
    self.model_config = model_config
    self.dataset_config = dataset_config

  def nn(self, t: Scalar, xt: Float[Array, '(T D)'], condition_info: TimeSeriesConditionInfo, *, context: Optional[PyTree] = None) -> Float[Array, 'D']:
    """Evaluate the time-dependent seq2seq NN on a flattened series state.

    Args:
      t: Scalar time in [t0, t1].
      xt: Flattened series state `(T*D,)`.
      condition_info: TimeSeriesConditionInfo bundling times, seq_length, and
        optional extra conditioning.
      context: Optional processed context to forward to the NN.

    Returns:
      Flattened NN output with shape `(T*D,)`.
    """
    assert isinstance(condition_info, TimeSeriesConditionInfo)
    times = condition_info.times
    values = einops.rearrange(xt, '... (T D) -> ... T D', T=condition_info.seq_length)
    series = TimeSeries(times=times, values=values)
    out: Float[Array, 'T D'] = self.seq_to_seq_nn(t, series, condition_info=condition_info.extra_conditioning, context=context)
    out_flat = einops.rearrange(out, '... T D -> ... (T D)')
    return out_flat

  def create_context(self, condition_info: TimeSeriesConditionInfo) -> Optional[PyTree]:
    """Create a processed context from `TimeSeriesConditionInfo`.

    Expects `condition_info.extra_conditioning` to be a `TimeSeries` aligned
    with `condition_info.times`. Delegates to the underlying seq-to-seq model's
    `create_context` when available.
    """
    extra = condition_info.extra_conditioning
    if extra is None:
      return None
    if not isinstance(extra, TimeSeries):
      raise ValueError("extra_conditioning must be a TimeSeries for create_context")
    if hasattr(self.seq_to_seq_nn, 'create_context') and callable(getattr(self.seq_to_seq_nn, 'create_context')):
      return self.seq_to_seq_nn.create_context(extra)
    return extra

  def drift_fn(self, t: Scalar, xt: Float[Array, '(T D)'], condition_info: TimeSeriesConditionInfo, *, context: Optional[PyTree] = None) -> Float[Array, 'D']:
    """Neural drift field for time-series SDE over flattened states.

    See `DiffusionModel.drift_fn` for argument semantics.
    """
    matching_target = self.model_config.matching_target
    nn_output = self.nn(t, xt, condition_info=condition_info, context=context)
    conversions = DiffusionModelConversions(self.components, t)
    if matching_target == "y1":
      drift = conversions.y1_to_drift(nn_output, xt)
    elif matching_target == "drift":
      drift = nn_output
    elif matching_target == "flow":
      drift = conversions.flow_to_drift(xt, nn_output)
    else:
      raise ValueError(f"Invalid matching target: {self.model_config.matching_target}")
    return drift

  def diffusion_fn(self, t: Scalar, xt: Float[Array, '(T D)']) -> AbstractSquareMatrix:
    """Linear SDE diffusion for the flattened time-series state."""
    return self.components.linear_sde.get_diffusion_coefficient(t, xt)

  def loss_fn(
    self,
    key: PRNGKeyArray,
    data: Annotated[TimeSeries, 'batch_size'],
    condition_info: Optional[PyTree] = None,
    *,
    context: Optional[PyTree] = None,
    debug: Optional[bool] = False
  ) -> Tuple[Scalar, Dict[str, Scalar]]:
    """Compute training losses on batched `TimeSeries` via bridge matching.

    Args:
      key: PRNGKey for sampling bridges and noise.
      data: Batched `TimeSeries` with shapes `(B, T)` and `(B, T, D)`.
      condition_info: Optional extra conditioning (batched or per-sample).
      context: Optional processed context to pass directly to the NN.
      debug: If true, enables additional assertions.

    Returns:
      (loss, metrics_dict) as described in `DiffusionModel.loss_fn`.
    """

    flat_data = einops.rearrange(data.values, '... T D -> ... (T D)')

    wrapped_condition_info = TimeSeriesConditionInfo(
      times=data.times,
      seq_length=data.values.shape[-2],
      extra_conditioning=condition_info
    )

    def nn_with_context(t, xt, *, condition_info=None):
      return self.nn(t, xt, condition_info=condition_info, context=context)

    losses = stochastic_interpolation_loss(
      self.model_config,
      self.components,
      nn_with_context,
      flat_data,
      condition_info=wrapped_condition_info,
      key=key,
      debug=debug
    )
    return losses['matching_loss'], losses

  def sample(
    self,
    key: PRNGKeyArray,
    times: Optional[Float[Array, 'T']] = None,
    condition_info: Optional[PyTree] = None,
    *,
    context: Optional[PyTree] = None,
    debug: Optional[bool] = False,
    method: Optional[Literal['ode', 'sde']] = None,
    noise_schedule: Optional[Callable[[Scalar], AbstractSquareMatrix]] = None,
    solver_params: Optional[Union[SDESolverParams, ODESolverParams]] = None,
    **kwargs
  ) -> TimeSeries:
    """Generate a `TimeSeries` sample with specified times or from context.

    If `times` is None, `condition_info` must be a `TimeSeriesConditionInfo`
    whose `times` field defines the evaluation grid.
    """

    if method is None:
      if self.model_config.matching_target == "flow":
        method = 'ode'
      else:
        method = 'sde'

    if times is None:
      assert condition_info is not None
      if isinstance(condition_info, TimeSeriesConditionInfo) == False:
        raise ValueError("Condition info must be a TimeSeriesConditionInfo if times is not provided")
      times = condition_info.times
      wrapped_condition_info = condition_info

    else:
      assert times.ndim == 1

      wrapped_condition_info = TimeSeriesConditionInfo(
        times=times,
        seq_length=times.shape[-1],
        extra_conditioning=condition_info
      )

    def nn_with_context(t, xt, *, condition_info=None):
      return self.nn(t, xt, condition_info=condition_info, context=context)

    if method == 'ode':
      out_values = probability_flow_ode_sample(
        self.model_config,
        self.components,
        nn_with_context,
        wrapped_condition_info,
        key,
        solver_params=solver_params,
        debug=debug)
    elif method == 'sde':
      out_values = neural_sde_sample(
        self.model_config,
        self.components,
        nn_with_context,
        wrapped_condition_info,
        key,
        noise_schedule=noise_schedule,
        solver_params=solver_params,
        debug=debug
      )
    else:
      raise ValueError(f"Invalid method: {method}")

    out_values_reshaped = einops.rearrange(out_values, '... (T D) -> ... T D', T=times.shape[-1])
    out_series = TimeSeries(times=times, values=out_values_reshaped)
    return out_series

  def log_prob(
    self,
    data: TimeSeries,
    condition_info: Optional[PyTree] = None,
    key: Optional[PRNGKeyArray] = None,
    *,
    context: Optional[PyTree] = None,
    debug: Optional[bool] = False
  ) -> Scalar:
    """Estimate log p(series) by integrating the probability-flow ODE.

    Args mirror `DiffusionModel.log_prob`, with `data` supplied as a
    `TimeSeries`.
    """

    flat_data = einops.rearrange(data.values, '... T D -> ... (T D)')

    wrapped_condition_info = TimeSeriesConditionInfo(
      times=data.times,
      seq_length=data.values.shape[-2],
      extra_conditioning=condition_info
    )

    def nn_with_context(t, xt, *, condition_info=None):
      return self.nn(t, xt, condition_info=condition_info, context=context)

    return diffusion_model_log_prob(
      self.model_config,
      self.components,
      nn_with_context,
      flat_data,
      condition_info=wrapped_condition_info,
      key=key,
      debug=debug
    ).log_prob
