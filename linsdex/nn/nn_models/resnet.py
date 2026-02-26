from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from linsdex.nn.nn_layers.layers import *
from linsdex.nn.nn_layers.resnet_blocks import GatedResBlock
from linsdex import AbstractBatchableObject, auto_vmap
from linsdex.nn.nn_models.nn_abstract import AbstractTimeDependentNeuralNet, AbstractHyperParams, AbstractNeuralNet
from linsdex.nn.nn_layers.time_condition import TimeFeatures

__all__ = ['ResNet',
           'ResNetHypers',
           'TimeDependentResNet',
           'TimeDependentResNetHypers']

################################################################################################################

class ResNetHypers(AbstractHyperParams):
  working_size: int # Hidden size of recurrent part of the GRU
  hidden_size: int # Number of layers in the GRU
  n_blocks: int # Number of layers in the GRU
  filter_shape: Optional[Tuple[int]] # Dimensionality of the intermediate sequence
  groups: Optional[int] # Number of layers in the GRU
  activation: Callable # Number of layers in the GRU

class ResNet(AbstractNeuralNet):
  """ResNet for 1d data"""

  blocks: tuple[GatedResBlock, ...]
  in_projection: WeightNormDense
  out_projection: WeightNormDense

  hypers: ResNetHypers = eqx.field(static=True)
  input_shape: Union[int, Tuple[int, int, int]] = eqx.field(static=True)
  cond_shape: Optional[Tuple[int]] = eqx.field(static=True)

  def __init__(self,
               input_shape: Union[Tuple[int], Tuple[int, int, int]],
               out_size: int,
               *,
               cond_shape: Optional[Tuple[int]] = None,
               hypers: ResNetHypers,
               key: PRNGKeyArray):
    """**Arguments**:

    - `input_shape`: The input size.  Output size is the same as input_shape.
    - `out_size`: The output size.  For images, this is the number of output
                  channels.
    - `cond_shape`: The size of the conditioning information.
    - `hypers`: The hyperparameters for the ResNet.
    - `key`: A `jax.random.PRNGKey` for initialization.
    """
    self.hypers = hypers

    if len(input_shape) not in [1, 3]:
      raise ValueError(f'Expected 1d or 3d input shape')

    image = False
    if len(input_shape) == 3:
      H, W, C = input_shape
      image = True
      assert hypers.filter_shape is not None, 'Must pass in filter shape when processing images'

    k1, k2, k3 = random.split(key, 3)

    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    self.input_shape = input_shape
    self.cond_shape = cond_shape

    if image == False:
      self.in_projection = WeightNormDense(in_size=input_shape[0],
                                          out_size=hypers.working_size,
                                          key=k1)
      working_shape = (hypers.working_size,)
    else:
      self.in_projection = ConvAndGroupNorm(input_shape=input_shape,
                                        out_size=hypers.working_size,
                                        filter_shape=hypers.filter_shape,
                                        groups=1,
                                        key=k1)
      working_shape = (H, W, hypers.working_size)

    def make_resblock(k):
      return GatedResBlock(input_shape=working_shape,
                          hidden_size=hypers.hidden_size,
                          groups=hypers.groups,
                          cond_shape=cond_shape,
                          activation=hypers.activation,
                          filter_shape=hypers.filter_shape,
                          key=k)

    keys = random.split(k2, hypers.n_blocks)
    self.blocks = eqx.filter_vmap(make_resblock)(keys)

    if image == False:
      self.out_projection = WeightNormDense(in_size=hypers.working_size,
                                            out_size=out_size,
                                            key=k3)
    else:
      self.out_projection = ConvAndGroupNorm(input_shape=working_shape,
                                           out_size=out_size,
                                           filter_shape=hypers.filter_shape,
                                           groups=1,
                                           key=k3)

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    return self.in_projection.batch_size

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'

    k1, k2, k3 = random.split(key, 3)

    # Input projection
    in_proj = self.in_projection.data_dependent_init(x, key=k1)
    x = eqx.filter_vmap(in_proj)(x)

    # Scan over the vmapped blocks
    params, state = eqx.partition(self.blocks, eqx.is_array)
    def scan_body(x, inputs):
      key, block_params = inputs
      block = eqx.combine(block_params, state)
      new_block = block.data_dependent_init(x, y, key=key)
      new_x = eqx.filter_vmap(new_block)(x, y)
      new_params, _ = eqx.partition(block, eqx.is_array)
      return new_x, new_params

    keys = random.split(k2, self.hypers.n_blocks)
    x, params = jax.lax.scan(scan_body, x, (keys, params))
    blocks = eqx.combine(params, state)

    out_proj = self.out_projection.data_dependent_init(x, key=k3)

    # Turn the new parameters into a new module
    get_in_proj = lambda tree: tree.in_projection
    get_blocks = lambda tree: tree.blocks
    get_out_proj = lambda tree: tree.out_projection

    updated_layer = eqx.tree_at(get_in_proj, self, in_proj)
    updated_layer = eqx.tree_at(get_blocks, updated_layer, blocks)
    updated_layer = eqx.tree_at(get_out_proj, updated_layer, out_proj)

    return updated_layer

  def __call__(
    self,
    x: Array,
    condition_info: Optional[Array] = None
  ) -> Array:
    """**Arguments:**

    - `x`: A JAX array with shape `(input_shape,)`.
    - `condition_info`: A JAX array with shape `(cond_shape,)`.

    **Returns:**

    A JAX array with shape `(input_shape,)`.
    """
    assert x.shape == self.input_shape

    # Input projection
    x = self.in_projection(x)

    # Resnet blocks
    dynamic, static = eqx.partition(self.blocks, eqx.is_array)

    def f(x, params):
        block = eqx.combine(params, static)
        return block(x, condition_info), None

    out, _ = jax.lax.scan(f, x, dynamic)

    # Output projection
    return self.out_projection(out)

################################################################################################################

class TimeDependentResNetHypers(AbstractHyperParams):
  working_size: int # Hidden size of recurrent part of the GRU
  hidden_size: int # Number of layers in the GRU
  n_blocks: int # Number of layers in the GRU
  filter_shape: Optional[Tuple[int]] # Dimensionality of the intermediate sequence
  groups: Optional[int] # Number of layers in the GRU
  activation: Callable # Number of layers in the GRU
  embedding_size: int # Number of layers in the GRU
  out_features: int # Number of time embedding features

class TimeDependentResNet(AbstractTimeDependentNeuralNet):
  """A time dependent version of a 1d resnet
  """

  blocks: tuple[GatedResBlock, ...]
  in_projection: Union[WeightNormDense, ConvAndGroupNorm]
  out_projection: Union[WeightNormDense, ConvAndGroupNorm]
  time_features: TimeFeatures

  hypers: TimeDependentResNetHypers = eqx.field(static=True)
  input_shape: Union[Tuple[int], Tuple[int, int, int]] = eqx.field(static=True)
  cond_shape: Optional[Tuple[int]] = eqx.field(static=True)

  def __init__(self,
               input_shape: Union[Tuple[int], Tuple[int, int, int]],
               out_size: int,
               *,
               cond_shape: Optional[Tuple[int]] = None,
               hypers: TimeDependentResNetHypers,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input size.
    - `out_size`: The output size.
    - `cond_shape`: The size of the conditioning information.
    - `hypers`: The hyperparameters for the TimeDependentResNet.
    - `key`: A `jax.random.PRNGKey` for initialization.
    """
    self.hypers = hypers

    if len(input_shape) not in [1, 3]:
      raise ValueError(f'Expected 1d or 3d input shape')

    image = False
    if len(input_shape) == 3:
      H, W, C = input_shape
      image = True
      assert hypers.filter_shape is not None, 'Must pass in filter shape when processing images'

    k1, k2, k3, k4 = random.split(key, 4)

    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    self.input_shape = input_shape

    self.time_features = TimeFeatures(embedding_size=hypers.embedding_size,
                                      out_features=hypers.out_features,
                                      key=k1,
                                      **kwargs)

    total_cond_size = hypers.out_features
    if cond_shape is not None:
      if len(cond_shape) != 1:
        raise ValueError(f'Expected 1d conditional input.')
      total_cond_size += cond_shape[0]

    self.cond_shape = (total_cond_size,)

    if image == False:
      self.in_projection = WeightNormDense(in_size=input_shape[0],
                                          out_size=hypers.working_size,
                                          key=k2)
      working_shape = (hypers.working_size,)
    else:
      self.in_projection = ConvAndGroupNorm(input_shape=input_shape,
                                        out_size=hypers.working_size,
                                        filter_shape=hypers.filter_shape,
                                        groups=1,
                                        key=k2)
      working_shape = (H, W, hypers.working_size)

    def make_resblock(k):
      return GatedResBlock(input_shape=working_shape,
                          hidden_size=hypers.hidden_size,
                          groups=hypers.groups,
                          cond_shape=self.cond_shape,
                          activation=hypers.activation,
                          filter_shape=hypers.filter_shape,
                          key=k)

    keys = random.split(k3, hypers.n_blocks)
    self.blocks = eqx.filter_vmap(make_resblock)(keys)

    if image == False:
      self.out_projection = WeightNormDense(in_size=hypers.working_size,
                                            out_size=out_size,
                                            key=k4)
    else:
      self.out_projection = ConvAndGroupNorm(input_shape=working_shape,
                                           out_size=out_size,
                                           filter_shape=hypers.filter_shape,
                                           groups=1,
                                           key=k4)

  @property
  def batch_size(self) -> Union[None, int, Tuple[int]]:
    return self.in_projection.batch_size

  def data_dependent_init(self,
                          t: Array,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `t`: The time to initialize the parameters with.
    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert t.ndim == 1
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'

    k1, k2, k3 = random.split(key, 3)

    # Prepare conditioning information
    h = eqx.filter_vmap(self.time_features)(t)
    if y is not None:
      h = jnp.concatenate([h, y], axis=-1)

    # Input projection
    in_proj = self.in_projection.data_dependent_init(x, key=k1)
    x = eqx.filter_vmap(in_proj)(x)

    # Scan over the vmapped blocks
    params, state = eqx.partition(self.blocks, eqx.is_array)
    def scan_body(x, inputs):
      key, block_params = inputs
      block = eqx.combine(block_params, state)
      new_block = block.data_dependent_init(x, h, key=key)
      new_x = eqx.filter_vmap(new_block)(x, h)
      new_params, _ = eqx.partition(block, eqx.is_array)
      return new_x, new_params

    keys = random.split(k2, self.hypers.n_blocks)
    x, params = jax.lax.scan(scan_body, x, (keys, params))
    blocks = eqx.combine(params, state)

    out_proj = self.out_projection.data_dependent_init(x, key=k3)

    # Turn the new parameters into a new module
    get_in_proj = lambda tree: tree.in_projection
    get_blocks = lambda tree: tree.blocks
    get_out_proj = lambda tree: tree.out_projection

    updated_layer = eqx.tree_at(get_in_proj, self, in_proj)
    updated_layer = eqx.tree_at(get_blocks, updated_layer, blocks)
    updated_layer = eqx.tree_at(get_out_proj, updated_layer, out_proj)

    return updated_layer

  def __call__(self,
               t: Array,
               x: Array,
               y: Optional[Array] = None,
               **kwargs) -> Array:
    """**Arguments:**

    - `t`: A JAX array with shape `()`.
    - `x`: A JAX array with shape `(input_shape,)`.
    - `y`: A JAX array with shape `(cond_shape,)`.

    **Returns:**

    A JAX array with shape `(input_shape,)`.
    """
    t = jnp.array(t)
    assert t.shape == ()
    assert x.shape == self.input_shape

    # Prepare conditioning information
    h = self.time_features(t)
    if y is not None:
      h = jnp.concatenate([h, y], axis=-1)

    # Input projection
    x = self.in_projection(x)

    # Resnet blocks
    dynamic, static = eqx.partition(self.blocks, eqx.is_array)

    def f(x, params):
        block = eqx.combine(params, static)
        return block(x, h), None

    out, _ = jax.lax.scan(f, x, dynamic)

    # Output projection
    return self.out_projection(out)
