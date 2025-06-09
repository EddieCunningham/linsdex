"""Time series representations and operations for diffusion-based models.

This module provides a flexible framework for representing and manipulating time series data,
especially in the context of diffusion models. The core class is TimeSeries, which represents
a sequence of observations over time, with support for batching, missing values, and
visualization.

The TimeSeries class inherits from AbstractBatchableObject, which provides common
functionality for objects that can be batched along one or more dimensions.
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterable, Literal, List, Annotated
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject
import numpy as np

__all__ = ['TimeSeries']

def _make_windowed_batches(obj, window_size: int, T: Optional[int] = None):
  """Create batched windows from a single TimeSeries object.

  This function takes a TimeSeries (or compatible object) and creates a batch of
  new TimeSeries objects by sliding a window of fixed size across the original series.

  Args:
    obj: The TimeSeries object to create windows from
    window_size: Size of each window
    T: Optional length of the series (will be inferred if not provided)

  Returns:
    A new TimeSeries with batched windows
  """
  if T is None:
    try:
      T = len(obj)
    except:
      T = obj.batch_size
  idx = jnp.arange(T - window_size + 1)[:, None] + jnp.arange(window_size)[None, :]
  return obj[idx]

################################################################################################################

class TimeSeries(AbstractBatchableObject):
  """A representation of time series data with support for batching and missing values.

  TimeSeries is the core data structure for representing temporal sequences in the
  diffusion-CRF framework. It combines three key components:

  1. times: Time points at which observations are made
  2. values: Observed values at each time point (potentially multi-dimensional)
  3. mask: Boolean mask indicating which time points have valid observations

  The class is designed to efficiently handle batches of time series, missing values,
  and provides utilities for visualization and windowing operations.

  Attributes:
    times: Array of time points
    values: Array of observed values at each time point (with feature dimension)
    mask: Boolean mask indicating which time points have observations
  """

  times: Float[Array, 'N']
  values: Float[Array, 'N D']
  mask: Bool[Array, 'N']  # True if an observation is present at that time step

  def __init__(self,
               times: Float[Array, 'N'],
               values: Float[Array, 'N D'],
               mask: Optional[Union[Bool[Array, 'N'], None]] = None):
    """Initialize a TimeSeries object.

    Args:
      times: Array of time points with shape (..., N) where ... represents optional batch dimensions
      values: Array of values with shape (..., N, D) where D is the feature dimension
      mask: Boolean mask with shape (..., N) indicating which time points have observations.
            If None, assumes all time points are observed

    Raises:
      ValueError: If the shapes of times and values are incompatible
    """
    if mask is None:
      mask = jnp.ones_like(times, dtype=bool)

    assert mask.shape == times.shape

    times_batch_size = times.shape[:-1]

    assert isinstance(values, jnp.ndarray) or isinstance(values, np.ndarray)
    values = jnp.array(values)
    if values.ndim != times.ndim + 1:
      raise ValueError(f"Values must have {times.ndim + 1} dimensions, got {values.ndim}")

    if times_batch_size != values.shape[:-2]:
      raise ValueError(f"Times and values must have the same batch size, got {times_batch_size} and {values.shape[:-2]}")

    if times.shape[-1] != values.shape[-2]:
      raise ValueError(f"Times and values must have the same number of time points, got {times.shape[-1]} and {values.shape[-2]}")

    self.times = times
    self.values = values
    self.mask = mask

  @property
  def batch_size(self):
    """Get the batch dimensions of this TimeSeries.

    Returns:
      - None if not batched (i.e., a single time series)
      - An integer if batched along a single dimension
      - A tuple of integers if batched along multiple dimensions
    """
    if self.times.ndim == 1:
      return None
    elif self.times.ndim == 2:
      return self.times.shape[0]
    else:
      return self.times.shape[:-1]

  @property
  def dim(self):
    return self.values.shape[-1]

  def __len__(self):
    """Return the length of the time series (number of time points).

    Returns:
      Integer length of the time series
    """
    return self.times.shape[-1]

  def is_fully_uncertain(self):
    """Return a mask for time points with no observations.

    Returns:
      Boolean array where True indicates no observation at that time point
    """
    # 0 if there is any observed dimension at that time step, 1 otherwise
    return ~self.mask

  def get_missing_observation_mask(self) -> Bool[Array, 'N']:
    """Return a mask identifying time points with missing observations.

    Returns:
      Boolean array where True indicates missing observation at that time point
    """
    return self.is_fully_uncertain()

  def make_windowed_batches(self, window_size: int):
    """Create a batch of time series from windows of the current series.

    This method creates a batch of TimeSeries objects by sliding a fixed-size window
    over the current series. Each window becomes a separate item in the batch.

    Args:
      window_size: Size of each window

    Returns:
      A new TimeSeries object with an added batch dimension for the windows
    """
    return _make_windowed_batches(self, window_size)

  def plot(self,
                index: Optional[Union[int, Literal['all']]] = None,
                axes: Optional[List] = None,
                fig: Optional['plt.Figure'] = None,
                show_plot: bool = True,
                add_title: bool = True,
                title: Optional[str] = None,
                line_colors: Optional[Union[str, List[str]]] = 'blue',
                line_alpha: float = 0.7,
                line_width: float = 1,
                marker_colors: Optional[Union[str, List[str]]] = None,
                marker_size: float = 25,
                marker_style: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None,
                fig_width: float = 6,
                fig_height_factor: float = 3,
                legend_loc: str = 'upper center',
                batch_color: str = 'blue',
                batch_line_width: float = 0.5,
                min_alpha: float = 0.1,
                max_alpha: float = 1.0,
                alpha_scaling: Literal['linear', 'sqrt', 'log'] = 'sqrt'):
    """Visualize the time series data.

    Creates a visualization of the time series with one subplot per dimension.
    For batched time series, can either plot a specific item or overlay all items.

    Args:
      index: Index of batch item to plot, or 'all' to overlay all items (default: None, same as 'all' if batched)
      axes: Optional matplotlib axes to plot on (if None, new axes will be created)
      fig: Optional matplotlib figure to plot on (if None, a new figure will be created)
      show_plot: Whether to call plt.show() after creating the plot
      add_title: Whether to add a title to the plot
      title: Optional custom title string
      line_colors: Color(s) for the connecting lines
      line_alpha: Alpha (transparency) for the connecting lines
      line_width: Width of the connecting lines
      marker_colors: Color(s) for the markers (defaults to line_colors if None)
      marker_size: Size of the markers
      marker_style: Marker style (e.g., 'o', 's', '^'). If None, no markers are shown.
      figsize: Optional figure size as (width, height) in inches
      fig_width: Width of the figure in inches (used if figsize is None)
      fig_height_factor: Height multiplier per dimension
      legend_loc: Location for the legend
      batch_color: Color for all samples
      batch_line_width: Line width for all samples
      min_alpha: Minimum transparency value for batched samples
      max_alpha: Maximum transparency value for batched samples
      alpha_scaling: Method to scale transparency with batch size ('linear', 'sqrt', 'log')

    Returns:
      tuple: (fig, axes) - The matplotlib figure and axes objects
    """
    from linsdex.series.plot import plot_series
    return plot_series(self,
                index,
                axes,
                fig,
                show_plot,
                add_title,
                title,
                line_colors,
                line_alpha,
                line_width,
                marker_colors,
                marker_size,
                marker_style,
                figsize,
                fig_width,
                fig_height_factor,
                legend_loc,
                batch_color,
                batch_line_width,
                min_alpha,
                max_alpha,
                alpha_scaling)

  @staticmethod
  def plot_multiple_series(series_list: List['TimeSeries'],
                          index: Optional[Union[int, Literal['all']]] = 'all',
                          titles: Optional[List[str]] = None,
                          show_plot: bool = True,
                          common_title: Optional[str] = None,
                          line_colors: Optional[Union[str, List[str]]] = 'blue',
                          line_alpha: float = 0.7,
                          line_width: float = 1,
                          marker_colors: Optional[Union[str, List[str]]] = None,
                          marker_size: float = 25,
                          marker_style: Optional[str] = None,
                          figsize: Optional[Tuple[float, float]] = None,
                          width_per_series: float = 6,
                          height_per_dim: float = 3,
                          batch_color: str = 'blue',
                          batch_line_width: float = 0.5,
                          min_alpha: float = 0.1,
                          max_alpha: float = 1.0,
                          alpha_scaling: Literal['linear', 'sqrt', 'log'] = 'sqrt',
                          use_max_dims: bool = False):
    """Create side-by-side plots of multiple TimeSeries objects for comparison.

    This method arranges multiple time series in columns, with each row
    representing a dimension of the data. This is useful for comparing
    different models, predicted vs. observed data, etc.

    Args:
      series_list: List of TimeSeries objects to compare
      index: Index of batch item to plot, or 'all' to overlay all items
      titles: Optional list of titles for each series (column)
      show_plot: Whether to call plt.show() after creating the plot
      common_title: Optional overall title for the plot
      line_colors: Color(s) for the connecting lines
      line_alpha: Alpha (transparency) for the connecting lines
      line_width: Width of the connecting lines
      marker_colors: Color(s) for the markers (defaults to line_colors if None)
      marker_size: Size of the markers
      marker_style: Marker style (e.g., 'o', 's', '^'). If None, no markers are shown.
      figsize: Optional figure size as (width, height) in inches
      width_per_series: Width in inches allocated for each series column
      height_per_dim: Height in inches allocated for each dimension row
      batch_color: Color for all samples
      batch_line_width: Line width for all samples
      min_alpha: Minimum transparency value for batched samples
      max_alpha: Maximum transparency value for batched samples
      alpha_scaling: Method to scale transparency with batch size ('linear', 'sqrt', 'log')
      use_max_dims: Whether to plot up to the maximum number of dimensions (True) or
                   only dimensions present in all series (False)

    Returns:
      tuple: (fig, axes) - The matplotlib figure and axes objects
    """
    from linsdex.series.plot import plot_multiple_series
    return plot_multiple_series(series_list,
                         index,
                         titles,
                         show_plot,
                         common_title,
                         line_colors,
                         line_alpha,
                         line_width,
                         marker_colors,
                         marker_size,
                         marker_style,
                         figsize,
                         width_per_series,
                         height_per_dim,
                         batch_color,
                         batch_line_width,
                         min_alpha,
                         max_alpha,
                         alpha_scaling,
                         use_max_dims)

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import jax
  import jax.numpy as jnp
  import jax.random as random
  import matplotlib.pyplot as plt

  import pickle
  # data = pickle.load(open('data_dump.pkl', 'rb'))
  # ts, values, mask = data['ts'], data['yts'], data['observation_mask']

  data = pickle.load(open('series.pkl', 'rb'))
  ts, values, mask = data.times, data.values, data.observation_mask

  mask = jnp.any(mask, axis=-1)
  series = TimeSeries(ts, values, mask)
  series.plot()
  import pdb; pdb.set_trace()