import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterable, Literal, List, Annotated, Dict
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import jax.tree_util as jtu
from linsdex.series.batchable_object import AbstractBatchableObject
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Helper functions for plotting

def _calculate_alpha(batch_size: int,
                   min_alpha: float = 0.1,
                   max_alpha: float = 1.0,
                   alpha_scaling: Literal['linear', 'sqrt', 'log'] = 'sqrt') -> float:
  """Calculate appropriate alpha value based on batch size.

  Args:
    batch_size: The number of samples in the batch
    min_alpha: Minimum alpha value
    max_alpha: Maximum alpha value
    alpha_scaling: Method to scale alpha with batch size ('linear', 'sqrt', 'log')

  Returns:
    Appropriate alpha value scaled by batch size
  """
  if batch_size <= 1:
    return max_alpha

  if alpha_scaling == 'linear':
    alpha = max_alpha / batch_size
  elif alpha_scaling == 'sqrt':
    alpha = max_alpha / math.sqrt(batch_size)
  elif alpha_scaling == 'log':
    alpha = max_alpha / (1 + math.log(batch_size))
  else:
    alpha = (min_alpha + max_alpha) / 2

  return max(min_alpha, min(max_alpha, alpha))

def _compute_y_ranges(values_list: List[np.ndarray],
                     masks_list: List[np.ndarray],
                     num_dims: int) -> List[Tuple[float, float]]:
  """Compute y-axis ranges for all dimensions, handling NaN and Inf values.

  Args:
    values_list: List of value arrays to consider
    masks_list: List of mask arrays corresponding to each value array
    num_dims: Number of dimensions to compute ranges for

  Returns:
    List of (min, max) tuples for each dimension
  """
  y_ranges = []

  for k in range(num_dims):
    all_observed_values = []

    for values, mask in zip(values_list, masks_list):
      if k < values.shape[1]:  # Check if this dimension exists
        # Get observed values for this dimension
        observed_values = values[:, k][mask]
        # Filter out NaN and Inf values
        observed_values = observed_values[np.isfinite(observed_values)]
        if len(observed_values) > 0:
          all_observed_values.extend(observed_values)

    if all_observed_values and len(all_observed_values) > 0:
      # Filter out any remaining NaN or Inf values
      finite_values = [v for v in all_observed_values if np.isfinite(v)]
      if len(finite_values) > 0:
        y_min = np.min(finite_values)
        y_max = np.max(finite_values)
        # Add a small buffer
        buffer = 0.1 * (y_max - y_min) if y_max > y_min else 0.1 * abs(y_max) if y_max != 0 else 0.1
        y_ranges.append((y_min - buffer, y_max + buffer))
      else:
        y_ranges.append((-1, 1))  # Default range if no finite data
    else:
      y_ranges.append((-1, 1))  # Default range if no data

  return y_ranges

def _style_axes(ax: plt.Axes,
               k: int,
               num_dims: int,
               is_first_col: bool = True,
               format_y_axis: bool = True) -> None:
  """Apply consistent styling to plot axes.

  Args:
    ax: The matplotlib axes to style
    k: The current dimension index
    num_dims: Total number of dimensions
    is_first_col: Whether this is the first column (for y-axis labels)
    format_y_axis: Whether to format the y-axis ticks
  """
  # Format y-axis ticks
  if format_y_axis:
    formatter = FuncFormatter(lambda x, pos: f"{x:.2f}")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(formatter)

  # Style y-tick labels
  ax.tick_params(axis='y', labelsize=8)
  ax.yaxis.set_tick_params(pad=1)
  for label in ax.get_yticklabels():
    label.set_horizontalalignment('right')

  # Handle x-tick labels visibility
  if k < num_dims - 1:
    plt.setp(ax.get_xticklabels(), visible=False)
  else:
    ax.tick_params(axis='x', labelsize=8)
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set_xlabel('Time', fontsize=10)

  # Add y-axis label (only for first column typically)
  if is_first_col:
    ax.set_ylabel(f"Dim {k}", fontsize=10)
    ax.yaxis.labelpad = 10

  # Ensure tick marks are visible
  ax.xaxis.set_tick_params(which='both', size=4, width=1, direction='out')

  # Remove any existing legend
  if ax.get_legend() is not None:
    ax.legend().remove()

def _set_y_limits(ax: plt.Axes,
                 k: int,
                 y_ranges: List[Tuple[float, float]]) -> None:
  """Set y-axis limits, handling cases where limits might not be finite.

  Args:
    ax: The matplotlib axes to set limits on
    k: The current dimension index
    y_ranges: List of (min, max) tuples for each dimension
  """
  if k < len(y_ranges):
    y_min, y_max = y_ranges[k]
    # Sanity check to ensure limits are finite
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
      y_min, y_max = -1, 1
    ax.set_ylim(y_min, y_max)
  else:
    ax.set_ylim(-1, 1)  # Default range

def _plot_timeseries(ax: plt.Axes,
                    times: np.ndarray,
                    values: np.ndarray,
                    mask: np.ndarray,
                    k: int,
                    color: str,
                    line_width: float,
                    alpha: float,
                    marker_style: Optional[str] = None,
                    marker_color: Optional[str] = None,
                    marker_size: float = 25,
                    label: Optional[str] = None) -> Optional[plt.Line2D]:
  """Plot a single time series for one dimension with optional markers.

  Args:
    ax: The matplotlib axes to plot on
    times: Time points array
    values: Values array
    mask: Observation mask
    k: Dimension index to plot
    color: Line color
    line_width: Line width
    alpha: Line transparency
    marker_style: Optional marker style
    marker_color: Optional marker color (defaults to line color)
    marker_size: Marker size
    label: Optional legend label

  Returns:
    The plotted line object if a label was provided, otherwise None
  """
  # Determine indices of observed points
  observed_indices = np.where(mask)[0]

  if len(observed_indices) <= 0:
    return None

  # Plot the line
  if label:
    line, = ax.plot(times[observed_indices], values[observed_indices, k],
                   color=color, linewidth=line_width, alpha=alpha, label=label)
  else:
    line, = ax.plot(times[observed_indices], values[observed_indices, k],
                   color=color, linewidth=line_width, alpha=alpha)

  # Add markers if specified
  if marker_style is not None:
    marker_color = marker_color if marker_color is not None else color
    ax.scatter(times[observed_indices], values[observed_indices, k],
              color=marker_color, marker=marker_style, s=marker_size)

  return line if label else None

def plot_series(self,
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
  """
  Create a visualization of a TimeSeries object.

  This method generates a plot for each dimension of the time series, showing:
  - Lines for observed values
  - Markers for observation points with the observation mask applied
  - Multiple samples if the TimeSeries is batched

  Args:
    self: The TimeSeries object to visualize
    index: Index of the sequence if the self is batched. If 'all', plot all samples.
            If None and self is batched, will plot all samples.
    axes: Optional list of axes to plot on (if None, new axes will be created)
    fig: Optional figure to plot on (if None, a new figure will be created)
    show_plot: Whether to call plt.show() after creating the plot
    add_title: Whether to add a title to the plot
    title: Optional title string to use instead of default
    line_colors: Color(s) for the connecting lines. Can be a string or a list of colors for multiple samples.
    line_alpha: Alpha (transparency) for the connecting lines
    line_width: Width of the connecting lines
    marker_colors: Color(s) for the markers. If None, will use line_colors.
    marker_size: Size of the markers
    marker_style: Marker style (e.g., 'o', 's', '^'). If None, no markers are shown.
    figsize: Directly specify figure size as (width, height) in inches
    fig_width: Width of the figure in inches (used if figsize is None)
    fig_height_factor: Multiplier for height per dimension (used if figsize is None)
    legend_loc: Location for the legend
    batch_color: Color for all samples (default: blue)
    batch_line_width: Line width for all samples (default: 0.5)
    min_alpha: Minimum alpha value for batched samples (default: 0.1)
    max_alpha: Maximum alpha value for batched samples (default: 1.0)
    alpha_scaling: Method to scale alpha with batch size ('linear', 'sqrt', 'log')

  Returns:
    tuple: (fig, axes) - The figure and axes objects used for the plot
  """
  # Convert marker_colors to match line_colors if not specified
  if marker_colors is None:
    marker_colors = batch_color

  # Check if we're dealing with a batched TimeSeries
  is_batched = self.batch_size is not None and isinstance(self.batch_size, int)
  plot_all_samples = is_batched and (index == 'all' or index is None)

  # Prepare samples to plot
  if plot_all_samples:
    samples_to_plot = [self[i] for i in range(self.batch_size)]
    effective_alpha = _calculate_alpha(self.batch_size, min_alpha, max_alpha, alpha_scaling)
  else:
    if is_batched and index is not None and index != 'all':
      samples_to_plot = [self[index]]
    else:
      samples_to_plot = [self]
    effective_alpha = max_alpha  # Use max_alpha for single samples

  # Get dimensions from first sample
  num_dims = samples_to_plot[0].values.shape[-1]

  # Check if we need to create new axes or use provided ones
  create_new_figure = fig is None or axes is None

  if create_new_figure:
    n_cols = 1
    n_rows = num_dims

    # Calculate figure size
    if figsize is None:
      figsize = (n_cols*fig_width, fig_height_factor*n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)

    # Set a publication-quality font
    plt.rcParams.update({
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'Palatino', 'DejaVu Serif', 'Times'],
      'mathtext.fontset': 'stix',
    })

    # Handle case where there's only one dimension (axes won't be array)
    if n_rows == 1:
      axes = [axes]

    # Add a title to the figure
    if add_title:
      title_text = title if title is not None else "Time Series Plot"
      fig.suptitle(title_text, fontsize=14, y=0.98)
      plt.subplots_adjust(top=0.95)  # Make room for title
  else:
    # Ensure axes is a list for consistent indexing
    if not isinstance(axes, list) and not isinstance(axes, np.ndarray):
      axes = [axes]

  # Prepare values for y-range calculation
  values_list = []
  masks_list = []

  for sample in samples_to_plot:
    values_list.append(np.array(sample.values))
    masks_list.append(np.array(sample.mask))

  # Calculate y-axis ranges
  y_ranges = _compute_y_ranges(values_list, masks_list, num_dims)

  # Track handles for legend
  legend_handles = []
  legend_labels = []

  # Plot each dimension
  for k in range(num_dims):
    ax = axes[k]

    # Plot all samples
    for i, sample in enumerate(samples_to_plot):
      times = np.array(sample.times)
      values = np.array(sample.values)
      mask = np.array(sample.mask)

      # Only add a legend entry for the first dimension and first sample (or all samples case)
      label = None
      if k == 0:
        if plot_all_samples and i == 0:
          label = "All Samples"
        elif not plot_all_samples:
          label = "Observations"

      line = _plot_timeseries(
          ax=ax,
          times=times,
          values=values,
          mask=mask,
          k=k,
          color=batch_color,
          line_width=batch_line_width,
          alpha=effective_alpha,
          marker_style=marker_style,
          marker_color=marker_colors,
          marker_size=marker_size,
          label=label
      )

      if line and label:
          legend_handles.append(line)
          legend_labels.append(label)

    # Set y-axis range and style the axes
    _set_y_limits(ax, k, y_ranges)
    _style_axes(ax, k, num_dims)

  # Add a legend if we have any legend handles
  if create_new_figure and legend_handles:
    fig.legend(handles=legend_handles, labels=legend_labels,
              loc=legend_loc, bbox_to_anchor=(0.5, 0.94),
              ncol=min(2, len(legend_handles)), fontsize=9,
              frameon=True, borderaxespad=0.)

  # Adjust layout
  if create_new_figure:
    plt.tight_layout(rect=[0, 0, 1, 0.95])

  if show_plot and create_new_figure:
    plt.show()
    if not fig._suptitle:  # Only close if not a custom figure we want to reuse
      plt.close()

  return fig, axes

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
  """
  Create side-by-side plots of multiple TimeSeries objects for comparison.

  This function arranges multiple time series in columns, with each row
  representing a dimension of the data.

  Args:
    series_list: List of TimeSeries objects to compare
    index: Index of the sequence if the series are batched. If 'all', plot all
            samples from each batched series overlaid on the same axes.
    titles: Optional list of titles for each series (column)
    show_plot: Whether to call plt.show() after creating the plot
    common_title: Optional overall title for the plot
    line_colors: Color(s) for the connecting lines. Can be a string or list of colors.
    line_alpha: Alpha (transparency) for the connecting lines
    line_width: Width of the connecting lines
    marker_colors: Color(s) for the markers. If None, will use line_colors.
    marker_size: Size of the markers
    marker_style: Marker style (e.g., 'o', 's', '^'). If None, no markers are shown.
    figsize: Directly specify figure size as (width, height) in inches
    width_per_series: Width in inches allocated for each series column
    height_per_dim: Height in inches allocated for each dimension row
    batch_color: Color for all samples (default: blue)
    batch_line_width: Line width for all samples (default: 0.5)
    min_alpha: Minimum alpha value for batched samples (default: 0.1)
    max_alpha: Maximum alpha value for batched samples (default: 1.0)
    alpha_scaling: Method to scale alpha with batch size ('linear', 'sqrt', 'log')
    use_max_dims: Whether to plot up to the maximum number of dimensions across
                  all series (True) or only dimensions present in all series (False)

  Returns:
    tuple: (fig, axes) - The figure and axes objects used for the plots
  """
  if not series_list:
    raise ValueError("No series provided for plotting")

  # Convert marker_colors to match batch_color if not specified
  if marker_colors is None:
    marker_colors = batch_color

  # Check if we need to plot all samples from batched series
  plot_all_samples = index == 'all'

  # Number of columns in the plot (one per series in series_list)
  n_series = len(series_list)

  # Find dimensions across all series, accounting for potential batching
  dims_list = []
  for series in series_list:
    if series.batch_size is not None and isinstance(series.batch_size, int) and plot_all_samples:
      dims_list.append(series[0].values.shape[-1])
    else:
      dims_list.append(series.values.shape[-1])

  # Use either the minimum or maximum dimensions based on use_max_dims flag
  plot_dims = max(dims_list) if use_max_dims else min(dims_list)

  if plot_dims == 0:
    raise ValueError("All series have 0 dimensions, cannot plot.")

  # Set a publication-quality font
  plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Palatino', 'DejaVu Serif', 'Times'],
    'mathtext.fontset': 'stix',
  })

  # Create figure with one column per series in series_list
  if figsize is None:
    figsize = (n_series*width_per_series, plot_dims*height_per_dim)

  fig, axes = plt.subplots(plot_dims, n_series,
                        figsize=figsize,
                        sharex='col', sharey='row')

  # Handle single dimension or single series case
  if plot_dims == 1:
    axes = np.array([axes])
  if n_series == 1:
    axes = axes.reshape(plot_dims, 1)

  # Set column titles if provided
  if titles is not None:
    for i, title in enumerate(titles[:n_series]):
      axes[0, i].set_title(title, fontsize=12)

  # Prepare data for y-range calculation
  all_values_list = []
  all_masks_list = []

  for k in range(plot_dims):
    values_for_dim = []
    masks_for_dim = []

    for i, series in enumerate(series_list):
      is_batched = series.batch_size is not None and isinstance(series.batch_size, int)

      if is_batched and plot_all_samples:
        # Collect from all samples in batch
        for j in range(series.batch_size):
          sample = series[j]
          if k < sample.values.shape[-1]:
            values_for_dim.append(np.array(sample.values))
            masks_for_dim.append(np.array(sample.mask))
      else:
        # Single series or specific index
        current_series = series
        if is_batched and index is not None and index != 'all':
          current_series = series[index]

        if k < current_series.values.shape[-1]:
          values_for_dim.append(np.array(current_series.values))
          masks_for_dim.append(np.array(current_series.mask))

    all_values_list.append(values_for_dim)
    all_masks_list.append(masks_for_dim)

  # Calculate y-axis ranges for each dimension
  y_ranges = []
  for k in range(plot_dims):
    y_ranges.append(_compute_y_ranges(all_values_list[k], all_masks_list[k], 1)[0])

  # Track legend handles and labels
  legend_handles = []
  legend_labels = []

  # Plot each series
  for i, series in enumerate(series_list):
    is_batched = series.batch_size is not None and isinstance(series.batch_size, int)

    # Prepare samples to plot for this series
    if is_batched and plot_all_samples:
      samples_to_plot = [series[j] for j in range(series.batch_size)]
      effective_alpha = _calculate_alpha(series.batch_size, min_alpha, max_alpha, alpha_scaling)
    else:
      if is_batched and index is not None and index != 'all':
        samples_to_plot = [series[index]]
      else:
        samples_to_plot = [series]
      effective_alpha = max_alpha  # Use max_alpha for single samples

    # Plot each dimension
    for k in range(plot_dims):
      ax = axes[k, i]

      # Skip plotting if this dimension doesn't exist for this series
      if k >= samples_to_plot[0].values.shape[-1]:
        # Hide tick labels for empty axes
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
          ax.set_ylabel(f"Dim {k}", fontsize=10, alpha=0.5)
        continue

      legend_added = False

      # Plot all samples for this series
      for j, sample in enumerate(samples_to_plot):
        times = np.array(sample.times)
        values = np.array(sample.values)
        mask = np.array(sample.mask)

        # Add legend only for first dimension and first sample of each series
        label = None
        if k == 0 and not legend_added:
          series_label = f"{titles[i]}" if titles is not None and i < len(titles) else f"Series {i+1}"
          label = series_label
          legend_added = True

        line = _plot_timeseries(
            ax=ax,
            times=times,
            values=values,
            mask=mask,
            k=k,
            color=batch_color,
            line_width=batch_line_width,
            alpha=effective_alpha,
            marker_style=marker_style,
            marker_color=marker_colors,
            marker_size=marker_size,
            label=label
        )

        if line and label:
            legend_handles.append(line)
            legend_labels.append(label)

      # Set y-axis range and style the axes
      _set_y_limits(ax, 0, [y_ranges[k]])
      _style_axes(ax, k, plot_dims, is_first_col=(i == 0))

  # Add a common legend if we have any legend handles
  if legend_handles:
    # Calculate number of columns based on the number of legend entries
    ncol = min(5, len(legend_handles))

    fig.legend(handles=legend_handles, labels=legend_labels,
              loc='upper center', bbox_to_anchor=(0.5, 0.99),
              ncol=ncol, fontsize=10, frameon=True, borderaxespad=0.)

  # Add overall title if provided
  if common_title:
    fig.suptitle(common_title, fontsize=18, y=0.995)

  # Calculate spacing that scales with dimensions
  w_base, h_base = 0.2, 0.3
  w_scale = max(0.25, 1.0 / n_series)
  h_scale = max(0.5, 1.0 / plot_dims)

  # Layout adjustments
  plt.tight_layout(rect=[0, 0, 1, 0.97])
  plt.subplots_adjust(
    wspace=w_base * w_scale,
    hspace=h_base * h_scale,
    top=0.95
  )

  if show_plot:
    plt.show()

  return fig, axes

