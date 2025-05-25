import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool
from typing import Union, Tuple, Optional
import pytest

from linsdex.series.series import TimeSeries
from linsdex.series.batchable_object import AbstractBatchableObject

class SimpleObject(AbstractBatchableObject):
  """Simple implementation of AbstractBatchableObject for testing."""
  data: Float[Array, "..."]

  @property
  def batch_size(self) -> Union[Tuple[int], int, None]:
    if len(self.data.shape) == 0:
      return None
    elif len(self.data.shape) == 1:
      return None  # Single dimension is not batch
    else:
      return self.data.shape[0]

class TestTimeSeries:
  """Tests for TimeSeries class."""

  def test_init_basic(self):
    """Test basic initialization with arrays."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([10.0, 11.0, 12.0, 13.0])[:,None]

    # Create TimeSeries
    ts = TimeSeries(times=times, values=values)

    # Check that attributes were set correctly
    assert jnp.array_equal(ts.times, times)
    assert jnp.array_equal(ts.values, values)
    assert jnp.array_equal(ts.mask, jnp.ones_like(times, dtype=bool))

  def test_init_with_mask(self):
    """Test initialization with explicit mask."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([10.0, 11.0, 12.0, 13.0])[:,None]
    mask = jnp.array([True, True, False, True])

    # Create TimeSeries
    ts = TimeSeries(times=times, values=values, mask=mask)

    # Check that attributes were set correctly
    assert jnp.array_equal(ts.times, times)
    assert jnp.array_equal(ts.values, values)
    assert jnp.array_equal(ts.mask, mask)

  def test_init_with_2d_values(self):
    """Test initialization with already 2D values."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])  # Already 2D

    # Create TimeSeries
    ts = TimeSeries(times=times, values=values)

    # Check that attributes were set correctly
    assert jnp.array_equal(ts.times, times)
    assert jnp.array_equal(ts.values, values)
    assert jnp.array_equal(ts.mask, jnp.ones_like(times, dtype=bool))

  def test_init_with_multi_feature_values(self):
    """Test initialization with 2D values with multiple features."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0, 20.0, 30.0],
                        [11.0, 21.0, 31.0],
                        [12.0, 22.0, 32.0],
                        [13.0, 23.0, 33.0]])  # 2D with multiple features

    # Create TimeSeries
    ts = TimeSeries(times=times, values=values)

    # Check that attributes were set correctly
    assert jnp.array_equal(ts.times, times)
    assert jnp.array_equal(ts.values, values)
    assert jnp.array_equal(ts.mask, jnp.ones_like(times, dtype=bool))

  def test_is_fully_uncertain(self):
    """Test is_fully_uncertain method."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])
    mask = jnp.array([True, False, True, False])

    ts = TimeSeries(times=times, values=values, mask=mask)

    # Check is_fully_uncertain
    expected = jnp.array([False, True, False, True])
    assert jnp.array_equal(ts.is_fully_uncertain(), expected)

  def test_get_missing_observation_mask(self):
    """Test get_missing_observation_mask method."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])
    mask = jnp.array([True, False, True, False])

    ts = TimeSeries(times=times, values=values, mask=mask)

    # get_missing_observation_mask should be the same as is_fully_uncertain
    expected = jnp.array([False, True, False, True])
    assert jnp.array_equal(ts.get_missing_observation_mask(), expected)

  def test_make_windowed_batches(self):
    """Test make_windowed_batches method."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = jnp.array([[10.0, 20.0],
                       [11.0, 21.0],
                       [12.0, 22.0],
                       [13.0, 23.0],
                       [14.0, 24.0]])

    ts = TimeSeries(times=times, values=values)

    # Create windowed batches with window size 3
    window_size = 3
    batched_ts = ts.make_windowed_batches(window_size)

    # Expected: 3 windows with a window size of 3
    # Window 1: [0, 1, 2]
    # Window 2: [1, 2, 3]
    # Window 3: [2, 3, 4]
    assert batched_ts.times.shape == (3, 3)

    # Check specific windows
    expected_times = jnp.array([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0]
    ])
    assert jnp.allclose(batched_ts.times, expected_times)

    # Check that values have the right shape and content
    assert batched_ts.values.shape == (3, 3, 2)

    expected_values = jnp.array([
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
        [[12.0, 22.0], [13.0, 23.0], [14.0, 24.0]]
    ])
    assert jnp.allclose(batched_ts.values, expected_values)

  def test_getitem(self):
    """Test the __getitem__ method inherited from AbstractBatchableObject."""
    times = jnp.array([[0.0, 1.0, 2.0],
                       [3.0, 4.0, 5.0]])
    values = jnp.array([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
                       [[13.0, 23.0], [14.0, 24.0], [15.0, 25.0]]])
    mask = jnp.ones_like(times, dtype=bool)

    ts = TimeSeries(times=times, values=values, mask=mask)

    # Get first batch
    ts_0 = ts[0]

    assert jnp.array_equal(ts_0.times, times[0])
    assert jnp.array_equal(ts_0.values, values[0])
    assert jnp.array_equal(ts_0.mask, mask[0])

    # Get using a slice
    ts_slice = ts[0:1]

    assert ts_slice.times.shape == (1, 3)
    assert ts_slice.values.shape == (1, 3, 2)
    assert jnp.array_equal(ts_slice.times, times[0:1])
    assert jnp.array_equal(ts_slice.values, values[0:1])
    assert jnp.array_equal(ts_slice.mask, mask[0:1])
