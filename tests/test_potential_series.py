import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool
from typing import Union, Tuple, Optional
import pytest

from linsdex.potential.gaussian.gaussian_potential_series import GaussianPotentialSeries
from linsdex.series.batchable_object import AbstractBatchableObject

class TestPotentialSeries:
  """Tests for PotentialSeries class."""

  def test_init_basic(self):
    """Test basic initialization with arrays."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])

    # Create ProbabilisticTimeSeries with default parameterization
    pts = GaussianPotentialSeries(ts=times, xts=values)

    # Check that attributes were set correctly
    assert jnp.array_equal(pts.times, times)
    assert len(pts) == 4
    assert pts.node_potentials is not None

  def test_init_with_standard_deviation(self):
    """Test initialization with standard deviation."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])
    std_dev = jnp.array([[0.1], [0.2], [0.3], [0.4]])

    # Create ProbabilisticTimeSeries
    pts = GaussianPotentialSeries(ts=times, xts=values, standard_deviation=std_dev)

    # Check that attributes were set correctly
    assert jnp.array_equal(pts.times, times)
    assert len(pts) == 4
    assert pts.node_potentials is not None

  def test_init_with_certainty(self):
    """Test initialization with certainty values."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])
    certainty = jnp.array([[10.0], [5.0], [3.33], [2.5]])

    # Create ProbabilisticTimeSeries
    pts = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)

    # Check that attributes were set correctly
    assert jnp.array_equal(pts.times, times)
    assert len(pts) == 4
    assert pts.node_potentials is not None

  def test_init_with_multi_feature_values(self):
    """Test initialization with 2D values with multiple features."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0, 20.0, 30.0],
                        [11.0, 21.0, 31.0],
                        [12.0, 22.0, 32.0],
                        [13.0, 23.0, 33.0]])
    certainty = jnp.array([[1.0, 2.0, 3.0],
                          [1.5, 2.5, 3.5],
                          [2.0, 3.0, 4.0],
                          [2.5, 3.5, 4.5]])

    # Create ProbabilisticTimeSeries
    pts = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)

    # Check that attributes were set correctly
    assert jnp.array_equal(pts.times, times)
    assert len(pts) == 4
    assert pts.node_potentials is not None

  def test_parameterizations(self):
    """Test different parameterizations."""
    times = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([[10.0], [11.0], [12.0]])
    certainty = jnp.array([[1.0], [2.0], [3.0]])

    # Test natural parameterization
    pts_nat = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty,
                                     parameterization='natural')
    assert pts_nat.node_potentials is not None

    # Test mixed parameterization
    pts_mixed = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty,
                                       parameterization='mixed')
    assert pts_mixed.node_potentials is not None

    # Test standard parameterization
    pts_std = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty,
                                     parameterization='standard')
    assert pts_std.node_potentials is not None

  def test_fully_certain_potentials(self):
    """Test creation of fully certain potentials."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])
    certainty = jnp.array([[jnp.inf], [1.0], [jnp.inf], [2.0]])

    pts = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)

    # Check is_fully_certain property
    expected_certain = jnp.array([True, False, True, False])
    assert jnp.array_equal(pts.is_fully_certain, expected_certain)

  def test_fully_uncertain_potentials(self):
    """Test creation of fully uncertain potentials."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([[10.0], [11.0], [12.0], [13.0]])
    certainty = jnp.array([[0.0], [1.0], [0.0], [2.0]])

    pts = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)

    # Check is_fully_uncertain property
    expected_uncertain = jnp.array([True, False, True, False])
    assert jnp.array_equal(pts.is_fully_uncertain, expected_uncertain)

  def test_batch_size_property(self):
    """Test batch_size property for different array dimensions."""
    times = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([[10.0], [11.0], [12.0]])

    # Single time series (no batching)
    pts_single = GaussianPotentialSeries(ts=times, xts=values)
    assert pts_single.batch_size is None

    # Batched time series
    batched_times = jnp.array([[0.0, 1.0, 2.0],
                              [3.0, 4.0, 5.0]])
    batched_values = jnp.array([[[10.0], [11.0], [12.0]],
                               [[13.0], [14.0], [15.0]]])

    pts_batched = GaussianPotentialSeries(ts=batched_times, xts=batched_values)
    assert pts_batched.batch_size == 2

  def test_conversion_methods(self):
    """Test to_mixed, to_nat, and to_std conversion methods."""
    times = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([[10.0], [11.0], [12.0]])
    certainty = jnp.array([[1.0], [2.0], [3.0]])

    pts = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)

    # Test conversions (should not raise errors)
    pts_mixed = pts.to_mixed()
    assert pts_mixed.node_potentials is not None
    assert jnp.array_equal(pts_mixed.times, times)

    pts_nat = pts.to_nat()
    assert pts_nat.node_potentials is not None
    assert jnp.array_equal(pts_nat.times, times)

    pts_std = pts.to_std()
    assert pts_std.node_potentials is not None
    assert jnp.array_equal(pts_std.times, times)

  def test_from_potentials_classmethod(self):
    """Test the from_potentials class method."""
    times = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([[10.0], [11.0], [12.0]])

    # Create a ProbabilisticTimeSeries normally
    pts_original = GaussianPotentialSeries(ts=times, xts=values)

    # Create another using from_potentials
    pts_from_potentials = GaussianPotentialSeries.from_potentials(
      ts=times,
      node_potentials=pts_original.node_potentials
    )

    assert jnp.array_equal(pts_from_potentials.times, times)
    assert pts_from_potentials.node_potentials is not None

  def test_make_windowed_batches(self):
    """Test make_windowed_batches method."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = jnp.array([[10.0, 20.0],
                       [11.0, 21.0],
                       [12.0, 22.0],
                       [13.0, 23.0],
                       [14.0, 24.0]])
    certainty = jnp.array([[1.0, 1.0],
                          [1.0, 1.0],
                          [1.0, 1.0],
                          [1.0, 1.0],
                          [1.0, 1.0]])

    pts = GaussianPotentialSeries(ts=times, xts=values, certainty=certainty)

    # Create windowed batches with window size 3
    window_size = 3
    batched_pts = pts.make_windowed_batches(window_size)

    # Expected: 3 windows with a window size of 3
    # Window 1: [0, 1, 2]
    # Window 2: [1, 2, 3]
    # Window 3: [2, 3, 4]
    assert batched_pts.times.shape == (3, 3)

    # Check specific windows
    expected_times = jnp.array([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0]
    ])
    assert jnp.allclose(batched_pts.times, expected_times)

  def test_getitem(self):
    """Test the __getitem__ method inherited from AbstractBatchableObject."""
    times = jnp.array([[0.0, 1.0, 2.0],
                       [3.0, 4.0, 5.0]])
    values = jnp.array([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
                       [[13.0, 23.0], [14.0, 24.0], [15.0, 25.0]]])

    pts = GaussianPotentialSeries(ts=times, xts=values)

    # Get first batch
    pts_0 = pts[0]

    assert jnp.array_equal(pts_0.times, times[0])
    assert len(pts_0) == 3
    assert pts_0.node_potentials is not None

    # Get using a slice
    pts_slice = pts[0:1]

    assert pts_slice.times.shape == (1, 3)
    assert jnp.array_equal(pts_slice.times, times[0:1])

  def test_error_conditions(self):
    """Test error conditions and edge cases."""
    times = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([[10.0], [11.0], [12.0]])

    # Test providing both standard_deviation and certainty (should raise error)
    std_dev = jnp.array([[0.1], [0.2], [0.3]])
    certainty = jnp.array([[10.0], [5.0], [3.33]])

    with pytest.raises(ValueError, match="Both standard_deviation and certainty cannot be provided"):
      GaussianPotentialSeries(ts=times, xts=values,
                             standard_deviation=std_dev, certainty=certainty)

    # Test mismatched shapes
    wrong_certainty = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Wrong shape

    with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
      GaussianPotentialSeries(ts=times, xts=values, certainty=wrong_certainty)

    # Test invalid parameterization
    certainty_for_test = jnp.array([[1.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match="Unknown parameterization"):
      GaussianPotentialSeries(ts=times, xts=values, certainty=certainty_for_test, parameterization='invalid')

  def test_init_with_linear_functional(self):
    """Test initialization with LinearFunctional."""
    from linsdex.linear_functional.linear_functional import LinearFunctional

    times = jnp.array([0.0, 1.0])
    dim = 2
    # Create a linear functional Ax + b where A=I, b=0 (identity)
    lf = LinearFunctional.identity(dim)
    # Broadcast to match times
    lf_batched = eqx.filter_vmap(lambda _: lf)(jnp.arange(2))

    # Create series
    pts = GaussianPotentialSeries(ts=times, xts=lf_batched)

    assert jnp.array_equal(pts.times, times)
    assert len(pts) == 2
    # Default parameterization is mixed when no certainty provided, so mu should be a LinearFunctional
    assert isinstance(pts.node_potentials.mu, LinearFunctional)

  def test_linear_functional_conversions(self):
    """Test conversions when using LinearFunctional."""
    from linsdex.linear_functional.linear_functional import LinearFunctional

    times = jnp.array([0.0, 1.0])
    dim = 2
    lf = LinearFunctional.identity(dim)
    lf_batched = eqx.filter_vmap(lambda _: lf)(jnp.arange(2))

    pts = GaussianPotentialSeries(ts=times, xts=lf_batched)

    # These should work without error
    pts_nat = pts.to_nat()
    pts_mixed = pts.to_mixed()
    pts_std = pts.to_std()

    assert isinstance(pts_nat.node_potentials.h, LinearFunctional)
    assert isinstance(pts_mixed.node_potentials.mu, LinearFunctional)
    assert isinstance(pts_std.node_potentials.mu, LinearFunctional)

  def test_scalar_ts_with_linear_functional(self):
    """Test initialization with scalar ts and LinearFunctional."""
    from linsdex.linear_functional.linear_functional import LinearFunctional

    time = jnp.array(0.5)
    dim = 2
    lf = LinearFunctional.identity(dim)

    pts = GaussianPotentialSeries(ts=time, xts=lf)

    assert pts.times.shape == (1,)
    assert len(pts) == 1
    assert isinstance(pts.node_potentials.mu, LinearFunctional)
