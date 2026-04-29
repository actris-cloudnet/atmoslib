import numpy as np

from atmoslib.thermodynamics import geometric_height


def test_zero_height():
    assert geometric_height(np.array(0.0)) == 0.0


def test_positive_height():
    result = geometric_height(np.array(1000.0))
    # Geometric height is always slightly larger than geopotential height
    assert result > 1000.0


def test_array_input():
    h = np.array([0.0, 1000.0, 10000.0])
    result = geometric_height(h)
    assert result.shape == h.shape
    assert np.all(np.diff(result) > 0)


def test_difference_grows_with_altitude():
    h = np.array([1000.0, 10000.0, 50000.0])
    result = geometric_height(h)
    diff = result - h
    # Correction grows with altitude
    assert np.all(np.diff(diff) > 0)
