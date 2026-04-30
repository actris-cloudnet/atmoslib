import pytest
from numpy.testing import assert_allclose

from atmoslib import isa_pressure


@pytest.mark.parametrize(
    "altitude,pressure",
    [
        (0, 101325),
        (1036.3, 89479),
        (2011.7, 79380),
        (3048.0, 69682),
        (4023.4, 61453),
        (5059.7, 53590),
    ],
)
def test_calc_pressure(altitude, pressure):
    assert_allclose(isa_pressure(altitude), pressure, atol=1)
