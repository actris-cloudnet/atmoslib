import numpy as np
import pytest
from numpy.testing import assert_allclose

from atmoslib import constants as con
from atmoslib import isa_air_density, isa_pressure, isa_temperature


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
def test_isa_pressure(altitude, pressure):
    assert_allclose(isa_pressure(altitude), pressure, atol=1)


def test_isa_temperature_known_values():
    assert isa_temperature(0) == pytest.approx(con.T_STD)
    assert isa_temperature(1000) == pytest.approx(con.T_STD - con.L0 * 1000, rel=1e-6)
    assert isa_temperature(2000) == pytest.approx(con.T_STD - con.L0 * 2000, rel=1e-6)


def test_isa_temperature_at_sea_level():
    assert isa_temperature(0) == pytest.approx(con.T_STD)


def test_isa_temperature_lapse_rate():
    # Temperature should decrease by 6.5 K per 1000 m
    t0 = isa_temperature(0)
    t1000 = isa_temperature(1000)
    assert t0 - t1000 == pytest.approx(6.5, rel=1e-6)


def test_isa_temperature_at_11km_raises():
    with pytest.raises(ValueError, match="Valid only up to 11 km"):
        isa_temperature(11000)


def test_isa_temperature_above_11km_raises():
    with pytest.raises(ValueError, match="Valid only up to 11 km"):
        isa_temperature(12000)


def test_isa_temperature_array_input():
    gph = np.array([0, 1000, 2000, 5000])
    expected = con.T_STD - con.L0 * gph
    assert_allclose(isa_temperature(gph), expected, atol=1e-6)


def test_isa_air_density_at_sea_level():
    assert isa_air_density(0) == pytest.approx(con.RHO_STD, rel=0.001)


def test_isa_air_density_decreases_with_height():
    rho_0 = isa_air_density(0)
    rho_1000 = isa_air_density(1000)
    rho_5000 = isa_air_density(5000)
    assert rho_0 > rho_1000 > rho_5000


def test_isa_air_density_at_11km_raises():
    with pytest.raises(ValueError, match="Valid only up to 11 km"):
        isa_air_density(11000)


def test_isa_air_density_array_input():
    gph = np.array([0, 1000, 2000, 5000])
    result = isa_air_density(gph)
    assert np.shape(result) == gph.shape
    assert np.all(np.diff(result) < 0)  # Density decreases with height


def test_isa_air_density_consistency_with_ideal_gas_law():
    # isa_air_density should equal isa_pressure / (R * isa_temperature)
    gph = np.array([0, 1000, 5000, 10000])
    expected = isa_pressure(gph) / (con.RS * isa_temperature(gph))
    assert_allclose(isa_air_density(gph), expected, rtol=1e-10)
