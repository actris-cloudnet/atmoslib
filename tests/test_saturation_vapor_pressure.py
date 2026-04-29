import numpy as np
import pytest

from atmoslib.thermodynamics import saturation_vapor_pressure


def test_liquid_at_triple_point():
    # Saturation vapor pressure at triple point ~611 Pa
    result = saturation_vapor_pressure(np.array(273.16))
    assert result == pytest.approx(611.0, rel=0.01)


def test_ice_at_triple_point():
    result = saturation_vapor_pressure(np.array(273.16), phase="ice")
    assert result == pytest.approx(611.0, rel=0.01)


def test_liquid_and_ice_agree_at_triple_point():
    t = np.array(273.16)
    liquid = saturation_vapor_pressure(t, phase="liquid")
    ice = saturation_vapor_pressure(t, phase="ice")
    np.testing.assert_allclose(liquid, ice, rtol=0.001)


def test_ice_below_liquid_below_freezing():
    t = np.array(253.15)  # -20°C
    liquid = saturation_vapor_pressure(t, phase="liquid")
    ice = saturation_vapor_pressure(t, phase="ice")
    assert ice < liquid


def test_mixed_uses_liquid_above_freezing():
    t = np.array(300.0)
    liquid = saturation_vapor_pressure(t, phase="liquid")
    mixed = saturation_vapor_pressure(t, phase="mixed")
    assert mixed == liquid


def test_mixed_uses_ice_below_freezing():
    t = np.array(250.0)
    ice = saturation_vapor_pressure(t, phase="ice")
    mixed = saturation_vapor_pressure(t, phase="mixed")
    assert mixed == ice


def test_mixed_array_switches_at_triple_point():
    t = np.array([300.0, 250.0])
    mixed = saturation_vapor_pressure(t, phase="mixed")
    liquid = saturation_vapor_pressure(t, phase="liquid")
    ice = saturation_vapor_pressure(t, phase="ice")
    assert mixed[0] == liquid[0]
    assert mixed[1] == ice[1]


def test_increases_with_temperature():
    t = np.array([253.15, 273.15, 293.15, 313.15])
    result = saturation_vapor_pressure(t)
    assert np.all(np.diff(result) > 0)


@pytest.mark.parametrize(
    "t_k, expected_pa",
    [
        # Reference values from standard tables
        (373.15, 101325.0),  # Boiling point at 1 atm
        (293.15, 2339.0),  # ~20°C
    ],
)
def test_known_liquid_values(t_k, expected_pa):
    result = saturation_vapor_pressure(np.array(t_k))
    assert result == pytest.approx(expected_pa, rel=0.02)
