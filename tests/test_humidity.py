import numpy as np
import pytest

from atmoslib import constants as con
from atmoslib.thermodynamics import (
    absolute_humidity,
    dew_point_temperature,
    latent_heat_of_vaporization,
    relative_humidity,
    saturation_vapor_pressure,
    specific_humidity,
    vapor_pressure,
)


def test_relative_humidity_at_saturation():
    t = np.array(293.15)
    p = np.array(101325.0)
    svp = saturation_vapor_pressure(t)
    # Specific humidity that produces vapor_pressure(p, q) == svp(t)
    q_sat = (svp * con.MW_RATIO) / (p - svp * (1 - con.MW_RATIO))
    rh = relative_humidity(t, p, q_sat)
    assert rh == pytest.approx(1.0, rel=1e-6)


def test_relative_humidity_known_value():
    # 25 degC, 101325 Pa, q = 0.010 kg/kg → RH ~50%
    t = np.array(298.15)
    p = np.array(101325.0)
    q = np.array(0.010)
    rh = relative_humidity(t, p, q)
    assert rh == pytest.approx(0.51, abs=0.03)


def test_specific_humidity_at_saturation():
    t = np.array(298.15)
    p = np.array(101325.0)
    svp = saturation_vapor_pressure(t)
    # Specific humidity that produces vapor_pressure(p, q) == svp(t)
    q_sat = (svp * con.MW_RATIO) / (p - svp * (1 - con.MW_RATIO))
    rh = np.array(1)
    q = specific_humidity(t, p, rh)
    assert q == q_sat


def test_specific_humidity_known_value():
    # 25 degC, 101325 Pa, q = 0.010 kg/kg → RH ~50%
    t = np.array(298.15)
    p = np.array(101325.0)
    rh = np.array(0.51)
    q = specific_humidity(t, p, rh)
    assert q == pytest.approx(0.010, abs=1e-4)


def test_relative_humidity_array_input():
    t = np.array([293.15, 283.15, 273.15])
    p = np.array([101325.0, 95000.0, 90000.0])
    q = np.array([0.010, 0.005, 0.001])
    rh = relative_humidity(t, p, q)
    assert rh.shape == t.shape
    assert np.all(rh > 0)
    assert np.all(rh < 1.5)  # generous bound for unsaturated cases


def test_absolute_humidity_known_value():
    # vp=1500 Pa, T=298.15 K → ah ≈ 0.01089 kg/m³
    ah = absolute_humidity(np.array(298.15), np.array(1500.0))
    assert ah == pytest.approx(0.01089, abs=1e-4)


def test_absolute_humidity_inverse_of_ideal_gas():
    t = np.array(290.0)
    vp = np.array(1200.0)
    ah = absolute_humidity(t, vp)
    np.testing.assert_allclose(vp, ah * con.RW * t, rtol=1e-6)


def test_dew_point_at_saturation_equals_temperature():
    t = np.array([293.15, 283.15, 273.15])
    rh = np.full_like(t, 1.0)
    td = dew_point_temperature(t, rh)
    np.testing.assert_allclose(td, t, atol=1e-6)


def test_dew_point_below_temperature():
    rng = np.random.default_rng(0)
    t = rng.uniform(273.15, 313.15, size=20)
    rh = rng.uniform(0.05, 0.99, size=20)
    td = dew_point_temperature(t, rh)
    assert np.all(td < t)


@pytest.mark.parametrize(
    "t_c, rh, expected_td_c",
    [
        # Reference values from psychrometric charts (Magnus formula)
        (25.0, 0.50, 13.85),
        (20.0, 0.60, 12.0),
        (30.0, 0.70, 23.9),
        (10.0, 0.80, 6.7),
    ],
)
def test_dew_point_known_values(t_c, rh, expected_td_c):
    t = np.array(t_c + 273.15)
    td = dew_point_temperature(t, np.array(rh))
    td_c = float(td) - 273.15
    assert td_c == pytest.approx(expected_td_c, abs=0.5)


def test_latent_heat_at_triple_point():
    lv = latent_heat_of_vaporization(np.array(273.16))
    assert float(lv) == pytest.approx(2.501e6, rel=1e-6)


def test_latent_heat_decreases_with_temperature():
    t = np.array([253.15, 273.15, 293.15, 313.15])
    lv = latent_heat_of_vaporization(t)
    assert np.all(np.diff(lv) < 0)


def test_vapor_pressure_consistent_with_humidity_helpers():
    # Round-trip: q -> vp -> ah -> back-derive vp
    t = np.array(290.0)
    p = np.array(95000.0)
    q = np.array(0.008)
    vp = vapor_pressure(p, q)
    ah = absolute_humidity(t, vp)
    np.testing.assert_allclose(vp, ah * con.RW * t, rtol=1e-6)
