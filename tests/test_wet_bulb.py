import numpy as np
import pytest

from atmoslib.thermodynamics import (
    calc_mixing_ratio,
    calc_saturation_vapor_pressure,
    calc_wet_bulb_temperature,
)


def _q_at_relative_humidity(t: float, p: float, rh: float) -> float:
    """Specific humidity (kg/kg) for a given temperature, pressure and RH."""
    svp = calc_saturation_vapor_pressure(np.array(t))
    vp = rh * svp
    w = calc_mixing_ratio(vp, np.array(p))
    return float(w / (1 + w))


def test_scalar_input_returns_scalar():
    t, p, q = 293.15, 101325.0, 0.010
    tw = calc_wet_bulb_temperature(np.array(t), np.array(p), np.array(q))
    assert tw.shape == ()
    assert np.isfinite(tw)


def test_1d_array_input():
    t = np.array([293.15, 283.15, 273.15])
    p = np.array([101325.0, 95000.0, 90000.0])
    q = np.array([0.010, 0.005, 0.001])
    tw = calc_wet_bulb_temperature(t, p, q)
    assert tw.shape == t.shape
    assert np.all(np.isfinite(tw))


def test_2d_array_input():
    t = np.full((4, 5), 293.15)
    p = np.full((4, 5), 101325.0)
    q = np.full((4, 5), 0.010)
    tw = calc_wet_bulb_temperature(t, p, q)
    assert tw.shape == t.shape
    assert np.all(np.isfinite(tw))


def test_scalar_and_array_agree():
    t, p, q = 293.15, 101325.0, 0.010
    tw_scalar = calc_wet_bulb_temperature(np.array(t), np.array(p), np.array(q))
    tw_array = calc_wet_bulb_temperature(
        np.array([t, t]), np.array([p, p]), np.array([q, q])
    )
    np.testing.assert_allclose(tw_array, float(tw_scalar))


def test_wet_bulb_not_above_dry_bulb():
    rng = np.random.default_rng(0)
    t = rng.uniform(253.15, 313.15, size=50)
    p = rng.uniform(70000.0, 105000.0, size=50)
    rh = rng.uniform(0.05, 1.0, size=50)
    q = np.array(
        [
            _q_at_relative_humidity(ti, pi, ri)
            for ti, pi, ri in zip(t, p, rh, strict=True)
        ]
    )
    tw = calc_wet_bulb_temperature(t, p, q)
    assert np.all(tw <= t + 1e-6)


def test_wet_bulb_equals_dry_bulb_at_saturation():
    t = np.array([293.15, 283.15, 273.15])
    p = np.array([101325.0, 95000.0, 90000.0])
    q = np.array(
        [_q_at_relative_humidity(ti, pi, 1.0) for ti, pi in zip(t, p, strict=True)]
    )
    tw = calc_wet_bulb_temperature(t, p, q)
    np.testing.assert_allclose(tw, t, atol=0.05)


def test_wet_bulb_drops_with_dryer_air():
    t = np.full(4, 293.15)
    p = np.full(4, 101325.0)
    rh = np.array([1.0, 0.75, 0.5, 0.25])
    q = np.array(
        [
            _q_at_relative_humidity(ti, pi, ri)
            for ti, pi, ri in zip(t, p, rh, strict=True)
        ]
    )
    tw = calc_wet_bulb_temperature(t, p, q)
    # Drier air -> lower wet-bulb -> strictly decreasing as RH decreases
    assert np.all(np.diff(tw) < 0)


@pytest.mark.parametrize(
    "t_c, p_pa, rh, expected_c",
    [
        # Reference values from psychrometric tables / online calculators
        (20.0, 101325.0, 0.50, 13.7),
        (30.0, 101325.0, 0.50, 22.0),
        (10.0, 101325.0, 0.80, 8.2),
        (25.0, 101325.0, 0.30, 13.8),
    ],
)
def test_known_psychrometric_values(t_c, p_pa, rh, expected_c):
    t = np.array(t_c + 273.15)
    p = np.array(p_pa)
    q = np.array(_q_at_relative_humidity(t_c + 273.15, p_pa, rh))
    tw = calc_wet_bulb_temperature(t, p, q)
    tw_c = float(tw) - 273.15
    assert tw_c == pytest.approx(expected_c, abs=1.0)
