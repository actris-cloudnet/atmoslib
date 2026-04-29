import numpy as np
import pytest

from atmoslib.thermodynamics import (
    equivalent_potential_temperature,
    potential_temperature,
    virtual_temperature,
)


def test_virtual_temperature_dry_air_equals_temperature():
    t = np.array([293.15, 283.15, 273.15])
    q = np.zeros_like(t)
    np.testing.assert_allclose(virtual_temperature(t, q), t)


def test_virtual_temperature_known_value():
    assert virtual_temperature(np.array(300.0), np.array(0.01)) == pytest.approx(
        301.83, abs=1e-6
    )


def test_virtual_temperature_above_dry_temperature():
    rng = np.random.default_rng(0)
    t = rng.uniform(263.15, 313.15, size=20)
    q = rng.uniform(1e-4, 0.025, size=20)
    assert np.all(virtual_temperature(t, q) > t)


def test_potential_temperature_at_reference_pressure():
    t = np.array([293.15, 250.0, 230.0])
    p = np.full_like(t, 101325.0)  # Default p0
    np.testing.assert_allclose(potential_temperature(t, p), t)


def test_potential_temperature_increases_aloft():
    # At lower pressure (higher altitude), theta > T for the same parcel
    t = np.array(280.0)
    p_low = np.array(50000.0)
    theta = potential_temperature(t, p_low)
    assert theta > t


def test_potential_temperature_known_value():
    # Air at 280 K, 800 hPa, brought to 1013.25 hPa
    # theta = 280 * (101325/80000)^(287.058/1004) ≈ 299.57
    theta = potential_temperature(np.array(280.0), np.array(80000.0))
    assert theta == pytest.approx(299.57, abs=0.1)


def test_potential_temperature_custom_p0():
    t = np.array(280.0)
    p = np.array(80000.0)
    theta_default = potential_temperature(t, p)
    theta_explicit = potential_temperature(t, p, p0=101325.0)
    np.testing.assert_allclose(theta_default, theta_explicit)


def test_equivalent_potential_temperature_dry_equals_potential():
    t = np.array(285.0)
    p = np.array(95000.0)
    q = np.array(0.0)
    theta = potential_temperature(t, p)
    theta_e = equivalent_potential_temperature(t, p, q)
    np.testing.assert_allclose(theta_e, theta)


def test_equivalent_potential_temperature_above_potential_when_moist():
    rng = np.random.default_rng(1)
    t = rng.uniform(273.15, 303.15, size=15)
    p = rng.uniform(70000.0, 101325.0, size=15)
    q = rng.uniform(1e-4, 0.020, size=15)
    theta = potential_temperature(t, p)
    theta_e = equivalent_potential_temperature(t, p, q)
    assert np.all(theta_e > theta)


def test_equivalent_potential_temperature_array_shape():
    t = np.full((3, 4), 290.0)
    p = np.full((3, 4), 95000.0)
    q = np.full((3, 4), 0.010)
    theta_e = equivalent_potential_temperature(t, p, q)
    assert theta_e.shape == t.shape
    assert np.all(np.isfinite(theta_e))
