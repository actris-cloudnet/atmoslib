import numpy as np
import pytest

from atmoslib import constants as con
from atmoslib.thermodynamics import barometric_pressure_profile


def test_surface_value_returned_at_first_level():
    t = np.array([288.15, 281.65, 275.15])
    q = np.array([0.005, 0.003, 0.001])
    z = np.array([0.0, 1000.0, 2000.0])
    p_sfc = np.array(101325.0)
    p = barometric_pressure_profile(t, q, z, p_sfc)
    assert p[0] == pytest.approx(p_sfc)


def test_pressure_decreases_with_height():
    t = np.array([288.15, 281.65, 275.15, 268.65, 262.15])
    q = np.array([0.008, 0.005, 0.003, 0.001, 0.0005])
    z = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    p = barometric_pressure_profile(t, q, z, np.array(101325.0))
    assert np.all(np.diff(p) < 0)


def test_isothermal_dry_matches_analytic_hypsometric():
    # In an isothermal, dry atmosphere the hypsometric equation reduces to
    # p(z) = p_sfc * exp(-g * z / (R_d * T))
    t = np.full(6, 273.15)
    q = np.zeros(6)
    z = np.linspace(0.0, 5000.0, 6)
    p_sfc = np.array(100000.0)
    p = barometric_pressure_profile(t, q, z, p_sfc)
    expected = p_sfc * np.exp(-con.G * z / (con.RS * 273.15))
    np.testing.assert_allclose(p, expected, rtol=1e-12)


def test_moist_atmosphere_lighter_than_dry():
    # Adding water vapor makes the air less dense, so pressure decreases
    # more slowly with height; pressure aloft should be higher than the dry case.
    t = np.full(4, 290.0)
    z = np.array([0.0, 1000.0, 2000.0, 3000.0])
    p_sfc = np.array(101325.0)
    p_dry = barometric_pressure_profile(t, np.zeros(4), z, p_sfc)
    p_moist = barometric_pressure_profile(t, np.full(4, 0.015), z, p_sfc)
    assert p_moist[0] == pytest.approx(p_dry[0])
    assert np.all(p_moist[1:] > p_dry[1:])


def test_2d_profile_matches_per_row_1d():
    rng = np.random.default_rng(42)
    n_time, n_lev = 5, 8
    t = rng.uniform(250.0, 295.0, size=(n_time, n_lev))
    q = rng.uniform(1e-4, 1e-2, size=(n_time, n_lev))
    z = np.linspace(0.0, 7000.0, n_lev)
    p_sfc = rng.uniform(95000.0, 102000.0, size=n_time)

    p_2d = barometric_pressure_profile(t, q, z, p_sfc)
    assert p_2d.shape == t.shape

    for i in range(n_time):
        p_1d = barometric_pressure_profile(t[i], q[i], z, p_sfc[i])
        np.testing.assert_allclose(p_2d[i], p_1d, rtol=1e-12)


def test_z_can_be_2d():
    # Heights may also vary per profile (e.g. terrain-following grids).
    t = np.array([[288.0, 282.0, 275.0], [290.0, 284.0, 277.0]])
    q = np.array([[0.006, 0.004, 0.002], [0.008, 0.005, 0.003]])
    z = np.array([[0.0, 1000.0, 2000.0], [50.0, 1100.0, 2200.0]])
    p_sfc = np.array([101325.0, 100800.0])
    p = barometric_pressure_profile(t, q, z, p_sfc)
    assert p.shape == t.shape
    np.testing.assert_allclose(p[:, 0], p_sfc)
    assert np.all(np.diff(p, axis=-1) < 0)


def test_single_level_returns_surface_pressure():
    t = np.array([288.15])
    q = np.array([0.005])
    z = np.array([0.0])
    p = barometric_pressure_profile(t, q, z, np.array(101325.0))
    assert p.shape == (1,)
    assert p[0] == pytest.approx(101325.0)


def test_known_reference_value():
    # ICAO standard atmosphere at 0 and 5500 m: ~101325 Pa and ~50500 Pa.
    # Using a piecewise-linear T profile with a 6.5 K/km lapse rate and dry air,
    # the integrated profile should be close to the ICAO value at 5500 m.
    z = np.linspace(0.0, 5500.0, 12)
    t = 288.15 - 0.0065 * z
    q = np.zeros_like(z)
    p = barometric_pressure_profile(t, q, z, np.array(101325.0))
    assert p[-1] == pytest.approx(50500.0, rel=0.01)
