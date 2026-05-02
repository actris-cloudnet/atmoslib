import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from atmoslib import constants as con
from atmoslib.thermodynamics import (
    air_density,
    c2k,
    dew_point_temperature,
    equivalent_potential_temperature,
    isa_altitude,
    isa_pressure,
    k2c,
    mixing_ratio,
    potential_temperature,
    relative_humidity,
    saturation_vapor_pressure,
    specific_humidity,
    vapor_pressure,
    virtual_temperature,
)

# Atmospheric ranges: troposphere temperatures, surface to ~10 km pressures,
# specific humidity up to a moist tropical maximum.
temperature = st.floats(min_value=200.0, max_value=330.0, allow_nan=False)
pressure = st.floats(min_value=20_000.0, max_value=110_000.0, allow_nan=False)
specific_hum = st.floats(min_value=0.0, max_value=0.04, allow_nan=False)
relative_hum = st.floats(min_value=0.01, max_value=1.0, allow_nan=False)


@given(temperature)
def test_c2k_k2c_round_trip(t):
    # Not bit-exact: 273.15 isn't representable in binary, so add-then-subtract
    # accumulates ~1 ULP of rounding. Within float64 precision is the contract.
    assert np.isclose(k2c(c2k(t)), t, rtol=0, atol=1e-12)
    assert np.isclose(c2k(k2c(t)), t, rtol=0, atol=1e-12)


@given(temperature, st.floats(min_value=0.1, max_value=20.0))
def test_saturation_vapor_pressure_monotonic_in_t(t, dt):
    assert saturation_vapor_pressure(t + dt) > saturation_vapor_pressure(t)


@given(temperature)
def test_mixed_phase_matches_branches(t):
    mixed = saturation_vapor_pressure(t, "mixed")
    if t < con.T0:
        assert mixed == saturation_vapor_pressure(t, "ice")
    else:
        assert mixed == saturation_vapor_pressure(t, "liquid")


@given(temperature, pressure, specific_hum)
def test_humidity_round_trip(t, p, q):
    rh = relative_humidity(t, p, q)
    q_back = specific_humidity(t, p, rh)
    assert np.isclose(q_back, q, rtol=1e-10, atol=1e-15)


@given(pressure, specific_hum)
def test_vapor_pressure_below_total_pressure(p, q):
    assert vapor_pressure(p, q) <= p


@given(temperature, specific_hum)
def test_virtual_temperature_at_least_temperature(t, q):
    assert virtual_temperature(t, q) >= t


@given(pressure, specific_hum)
def test_mixing_ratio_round_trip(p, q):
    vp = vapor_pressure(p, q)
    mr = mixing_ratio(vp, p)
    assert np.isclose(mr, q / (1 - q), rtol=1e-12)


@given(temperature, pressure, specific_hum)
def test_theta_e_at_least_theta(t, p, q):
    theta = potential_temperature(t, p)
    theta_e = equivalent_potential_temperature(t, p, q)
    assert theta_e >= theta - 1e-9


@given(temperature, pressure, specific_hum)
def test_air_density_positive(t, p, q):
    mr = q / (1 - q)
    assert air_density(t, p, mr) > 0


@given(temperature)
def test_dew_point_at_saturation_equals_t(t):
    td = dew_point_temperature(t, np.array(1.0))
    assert np.isclose(td, t, rtol=1e-6)


@given(st.floats(min_value=0.0, max_value=10_999.0, allow_nan=False))
def test_isa_altitude_pressure_round_trip(z):
    # `isa_altitude` and `isa_pressure` only form an exact inverse pair when
    # the sea-level standard temperature `T_STD` is passed to `isa_altitude`
    # (the formula treats `t` as the temperature at sea level, not at z).
    p = isa_pressure(np.array(z))
    z_back = isa_altitude(np.array(con.T_STD), p)
    assert np.isclose(z_back, z, rtol=1e-9, atol=1e-6)


@given(temperature, pressure, relative_hum)
def test_relative_humidity_round_trip(t, p, rh):
    q = specific_humidity(t, p, rh)
    rh_back = relative_humidity(t, p, q)
    assert np.isclose(rh_back, rh, rtol=1e-10)


@settings(max_examples=50)
@given(
    st.lists(temperature, min_size=2, max_size=10, unique=True).map(sorted),
)
def test_saturation_vapor_pressure_strictly_increasing(temps):
    arr = np.array(temps)
    svp = saturation_vapor_pressure(arr)
    assert np.all(np.diff(svp) > 0)
