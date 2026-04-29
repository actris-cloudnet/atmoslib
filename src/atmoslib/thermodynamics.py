import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy import ma

from atmoslib import constants as con

logger = logging.getLogger(__name__)

# (R_vapor / R_dry - 1), used in moist-air virtual-temperature corrections
_EPSILON_V = (1 - con.MW_RATIO) / con.MW_RATIO


def c2k(t: npt.NDArray) -> npt.NDArray:
    """Converts Celsius to Kelvins."""
    return ma.array(t) + 273.15


def k2c(t: npt.NDArray) -> npt.NDArray:
    """Converts Kelvins to Celsius."""
    return ma.array(t) - 273.15


def vapor_pressure(
    pressure: npt.NDArray, specific_humidity: npt.NDArray
) -> npt.NDArray:
    """Calculate vapor pressure of water from pressure and specific humidity.

    Args:
        pressure: Pressure (Pa).
        specific_humidity: Specific humidity (kg kg-1).

    Returns:
        Vapor pressure (Pa).

    References:
        Cai, J. (2019). Humidity Measures.
        https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    """
    return (
        specific_humidity
        * pressure
        / (con.MW_RATIO + (1 - con.MW_RATIO) * specific_humidity)
    )


def saturation_vapor_pressure(
    temperature: npt.NDArray,
    phase: Literal["liquid", "ice", "mixed"] = "liquid",
) -> npt.NDArray:
    """Goff-Gratch formula for saturation vapor pressure adopted by WMO.

    Args:
        temperature: Temperature (K).
        phase: ``"liquid"`` for over water, ``"ice"`` for over ice, or
            ``"mixed"`` to automatically select ice below 273.16 K and
            liquid at or above.

    Returns:
        Saturation vapor pressure (Pa).

    References:
        Vömel, H. (2016). Saturation vapor pressure formulations.
        http://cires1.colorado.edu/~voemel/vp.html
    """
    ratio = con.T0 / temperature
    inv_ratio = ratio**-1

    if phase == "ice":
        return _svp_ice(ratio, inv_ratio)
    if phase == "mixed":
        return np.where(
            temperature < con.T0,
            _svp_ice(ratio, inv_ratio),
            _svp_liquid(ratio, inv_ratio),
        )
    return _svp_liquid(ratio, inv_ratio)


def _svp_liquid(ratio: npt.NDArray, inv_ratio: npt.NDArray) -> npt.NDArray:
    return (
        10
        ** (
            10.79574 * (1 - ratio)
            - 5.02800 * np.log10(inv_ratio)
            + 1.50475e-4 * (1 - 10 ** (-8.2969 * (inv_ratio - 1)))
            + 0.42873e-3 * (10 ** (4.76955 * (1 - ratio)) - 1)
            + 0.78614
        )
    ) * con.HPA_TO_PA


def _svp_ice(ratio: npt.NDArray, inv_ratio: npt.NDArray) -> npt.NDArray:
    return (
        10
        ** (
            -9.09718 * (ratio - 1)
            - 3.56654 * np.log10(ratio)
            + 0.876793 * (1 - inv_ratio)
            + np.log10(6.1071)
        )
    ) * con.HPA_TO_PA


def mixing_ratio(vp: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
    """Calculate mixing ratio from partial vapor pressure and pressure.

    Args:
        vp: Partial pressure of water vapor (Pa).
        pressure: Atmospheric pressure (Pa).

    Returns:
        Mixing ratio (kg kg-1).
    """
    return con.MW_RATIO * vp / (pressure - vp)


def relative_humidity(t: npt.NDArray, p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Calculate relative humidity over liquid water.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        q: Specific humidity (kg kg-1).

    Returns:
        Relative humidity (1).
    """
    return vapor_pressure(p, q) / saturation_vapor_pressure(t)


def absolute_humidity(t: npt.NDArray, vp: npt.NDArray) -> npt.NDArray:
    """Calculate absolute humidity from temperature and vapor pressure.

    Args:
        t: Temperature (K).
        vp: Water vapor pressure (Pa).

    Returns:
        Absolute humidity (kg m-3).
    """
    return vp / (con.RW * t)


def dew_point(t: npt.NDArray, rh: npt.NDArray) -> npt.NDArray:
    """Calculate dew point temperature using the Magnus formula.

    Args:
        t: Temperature (K).
        rh: Relative humidity (1).

    Returns:
        Dew point temperature (K).

    References:
        Alduchov, O. A., & Eskridge, R. E. (1996). Improved Magnus form
        approximation of saturation vapor pressure. J. Appl. Meteor., 35,
        601-609. https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2
    """
    a = 17.625
    b = 243.04
    tc = k2c(t)
    alpha = np.log(rh) + (a * tc) / (b + tc)
    return c2k((b * alpha) / (a - alpha))


def latent_heat_vaporization(t: npt.NDArray) -> npt.NDArray:
    """Calculate temperature-dependent latent heat of vaporization.

    Args:
        t: Temperature (K).

    Returns:
        Latent heat of vaporization (J kg-1).
    """
    return con.LATENT_HEAT_0 - 2420.0 * (t - con.T0)


def air_density(
    pressure: npt.NDArray,
    temperature: npt.NDArray,
    mr: npt.NDArray,
) -> npt.NDArray:
    """Calculate moist-air density.

    Args:
        pressure: Pressure (Pa).
        temperature: Temperature (K).
        mr: Water vapor mixing ratio (kg kg-1).

    Returns:
        Air density (kg m-3).
    """
    return pressure / (con.RS * temperature * (1 + _EPSILON_V * mr))


def virtual_temperature(t: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Calculate virtual temperature from temperature and specific humidity.

    Args:
        t: Temperature (K).
        q: Specific humidity (kg kg-1).

    Returns:
        Virtual temperature (K).
    """
    return t * (1 + _EPSILON_V * q)


def potential_temperature(
    t: npt.NDArray, p: npt.NDArray, p0: float = con.P0
) -> npt.NDArray:
    """Calculate potential temperature.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        p0: Reference pressure (Pa). Defaults to standard sea-level pressure.

    Returns:
        Potential temperature (K).
    """
    return t * (p0 / p) ** (con.RS / con.CP_DRY)


def equivalent_potential_temperature(
    t: npt.NDArray, p: npt.NDArray, q: npt.NDArray
) -> npt.NDArray:
    """Calculate equivalent potential temperature.

    Uses the first-order linearization of the simplified moist-adiabatic
    expression: ``theta_e ≈ theta * (1 + Lv * r / (Cp * T))``. Accurate enough
    for most boundary-layer applications; for higher precision see Bolton
    (1980), https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        q: Specific humidity (kg kg-1).

    Returns:
        Equivalent potential temperature (K).
    """
    theta = potential_temperature(t, p)
    vp = vapor_pressure(p, q)
    mr = mixing_ratio(vp, p)
    lv = latent_heat_vaporization(t)
    return theta * (1 + lv * mr / (con.CP_DRY * t))


def wet_bulb_temperature(t: npt.NDArray, p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Calculate wet-bulb temperature iteratively.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        q: Specific humidity (kg/kg).

    Returns:
        Wet-bulb temperature (K).

    References:
        Al-Ismaili, A. M., & Al-Azri, N. A. (2016). Simple Iterative Approach to
        Calculate Wet-Bulb Temperature for Estimating Evaporative Cooling
        Efficiency. Int. J. Agric. Innovations Res., 4, 1013-1018.
    """
    td = k2c(t)
    vp = vapor_pressure(p, q)
    W = mixing_ratio(vp, p)
    L_v_0 = con.LATENT_HEAT_0  # Latent heat of vaporization at 0degC (J kg-1)

    def f(tw: npt.NDArray) -> npt.NDArray:
        svp = saturation_vapor_pressure(c2k(tw))
        W_s = mixing_ratio(svp, p)
        C_p_w = 0.0265 * tw**2 - 1.7688 * tw + 4205.6  # Eq. 6 (J kg-1 C-1)
        C_p_wv = 0.0016 * td**2 + 0.1546 * td + 1858.7  # Eq. 7 (J kg-1 C-1)
        C_p_da = 0.0667 * ((td + tw) / 2) + 1005  # Eq. 8 (J kg-1 C-1)
        a = (L_v_0 - (C_p_w - C_p_wv) * tw) * W_s - C_p_da * (td - tw)
        b = L_v_0 + C_p_wv * td - C_p_w * tw
        return a / b - W

    min_err = 1e-6 * np.maximum(np.abs(td), 1)
    delta = 1e-8
    tw = td
    max_iter = 20
    for _ in range(max_iter):
        f_tw = f(tw)
        if np.all(np.abs(f_tw) < min_err):
            break
        df_tw = (f(tw + delta) - f_tw) / delta
        tw = tw - f_tw / df_tw
    else:
        msg = (
            "Wet-bulb temperature didn't converge after %d iterations: "
            "error min %g, max %g, mean %g, median %g"
        )
        logger.warning(
            msg, max_iter, np.min(f_tw), np.max(f_tw), np.mean(f_tw), np.median(f_tw)
        )

    return c2k(tw)


def adiabatic_dlwc_dz(temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
    """Return adiabatic vertical gradient of liquid water content (dLWC/dz).

    Calculates the theoretical adiabatic rate of increase of LWC with height,
    given the cloud base temperature and pressure.

    Args:
        temperature: Temperature of cloud base (K).
        pressure: Pressure of cloud base (Pa).

    Returns:
        dlwc/dz (kg m-3 m-1).

    References:
        Brenguier, 1991, https://doi.org/10.1175/1520-0469(1991)048<0264:POTCPA>2.0.CO;2
    """
    svp = saturation_vapor_pressure(temperature)
    svp_mr = mixing_ratio(svp, pressure)
    rho = air_density(pressure, temperature, svp_mr)
    Lv = latent_heat_vaporization(temperature)

    qs = svp_mr  # kg kg-1
    pa = rho  # kg m-3
    es = svp  # Pa
    P = pressure  # Pa
    T = temperature  # K

    # See Appendix B in Brenguier (1991) for the derivation
    dqs_dp = (
        -(1 - (con.CP_DRY * T) / (con.MW_RATIO * Lv))
        * (((con.CP_DRY * T) / (con.MW_RATIO * Lv)) + ((Lv * qs * pa) / (P - es))) ** -1
        * (con.MW_RATIO * es)
        * (P - es) ** -2
    )

    # Hydrostatic equation to convert dqs_dp to dqs_dz
    dqs_dz = dqs_dp * rho * -con.G

    return dqs_dz * rho


def geometric_height(gph: npt.NDArray) -> npt.NDArray:
    """Convert geopotential height to geometric height.

    Args:
        gph: Geopotential height (m).

    Returns:
        Geometric height (m).

    References:
        ECMWF (2023). ERA5: compute pressure and geopotential on model levels,
        geopotential height and geometric height. https://confluence.ecmwf.int/x/JJh0CQ
    """
    return con.EARTH_RADIUS * gph / (con.EARTH_RADIUS - gph)


def isa_altitude(temperature: float, pressure: float) -> float:
    """Calculate altitude from observed pressure and temperature.

    Uses the International Standard Atmosphere (ISA) hypsometric formula.

    Args:
        temperature: Observed temperature (K).
        pressure: Observed atmospheric pressure (Pa).

    Returns:
        Altitude (m).
    """
    L = 0.0065  # Temperature lapse rate (K/m)
    return (temperature / L) * (1 - (pressure / con.P0) ** (con.RS * L / con.G))
