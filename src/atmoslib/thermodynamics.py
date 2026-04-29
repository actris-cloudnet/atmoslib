import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.constants
from numpy import ma

from atmoslib import constants as con

logger = logging.getLogger(__name__)


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
    return pressure / (con.RS * temperature * (0.6 * mr + 1))


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

    e = 0.622
    Cp = 1004  # J kg-1 K-1
    Lv = 2.45e6  # J kg-1 = Pa m3 kg-1
    qs = svp_mr  # kg kg-1
    pa = rho  # kg m-3
    es = svp  # Pa
    P = pressure  # Pa
    T = temperature  # K

    # See Appendix B in Brenguier (1991) for the derivation
    dqs_dp = (
        -(1 - (Cp * T) / (e * Lv))
        * (((Cp * T) / (e * Lv)) + ((Lv * qs * pa) / (P - es))) ** -1
        * (e * es)
        * (P - es) ** -2
    )

    # Hydrostatic equation to convert dqs_dp to dqs_dz
    dqs_dz = dqs_dp * rho * -scipy.constants.g

    return dqs_dz * rho


def geometric_height(height: npt.NDArray) -> npt.NDArray:
    """Convert geopotential height to geometric height.

    Args:
        height: Geopotential height (m).

    Returns:
        Geometric height (m).

    References:
        ECMWF (2023). ERA5: compute pressure and geopotential on model levels,
        geopotential height and geometric height. https://confluence.ecmwf.int/x/JJh0CQ
    """
    return con.EARTH_RADIUS * height / (con.EARTH_RADIUS - height)


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
    L_v_0 = 2501e3  # Latent heat of vaporization at 0degC (J kg-1)

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
