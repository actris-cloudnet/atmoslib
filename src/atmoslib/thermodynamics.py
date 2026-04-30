import logging
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy import ma

from atmoslib import constants as con

logger = logging.getLogger(__name__)

# (R_vapor / R_dry - 1), used in moist-air virtual-temperature corrections
_EPSILON_V = (1 - con.MW_RATIO) / con.MW_RATIO

PHASE: TypeAlias = Literal["liquid", "ice", "mixed"]


def c2k(t: npt.NDArray) -> npt.NDArray:
    """Converts Celsius to Kelvins."""
    return ma.array(t) + 273.15


def k2c(t: npt.NDArray) -> npt.NDArray:
    """Converts Kelvins to Celsius."""
    return ma.array(t) - 273.15


def vapor_pressure(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Calculate vapor pressure of water from pressure and specific humidity.

    Args:
        p: Pressure (Pa).
        q: Specific humidity (kg kg-1).

    Returns:
        Vapor pressure (Pa).

    References:
        Cai, J. (2019). Humidity Measures.
        https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    """
    return q * p / (con.MW_RATIO + (1 - con.MW_RATIO) * q)


def saturation_vapor_pressure(t: npt.NDArray, phase: PHASE = "liquid") -> npt.NDArray:
    """Goff-Gratch formula for saturation vapor pressure adopted by WMO.

    Args:
        t: Temperature (K).
        phase: ``"liquid"`` for over water (default), ``"ice"`` for over ice,
            or ``"mixed"`` to automatically select ice below 273.16 K and
            liquid at or above.

    Returns:
        Saturation vapor pressure (Pa).

    References:
        Vömel, H. (2016). Saturation vapor pressure formulations.
        http://cires1.colorado.edu/~voemel/vp.html
    """
    ratio = con.T0 / t
    inv_ratio = ratio**-1

    if phase == "liquid":
        return _svp_liquid(ratio, inv_ratio)
    if phase == "ice":
        return _svp_ice(ratio, inv_ratio)
    if phase == "mixed":
        return np.where(
            t < con.T0,
            _svp_ice(ratio, inv_ratio),
            _svp_liquid(ratio, inv_ratio),
        )
    msg = "phase should be liquid, ice or mixed"
    raise ValueError(msg)


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


def mixing_ratio(vp: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
    """Calculate mixing ratio from partial vapor pressure and pressure.

    Args:
        vp: Partial pressure of water vapor (Pa).
        p: Atmospheric pressure (Pa).

    Returns:
        Mixing ratio (kg kg-1).
    """
    return con.MW_RATIO * vp / (p - vp)


def relative_humidity(
    t: npt.NDArray, p: npt.NDArray, q: npt.NDArray, phase: PHASE = "liquid"
) -> npt.NDArray:
    """Calculate relative humidity from specific humidity.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        q: Specific humidity (kg kg-1).
        phase: ``"liquid"`` for over water (default), ``"ice"`` for over ice,
            or ``"mixed"`` to automatically select ice below 273.16 K and
            liquid at or above.

    Returns:
        Relative humidity (1).
    """
    return vapor_pressure(p, q) / saturation_vapor_pressure(t, phase)


def specific_humidity(
    t: npt.NDArray, p: npt.NDArray, rh: npt.NDArray, phase: PHASE = "liquid"
) -> npt.NDArray:
    """Calculate specific humidity from relative humidity.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        rh: Relative humidity (1).
        phase: ``"liquid"`` for over water (default), ``"ice"`` for over ice,
            or ``"mixed"`` to automatically select ice below 273.16 K and
            liquid at or above.

    Returns:
        Specific humidity (kg kg-1).
    """
    vp = rh * saturation_vapor_pressure(t, phase)
    return (con.MW_RATIO * vp) / (p - (1 - con.MW_RATIO) * vp)


def absolute_humidity(t: npt.NDArray, vp: npt.NDArray) -> npt.NDArray:
    """Calculate absolute humidity from temperature and vapor pressure.

    Args:
        t: Temperature (K).
        vp: Water vapor pressure (Pa).

    Returns:
        Absolute humidity (kg m-3).
    """
    return vp / (con.RW * t)


def dew_point_temperature(t: npt.NDArray, rh: npt.NDArray) -> npt.NDArray:
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


def latent_heat_of_vaporization(t: npt.NDArray) -> npt.NDArray:
    """Calculate temperature-dependent latent heat of vaporization.

    Args:
        t: Temperature (K).

    Returns:
        Latent heat of vaporization (J kg-1).
    """
    return con.LATENT_HEAT_0 - 2420.0 * (t - con.T0)


def air_density(
    t: npt.NDArray,
    p: npt.NDArray,
    mr: npt.NDArray,
) -> npt.NDArray:
    """Calculate moist-air density.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        mr: Water vapor mixing ratio (kg kg-1).

    Returns:
        Air density (kg m-3).
    """
    return p / (con.RS * t * (1 + _EPSILON_V * mr))


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
    lv = latent_heat_of_vaporization(t)
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


def adiabatic_lwc_gradient(t: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
    """Return adiabatic vertical gradient of liquid water content (dLWC/dz).

    Calculates the theoretical adiabatic rate of increase of LWC with height,
    given the cloud base temperature and pressure.

    Args:
        t: Temperature of cloud base (K).
        p: Pressure of cloud base (Pa).

    Returns:
        dlwc/dz (kg m-3 m-1).

    References:
        Brenguier, 1991, https://doi.org/10.1175/1520-0469(1991)048<0264:POTCPA>2.0.CO;2
    """
    svp = saturation_vapor_pressure(t)
    svp_mr = mixing_ratio(svp, p)
    rho = air_density(t, p, svp_mr)
    Lv = latent_heat_of_vaporization(t)

    qs = svp_mr  # kg kg-1
    pa = rho  # kg m-3
    es = svp  # Pa

    # See Appendix B in Brenguier (1991) for the derivation
    dqs_dp = (
        -(1 - (con.CP_DRY * t) / (con.MW_RATIO * Lv))
        * (((con.CP_DRY * t) / (con.MW_RATIO * Lv)) + ((Lv * qs * pa) / (p - es))) ** -1
        * (con.MW_RATIO * es)
        * (p - es) ** -2
    )

    # Hydrostatic equation to convert dqs_dp to dqs_dz
    dqs_dz = dqs_dp * rho * -con.G

    return dqs_dz * rho


def geometric_height(gph: npt.NDArray) -> npt.NDArray:
    """Convert geopotential height to geometric height.

    Args:
        gph: Geopotential height (gpm).

    Returns:
        Geometric height (m).

    References:
        ECMWF (2023). ERA5: compute pressure and geopotential on model levels,
        geopotential height and geometric height. https://confluence.ecmwf.int/x/JJh0CQ
    """
    return con.EARTH_RADIUS * gph / (con.EARTH_RADIUS - gph)


def hydrostatic_pressure(
    t: npt.NDArray,
    q: npt.NDArray,
    z: npt.NDArray,
    p_sfc: npt.NDArray,
) -> npt.NDArray:
    """Integrate pressure profile from a surface value via the hypsometric equation.

    Levels are integrated along the last axis using the mean virtual temperature
    of each adjacent pair of levels.

    Args:
        t: Temperature at each level (K). Last axis is vertical.
        q: Specific humidity at each level (kg kg-1). Same shape as ``t``.
        z: Geometric height of each level (m). Broadcasts against ``t`` along
            the last axis (typically a 1-D array of size ``t.shape[-1]``).
        p_sfc: Pressure at the lowest level (Pa). Shape must match ``t.shape[:-1]``.

    Returns:
        Pressure at each level (Pa), same shape as ``t``.

    References:
        Wallace, J. M., & Hobbs, P. V. (2006). Atmospheric Science: An
        Introductory Survey, 2nd ed., Section 3.2.
    """
    tv = virtual_temperature(t, q)
    tv_half = (tv[..., :-1] + tv[..., 1:]) / 2
    dz = np.diff(z, axis=-1)
    dp_ratio = np.exp(-con.G * dz / (con.RS * tv_half))
    tmp = np.insert(dp_ratio, 0, p_sfc, axis=-1)
    return np.cumprod(tmp, axis=-1)


def isa_altitude(t: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
    """Calculate altitude from observed pressure and temperature.

    Uses the International Standard Atmosphere (ISA) hypsometric formula. Only
    valid in troposphere up to 11 km.

    Args:
        t: Observed temperature (K).
        p: Observed atmospheric pressure (Pa).

    Returns:
        Geopotential height (gpm).
    """
    return (t / con.L0) * (1 - (p / con.P0) ** (con.RS * con.L0 / con.G))


def isa_pressure(gph: npt.NDArray) -> npt.NDArray:
    """Calculate atmospheric pressure at given altitude.

    Uses the International Standard Atmosphere (ISA) hypsometric formula. Only
    valid in troposphere up to 11 km.

    Args:
        gph: Geopotential height (gpm).

    Returns:
        Atmospheric pressure (Pa).
    """
    if np.any(gph >= 11_000):
        msg = "Valid only up to 11 km"
        raise ValueError(msg)
    return con.P0 * (1 - con.L0 * gph / con.T_STD) ** (con.G / (con.RS * con.L0))
