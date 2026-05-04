"""Atmospheric attenuation coefficients (ITU-R P.676 / P.838 / P.840)."""

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from atmoslib import constants as con

POLARIZATION: TypeAlias = Literal["horizontal", "vertical", "circular"]

_TAU: dict[POLARIZATION, float] = {
    "horizontal": 0.0,  # 0 degrees
    "vertical": np.pi / 2,  # 90 degrees
    "circular": np.pi / 4,  # 45 degrees
}


def liquid_water_specific_attenuation(
    t: npt.NDArray, f: float | np.floating
) -> npt.NDArray:
    """Calculate cloud liquid water specific attenuation coefficient.

    Valid for frequency up to 200 GHz.

    Args:
        t: Temperature (K).
        f: Frequency (GHz).

    Returns:
        Cloud liquid water specific attenuation coefficient ((dB km-1)/(g m-3)).

    References:
        ITU-R P.840-9: Attenuation due to clouds and fog.
        https://www.itu.int/rec/R-REC-P.840-9-202308-I/en
    """
    theta1 = 300 / t - 1
    e0 = 77.66 + 103.3 * theta1
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146 * theta1 + 316 * theta1**2
    fs = 39.8 * fp
    ei = f * (e0 - e1) / (fp * (1 + (f / fp) ** 2)) + f * (e1 - e2) / (
        fs * (1 + (f / fs) ** 2)
    )
    er = (e0 - e1) / (1 + (f / fp) ** 2) + (e1 - e2) / (1 + (f / fs) ** 2) + e2
    eta = (2 + er) / ei
    return 0.819 * f / (ei * (1 + eta**2))


def gas_specific_attenuation(
    t: npt.NDArray,
    p: npt.NDArray,
    e: npt.NDArray,
    f: float | np.floating,
) -> npt.NDArray:
    """Calculate specific attenuation due to dry air and water vapor.

    Valid for frequency up to 1000 GHz.

    Args:
        t: Temperature (K).
        p: Pressure (Pa).
        e: Water vapor partial pressure (Pa).
        f: Frequency (GHz).

    Returns:
        Gas specific attenuation (dB km-1).

    References:
        ITU-R P.676-13: Attenuation by atmospheric gases and related effects.
        https://www.itu.int/rec/R-REC-P.676-13-202208-I/en
    """
    p = p * con.PA_TO_HPA
    e = e * con.PA_TO_HPA
    pd = p - e
    theta = 300 / t
    oxygen = _oxygen_refractivity(pd, e, f, theta)
    vapor = _vapor_refractivity(pd, e, f, theta)
    return 0.1820 * f * (oxygen + vapor)


def rain_specific_attenuation(
    r: float | npt.NDArray,
    f: float | np.floating,
    polarization: POLARIZATION = "horizontal",
    elevation: float = 90.0,
) -> npt.NDArray:
    """Calculate rain specific attenuation.

    Valid for frequency 1-1000 GHz.

    Args:
        r: Rain rate (mm h-1).
        f: Frequency (GHz).
        polarization: Wave polarization. ``"horizontal"`` (default),
            ``"vertical"``, or ``"circular"``.
        elevation: Path elevation angle from the horizontal (degrees).
            Defaults to 90 (zenith), for which the polarization choice has
            no effect. Use 0 for a horizontal path (e.g., terrestrial link).

    Returns:
        Rain specific attenuation (dB km-1).

    References:
        ITU-R P.838-3: Specific attenuation model for rain for use in
        prediction methods.
        https://www.itu.int/rec/R-REC-P.838-3-200503-I/en
    """
    log_f = np.log10(f)
    k_h = 10 ** _p838_log_sum(log_f, *_P838_K_H)
    k_v = 10 ** _p838_log_sum(log_f, *_P838_K_V)
    a_h = _p838_log_sum(log_f, *_P838_ALPHA_H)
    a_v = _p838_log_sum(log_f, *_P838_ALPHA_V)

    weight = np.cos(np.deg2rad(elevation)) ** 2 * np.cos(2 * _TAU[polarization])

    kh_ah = k_h * a_h
    kv_av = k_v * a_v
    k = (k_h + k_v + (k_h - k_v) * weight) / 2
    alpha = (kh_ah + kv_av + (kh_ah - kv_av) * weight) / (2 * k)
    return k * r**alpha


def _p838_log_sum(
    log_f: float | np.floating,
    gaussians: npt.NDArray,
    m: float,
    c: float,
) -> npt.NDArray:
    a, b, sigma = gaussians.T
    return np.sum(a * np.exp(-(((log_f - b) / sigma) ** 2))) + m * log_f + c


def _line_shape(
    f: float | np.floating,
    center: npt.NDArray,
    width: npt.NDArray,
    correction: npt.NDArray | float,
) -> npt.NDArray:
    return (
        f
        / center
        * (
            (width - correction * (center - f)) / ((center - f) ** 2 + width**2)
            + (width - correction * (center + f)) / ((center + f) ** 2 + width**2)
        )
    )


def _oxygen_refractivity(
    pd: npt.NDArray,
    e: npt.NDArray,
    f: float | np.floating,
    theta: npt.NDArray,
) -> npt.NDArray:
    f0, a1, a2, a3, a4, a5, a6 = _OXYGEN_TABLE[:, :, np.newaxis, np.newaxis]
    strength = a1 * 1e-7 * pd * theta**3 * np.exp(a2 * (1 - theta))
    width = a3 * 1e-4 * (pd * theta ** (0.8 - a4) + 1.1 * e * theta)
    width = np.sqrt(width**2 + 2.25e-6)
    correction = (a5 + a6 * theta) * 1e-4 * (pd + e) * theta**0.8
    shape = _line_shape(f, f0, width, correction)
    d = 5.6e-4 * (pd + e) * theta**0.8
    continuum = (
        f
        * pd
        * theta**2
        * (
            6.14e-5 / (d * (1 + (f / d) ** 2))
            + ((1.4e-12 * pd * theta**1.5) / (1 + 1.9e-5 * f**1.5))
        )
    )
    return np.sum(strength * shape, axis=0) + continuum


def _vapor_refractivity(
    pd: npt.NDArray,
    e: npt.NDArray,
    f: float | np.floating,
    theta: npt.NDArray,
) -> npt.NDArray:
    f0, b1, b2, b3, b4, b5, b6 = _VAPOR_TABLE[:, :, np.newaxis, np.newaxis]
    strength = b1 * 1e-1 * e * theta**3.5 * np.exp(b2 * (1 - theta))
    width = b3 * 1e-4 * (pd * theta**b4 + b5 * e * theta**b6)
    width = 0.535 * width + np.sqrt(0.217 * width**2 + (2.1316e-12 * f0**2) / theta)
    correction = 0.0
    shape = _line_shape(f, f0, width, correction)
    return np.sum(strength * shape, axis=0)


# Spectroscopic data for oxygen attenuation lines (ITU-R P.676-13 Table 1).
_OXYGEN_TABLE = np.array(
    [
        [50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850],
        [50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800],
        [51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729],
        [52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640],
        [52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526],
        [53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206],
        [53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085],
        [54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750],
        [54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654],
        [55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952],
        [55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135],
        [56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978],
        [56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547],
        [56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451],
        [57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056],
        [58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436],
        [58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273],
        [59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309],
        [59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776],
        [60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699],
        [60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825],
        [61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584],
        [61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619],
        [62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759],
        [62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844],
        [62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675],
        [63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139],
        [64.127775, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895],
        [64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590],
        [65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680],
        [65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002],
        [66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091],
        [66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393],
        [67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475],
        [67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545],
        [68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600],
        [68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650],
        [118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079],
        [368.498246, 67.400, 0.048, 16.400, 0.0, 0.000, 0.000],
        [424.763020, 637.700, 0.044, 16.400, 0.0, 0.000, 0.000],
        [487.249273, 237.400, 0.049, 16.000, 0.0, 0.000, 0.000],
        [715.392902, 98.100, 0.145, 16.000, 0.0, 0.000, 0.000],
        [773.839490, 572.300, 0.141, 16.200, 0.0, 0.000, 0.000],
        [834.145546, 183.100, 0.145, 14.700, 0.0, 0.000, 0.000],
    ]
).T

# Spectroscopic data for water vapor attenuation lines (ITU-R P.676-13 Table 2).
_VAPOR_TABLE = np.array(
    [
        [22.235080, 0.1079, 2.144, 26.38, 0.76, 5.087, 1.00],
        [67.803960, 0.0011, 8.732, 28.58, 0.69, 4.930, 0.82],
        [119.995940, 0.0007, 8.353, 29.48, 0.70, 4.780, 0.79],
        [183.310087, 2.273, 0.668, 29.06, 0.77, 5.022, 0.85],
        [321.225630, 0.0470, 6.179, 24.04, 0.67, 4.398, 0.54],
        [325.152888, 1.514, 1.541, 28.23, 0.64, 4.893, 0.74],
        [336.227764, 0.0010, 9.825, 26.93, 0.69, 4.740, 0.61],
        [380.197353, 11.67, 1.048, 28.11, 0.54, 5.063, 0.89],
        [390.134508, 0.0045, 7.347, 21.52, 0.63, 4.810, 0.55],
        [437.346667, 0.0632, 5.048, 18.45, 0.60, 4.230, 0.48],
        [439.150807, 0.9098, 3.595, 20.07, 0.63, 4.483, 0.52],
        [443.018343, 0.1920, 5.048, 15.55, 0.60, 5.083, 0.50],
        [448.001085, 10.41, 1.405, 25.64, 0.66, 5.028, 0.67],
        [470.888999, 0.3254, 3.597, 21.34, 0.66, 4.506, 0.65],
        [474.689092, 1.260, 2.379, 23.20, 0.65, 4.804, 0.64],
        [488.490108, 0.2529, 2.852, 25.86, 0.69, 5.201, 0.72],
        [503.568532, 0.0372, 6.731, 16.12, 0.61, 3.980, 0.43],
        [504.482692, 0.0124, 6.731, 16.12, 0.61, 4.010, 0.45],
        [547.676440, 0.9785, 0.158, 26.00, 0.70, 4.500, 1.00],
        [552.020960, 0.1840, 0.158, 26.00, 0.70, 4.500, 1.00],
        [556.935985, 497.0, 0.159, 30.86, 0.69, 4.552, 1.00],
        [620.700807, 5.015, 2.391, 24.38, 0.71, 4.856, 0.68],
        [645.766085, 0.0067, 8.633, 18.00, 0.60, 4.000, 0.50],
        [658.005280, 0.2732, 7.816, 32.10, 0.69, 4.140, 1.00],
        [752.033113, 243.4, 0.396, 30.86, 0.68, 4.352, 0.84],
        [841.051732, 0.0134, 8.177, 15.90, 0.33, 5.760, 0.45],
        [859.965698, 0.1325, 8.055, 30.60, 0.68, 4.090, 0.84],
        [899.303175, 0.0547, 7.914, 29.85, 0.68, 4.530, 0.90],
        [902.611085, 0.0386, 8.429, 28.65, 0.70, 5.100, 0.95],
        [906.205957, 0.1836, 5.110, 24.08, 0.70, 4.700, 0.53],
        [916.171582, 8.400, 1.441, 26.73, 0.70, 5.150, 0.78],
        [923.112692, 0.0079, 10.293, 29.00, 0.70, 5.000, 0.80],
        [970.315022, 9.009, 1.919, 25.50, 0.64, 4.940, 0.67],
        [987.926764, 134.6, 0.257, 29.85, 0.68, 4.550, 0.90],
        [1780.000000, 17506.0, 0.952, 196.3, 2.00, 24.15, 5.00],
    ]
).T

# Coefficients for rain specific attenuation (ITU-R P.838-3 Table 1).
_P838_K_H = (
    np.array(
        [
            [-5.33980, -0.10008, 1.13098],
            [-0.35351, 1.26970, 0.45400],
            [-0.23789, 0.86036, 0.15354],
            [-0.94158, 0.64552, 0.16817],
        ]
    ),
    -0.18961,
    0.71147,
)

# Coefficients for rain specific attenuation (ITU-R P.838-3 Table 2).
_P838_K_V = (
    np.array(
        [
            [-3.80595, 0.56934, 0.81061],
            [-3.44965, -0.22911, 0.51059],
            [-0.39902, 0.73042, 0.11899],
            [0.50167, 1.07319, 0.27195],
        ]
    ),
    -0.16398,
    0.63297,
)

# Coefficients for rain specific attenuation (ITU-R P.838-3 Table 3).
_P838_ALPHA_H = (
    np.array(
        [
            [-0.14318, 1.82442, -0.55187],
            [0.29591, 0.77564, 0.19822],
            [0.32177, 0.63773, 0.13164],
            [-5.37610, -0.96230, 1.47828],
            [16.1721, -3.29980, 3.43990],
        ]
    ),
    0.67849,
    -1.95537,
)

# Coefficients for rain specific attenuation (ITU-R P.838-3 Table 4).
_P838_ALPHA_V = (
    np.array(
        [
            [-0.07771, 2.33840, -0.76284],
            [0.56727, 0.95545, 0.54039],
            [-0.20238, 1.14520, 0.26809],
            [-48.2991, 0.791669, 0.116226],
            [48.5833, 0.791459, 0.116479],
        ]
    ),
    -0.053739,
    0.83433,
)
