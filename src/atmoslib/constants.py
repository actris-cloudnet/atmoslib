"""Physical constants used in atmospheric thermodynamic calculations."""

from typing import Final

T0: Final = 273.16
"Triple point of water (K)"

MW_RATIO: Final = 0.62198
"Ratio of the molecular weight of water vapor to dry air"

RS: Final = 287.058
"Specific gas constant for dry air (J kg-1 K-1)"

RW: Final = RS / MW_RATIO
"Specific gas constant for water vapor (J kg-1 K-1)"

CP_DRY: Final = 1004.0
"""Specific heat of dry air at constant pressure (J kg-1 K-1).
Value from Wallace & Hobbs (2006), Atmospheric Science: An Introductory
Survey, 2nd ed., Appendix A. The AMS Glossary / NIST experimental value
at 273 K is 1005.7 +/- 2.5; the 0.2% difference is negligible here.
Latent heat of vaporization at the triple point T0 (J kg-1)."""

LATENT_HEAT_0: Final = 2.501e6
"Specific latent heat of vaporization of water at 0 degC (J kg-1)"

P0: Final = 101325
"Standard atmospheric pressure at sea level (Pa)"

RHO_STD: Final = 1.225
"Standard air density at sea level (kg m-3)"

P_REF: Final = 100000
"""Meteorological reference pressure (Pa, 1000 hPa). Distinct from P0
(ICAO standard sea-level pressure)."""

T_STD: Final = 288.15
"Standard temperature at sea level (K)"

G: Final = 9.80665
"Standard gravitational acceleration (m s-2)"

HPA_TO_PA: Final = 100
PA_TO_HPA: Final = 1 / HPA_TO_PA

EARTH_RADIUS: Final = 6_371_229
"Radius of the Earth (m) as assumed in ECMWF IFS"

L0: Final = 0.0065
"Temperature lapse rate in troposphere (K m-1)"
