"""Physical constants used in atmospheric thermodynamic calculations."""

from typing import Final

# Triple point of water (K)
T0: Final = 273.16

# Ratio of the molecular weight of water vapor to dry air
MW_RATIO: Final = 0.62198

# Specific gas constant for dry air (J kg-1 K-1)
RS: Final = 287.058

# Specific gas constant for water vapor (J kg-1 K-1)
RW: Final = RS / MW_RATIO

# Specific heat of dry air at constant pressure (J kg-1 K-1)
CP_DRY: Final = 1004.0

# Latent heat of vaporization at the triple point T0 (J kg-1)
LATENT_HEAT_0: Final = 2.501e6

# Standard atmospheric pressure at sea level (Pa)
P0: Final = 101325

# Standard temperature at sea level (K)
T_STD: Final = 288.15

# Standard gravitational acceleration (m s-2)
G: Final = 9.80665

HPA_TO_PA: Final = 100
PA_TO_HPA: Final = 1 / HPA_TO_PA

# Radius of the Earth (m) as assumed in ECMWF IFS
EARTH_RADIUS: Final = 6_371_229

# Temperature lapse rate in troposphere (K m-1)
L0: Final = 0.0065
