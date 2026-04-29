"""Physical constants used in atmospheric thermodynamic calculations."""

from typing import Final

# Triple point of water (K)
T0: Final = 273.16

# Ratio of the molecular weight of water vapor to dry air
MW_RATIO: Final = 0.62198

# Specific gas constant for dry air (J kg-1 K-1)
RS: Final = 287.058

# Standard atmospheric pressure at sea level (Pa)
P0: Final = 101325

# Standard gravitational acceleration (m s-2)
G: Final = 9.80665

HPA_TO_PA: Final = 100
PA_TO_HPA: Final = 1 / HPA_TO_PA

# Radius of the Earth (m) as assumed in ECMWF IFS
EARTH_RADIUS: Final = 6_371_229
