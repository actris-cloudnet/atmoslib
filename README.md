# atmoslib

[![Run tests](https://github.com/actris-cloudnet/atmoslib/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/atmoslib/actions/workflows/test.yml)

Python library for atmospheric thermodynamic calculations.

## Installation

```sh
pip install atmoslib
```

## Usage

```python
import numpy as np
import atmoslib

t = np.array([293.15, 283.15, 273.15])    # temperature (K)
p = np.array([101325.0, 95000.0, 90000.0]) # pressure (Pa)
q = np.array([0.010, 0.005, 0.001])        # specific humidity (kg/kg)

tw = atmoslib.wet_bulb_temperature(t, p, q)
```

All inputs accept scalars or NumPy arrays of any shape (broadcasting follows
NumPy rules).

## Available functions

| Function                        | Description                                                |
| ------------------------------- | ---------------------------------------------------------- |
| `wet_bulb_temperature(t, p, q)` | Wet-bulb temperature from `t`, `p`, `q` (K)                |
| `vapor_pressure(p, q)`          | Vapor pressure of water (Pa)                               |
| `saturation_vapor_pressure(t)`  | Saturation vapor pressure over water (Pa, Goff-Gratch)     |
| `mixing_ratio(vp, p)`           | Mixing ratio (kg/kg)                                       |
| `air_density(p, t, mr)`         | Moist-air density (kg/m³)                                  |
| `adiabatic_dlwc_dz(t, p)`       | Adiabatic vertical gradient of LWC at cloud base (kg/m³/m) |
| `geometric_height(height)`      | Geopotential height to geometric height (m, ECMWF)         |
| `isa_altitude(t, p)`            | Altitude from pressure and temperature (m, ISA)            |
| `c2k(temp)` / `k2c(temp)`       | Celsius ↔ Kelvin conversion                               |

See the docstrings for argument details and references.

## License

MIT
