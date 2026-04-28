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

tw = atmoslib.calc_wet_bulb_temperature(t, p, q)
```

All inputs accept scalars or NumPy arrays of any shape (broadcasting follows
NumPy rules).

## Available functions

| Function                                   | Description                                            |
| ------------------------------------------ | ------------------------------------------------------ |
| `calc_wet_bulb_temperature(t, p, q)`       | Wet-bulb temperature from `t`, `p`, `q` (K)            |
| `calc_vapor_pressure(p, q)`                | Vapor pressure of water (Pa)                           |
| `calc_saturation_vapor_pressure(t)`        | Saturation vapor pressure over water (Pa, Goff-Gratch) |
| `calc_mixing_ratio(vp, p)`                 | Mixing ratio (kg/kg)                                   |
| `calc_air_density(p, t, svp_mixing_ratio)` | Air density (kg/m³)                                    |
| `calc_lwc_change_rate(t, p)`               | Adiabatic dLWC/dz at cloud base (kg/m³/m)              |
| `calc_altitude(t, p)`                      | Altitude from pressure and temperature (m, ISA)        |
| `c2k(temp)` / `k2c(temp)`                  | Celsius ↔ Kelvin conversion                           |

See the docstrings for argument details and references.

## License

MIT
