<h1>
  <img src="logo.png" alt="" height="48" align="absmiddle">
  &nbsp;atmoslib
</h1>

[![Run tests](https://github.com/actris-cloudnet/atmoslib/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/atmoslib/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/atmoslib.svg)](https://pypi.org/project/atmoslib/)

Python library for atmospheric thermodynamics and microwave attenuation
calculations. This library is used in
[CloudnetPy](https://github.com/actris-cloudnet/cloudnetpy) and related
projects with focus on good performance and minimal dependencies.

## Installation

```sh
pip install atmoslib
```

## Usage

```python
import numpy as np
import atmoslib

t = np.array([293.15, 283.15, 273.15])     # temperature (K)
p = np.array([101325.0, 95000.0, 90000.0]) # pressure (Pa)
q = np.array([0.010, 0.005, 0.001])        # specific humidity (kg/kg)

tw = atmoslib.wet_bulb_temperature(t, p, q)
```

All inputs accept scalars or NumPy arrays of any shape (broadcasting follows
NumPy rules).

## Available functions

| Function                                    | Description                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| `wet_bulb_temperature(t, p, q)`             | Wet-bulb temperature (K)                                                 |
| `vapor_pressure(p, q)`                      | Vapor pressure of water (Pa)                                             |
| `saturation_vapor_pressure(t)`              | Saturation vapor pressure (Pa, Goff-Gratch, liquid/ice/mixed)            |
| `relative_humidity(t, p, q)`                | Relative humidity (1, liquid/ice/mixed)                                  |
| `specific_humidity(t, p, rh)`               | Specific humidity (kg/kg, liquid/ice/mixed)                              |
| `absolute_humidity(t, vp)`                  | Absolute humidity (kg/m³)                                                |
| `dew_point_temperature(t, rh)`              | Dew-point temperature (K, Magnus)                                        |
| `mixing_ratio(vp, p)`                       | Mixing ratio (kg/kg)                                                     |
| `latent_heat_of_vaporization(t)`            | Temperature-dependent latent heat of vaporization (J/kg)                 |
| `virtual_temperature(t, q)`                 | Virtual temperature (K)                                                  |
| `potential_temperature(t, p)`               | Potential temperature (K)                                                |
| `equivalent_potential_temperature(t, p, q)` | Equivalent potential temperature (K, Bolton 1980)                        |
| `air_density(t, p, mr)`                     | Moist-air density (kg/m³)                                                |
| `adiabatic_lwc_gradient(t, p)`              | Adiabatic vertical gradient of LWC at cloud base (kg/m³/m)               |
| `hydrostatic_pressure(t, q, z, p_sfc)`      | Pressure profile from surface value via hypsometric equation (Pa)        |
| `geometric_height(gph)`                     | Geopotential height to geometric height (m, ECMWF)                       |
| `isa_altitude(t, p)`                        | Altitude from pressure and temperature (gpm, ISA)                        |
| `isa_pressure(gph)`                         | Pressure from geopotential height (Pa, ISA)                              |
| `c2k(t)` / `k2c(t)`                         | Celsius ↔ Kelvin conversion                                             |
| `liquid_water_specific_attenuation(t, f)`   | Cloud liquid water specific attenuation ((dB km⁻¹)/(g m⁻³), ITU-R P.840) |
| `gas_specific_attenuation(t, p, e, f)`      | Dry-air + water-vapor specific attenuation (dB km⁻¹, ITU-R P.676)        |

See the docstrings for argument details and references.

## License

MIT
