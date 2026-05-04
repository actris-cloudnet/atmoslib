<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.svg?v=2">
    <img src="logo.svg?v=2" alt="" height="45" align="absmiddle">
  </picture>
  atmoslib
</h1>

[![CI](https://github.com/actris-cloudnet/atmoslib/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/atmoslib/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/atmoslib.svg)](https://pypi.org/project/atmoslib/)

Python library for atmospheric thermodynamics and microwave attenuation
calculations. This library is used in
[CloudnetPy](https://github.com/actris-cloudnet/cloudnetpy) and related
projects with focus on performance and minimal dependencies.

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

Most inputs accept scalars or NumPy arrays of any shape (broadcasting follows
NumPy rules).

## Available functions

| Function                                    | Description                                                                   |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| `vapor_pressure(p, q)`                      | Vapor pressure of water (Pa)                                                  |
| `saturation_vapor_pressure(t)`              | Saturation vapor pressure (Pa, Goff-Gratch, liquid/ice/mixed)                 |
| `mixing_ratio(vp, p)`                       | Mixing ratio (kg kg⁻¹)                                                        |
| `specific_humidity(t, p, rh)`               | Specific humidity (kg kg⁻¹, liquid/ice/mixed)                                 |
| `relative_humidity(t, p, q)`                | Relative humidity (0–1, liquid/ice/mixed)                                     |
| `absolute_humidity(t, vp)`                  | Absolute humidity (kg m⁻³)                                                    |
| `dew_point_temperature(t, rh)`              | Dew-point temperature (K, Magnus)                                             |
| `virtual_temperature(t, q)`                 | Virtual temperature (K)                                                       |
| `potential_temperature(t, p)`               | Potential temperature (K)                                                     |
| `equivalent_potential_temperature(t, p, q)` | Equivalent potential temperature (K, Bolton 1980)                             |
| `wet_bulb_temperature(t, p, q)`             | Wet-bulb temperature (K, iterative)                                           |
| `latent_heat_of_vaporization(t)`            | Latent heat of vaporization (J kg⁻¹)                                          |
| `air_density(t, p, mr)`                     | Moist-air density (kg m⁻³)                                                    |
| `adiabatic_lwc_gradient(t, p)`              | Adiabatic vertical gradient of LWC at cloud base (kg m⁻³ m⁻¹, Brenguier 1991) |
| `hydrostatic_pressure(t, q, z, p_sfc)`      | Pressure profile from surface value via hypsometric equation (Pa)             |
| `isa_pressure(gph)`                         | Pressure from geopotential height (Pa, ISA)                                   |
| `isa_altitude(t, p)`                        | Geopotential height from pressure and temperature (gpm, ISA)                  |
| `geometric_height(gph)`                     | Geometric height from geopotential height (m, ECMWF)                          |
| `c2k(t)` / `k2c(t)`                         | Celsius ↔ Kelvin conversion                                                  |
| `liquid_water_specific_attenuation(t, f)`   | Cloud liquid water specific attenuation (dB km⁻¹ per g m⁻³, ITU-R P.840)      |
| `gas_specific_attenuation(t, p, e, f)`      | Dry-air + water-vapor specific attenuation (dB km⁻¹, ITU-R P.676)             |
| `rain_specific_attenuation(R, f)`           | Rain specific attenuation (dB km⁻¹, ITU-R P.838)                              |

See the docstrings for argument details and references.

## License

MIT
