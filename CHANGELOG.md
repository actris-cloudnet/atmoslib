# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2.3.1 – 2026-05-02

- Seed wet-bulb solver with Stull (2011) initial guess

## 2.3.0 – 2026-05-02

- Use 1000 hPa as reference pressure for potential temperature

## 2.2.0 – 2026-05-02

- Use Bolton (1980) for equivalent potential temperature

## 2.1.0 – 2026-05-01

- Add `isa_pressure` function
- Add `specific_humidity` function
- Add `phase` argument to `relative_humidity`

## 2.0.0 – 2026-04-30

- Standardize public API: parameter orders and function names
- Add barometric pressure profile function

## 1.1.0 – 2026-04-29

- Add ITU-R P.676 and P.840 attenuation functions

## 1.0.1 – 2026-04-29

- Standardize parameter names to short symbols

## 1.0.0 – 2026-04-29

Initial release

## 0.0.1 – 2026-04-28

- Initial release with wet-bulb temperature calculation
