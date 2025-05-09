# Data Description: Smart Factory Energy Dataset

This document outlines the meaning, units, and context of all columns in the dataset used for predicting equipment energy consumption in a smart manufacturing facility.



## Target Variable

* equipment_energy_consumption:
  Description: Total energy (kWh) used by manufacturing equipment at each timestamp.
  Type: Numeric (float)



## Auxiliary Energy Feature

* lighting_energy:
  Description: Energy used by factory lighting systems (kWh).
  Type: Numeric (float)



## Timestamp

* timestamp:
  Description: Datetime of the observation (e.g., '2016-01-11 17:00:00').
  Type: String → Parsed as datetime object


## Zone Sensor Features (per zone: 1 to 9)

For each zone:

* zoneN_temperature:
  Description: Measured air temperature in zone N (°C).
  Type: Numeric (float)

* zoneN_humidity:
  Description: Relative humidity (%) in zone N.
  Type: Numeric (float)

Zones may represent separate physical areas such as production lines, assembly zones, or material storage sections.


## Outdoor Environment Features

* outdoor_temperature:
  Description: External ambient air temperature (°C).
  Type: Numeric (float)

* outdoor_humidity:
  Description: External relative humidity (%).
  Type: Numeric (float)

* atmospheric_pressure:
  Description: Barometric pressure measured in millibars.
  Type: Numeric (float)

* wind_speed:
  Description: Wind speed outside the facility (m/s).
  Type: Numeric (float)

* visibility_index:
  Description: Proxy for fog or obstruction (higher = clearer).
  Type: Numeric (float)

* dew_point:
  Description: Temperature (°C) at which air reaches 100% humidity.
  Type: Numeric (float)



## Random Variables

* random_variable1, random_variable2:
  Description: Unknown origin; likely synthetic or derived noise. Included for feature selection testing.
  Type: Numeric (float)

---

## Engineered Features (added in preprocessing)

* hour: Hour of the day (0–23)
* weekday: Day of the week (0 = Monday)
* is_weekend: Binary flag for weekends
* month: Month number (1–12)
* temp_mean, temp_std: Aggregated mean and standard deviation of all zone temperatures
* humid_mean, humid_std: Aggregated mean and standard deviation of all zone humidities
* temp_humid_interaction: Product of `temp_mean` and `humid_mean`

These features help model daily/weekly cycles and overall environmental conditions in a compressed form.
