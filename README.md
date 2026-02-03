# 2026MCM_A-2625441

# Smartphone Battery Depletion Modeling

This repository contains a data-driven, scenario-based modeling framework for analyzing
smartphone battery power consumption and relative Time-to-Empty (TTE).
The model decomposes total power consumption into interpretable hardware components
and evaluates how different usage patterns affect battery drain.

## Model Overview
The total power consumption is modeled as
\[
P = P_{\text{screen}} + P_{\text{CPU}} + P_{\text{data usage}} + P_{\text{GPS}} + P_{\text{base}},
\]
combined with a battery capacity model that accounts for temperature effects and long-term aging.
Scenario-level average power is used to compare relative battery lifetime across usage patterns.

## Scripts Description
- **battery_health_analysis.py**  
  Analyzes battery capacity degradation, equivalent cycles, and effective maximum capacity.

- **cpu_test.py**  
  Constructs and validates the CPU power consumption model under controlled scenarios.

- **datausage.py**
  Performs data preprocessing, binning, and regression for data usage power modeling.
  
- **data_usage_test.py**  
  Implements and tests the saturating power model for mobile data usage.

- **screentest.py**

  Constructs and validates the screen power consumption model under controlled scenarios.


- **datasetcleaning_pscreenregression.py**  
  Performs data preprocessing, binning, and regression for screen brightness and cpu power modeling.

## How to Use
1. Run data preprocessing and regression scripts to fit sub-model parameters.
2. Use the fitted models to estimate scenario-level power consumption.
3. Compare relative battery drain and infer Time-to-Empty across usage scenarios.

## Key Findings
- CPU-intensive workloads are the primary driver of rapid battery drain.
- GPS usage introduces a persistent power draw comparable to high CPU load.
- Data usage exhibits saturation behavior, contributing less marginal drain at high volumes.
- Relative ranking of battery-draining activities is robust to modeling assumptions.

## Notes
- The model focuses on scenario-level average behavior rather than fine-grained transients.
- Absolute TTE values depend on battery capacity assumptions; relative comparisons are emphasized.

## License
This project is intended for academic and educational use.
