# Theoretical Foundation of sd-bit Simulation

## Overview

The Stochastic-Decay Bit (sd-bit) represents a novel approach to probabilistic computing that leverages controlled radioactive decay for temperature-independent random number generation. This document outlines the theoretical framework underlying both the Monte Carlo and GEANT4 simulations.

## Physical Principles

### Radioactive Decay Statistics

The sd-bit operates on the principle that radioactive decay follows Poisson statistics with a characteristic decay rate λ:

```
P(n decays in time t) = (λt)^n * exp(-λt) / n!
```

This provides a fundamental source of randomness that is independent of temperature, unlike thermal noise sources.

### Energy Landscape Model

The sd-bit consists of a bistable memory element with two states (0 and 1) separated by energy barriers. The energy landscape is characterized by:

- **Base barrier height**: ΔE (symmetric without bias)
- **Bias voltage**: V_bias (tilts the landscape)
- **Asymmetric barriers**:
  - E_barrier,0→1 = ΔE - |V_bias| (lower, favored direction)
  - E_barrier,1→0 = ΔE + |V_bias| (higher, unfavored direction)

### State Transition Probability

When a decay event deposits energy E_d in the core, the probability of a state flip is given by:

```
P_flip = 1 / (1 + exp(-β_eff * (E_d - E_barrier)))
```

where β_eff is the effective inverse temperature parameter derived from the decay statistics rather than thermal energy.

### Effective Beta Parameter

The effective β parameter is given by:

```
β_eff = λ * dt / <E_d>
```

where:
- λ is the decay rate
- dt is the time step
- <E_d> is the average energy per decay event

This parameter determines the steepness of the sigmoid transition probability.

## Steady-State Probability

In steady state, the probability of finding the sd-bit in state 1 follows:

```
P(S=1) = 1 / (1 + exp(-β_eff * (V_bias - V_offset)))
```

This sigmoid function allows tunable probability control through the bias voltage, independent of temperature.

## Simulation Approaches

### Monte Carlo Method

The Monte Carlo simulation captures the essential physics through:

1. **Poisson event generation**: Random decay times with rate λ
2. **Energy deposition**: Exponential distribution for realistic spectra
3. **Cumulative effects**: Sliding window for energy accumulation
4. **State evolution**: Probabilistic flips based on energy barriers

### GEANT4 Method

The GEANT4 simulation provides detailed particle physics including:

1. **Nanoscale geometry**: Realistic device dimensions and materials
2. **Particle transport**: Full electromagnetic and nuclear processes
3. **Energy deposition**: Precise tracking of energy loss in silicon
4. **Statistical accuracy**: Large event samples for validation

## Key Parameters

### Physical Parameters
- **Decay rate (λ)**: 10³ - 10⁸ Hz (activity dependent)
- **Average decay energy (<E_d>)**: 0.01 - 10 eV (isotope dependent)
- **Barrier height (ΔE)**: 0.01 - 0.1 eV (device dependent)
- **Bias voltage (V_bias)**: ±0.1 V (tunable)

### Simulation Parameters
- **Time step (dt)**: 1 ns (Monte Carlo resolution)
- **Total time**: 1 μs - 1 ms (statistics dependent)
- **Energy window**: 100 ns (event accumulation)
- **Geometry scale**: 10-100 nm (device dimensions)

## Validation Criteria

### Theoretical Consistency
- Sigmoid probability curve vs. bias voltage
- Poisson decay statistics
- Energy conservation

### Physical Realism
- Realistic decay spectra (Am-241: 5.486 MeV α, Tritium: 18.6 keV β)
- Appropriate cross-sections and stopping powers
- Nanoscale geometry effects

### Performance Metrics
- Flip rate vs. activity
- Temperature independence
- Crosstalk between adjacent bits

## Applications

### Probabilistic Processing Units (PPUs)
Arrays of sd-bits can implement:
- Ising model optimization
- Monte Carlo sampling
- Bayesian inference
- Stochastic neural networks

### Advantages over Thermal p-bits
- Temperature independence
- Stable operation in harsh environments
- Predictable decay statistics
- No thermal calibration required

## Future Extensions

### Multi-bit Arrays
- Crosstalk modeling
- Collective behavior
- Scaling laws

### Advanced Physics
- Quantum effects in nanoscale devices
- Radiation damage and aging
- Shielding optimization

### System Integration
- CMOS compatibility
- Power consumption analysis
- Fabrication considerations