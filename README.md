# sd-bit Simulation Suite

A comprehensive simulation framework for the **Stochastic-Decay Bit (sd-bit)** proposed in the AthermalAI white paper. This repository provides both Monte Carlo and GEANT4 implementations for modeling temperature-independent probabilistic computing using controlled radioactive decay.

## Overview

The sd-bit is a hardware primitive that uses quantum decay events for athermal random number generation. Unlike thermal p-bits that depend on temperature (β = 1/kT), sd-bits achieve temperature independence through radioactive decay statistics with an effective β derived from decay rate λ and average decay energy E_d.

## Repository Structure

```
sd-bit-simulation/
├── monte-carlo/           # Python Monte Carlo simulation
│   ├── sd_bit_simulation.py
│   ├── requirements.txt
│   └── examples/
├── geant4/               # GEANT4 nanoscale particle simulation
│   ├── CMakeLists.txt
│   ├── sd_bit.cc
│   ├── include/
│   ├── src/
│   ├── macros/
│   └── analysis/
├── docs/                 # Documentation and theory
├── results/              # Sample outputs and validation
└── scripts/              # Utility scripts
```

## Features

### Monte Carlo Simulation
- **Poisson-distributed decay events** with tunable decay rate λ
- **Exponential energy distribution** for realistic beta/alpha spectra
- **Asymmetric energy barriers** with bias voltage control
- **Time-resolved state evolution** with configurable resolution
- **Statistical validation** against theoretical sigmoid function

### GEANT4 Simulation
- **Nanoscale geometry modeling** (100 nm³ silicon core)
- **Realistic particle physics** using low-energy electromagnetic processes
- **Isotope source modeling** (Am-241 alpha or Tritium beta)
- **Tungsten collimator** for particle direction and crosstalk prevention
- **Energy deposition scoring** with flip probability calculation

## Quick Start

### Monte Carlo Simulation

```bash
cd monte-carlo
pip install -r requirements.txt
python sd_bit_simulation.py
```

### GEANT4 Simulation

```bash
cd geant4
mkdir build && cd build
cmake ..
make -j4
./sd_bit -macro ../macros/run.mac
```

## Key Parameters

- **λ (decay rate)**: 1e8 Hz (demo) to 1e3 Hz (realistic low-activity)
- **E_d (decay energy)**: Exponential distribution, mean ~0.05 eV
- **V_bias**: Bias voltage for energy landscape tilting
- **Barriers**: Asymmetric (0.02 eV for 0→1, 0.08 eV for 1→0)

## Theoretical Foundation

The sd-bit probability follows:
```
P(S=1) = 1 / (1 + exp(-β_eff * (V_bias - V_offset)))
```

Where β_eff derives from λ and E_d, providing temperature-independent operation.

## Applications

- **Athermal random number generation**
- **Probabilistic processing units (PPUs)**
- **Ising model optimization**
- **Monte Carlo sampling acceleration**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this simulation suite in your research, please cite:
```
AthermalAI White Paper: "Stochastic-Decay Bits for Temperature-Independent Probabilistic Computing"
```

## Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.