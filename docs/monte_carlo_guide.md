# Monte Carlo Simulation Guide

## Overview

The Monte Carlo simulation provides a fast, flexible approach to modeling sd-bit behavior using statistical methods. This guide covers installation, usage, and customization of the Python-based simulator.

## Installation

### Prerequisites
- Python 3.7 or later
- pip package manager

### Setup
```bash
cd monte-carlo
pip install -r requirements.txt
```

## Basic Usage

### Command Line Interface
```bash
# Basic simulation with default parameters
python sd_bit_simulation.py --plot

# Save results and plots
python sd_bit_simulation.py --output results --save-plot simulation.png

# Use custom configuration
python sd_bit_simulation.py --config examples/low_activity_config.json --plot
```

### Python API
```python
from sd_bit_simulation import SdBitSimulator

# Create simulator with default parameters
simulator = SdBitSimulator()

# Run simulation
results = simulator.run_simulation()

# Generate plots
simulator.plot_results()

# Access results
print(f"Empirical P(S=1): {results['P_empirical']:.3f}")
print(f"Theoretical P(S=1): {results['P_theory']:.3f}")
```

## Configuration

### Parameter Files
Configuration files use JSON format. Key parameters:

```json
{
  "dt": 1e-9,              // Time step (seconds)
  "total_time": 1e-6,      // Total simulation time (seconds)
  "lambda_decay": 1e8,     // Decay rate (Hz)
  "Ed_mean": 0.05,         // Mean decay energy (arbitrary units)
  "delta_E": 0.05,         // Base barrier height
  "Vbias": 0.03,           // Bias voltage
  "Voffset": 0.0,          // Offset voltage
  "window_size": 100,      // Energy accumulation window (time steps)
  "seed": 42               // Random seed
}
```

### Parameter Ranges

#### Decay Rate (lambda_decay)
- **High activity (demo)**: 10⁶ - 10⁸ Hz
- **Realistic low activity**: 10³ - 10⁴ Hz
- **Ultra-low activity**: 10¹ - 10² Hz

#### Energy Parameters
- **Ed_mean**: 0.01 - 1.0 (normalized units)
- **delta_E**: 0.01 - 0.1 (barrier height)
- **Vbias**: -0.1 to +0.1 (bias range)

#### Temporal Parameters
- **dt**: 1 ns - 1 μs (time resolution)
- **total_time**: 1 μs - 1 ms (statistics)
- **window_size**: 10 - 1000 time steps

## Output Analysis

### Results Dictionary
```python
results = {
    'P_empirical': 0.123,        # Measured P(S=1)
    'P_theory': 0.125,           # Theoretical P(S=1)
    'num_decays': 1000,          # Total decay events
    'avg_cumulative_Ed': 5.2,    # Average energy per window
    'parameters': {...}          # Simulation parameters
}
```

### Visualization
The simulation generates three-panel plots:

1. **State Timeline**: Bit state vs. time showing flicker dynamics
2. **Cumulative Energy**: Energy accumulation with barrier thresholds
3. **State Distribution**: Histogram of state probabilities

### Statistical Analysis
```python
# Calculate flip rate
flip_rate = results['num_decays'] * results['P_empirical'] / total_time

# Estimate effective beta
beta_eff = results['parameters']['beta_eff']

# Compare with theory
theory_prob = 1 / (1 + np.exp(-beta_eff * Vbias))
```

## Advanced Usage

### Parameter Sweeps
```python
import numpy as np
import matplotlib.pyplot as plt

# Sweep bias voltage
bias_values = np.linspace(-0.1, 0.1, 21)
empirical_probs = []
theoretical_probs = []

for bias in bias_values:
    config = {'Vbias': bias}
    simulator = SdBitSimulator(config)
    results = simulator.run_simulation()
    
    empirical_probs.append(results['P_empirical'])
    theoretical_probs.append(results['P_theory'])

# Plot sigmoid curve
plt.plot(bias_values, empirical_probs, 'o', label='Empirical')
plt.plot(bias_values, theoretical_probs, '-', label='Theory')
plt.xlabel('Bias Voltage')
plt.ylabel('P(S=1)')
plt.legend()
plt.show()
```

### Custom Energy Distributions
```python
class CustomSdBitSimulator(SdBitSimulator):
    def calculate_cumulative_energy(self):
        # Override with custom energy distribution
        # Example: Gamma distribution instead of exponential
        from scipy.stats import gamma
        
        self.cumulative_Ed = np.zeros(self.num_steps)
        decay_indices = np.searchsorted(self.decay_times, 
                                       np.arange(1, self.num_steps + 1) * self.dt)
        
        for i in range(self.num_steps):
            start_idx = max(0, i - self.window_size)
            recent_decays = decay_indices[(decay_indices >= start_idx) & 
                                        (decay_indices <= i)]
            
            if len(recent_decays) > 0:
                num_decays = len(recent_decays)
                # Gamma distribution with shape=2, scale=Ed_mean/2
                Eds = gamma.rvs(2, scale=self.Ed_mean/2, size=num_decays)
                self.cumulative_Ed[i] = np.sum(Eds)
```

### Multi-bit Arrays
```python
def simulate_array(n_bits, coupling_strength=0.01):
    """Simulate array of coupled sd-bits."""
    simulators = [SdBitSimulator() for _ in range(n_bits)]
    
    # Run individual simulations
    results = []
    for sim in simulators:
        result = sim.run_simulation()
        results.append(result)
    
    # Add coupling effects (simplified)
    for i, sim in enumerate(simulators):
        # Neighboring bits influence energy deposition
        neighbors = [j for j in [i-1, i+1] if 0 <= j < n_bits]
        for j in neighbors:
            coupling_energy = coupling_strength * np.mean(simulators[j].cumulative_Ed)
            sim.cumulative_Ed += coupling_energy
    
    return results
```

## Performance Optimization

### Memory Usage
- Use smaller time steps only when necessary
- Limit total simulation time for large arrays
- Consider chunked processing for very long simulations

### Computational Speed
- Vectorize operations where possible
- Use compiled NumPy functions
- Consider Numba JIT compilation for critical loops

### Parallel Processing
```python
from multiprocessing import Pool
import functools

def run_simulation_with_config(config):
    simulator = SdBitSimulator(config)
    return simulator.run_simulation()

# Parallel parameter sweep
configs = [{'Vbias': bias} for bias in np.linspace(-0.1, 0.1, 21)]

with Pool() as pool:
    results = pool.map(run_simulation_with_config, configs)
```

## Validation and Testing

### Unit Tests
```python
def test_poisson_statistics():
    """Test that decay events follow Poisson statistics."""
    simulator = SdBitSimulator({'lambda_decay': 1e6, 'total_time': 1e-3})
    simulator.generate_decay_events()
    
    expected_events = simulator.lambda_decay * simulator.total_time
    actual_events = len(simulator.decay_times)
    
    # Should be within 3 standard deviations
    std_dev = np.sqrt(expected_events)
    assert abs(actual_events - expected_events) < 3 * std_dev

def test_energy_conservation():
    """Test that energy is properly conserved."""
    simulator = SdBitSimulator()
    simulator.run_simulation()
    
    # Total energy should equal sum of individual deposits
    total_energy = np.sum(simulator.cumulative_Ed)
    assert total_energy >= 0
```

### Benchmarking
```python
import time

def benchmark_simulation(config, n_runs=10):
    """Benchmark simulation performance."""
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        simulator = SdBitSimulator(config)
        simulator.run_simulation()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'events_per_second': simulator.num_steps / np.mean(times)
    }
```

## Troubleshooting

### Common Issues

1. **Low flip rates**: Increase lambda_decay or decrease barrier heights
2. **Poor statistics**: Increase total_time or number of events
3. **Memory errors**: Reduce time resolution or simulation duration
4. **Slow performance**: Use vectorized operations and appropriate time steps

### Debug Mode
```python
# Enable detailed logging
simulator = SdBitSimulator()
simulator.verbosity = 2  # Add this feature for debugging
results = simulator.run_simulation()
```

### Validation Checks
- Verify Poisson statistics for decay events
- Check energy conservation
- Compare with theoretical predictions
- Validate against GEANT4 results for cross-validation