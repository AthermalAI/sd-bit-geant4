#!/usr/bin/env python3
"""
Monte Carlo Simulation of Stochastic-Decay Bit (sd-bit)

This script simulates the core probabilistic dynamics of an sd-bit using:
- Poisson-distributed radioactive decay events
- Exponential energy distribution per decay
- Asymmetric energy barriers with bias voltage control
- Time-resolved state evolution with statistical analysis

Based on the AthermalAI white paper theoretical framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import argparse
import json
from pathlib import Path

class SdBitSimulator:
    """Monte Carlo simulator for stochastic-decay bit dynamics."""
    
    def __init__(self, config=None):
        """Initialize simulator with configuration parameters."""
        # Default parameters (tunable based on paper's conceptual model)
        self.dt = 1e-9  # Time step (1 ns)
        self.total_time = 1e-6  # Total simulation time (1 μs)
        self.lambda_decay = 1e8  # Mean decay rate (100 MHz; high for demo)
        self.Ed_mean = 0.05  # Mean energy per decay (arbitrary units, ~eV scale)
        self.delta_E = 0.05  # Symmetric base barrier
        self.Vbias = 0.03  # Bias voltage (tilts toward state '1')
        self.Voffset = 0.0  # Symmetry offset
        self.window_size = 100  # Energy accumulation window (ns)
        self.seed = 42  # Random seed for reproducibility
        
        # Override with config if provided
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Derived parameters
        self.num_steps = int(self.total_time / self.dt)
        self.Ed_scale = self.Ed_mean
        self.beta_eff = self.lambda_decay * self.dt / self.Ed_mean
        
        # Energy landscape
        self.E0 = 0.0 + self.Vbias  # State 0 energy
        self.E1 = -self.Vbias       # State 1 energy
        
        # Asymmetric barriers
        self.E_barrier_0to1 = self.delta_E - abs(self.Vbias)  # Lower barrier
        self.E_barrier_1to0 = self.delta_E + abs(self.Vbias)  # Higher barrier
        
        # Results storage
        self.state = None
        self.cumulative_Ed = None
        self.decay_times = None
        self.P_empirical = None
        self.P_theory = None
    
    def generate_decay_events(self):
        """Generate Poisson-distributed decay event times."""
        np.random.seed(self.seed)
        
        # Generate more events than expected to ensure coverage
        num_expected_decays = int(self.lambda_decay * self.total_time * 1.5)
        decay_intervals = np.random.exponential(1 / self.lambda_decay, num_expected_decays)
        decay_times = np.cumsum(decay_intervals)
        
        # Keep only events within simulation time
        self.decay_times = decay_times[decay_times < self.total_time]
        
        return len(self.decay_times)
    
    def calculate_cumulative_energy(self):
        """Calculate cumulative energy deposition using sliding window."""
        self.cumulative_Ed = np.zeros(self.num_steps)
        
        # Map decay times to step indices
        decay_indices = np.searchsorted(self.decay_times, np.arange(1, self.num_steps + 1) * self.dt)
        
        for i in range(self.num_steps):
            start_idx = max(0, i - self.window_size)
            recent_decays = decay_indices[(decay_indices >= start_idx) & (decay_indices <= i)]
            
            if len(recent_decays) > 0:
                num_decays = len(recent_decays)
                Eds = np.random.exponential(self.Ed_scale, num_decays)
                self.cumulative_Ed[i] = np.sum(Eds)
    
    def simulate_state_evolution(self):
        """Simulate sd-bit state evolution using Monte Carlo method."""
        self.state = np.zeros(self.num_steps, dtype=int)
        self.state[0] = 0  # Start in state '0'
        
        np.random.seed(self.seed)
        
        for t in range(1, self.num_steps):
            current_state = self.state[t - 1]
            
            if current_state == 0:
                # Check for 0→1 transition
                if self.cumulative_Ed[t] > self.E_barrier_0to1:
                    excess = (self.cumulative_Ed[t] - self.E_barrier_0to1) * self.beta_eff
                    flip_prob = 1 / (1 + np.exp(-excess))  # Logistic function
                    self.state[t] = 1 if np.random.rand() < flip_prob else 0
                else:
                    self.state[t] = 0
            else:
                # Check for 1→0 transition
                if self.cumulative_Ed[t] > self.E_barrier_1to0:
                    excess = (self.cumulative_Ed[t] - self.E_barrier_1to0) * self.beta_eff
                    flip_prob = 1 / (1 + np.exp(-excess))
                    self.state[t] = 0 if np.random.rand() < flip_prob else 1
                else:
                    self.state[t] = 1
    
    def calculate_probabilities(self):
        """Calculate empirical and theoretical probabilities."""
        self.P_empirical = np.mean(self.state)
        self.P_theory = 1 / (1 + np.exp(-self.beta_eff * (self.Vbias - self.Voffset)))
    
    def run_simulation(self):
        """Run complete simulation pipeline."""
        print("Generating decay events...")
        num_decays = self.generate_decay_events()
        print(f"Generated {num_decays} decay events")
        
        print("Calculating cumulative energy...")
        self.calculate_cumulative_energy()
        
        print("Simulating state evolution...")
        self.simulate_state_evolution()
        
        print("Calculating probabilities...")
        self.calculate_probabilities()
        
        return self.get_results()
    
    def get_results(self):
        """Return simulation results as dictionary."""
        return {
            'P_empirical': self.P_empirical,
            'P_theory': self.P_theory,
            'num_decays': len(self.decay_times),
            'avg_cumulative_Ed': np.mean(self.cumulative_Ed),
            'parameters': {
                'lambda_decay': self.lambda_decay,
                'Ed_mean': self.Ed_mean,
                'Vbias': self.Vbias,
                'beta_eff': self.beta_eff,
                'E_barrier_0to1': self.E_barrier_0to1,
                'E_barrier_1to0': self.E_barrier_1to0
            }
        }
    
    def plot_results(self, save_path=None):
        """Generate comprehensive visualization of simulation results."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        time_ns = np.arange(self.num_steps) * self.dt * 1e9  # Time in ns
        
        # State timeline
        ax1.plot(time_ns, self.state, linewidth=0.5, alpha=0.7, color='blue')
        ax1.set_ylabel('State (0/1)')
        ax1.set_title('sd-bit State Fluctuations Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # Cumulative energy
        ax2.plot(time_ns, self.cumulative_Ed, linewidth=0.5, alpha=0.7, color='orange')
        ax2.axhline(self.E_barrier_0to1, color='g', linestyle='--', 
                   label=f'Barrier 0→1 ({self.E_barrier_0to1:.3f})')
        ax2.axhline(self.E_barrier_1to0, color='r', linestyle='--', 
                   label=f'Barrier 1→0 ({self.E_barrier_1to0:.3f})')
        ax2.set_ylabel('Cumulative Ed (arb. units)')
        ax2.set_title('Energy from Decay Events')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # State histogram
        ax3.hist(self.state, bins=2, density=True, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.axvline(self.P_empirical, color='r', linestyle='--', linewidth=2,
                   label=f'Empirical P(S=1) = {self.P_empirical:.3f}')
        ax3.axvline(self.P_theory, color='g', linestyle='--', linewidth=2,
                   label=f'Theoretical P(S=1) = {self.P_theory:.3f}')
        ax3.set_xlabel('State')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('State Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results, output_path):
    """Save simulation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def main():
    """Main simulation runner with command-line interface."""
    parser = argparse.ArgumentParser(description='sd-bit Monte Carlo Simulation')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Run simulation
    simulator = SdBitSimulator(config)
    results = simulator.run_simulation()
    
    # Print results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Empirical P(S=1): {results['P_empirical']:.3f}")
    print(f"Theoretical P(S=1): {results['P_theory']:.3f}")
    print(f"Number of decay events: {results['num_decays']}")
    print(f"Average cumulative Ed: {results['avg_cumulative_Ed']:.3f}")
    print(f"Effective beta: {results['parameters']['beta_eff']:.3f}")
    print("="*50)
    
    # Save results
    save_results(results, output_dir / 'simulation_results.json')
    
    # Generate plots
    if args.plot or args.save_plot:
        plot_path = args.save_plot if args.save_plot else None
        if plot_path and not plot_path.endswith(('.png', '.pdf', '.svg')):
            plot_path = output_dir / f"{plot_path}.png"
        elif not plot_path and args.plot:
            plot_path = output_dir / 'sd_bit_simulation.png'
        
        simulator.plot_results(plot_path)

if __name__ == "__main__":
    main()