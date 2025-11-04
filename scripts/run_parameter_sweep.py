#!/usr/bin/env python3
"""
Parameter sweep script for sd-bit simulations

This script runs systematic parameter sweeps to characterize sd-bit behavior
across different operating conditions and validates theoretical predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys
import os

# Add monte-carlo directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monte-carlo'))

try:
    from sd_bit_simulation import SdBitSimulator
except ImportError:
    print("Error: Could not import SdBitSimulator")
    print("Make sure you're running from the repository root or monte-carlo directory")
    sys.exit(1)

def run_single_simulation(params):
    """Run a single simulation with given parameters."""
    config, sweep_param, sweep_value = params
    
    # Update config with sweep parameter
    config[sweep_param] = sweep_value
    
    try:
        simulator = SdBitSimulator(config)
        results = simulator.run_simulation()
        
        return {
            'sweep_value': sweep_value,
            'P_empirical': results['P_empirical'],
            'P_theory': results['P_theory'],
            'num_decays': results['num_decays'],
            'avg_cumulative_Ed': results['avg_cumulative_Ed'],
            'success': True
        }
    except Exception as e:
        return {
            'sweep_value': sweep_value,
            'error': str(e),
            'success': False
        }

def bias_voltage_sweep(base_config, bias_range=(-0.1, 0.1), n_points=21, n_processes=None):
    """Sweep bias voltage to generate sigmoid curve."""
    print("Running bias voltage sweep...")
    
    bias_values = np.linspace(bias_range[0], bias_range[1], n_points)
    
    # Prepare parameters for parallel processing
    params_list = [(base_config.copy(), 'Vbias', bias) for bias in bias_values]
    
    # Run simulations in parallel
    if n_processes is None:
        n_processes = min(cpu_count(), len(params_list))
    
    with Pool(n_processes) as pool:
        results = pool.map(run_single_simulation, params_list)
    
    # Extract successful results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if failed_results:
        print(f"Warning: {len(failed_results)} simulations failed")
        for result in failed_results:
            print(f"  Bias = {result['sweep_value']}: {result['error']}")
    
    return successful_results

def decay_rate_sweep(base_config, rate_range=(1e3, 1e8), n_points=10, n_processes=None):
    """Sweep decay rate to study activity dependence."""
    print("Running decay rate sweep...")
    
    # Logarithmic spacing for decay rates
    rate_values = np.logspace(np.log10(rate_range[0]), np.log10(rate_range[1]), n_points)
    
    # Prepare parameters
    params_list = [(base_config.copy(), 'lambda_decay', rate) for rate in rate_values]
    
    # Run simulations
    if n_processes is None:
        n_processes = min(cpu_count(), len(params_list))
    
    with Pool(n_processes) as pool:
        results = pool.map(run_single_simulation, params_list)
    
    successful_results = [r for r in results if r['success']]
    return successful_results

def energy_sweep(base_config, energy_range=(0.01, 0.2), n_points=15, n_processes=None):
    """Sweep average decay energy."""
    print("Running decay energy sweep...")
    
    energy_values = np.linspace(energy_range[0], energy_range[1], n_points)
    
    params_list = [(base_config.copy(), 'Ed_mean', energy) for energy in energy_values]
    
    if n_processes is None:
        n_processes = min(cpu_count(), len(params_list))
    
    with Pool(n_processes) as pool:
        results = pool.map(run_single_simulation, params_list)
    
    successful_results = [r for r in results if r['success']]
    return successful_results

def barrier_height_sweep(base_config, barrier_range=(0.01, 0.15), n_points=12, n_processes=None):
    """Sweep barrier height to study switching characteristics."""
    print("Running barrier height sweep...")
    
    barrier_values = np.linspace(barrier_range[0], barrier_range[1], n_points)
    
    params_list = [(base_config.copy(), 'delta_E', barrier) for barrier in barrier_values]
    
    if n_processes is None:
        n_processes = min(cpu_count(), len(params_list))
    
    with Pool(n_processes) as pool:
        results = pool.map(run_single_simulation, params_list)
    
    successful_results = [r for r in results if r['success']]
    return successful_results

def plot_bias_sweep_results(results, save_path=None):
    """Plot sigmoid curve from bias voltage sweep."""
    if not results:
        print("No results to plot")
        return None
    
    # Sort by bias voltage
    results = sorted(results, key=lambda x: x['sweep_value'])
    
    bias_values = [r['sweep_value'] for r in results]
    empirical_probs = [r['P_empirical'] for r in results]
    theoretical_probs = [r['P_theory'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(bias_values, empirical_probs, 'o-', label='Empirical', markersize=6, linewidth=2)
    ax.plot(bias_values, theoretical_probs, '--', label='Theoretical', linewidth=2)
    
    ax.set_xlabel('Bias Voltage (V)')
    ax.set_ylabel('P(S=1)')
    ax.set_title('sd-bit Probability vs Bias Voltage')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add text box with statistics
    rmse = np.sqrt(np.mean([(emp - theo)**2 for emp, theo in zip(empirical_probs, theoretical_probs)]))
    textstr = f'RMSE = {rmse:.3f}\nPoints = {len(results)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bias sweep plot saved to {save_path}")
    
    return fig

def plot_rate_sweep_results(results, save_path=None):
    """Plot flip rate vs decay rate."""
    if not results:
        return None
    
    results = sorted(results, key=lambda x: x['sweep_value'])
    
    decay_rates = [r['sweep_value'] for r in results]
    empirical_probs = [r['P_empirical'] for r in results]
    num_decays = [r['num_decays'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Probability vs decay rate
    ax1.semilogx(decay_rates, empirical_probs, 'o-', markersize=6, linewidth=2)
    ax1.set_xlabel('Decay Rate (Hz)')
    ax1.set_ylabel('P(S=1)')
    ax1.set_title('Probability vs Decay Rate')
    ax1.grid(True, alpha=0.3)
    
    # Number of decays vs decay rate
    ax2.loglog(decay_rates, num_decays, 's-', markersize=6, linewidth=2, color='orange')
    ax2.set_xlabel('Decay Rate (Hz)')
    ax2.set_ylabel('Number of Decay Events')
    ax2.set_title('Event Statistics vs Decay Rate')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rate sweep plot saved to {save_path}")
    
    return fig

def plot_energy_sweep_results(results, save_path=None):
    """Plot results from energy sweep."""
    if not results:
        return None
    
    results = sorted(results, key=lambda x: x['sweep_value'])
    
    energies = [r['sweep_value'] for r in results]
    empirical_probs = [r['P_empirical'] for r in results]
    theoretical_probs = [r['P_theory'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(energies, empirical_probs, 'o-', label='Empirical', markersize=6, linewidth=2)
    ax.plot(energies, theoretical_probs, '--', label='Theoretical', linewidth=2)
    
    ax.set_xlabel('Average Decay Energy (arb. units)')
    ax.set_ylabel('P(S=1)')
    ax.set_title('Probability vs Average Decay Energy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Energy sweep plot saved to {save_path}")
    
    return fig

def plot_barrier_sweep_results(results, save_path=None):
    """Plot results from barrier height sweep."""
    if not results:
        return None
    
    results = sorted(results, key=lambda x: x['sweep_value'])
    
    barriers = [r['sweep_value'] for r in results]
    empirical_probs = [r['P_empirical'] for r in results]
    theoretical_probs = [r['P_theory'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(barriers, empirical_probs, 'o-', label='Empirical', markersize=6, linewidth=2)
    ax.plot(barriers, theoretical_probs, '--', label='Theoretical', linewidth=2)
    
    ax.set_xlabel('Barrier Height (arb. units)')
    ax.set_ylabel('P(S=1)')
    ax.set_title('Probability vs Barrier Height')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Barrier sweep plot saved to {save_path}")
    
    return fig

def save_sweep_results(results, filename, sweep_parameter):
    """Save sweep results to JSON file."""
    output_data = {
        'sweep_parameter': sweep_parameter,
        'results': results,
        'summary': {
            'total_points': len(results),
            'successful_points': len([r for r in results if r['success']]),
            'failed_points': len([r for r in results if not r['success']])
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    """Main parameter sweep runner."""
    parser = argparse.ArgumentParser(description='sd-bit Parameter Sweep Analysis')
    parser.add_argument('--config', type=str, help='Base configuration file')
    parser.add_argument('--output-dir', type=str, default='sweep_results', 
                       help='Output directory')
    parser.add_argument('--sweep', type=str, choices=['bias', 'rate', 'energy', 'barrier', 'all'],
                       default='all', help='Type of parameter sweep')
    parser.add_argument('--n-points', type=int, default=21, 
                       help='Number of points in sweep')
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of parallel processes')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load base configuration
    if args.config:
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    else:
        # Use default configuration
        base_config = {
            'dt': 1e-9,
            'total_time': 1e-6,
            'lambda_decay': 1e8,
            'Ed_mean': 0.05,
            'delta_E': 0.05,
            'Vbias': 0.03,
            'Voffset': 0.0,
            'window_size': 100,
            'seed': 42
        }
    
    print("Starting parameter sweep analysis...")
    print(f"Base configuration: {base_config}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel processes: {args.n_processes or cpu_count()}")
    
    # Run sweeps based on selection
    if args.sweep in ['bias', 'all']:
        print("\n" + "="*50)
        bias_results = bias_voltage_sweep(base_config, n_points=args.n_points, 
                                        n_processes=args.n_processes)
        save_sweep_results(bias_results, output_dir / 'bias_sweep.json', 'Vbias')
        
        if args.plot:
            plot_bias_sweep_results(bias_results, output_dir / 'bias_sweep.png')
    
    if args.sweep in ['rate', 'all']:
        print("\n" + "="*50)
        rate_results = decay_rate_sweep(base_config, n_points=args.n_points,
                                      n_processes=args.n_processes)
        save_sweep_results(rate_results, output_dir / 'rate_sweep.json', 'lambda_decay')
        
        if args.plot:
            plot_rate_sweep_results(rate_results, output_dir / 'rate_sweep.png')
    
    if args.sweep in ['energy', 'all']:
        print("\n" + "="*50)
        energy_results = energy_sweep(base_config, n_points=args.n_points,
                                    n_processes=args.n_processes)
        save_sweep_results(energy_results, output_dir / 'energy_sweep.json', 'Ed_mean')
        
        if args.plot:
            plot_energy_sweep_results(energy_results, output_dir / 'energy_sweep.png')
    
    if args.sweep in ['barrier', 'all']:
        print("\n" + "="*50)
        barrier_results = barrier_height_sweep(base_config, n_points=args.n_points,
                                             n_processes=args.n_processes)
        save_sweep_results(barrier_results, output_dir / 'barrier_sweep.json', 'delta_E')
        
        if args.plot:
            plot_barrier_sweep_results(barrier_results, output_dir / 'barrier_sweep.png')
    
    print("\n" + "="*50)
    print("Parameter sweep analysis complete!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()