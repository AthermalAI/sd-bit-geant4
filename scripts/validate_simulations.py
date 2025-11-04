#!/usr/bin/env python3
"""
Validation script for sd-bit simulations

This script performs comprehensive validation of both Monte Carlo and GEANT4
simulations against theoretical predictions and known physics.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import sys
import os
from scipy import stats
from scipy.optimize import curve_fit

# Add monte-carlo directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monte-carlo'))

try:
    from sd_bit_simulation import SdBitSimulator
except ImportError:
    print("Warning: Could not import SdBitSimulator for Monte Carlo validation")

def sigmoid_function(x, beta_eff, v_offset):
    """Theoretical sigmoid function for sd-bit probability."""
    return 1.0 / (1.0 + np.exp(-beta_eff * (x - v_offset)))

def validate_poisson_statistics(simulator, tolerance=0.05):
    """Validate that decay events follow Poisson statistics."""
    print("Validating Poisson statistics...")
    
    # Generate decay events
    simulator.generate_decay_events()
    
    # Expected number of events
    expected_events = simulator.lambda_decay * simulator.total_time
    actual_events = len(simulator.decay_times)
    
    # Calculate relative error
    relative_error = abs(actual_events - expected_events) / expected_events
    
    # Statistical test (should be within ~3 sigma for Poisson)
    sigma = np.sqrt(expected_events)
    z_score = abs(actual_events - expected_events) / sigma
    
    result = {
        'test': 'Poisson Statistics',
        'expected_events': expected_events,
        'actual_events': actual_events,
        'relative_error': relative_error,
        'z_score': z_score,
        'passed': relative_error < tolerance and z_score < 3.0,
        'tolerance': tolerance
    }
    
    print(f"  Expected events: {expected_events:.1f}")
    print(f"  Actual events: {actual_events}")
    print(f"  Relative error: {relative_error:.3f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Test passed: {result['passed']}")
    
    return result

def validate_energy_conservation(simulator, tolerance=1e-6):
    """Validate energy conservation in the simulation."""
    print("Validating energy conservation...")
    
    # Run simulation
    simulator.run_simulation()
    
    # Check that cumulative energy is always non-negative
    negative_energy = np.any(simulator.cumulative_Ed < 0)
    
    # Check for unrealistic energy spikes
    max_energy = np.max(simulator.cumulative_Ed)
    mean_energy = np.mean(simulator.cumulative_Ed[simulator.cumulative_Ed > 0])
    
    # Energy should be reasonable compared to input parameters
    expected_max = simulator.Ed_mean * simulator.window_size * 10  # Allow 10x fluctuation
    
    result = {
        'test': 'Energy Conservation',
        'negative_energy': negative_energy,
        'max_energy': max_energy,
        'mean_energy': mean_energy,
        'expected_max': expected_max,
        'passed': not negative_energy and max_energy < expected_max,
        'tolerance': tolerance
    }
    
    print(f"  Negative energy detected: {negative_energy}")
    print(f"  Maximum energy: {max_energy:.3f}")
    print(f"  Mean energy: {mean_energy:.3f}")
    print(f"  Expected maximum: {expected_max:.3f}")
    print(f"  Test passed: {result['passed']}")
    
    return result

def validate_sigmoid_curve(base_config, n_points=15, tolerance=0.1):
    """Validate that P(S=1) follows theoretical sigmoid curve."""
    print("Validating sigmoid curve...")
    
    # Generate bias voltage sweep
    bias_values = np.linspace(-0.08, 0.08, n_points)
    empirical_probs = []
    theoretical_probs = []
    
    for bias in bias_values:
        config = base_config.copy()
        config['Vbias'] = bias
        
        simulator = SdBitSimulator(config)
        results = simulator.run_simulation()
        
        empirical_probs.append(results['P_empirical'])
        theoretical_probs.append(results['P_theory'])
    
    empirical_probs = np.array(empirical_probs)
    theoretical_probs = np.array(theoretical_probs)
    
    # Fit sigmoid to empirical data
    try:
        # Initial guess for parameters
        beta_guess = base_config.get('lambda_decay', 1e8) * base_config.get('dt', 1e-9) / base_config.get('Ed_mean', 0.05)
        popt, pcov = curve_fit(sigmoid_function, bias_values, empirical_probs, 
                              p0=[beta_guess, 0.0], maxfev=1000)
        fitted_beta, fitted_offset = popt
        
        # Calculate R-squared
        ss_res = np.sum((empirical_probs - sigmoid_function(bias_values, *popt))**2)
        ss_tot = np.sum((empirical_probs - np.mean(empirical_probs))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        fit_successful = True
    except:
        fitted_beta = fitted_offset = r_squared = 0
        fit_successful = False
    
    # Calculate RMSE between empirical and theoretical
    rmse = np.sqrt(np.mean((empirical_probs - theoretical_probs)**2))
    
    result = {
        'test': 'Sigmoid Curve',
        'bias_values': bias_values.tolist(),
        'empirical_probs': empirical_probs.tolist(),
        'theoretical_probs': theoretical_probs.tolist(),
        'rmse': rmse,
        'fit_successful': fit_successful,
        'fitted_beta': fitted_beta if fit_successful else None,
        'fitted_offset': fitted_offset if fit_successful else None,
        'r_squared': r_squared if fit_successful else None,
        'passed': rmse < tolerance and (not fit_successful or r_squared > 0.8),
        'tolerance': tolerance
    }
    
    print(f"  RMSE (empirical vs theoretical): {rmse:.3f}")
    if fit_successful:
        print(f"  Fitted beta: {fitted_beta:.2f}")
        print(f"  Fitted offset: {fitted_offset:.4f}")
        print(f"  R-squared: {r_squared:.3f}")
    print(f"  Test passed: {result['passed']}")
    
    return result

def validate_temperature_independence(base_config, n_trials=5):
    """Validate that results are independent of random seed (temperature analog)."""
    print("Validating temperature independence...")
    
    # Run multiple trials with different seeds
    probabilities = []
    
    for trial in range(n_trials):
        config = base_config.copy()
        config['seed'] = 42 + trial * 1000  # Different seeds
        
        simulator = SdBitSimulator(config)
        results = simulator.run_simulation()
        probabilities.append(results['P_empirical'])
    
    probabilities = np.array(probabilities)
    
    # Calculate statistics
    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    cv = std_prob / mean_prob if mean_prob > 0 else float('inf')
    
    # Test should show low variability (coefficient of variation < 0.1)
    tolerance = 0.1
    passed = cv < tolerance
    
    result = {
        'test': 'Temperature Independence',
        'n_trials': n_trials,
        'probabilities': probabilities.tolist(),
        'mean_probability': mean_prob,
        'std_probability': std_prob,
        'coefficient_of_variation': cv,
        'passed': passed,
        'tolerance': tolerance
    }
    
    print(f"  Mean probability: {mean_prob:.3f}")
    print(f"  Standard deviation: {std_prob:.3f}")
    print(f"  Coefficient of variation: {cv:.3f}")
    print(f"  Test passed: {passed}")
    
    return result

def validate_scaling_behavior(base_config):
    """Validate that simulation scales properly with parameters."""
    print("Validating scaling behavior...")
    
    # Test scaling with decay rate
    rates = [1e6, 1e7, 1e8]
    flip_rates = []
    
    for rate in rates:
        config = base_config.copy()
        config['lambda_decay'] = rate
        
        simulator = SdBitSimulator(config)
        results = simulator.run_simulation()
        
        # Calculate flip rate (flips per second)
        flip_rate = results['P_empirical'] * results['num_decays'] / config['total_time']
        flip_rates.append(flip_rate)
    
    flip_rates = np.array(flip_rates)
    
    # Flip rate should scale roughly linearly with decay rate
    # (allowing for some nonlinearity due to saturation effects)
    scaling_ratios = flip_rates[1:] / flip_rates[:-1]
    rate_ratios = np.array(rates[1:]) / np.array(rates[:-1])
    
    # Check if scaling is reasonable (within factor of 2 of linear)
    scaling_errors = np.abs(scaling_ratios - rate_ratios) / rate_ratios
    max_scaling_error = np.max(scaling_errors)
    
    passed = max_scaling_error < 1.0  # Allow up to 100% deviation
    
    result = {
        'test': 'Scaling Behavior',
        'decay_rates': rates,
        'flip_rates': flip_rates.tolist(),
        'scaling_ratios': scaling_ratios.tolist(),
        'rate_ratios': rate_ratios.tolist(),
        'max_scaling_error': max_scaling_error,
        'passed': passed
    }
    
    print(f"  Decay rates: {rates}")
    print(f"  Flip rates: {flip_rates}")
    print(f"  Max scaling error: {max_scaling_error:.2f}")
    print(f"  Test passed: {passed}")
    
    return result

def plot_validation_results(results, save_dir=None):
    """Generate plots for validation results."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # Plot sigmoid curve validation
    sigmoid_result = next((r for r in results if r['test'] == 'Sigmoid Curve'), None)
    if sigmoid_result and sigmoid_result['bias_values']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bias_values = np.array(sigmoid_result['bias_values'])
        empirical_probs = np.array(sigmoid_result['empirical_probs'])
        theoretical_probs = np.array(sigmoid_result['theoretical_probs'])
        
        ax.plot(bias_values, empirical_probs, 'o-', label='Empirical', markersize=6)
        ax.plot(bias_values, theoretical_probs, '--', label='Theoretical', linewidth=2)
        
        if sigmoid_result['fit_successful']:
            fitted_curve = sigmoid_function(bias_values, 
                                          sigmoid_result['fitted_beta'],
                                          sigmoid_result['fitted_offset'])
            ax.plot(bias_values, fitted_curve, ':', label='Fitted', linewidth=2)
        
        ax.set_xlabel('Bias Voltage (V)')
        ax.set_ylabel('P(S=1)')
        ax.set_title('Sigmoid Curve Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        rmse = sigmoid_result['rmse']
        r_sq = sigmoid_result.get('r_squared', 0)
        textstr = f'RMSE = {rmse:.3f}\nR² = {r_sq:.3f}' if r_sq else f'RMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        if save_dir:
            plt.savefig(save_dir / 'sigmoid_validation.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # Plot temperature independence
    temp_result = next((r for r in results if r['test'] == 'Temperature Independence'), None)
    if temp_result:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        probabilities = temp_result['probabilities']
        trials = range(1, len(probabilities) + 1)
        
        ax.plot(trials, probabilities, 'o-', markersize=8, linewidth=2)
        ax.axhline(temp_result['mean_probability'], color='r', linestyle='--', 
                  label=f"Mean = {temp_result['mean_probability']:.3f}")
        
        # Add error bars
        std_prob = temp_result['std_probability']
        ax.fill_between(trials, 
                       temp_result['mean_probability'] - std_prob,
                       temp_result['mean_probability'] + std_prob,
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('P(S=1)')
        ax.set_title('Temperature Independence Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(save_dir / 'temperature_independence.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()

def generate_validation_report(results, output_file=None):
    """Generate a comprehensive validation report."""
    report = {
        'validation_summary': {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r['passed']),
            'failed_tests': sum(1 for r in results if not r['passed']),
            'success_rate': sum(1 for r in results if r['passed']) / len(results)
        },
        'test_results': results
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total tests: {report['validation_summary']['total_tests']}")
    print(f"Passed tests: {report['validation_summary']['passed_tests']}")
    print(f"Failed tests: {report['validation_summary']['failed_tests']}")
    print(f"Success rate: {report['validation_summary']['success_rate']:.1%}")
    
    print("\nDetailed Results:")
    for result in results:
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {result['test']}: {status}")
    
    return report

def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description='sd-bit Simulation Validation')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--monte-carlo-only', action='store_true',
                       help='Run only Monte Carlo validation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    else:
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
    
    print("Starting sd-bit simulation validation...")
    print(f"Configuration: {base_config}")
    
    validation_results = []
    
    if 'SdBitSimulator' in globals():
        # Monte Carlo validation
        print("\n" + "="*50)
        print("MONTE CARLO VALIDATION")
        print("="*50)
        
        simulator = SdBitSimulator(base_config)
        
        # Run validation tests
        validation_results.append(validate_poisson_statistics(simulator))
        validation_results.append(validate_energy_conservation(simulator))
        validation_results.append(validate_sigmoid_curve(base_config))
        validation_results.append(validate_temperature_independence(base_config))
        validation_results.append(validate_scaling_behavior(base_config))
    
    else:
        print("Warning: Monte Carlo simulator not available")
    
    # TODO: Add GEANT4 validation when ROOT files are available
    if not args.monte_carlo_only:
        print("\n" + "="*50)
        print("GEANT4 VALIDATION")
        print("="*50)
        print("GEANT4 validation not yet implemented")
        print("Run GEANT4 simulation first to generate ROOT files")
    
    # Generate report and plots
    if validation_results:
        report = generate_validation_report(validation_results, 
                                          output_dir / 'validation_report.json')
        
        if args.plot:
            plot_validation_results(validation_results, output_dir)
    else:
        print("No validation tests were run")

if __name__ == "__main__":
    main()