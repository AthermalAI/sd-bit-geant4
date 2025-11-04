# Contributing to sd-bit Simulation Suite

We welcome contributions to the sd-bit simulation project! This document provides guidelines for contributing code, documentation, and other improvements.

## Getting Started

### Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/sd-bit-simulation.git
   cd sd-bit-simulation
   ```
3. **Set up development environment**:
   ```bash
   # Monte Carlo simulation
   cd monte-carlo
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   
   # GEANT4 simulation (requires GEANT4 installation)
   cd ../geant4
   mkdir build && cd build
   cmake ..
   make
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the coding standards below
3. **Test your changes** thoroughly
4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request** on GitHub

## Coding Standards

### Python Code (Monte Carlo)

- **PEP 8 compliance**: Use `flake8` or `black` for formatting
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: Follow NumPy/SciPy docstring conventions
- **Testing**: Write unit tests using `pytest`

Example:
```python
def calculate_flip_probability(energy_deposited: float, 
                             barrier_height: float,
                             beta_eff: float) -> float:
    """
    Calculate state flip probability using logistic function.
    
    Parameters
    ----------
    energy_deposited : float
        Energy deposited in the core (arbitrary units)
    barrier_height : float
        Energy barrier height (same units as energy_deposited)
    beta_eff : float
        Effective inverse temperature parameter
    
    Returns
    -------
    float
        Flip probability between 0 and 1
    
    Examples
    --------
    >>> prob = calculate_flip_probability(0.1, 0.05, 2.0)
    >>> assert 0 <= prob <= 1
    """
    if energy_deposited <= barrier_height:
        return 0.0
    
    excess = energy_deposited - barrier_height
    return 1.0 / (1.0 + np.exp(-beta_eff * excess))
```

### C++ Code (GEANT4)

- **Google C++ Style Guide**: Follow Google's style conventions
- **Modern C++**: Use C++17 features where appropriate
- **RAII**: Proper resource management
- **Documentation**: Doxygen-style comments

Example:
```cpp
/**
 * @brief Calculate energy deposition in sensitive volume
 * @param step Geant4 step containing energy information
 * @return Energy deposited in this step (MeV)
 */
G4double SdBitSensitiveDetector::GetEnergyDeposition(const G4Step* step) const
{
    if (!step) {
        G4cerr << "Warning: Null step pointer in GetEnergyDeposition" << G4endl;
        return 0.0;
    }
    
    return step->GetTotalEnergyDeposit();
}
```

## Testing

### Monte Carlo Tests

Run the test suite:
```bash
cd monte-carlo
python -m pytest tests/ -v
```

Add new tests in the `tests/` directory:
```python
import pytest
import numpy as np
from sd_bit_simulation import SdBitSimulator

def test_poisson_statistics():
    """Test that decay events follow Poisson distribution."""
    config = {'lambda_decay': 1e6, 'total_time': 1e-3, 'seed': 42}
    simulator = SdBitSimulator(config)
    simulator.generate_decay_events()
    
    expected = config['lambda_decay'] * config['total_time']
    actual = len(simulator.decay_times)
    
    # Should be within 3 standard deviations
    assert abs(actual - expected) < 3 * np.sqrt(expected)

def test_energy_conservation():
    """Test that energy is conserved in simulation."""
    simulator = SdBitSimulator()
    simulator.run_simulation()
    
    # All energies should be non-negative
    assert np.all(simulator.cumulative_Ed >= 0)
```

### GEANT4 Tests

Create validation macros in `geant4/macros/test/`:
```bash
# test_geometry.mac
/geometry/test/run
/vis/drawVolume
```

## Documentation

### Adding Documentation

- **Theory**: Add mathematical derivations to `docs/theory.md`
- **User guides**: Update relevant guide files in `docs/`
- **API documentation**: Use docstrings for automatic generation
- **Examples**: Add working examples to `examples/` directories

### Documentation Style

- **Clear and concise**: Explain concepts simply
- **Code examples**: Include runnable code snippets
- **Mathematical notation**: Use LaTeX for equations
- **Cross-references**: Link related sections

Example documentation:
```markdown
## Energy Barrier Model

The sd-bit uses asymmetric energy barriers controlled by bias voltage:

```math
E_{\text{barrier},0 \to 1} = \Delta E - |V_{\text{bias}}|
```

```math
E_{\text{barrier},1 \to 0} = \Delta E + |V_{\text{bias}}|
```

### Implementation

```python
def calculate_barriers(delta_E, V_bias):
    """Calculate asymmetric energy barriers."""
    barrier_0to1 = delta_E - abs(V_bias)
    barrier_1to0 = delta_E + abs(V_bias)
    return barrier_0to1, barrier_1to0
```
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs. actual behavior**
- **System information** (OS, Python/GEANT4 version)
- **Minimal example** that demonstrates the bug

### Feature Requests

For new features, please provide:
- **Use case description**: Why is this feature needed?
- **Proposed implementation**: How should it work?
- **Backward compatibility**: Will it break existing code?
- **Testing strategy**: How can it be validated?

### Code Contributions

We welcome:
- **Bug fixes**: Corrections to existing functionality
- **New features**: Extensions to simulation capabilities
- **Performance improvements**: Optimizations and speedups
- **Documentation**: Improvements to guides and examples
- **Tests**: Additional validation and unit tests

### Priority Areas

Current areas where contributions are especially welcome:
- **Multi-bit array simulations**: Crosstalk and collective behavior
- **Advanced physics models**: Quantum effects, radiation damage
- **Performance optimization**: Parallel processing, GPU acceleration
- **Validation tools**: Comparison with experimental data
- **User interface**: GUI tools for parameter exploration

## Code Review Process

### Pull Request Guidelines

1. **Descriptive title**: Clearly describe the change
2. **Detailed description**: Explain what and why
3. **Link issues**: Reference related issue numbers
4. **Test coverage**: Include tests for new functionality
5. **Documentation updates**: Update relevant docs

### Review Criteria

Pull requests will be evaluated on:
- **Correctness**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Style**: Does it follow coding standards?
- **Performance**: Are there any performance regressions?
- **Compatibility**: Does it maintain backward compatibility?

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Maintainer review**: Core team reviews code
3. **Community feedback**: Other contributors may comment
4. **Revisions**: Address feedback and update PR
5. **Merge**: Approved PRs are merged to main branch

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:
- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember that everyone is learning
- **Be collaborative**: Work together toward common goals

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and discussions
- **Discussions**: For general questions and ideas

### Recognition

Contributors will be recognized through:
- **Contributor list**: Added to README and documentation
- **Release notes**: Contributions mentioned in releases
- **Authorship**: Significant contributions may warrant co-authorship

## Development Roadmap

### Short-term Goals (3-6 months)
- Complete GEANT4 validation suite
- Add multi-threading support
- Implement array simulations
- Performance benchmarking

### Medium-term Goals (6-12 months)
- GPU acceleration for Monte Carlo
- Advanced physics models
- Experimental validation
- User interface development

### Long-term Goals (1+ years)
- Integration with quantum simulators
- Machine learning applications
- Commercial device modeling
- Educational tools and tutorials

## Getting Help

If you need help with development:
- **Check documentation**: Start with the guides in `docs/`
- **Search issues**: Look for similar problems or questions
- **Ask questions**: Open a discussion or issue
- **Contact maintainers**: Reach out to core team members

Thank you for contributing to the sd-bit simulation project!