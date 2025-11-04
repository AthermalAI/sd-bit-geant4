# GEANT4 Simulation Guide

## Overview

The GEANT4 simulation provides detailed particle physics modeling of sd-bit devices at the nanoscale. This guide covers installation, compilation, and usage of the C++ simulation framework.

## Prerequisites

### System Requirements
- Linux, macOS, or Windows with WSL
- GCC 7.0+ or Clang 5.0+
- CMake 3.16+
- GEANT4 11.0+ with visualization and analysis support

### GEANT4 Installation

#### Option 1: Package Manager (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install geant4 geant4-data libgeant4-dev
```

#### Option 2: Source Build
```bash
# Download GEANT4 source
wget https://geant4-data.web.cern.ch/releases/geant4-v11.1.3.tar.gz
tar -xzf geant4-v11.1.3.tar.gz

# Build with required features
mkdir geant4-build && cd geant4-build
cmake -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_USE_QT=ON \
      -DGEANT4_USE_OPENGL_X11=ON \
      -DGEANT4_BUILD_MULTITHREADED=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local/geant4 \
      ../geant4-v11.1.3

make -j$(nproc)
sudo make install
```

#### Environment Setup
```bash
# Add to ~/.bashrc
source /usr/local/geant4/bin/geant4.sh
export G4NEUTRONHPDATA=/usr/local/geant4/share/Geant4-11.1.3/data/G4NDL4.6
export G4LEDATA=/usr/local/geant4/share/Geant4-11.1.3/data/G4EMLOW8.2
```

## Compilation

### Build Process
```bash
cd geant4
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### CMake Options
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..

# Enable multithreading
cmake -DGEANT4_BUILD_MULTITHREADED=ON ..
```

## Basic Usage

### Interactive Mode
```bash
./sd_bit
# Opens visualization window
# Use GUI or command line interface
```

### Batch Mode
```bash
./sd_bit macros/run.mac
```

### Macro Commands
```bash
# Basic simulation
/run/beamOn 100000

# Set random seed
/random/setSeeds 12345 67890

# Configure analysis
/analysis/setFileName output_file
/analysis/h1/setTitle 0 "Energy Deposition"
```

## Geometry Configuration

### Device Dimensions
The default geometry includes:
- **World volume**: 1 μm³ vacuum box
- **Fluctuation core**: 100 nm³ silicon (sensitive detector)
- **Isotope source**: 50 nm³ Am-241 or Tritium
- **Collimator**: 20 nm diameter tungsten tunnel

### Customization
Modify `SdBitDetectorConstruction.cc`:

```cpp
// Change core size
static constexpr G4double kCoreSize = 25.0; // nanometers

// Different isotope
fIsotopeSourceLog = new G4LogicalVolume(sourceBox, fTritiumMat, "SourceLog");

// Add shielding layer
G4Box* shieldBox = new G4Box("Shield", 100*nm, 100*nm, 10*nm);
G4LogicalVolume* shieldLog = new G4LogicalVolume(shieldBox, fLeadMat, "ShieldLog");
```

## Physics Configuration

### Physics Lists
The simulation uses optimized physics for low-energy particles:

```cpp
// Low-energy electromagnetic processes
RegisterPhysics(new G4EmLivermorePhysics());

// Radioactive decay
RegisterPhysics(new G4RadioactiveDecayPhysics());

// Ion physics for alphas
RegisterPhysics(new G4IonPhysics());
```

### Production Cuts
Very low cuts for nanoscale precision:
```cpp
SetCutValue(0.1 * eV, "gamma");
SetCutValue(0.1 * eV, "e-");
SetCutValue(0.1 * eV, "e+");
```

### Custom Physics
Add specialized processes:
```cpp
// In PhysicsList constructor
RegisterPhysics(new G4OpticalPhysics());  // For scintillation
RegisterPhysics(new G4HadronElasticPhysics());  // For neutron interactions
```

## Primary Generation

### Isotope Configuration

#### Am-241 (Alpha Emitter)
```cpp
// 5.486 MeV alpha particles
G4ParticleDefinition* ion = G4IonTable::Instance()->GetIon(95, 241, 0);
gun->SetParticleDefinition(ion);
gun->SetParticleEnergy(0 * MeV);  // At rest for decay
```

#### Tritium (Beta Emitter)
```cpp
// 18.6 keV beta spectrum
G4ParticleDefinition* tritium = G4IonTable::Instance()->GetIon(1, 3, 0);
gun->SetParticleDefinition(tritium);
gun->SetParticleEnergy(0 * MeV);
```

### Custom Sources
```cpp
// Uniform energy distribution
G4double energy = G4UniformRand() * 10 * MeV;
gun->SetParticleEnergy(energy);

// Gaussian spatial distribution
G4double sigma = 5 * nanometer;
G4double x = G4RandGauss::shoot(0, sigma);
G4double y = G4RandGauss::shoot(0, sigma);
gun->SetParticlePosition(G4ThreeVector(x, y, -50*nm));
```

## Analysis and Output

### Histogram Configuration
```cpp
// Energy deposition spectrum
fAnalysisManager->CreateH1("Edep", "Energy Deposition", 1000, 0, 10*MeV);

// Flip probability vs energy
fAnalysisManager->CreateH1("FlipProb", "Flip Probability", 100, 0, 1);

// Spatial distribution
fAnalysisManager->CreateH2("Position", "Hit Position", 100, -50, 50, 100, -50, 50);
```

### Ntuple Data
```cpp
// Create detailed event record
fAnalysisManager->CreateNtuple("Events", "Event Data");
fAnalysisManager->CreateNtupleDColumn("EventID");
fAnalysisManager->CreateNtupleDColumn("EnergyDep");
fAnalysisManager->CreateNtupleDColumn("PosX");
fAnalysisManager->CreateNtupleDColumn("PosY");
fAnalysisManager->CreateNtupleDColumn("PosZ");
fAnalysisManager->CreateNtupleIColumn("ParticleID");
fAnalysisManager->FinishNtuple();
```

### ROOT Output Analysis
```python
# Python analysis script
import uproot
import numpy as np
import matplotlib.pyplot as plt

# Load ROOT file
file = uproot.open("sd_bit_simulation.root")

# Extract histograms
edep_hist = file["Edep"]
values, edges = edep_hist.to_numpy()

# Plot energy spectrum
plt.figure(figsize=(10, 6))
plt.step(edges[:-1], values, where='post')
plt.xlabel('Energy Deposition (MeV)')
plt.ylabel('Counts')
plt.yscale('log')
plt.show()
```

## Advanced Features

### Multithreading
```cpp
// Enable MT in main()
auto runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::MT);
runManager->SetNumberOfThreads(4);
```

### Visualization
```bash
# In macro file
/vis/open OGL 800x600
/vis/drawVolume
/vis/scene/add/trajectories
/vis/scene/add/hits
/vis/viewer/set/viewpointVector 1 1 1
```

### Sensitive Detector Customization
```cpp
// Custom hit processing
G4bool SdBitSensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    G4double edep = step->GetTotalEnergyDeposit();
    G4ThreeVector pos = step->GetPreStepPoint()->GetPosition();
    
    // Custom flip probability calculation
    G4double barrier = CalculateLocalBarrier(pos);
    G4double flipProb = 1.0 / (1.0 + exp(-(edep - barrier) / kT_eff));
    
    // Record in analysis
    fAnalysisManager->FillH1(0, edep);
    fAnalysisManager->FillH1(1, flipProb);
    
    return true;
}
```

### Field Configuration
```cpp
// Add electric field for bias voltage
class SdBitElectricField : public G4ElectricField
{
public:
    void GetFieldValue(const G4double Point[4], G4double* field) const override
    {
        field[0] = 0;  // Ex
        field[1] = 0;  // Ey  
        field[2] = fBiasVoltage / fDeviceThickness;  // Ez
    }
private:
    G4double fBiasVoltage = 0.1 * volt;
    G4double fDeviceThickness = 100 * nanometer;
};
```

## Performance Optimization

### Memory Management
```cpp
// Efficient hit collection
class SdBitHit : public G4VHit
{
private:
    G4double fEdep;
    G4ThreeVector fPos;
    G4double fTime;
    
public:
    // Inline accessors for speed
    inline G4double GetEdep() const { return fEdep; }
    inline void AddEdep(G4double de) { fEdep += de; }
};
```

### Computational Efficiency
```bash
# Compile with optimizations
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..

# Use appropriate cuts
/cuts/setLowEdge 1 keV

# Limit secondary production
/process/em/setSecBiasing eBrem target 0.1
```

### Parallel Processing
```bash
# Run multiple jobs
for i in {1..10}; do
    ./sd_bit macros/run_${i}.mac &
done
wait

# Merge ROOT files
hadd combined_results.root sd_bit_*.root
```

## Validation and Testing

### Physics Validation
```cpp
// Test energy conservation
G4double totalEnergyIn = primaryEnergy;
G4double totalEnergyOut = sumSecondaryEnergies + energyDeposited;
assert(abs(totalEnergyIn - totalEnergyOut) < tolerance);

// Validate decay spectra
// Compare with ENSDF data for Am-241
G4double expectedAlphaEnergy = 5.486 * MeV;
G4double measuredAlphaEnergy = GetMeanAlphaEnergy();
assert(abs(expectedAlphaEnergy - measuredAlphaEnergy) < 0.1 * MeV);
```

### Geometry Validation
```bash
# Check for overlaps
/geometry/test/run

# Visualize geometry
/vis/drawVolume
/vis/viewer/set/style wireframe
```

### Statistical Tests
```python
# Chi-square test for Poisson statistics
from scipy.stats import chisquare

observed_counts = histogram_data
expected_rate = decay_rate * bin_width
expected_counts = [expected_rate] * len(observed_counts)

chi2, p_value = chisquare(observed_counts, expected_counts)
print(f"Chi-square test: p-value = {p_value}")
```

## Troubleshooting

### Common Issues

1. **Compilation errors**: Check GEANT4 environment variables
2. **Visualization problems**: Verify Qt/OpenGL installation
3. **Physics warnings**: Review production cuts and physics lists
4. **Memory issues**: Reduce event count or optimize data structures

### Debug Mode
```bash
# Compile with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Run with GDB
gdb ./sd_bit
(gdb) run macros/run.mac
```

### Performance Profiling
```bash
# Profile with gprof
cmake -DCMAKE_CXX_FLAGS="-pg" ..
make
./sd_bit macros/run.mac
gprof ./sd_bit gmon.out > profile.txt
```

## Integration with Monte Carlo

### Cross-Validation
Compare key metrics between simulations:
- Energy deposition spectra
- Flip probability curves
- Statistical distributions

### Parameter Mapping
```python
# Convert GEANT4 results to Monte Carlo parameters
geant4_edep_mean = np.mean(energy_deposition_data)
monte_carlo_Ed_mean = geant4_edep_mean / normalization_factor

geant4_flip_rate = flip_events / total_events
monte_carlo_lambda_eff = geant4_flip_rate / time_window
```

### Hybrid Approach
Use GEANT4 for detailed physics, Monte Carlo for system-level behavior:
1. Run GEANT4 to characterize single-bit response
2. Extract effective parameters
3. Use Monte Carlo for large arrays and long-term statistics