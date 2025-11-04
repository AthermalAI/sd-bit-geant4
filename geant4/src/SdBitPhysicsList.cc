/**
 * @file SdBitPhysicsList.cc
 * @brief Implementation of sd-bit physics list
 */

#include "SdBitPhysicsList.hh"

#include "G4EmLivermorePhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4StoppingPhysics.hh"

#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

SdBitPhysicsList::SdBitPhysicsList()
    : G4VModularPhysicsList(),
      fCutForGamma(0.1 * eV),
      fCutForElectron(0.1 * eV),
      fCutForPositron(0.1 * eV),
      fCutForProton(0.1 * eV),
      fCutForAlpha(0.1 * eV)
{
    SetVerboseLevel(1);

    // Electromagnetic physics (low-energy optimized)
    RegisterPhysics(new G4EmLivermorePhysics());

    // Radioactive decay physics
    RegisterPhysics(new G4RadioactiveDecayPhysics());

    // General decay physics
    RegisterPhysics(new G4DecayPhysics());

    // Ion physics for alpha particles
    RegisterPhysics(new G4IonPhysics());

    // Extra electromagnetic processes
    RegisterPhysics(new G4EmExtraPhysics());

    // Stopping physics for ions
    RegisterPhysics(new G4StoppingPhysics());
}

SdBitPhysicsList::~SdBitPhysicsList()
{
}

void SdBitPhysicsList::SetCuts()
{
    // Set production cuts for different particle types
    // Very low cuts for nanoscale precision
    
    SetCutValue(fCutForGamma, "gamma");
    SetCutValue(fCutForElectron, "e-");
    SetCutValue(fCutForPositron, "e+");
    SetCutValue(fCutForProton, "proton");
    SetCutValue(fCutForAlpha, "alpha");

    // Set cuts for specific regions if needed
    // (can be implemented for different detector components)

    if (verboseLevel > 0) {
        DumpCutValuesTable();
    }
}