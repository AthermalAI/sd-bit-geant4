/**
 * @file SdBitPhysicsList.hh
 * @brief Physics list for sd-bit simulation
 * 
 * Defines the physics processes relevant for sd-bit simulation:
 * - Low-energy electromagnetic processes for alpha/beta particles
 * - Radioactive decay processes
 * - Appropriate production cuts for nanoscale geometry
 */

#ifndef SDBIT_PHYSICS_LIST_HH
#define SDBIT_PHYSICS_LIST_HH

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

/**
 * @class SdBitPhysicsList
 * @brief Custom physics list optimized for sd-bit simulation
 */
class SdBitPhysicsList : public G4VModularPhysicsList
{
public:
    SdBitPhysicsList();
    ~SdBitPhysicsList() override;

    void SetCuts() override;

private:
    // Cut values
    G4double fCutForGamma;
    G4double fCutForElectron;
    G4double fCutForPositron;
    G4double fCutForProton;
    G4double fCutForAlpha;
};

#endif