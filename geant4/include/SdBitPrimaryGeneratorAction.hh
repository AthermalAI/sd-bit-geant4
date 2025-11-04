/**
 * @file SdBitPrimaryGeneratorAction.hh
 * @brief Primary generator for radioactive decay events
 */

#ifndef SDBIT_PRIMARY_GENERATOR_ACTION_HH
#define SDBIT_PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

class G4Event;
class G4ParticleDefinition;

/**
 * @class SdBitPrimaryGeneratorAction
 * @brief Generate primary particles from radioactive decay
 */
class SdBitPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    SdBitPrimaryGeneratorAction();
    ~SdBitPrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    // Configuration methods
    void SetIsotopeType(const G4String& isotope) { fIsotopeType = isotope; }
    void SetSourceActivity(G4double activity) { fSourceActivity = activity; }

private:
    G4ParticleGun* fParticleGun;
    G4String fIsotopeType;
    G4double fSourceActivity;

    // Particle definitions
    G4ParticleDefinition* fAmericium241;
    G4ParticleDefinition* fTritium;

    void SetupAmericium241();
    void SetupTritium();
};

#endif