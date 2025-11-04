/**
 * @file SdBitSteppingAction.cc
 * @brief Implementation of stepping action
 */

#include "SdBitSteppingAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

SdBitSteppingAction::SdBitSteppingAction()
    : G4UserSteppingAction(),
      fTrackParticles(false)
{
}

SdBitSteppingAction::~SdBitSteppingAction()
{
}

void SdBitSteppingAction::UserSteppingAction(const G4Step* step)
{
    if (!fTrackParticles) return;

    // Get step information
    G4Track* track = step->GetTrack();
    G4VPhysicalVolume* volume = track->GetVolume();
    
    if (!volume) return;

    G4String volumeName = volume->GetName();
    G4String particleName = track->GetDefinition()->GetParticleName();
    G4double energy = track->GetKineticEnergy();
    G4double edep = step->GetTotalEnergyDeposit();

    // Track particles entering the fluctuation core
    if (volumeName == "FluctuationCore" && edep > 0) {
        G4cout << "Particle " << particleName 
               << " deposited " << G4BestUnit(edep, "Energy")
               << " in core (KE = " << G4BestUnit(energy, "Energy") << ")"
               << G4endl;
    }
}