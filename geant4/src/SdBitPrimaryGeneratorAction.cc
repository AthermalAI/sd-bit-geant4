/**
 * @file SdBitPrimaryGeneratorAction.cc
 * @brief Implementation of primary generator action
 */

#include "SdBitPrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"

SdBitPrimaryGeneratorAction::SdBitPrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(nullptr),
      fIsotopeType("Am241"),
      fSourceActivity(1.0 * becquerel),
      fAmericium241(nullptr),
      fTritium(nullptr)
{
    fParticleGun = new G4ParticleGun(1);

    // Setup default isotope (Am-241)
    SetupAmericium241();
}

SdBitPrimaryGeneratorAction::~SdBitPrimaryGeneratorAction()
{
    delete fParticleGun;
}

void SdBitPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    // Set particle position in the isotope source
    G4ThreeVector position(0., 0., -75 * nanometer);
    fParticleGun->SetParticlePosition(position);

    // Generate random direction (isotropic)
    G4double cosTheta = 2 * G4UniformRand() - 1;
    G4double sinTheta = std::sqrt(1 - cosTheta * cosTheta);
    G4double phi = 2 * pi * G4UniformRand();
    
    G4ThreeVector direction(sinTheta * std::cos(phi),
                           sinTheta * std::sin(phi),
                           cosTheta);
    fParticleGun->SetParticleMomentumDirection(direction);

    // Generate primary vertex
    fParticleGun->GeneratePrimaryVertex(event);
}

void SdBitPrimaryGeneratorAction::SetupAmericium241()
{
    // Get Am-241 ion from ion table
    G4IonTable* ionTable = G4IonTable::GetIonTable();
    fAmericium241 = ionTable->GetIon(95, 241, 0); // Z=95, A=241, excitation=0

    fParticleGun->SetParticleDefinition(fAmericium241);
    fParticleGun->SetParticleEnergy(0 * MeV); // At rest for radioactive decay
    
    G4cout << "Primary generator configured for Am-241 decay" << G4endl;
}

void SdBitPrimaryGeneratorAction::SetupTritium()
{
    // Get Tritium (H-3) ion from ion table
    G4IonTable* ionTable = G4IonTable::GetIonTable();
    fTritium = ionTable->GetIon(1, 3, 0); // Z=1, A=3, excitation=0

    fParticleGun->SetParticleDefinition(fTritium);
    fParticleGun->SetParticleEnergy(0 * MeV); // At rest for radioactive decay
    
    G4cout << "Primary generator configured for Tritium decay" << G4endl;
}