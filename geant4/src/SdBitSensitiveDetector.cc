/**
 * @file SdBitSensitiveDetector.cc
 * @brief Implementation of sd-bit sensitive detector
 */

#include "SdBitSensitiveDetector.hh"

#include "G4AnalysisManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "Randomize.hh"

#include <cmath>

SdBitSensitiveDetector::SdBitSensitiveDetector(const G4String& name)
    : G4VSensitiveDetector(name),
      fAnalysisManager(nullptr),
      fVbias(0.03 * volt),
      fVoffset(0.0 * volt),
      fDeltaE(0.05 * eV),
      fBetaEff(2.0),
      fBarrier0to1(0.0),
      fBarrier1to0(0.0),
      fEventID(0),
      fFlipCount0to1(0),
      fFlipCount1to0(0),
      fTotalEnergyDeposited(0.0),
      fEdepHistID(-1),
      fFlipHistID(-1),
      fBarrierHistID(-1)
{
    // Get analysis manager
    fAnalysisManager = G4AnalysisManager::Instance();

    // Create histograms
    fEdepHistID = fAnalysisManager->CreateH1("Edep", 
                                             "Energy Deposition in Core", 
                                             1000, 0., 10 * MeV);
    
    fFlipHistID = fAnalysisManager->CreateH1("Flips", 
                                             "State Flip Events", 
                                             2, -0.5, 1.5);
    
    fBarrierHistID = fAnalysisManager->CreateH1("Barriers", 
                                                "Energy vs Barriers", 
                                                1000, 0., 1 * eV);

    // Calculate initial barriers
    UpdateBarriers();
}

SdBitSensitiveDetector::~SdBitSensitiveDetector()
{
}

void SdBitSensitiveDetector::Initialize(G4HCofThisEvent*)
{
    // Reset event counters
    fTotalEnergyDeposited = 0.0;
}

G4bool SdBitSensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    // Get energy deposition
    G4double edep = step->GetTotalEnergyDeposit();
    
    if (edep > 0.) {
        fTotalEnergyDeposited += edep;
        
        // Fill energy deposition histogram
        fAnalysisManager->FillH1(fEdepHistID, edep);
        
        // Fill barrier comparison histogram
        fAnalysisManager->FillH1(fBarrierHistID, edep);
        
        // Check for potential state flips
        // Assume current state is 0 (can be extended to track actual state)
        if (edep > fBarrier0to1) {
            G4double flipProb = CalculateFlipProbability(edep, fBarrier0to1);
            
            if (G4UniformRand() < flipProb) {
                fFlipCount0to1++;
                fAnalysisManager->FillH1(fFlipHistID, 1.0); // Flip to state 1
                
                // Print flip event (optional, for debugging)
                if (verboseLevel > 1) {
                    G4cout << "State flip 0→1: Edep = " << G4BestUnit(edep, "Energy")
                           << ", Probability = " << flipProb << G4endl;
                }
            }
        }
        
        // Check for 1→0 flip (higher barrier)
        if (edep > fBarrier1to0) {
            G4double flipProb = CalculateFlipProbability(edep, fBarrier1to0);
            
            if (G4UniformRand() < flipProb) {
                fFlipCount1to0++;
                fAnalysisManager->FillH1(fFlipHistID, 0.0); // Flip to state 0
                
                if (verboseLevel > 1) {
                    G4cout << "State flip 1→0: Edep = " << G4BestUnit(edep, "Energy")
                           << ", Probability = " << flipProb << G4endl;
                }
            }
        }
    }
    
    return true;
}

void SdBitSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
    fEventID++;
    
    // Print event summary (optional)
    if (verboseLevel > 0 && fEventID % 10000 == 0) {
        G4cout << "Event " << fEventID 
               << ": Total Edep = " << G4BestUnit(fTotalEnergyDeposited, "Energy")
               << ", Flips 0→1: " << fFlipCount0to1
               << ", Flips 1→0: " << fFlipCount1to0 << G4endl;
    }
}

void SdBitSensitiveDetector::UpdateBarriers()
{
    // Calculate asymmetric barriers based on bias voltage
    fBarrier0to1 = fDeltaE - std::abs(fVbias);  // Lower barrier (favored direction)
    fBarrier1to0 = fDeltaE + std::abs(fVbias);  // Higher barrier (unfavored direction)
    
    // Ensure barriers are positive
    if (fBarrier0to1 < 0) fBarrier0to1 = 0.001 * eV;
    if (fBarrier1to0 < 0) fBarrier1to0 = 0.001 * eV;
    
    G4cout << "Updated barriers: 0→1 = " << G4BestUnit(fBarrier0to1, "Energy")
           << ", 1→0 = " << G4BestUnit(fBarrier1to0, "Energy") << G4endl;
}

G4double SdBitSensitiveDetector::CalculateFlipProbability(G4double energyDeposited, 
                                                          G4double barrier)
{
    // Calculate excess energy above barrier
    G4double excess = energyDeposited - barrier;
    
    if (excess <= 0) {
        return 0.0;
    }
    
    // Logistic function based on effective beta
    G4double exponent = fBetaEff * excess / eV; // Normalize to eV
    G4double probability = 1.0 / (1.0 + std::exp(-exponent));
    
    return probability;
}