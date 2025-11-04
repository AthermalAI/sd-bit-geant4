/**
 * @file SdBitSensitiveDetector.hh
 * @brief Sensitive detector for sd-bit energy deposition scoring
 * 
 * Records energy deposition in the fluctuation core and calculates
 * flip probabilities based on the tilted energy landscape model.
 */

#ifndef SDBIT_SENSITIVE_DETECTOR_HH
#define SDBIT_SENSITIVE_DETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4TouchableHistory.hh"
#include "globals.hh"

class G4AnalysisManager;

/**
 * @class SdBitSensitiveDetector
 * @brief Sensitive detector for energy deposition and flip probability calculation
 */
class SdBitSensitiveDetector : public G4VSensitiveDetector
{
public:
    SdBitSensitiveDetector(const G4String& name);
    ~SdBitSensitiveDetector() override;

    void Initialize(G4HCofThisEvent* hitCollection) override;
    G4bool ProcessHits(G4Step* step, G4TouchableHistory* history) override;
    void EndOfEvent(G4HCofThisEvent* hitCollection) override;

    // Configuration methods
    void SetBiasVoltage(G4double voltage) { fVbias = voltage; }
    void SetBarrierHeight(G4double barrier) { fDeltaE = barrier; }
    void SetEffectiveBeta(G4double beta) { fBetaEff = beta; }

private:
    // Analysis manager
    G4AnalysisManager* fAnalysisManager;

    // sd-bit parameters
    G4double fVbias;        // Bias voltage
    G4double fVoffset;      // Offset voltage
    G4double fDeltaE;       // Base barrier height
    G4double fBetaEff;      // Effective beta parameter

    // Energy barriers (calculated from bias)
    G4double fBarrier0to1;  // 0→1 transition barrier
    G4double fBarrier1to0;  // 1→0 transition barrier

    // Event counters
    G4int fEventID;
    G4int fFlipCount0to1;
    G4int fFlipCount1to0;
    G4double fTotalEnergyDeposited;

    // Histogram IDs
    G4int fEdepHistID;
    G4int fFlipHistID;
    G4int fBarrierHistID;

    // Helper methods
    void UpdateBarriers();
    G4double CalculateFlipProbability(G4double energyDeposited, G4double barrier);
};

#endif