/**
 * @file SdBitRunAction.cc
 * @brief Implementation of run action
 */

#include "SdBitRunAction.hh"

#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

SdBitRunAction::SdBitRunAction()
    : G4UserRunAction(),
      fAnalysisManager(nullptr)
{
    // Get analysis manager
    fAnalysisManager = G4AnalysisManager::Instance();
    fAnalysisManager->SetVerboseLevel(1);
    fAnalysisManager->SetNtupleMerging(true);

    // Set output file name
    fAnalysisManager->SetFileName("sd_bit_simulation");
}

SdBitRunAction::~SdBitRunAction()
{
}

void SdBitRunAction::BeginOfRunAction(const G4Run* run)
{
    G4cout << "### Run " << run->GetRunID() << " start." << G4endl;

    // Open output file
    fAnalysisManager->OpenFile();

    // Create additional histograms if needed
    G4int h1Id = fAnalysisManager->CreateH1("ProbabilityVsBias", 
                                            "P(S=1) vs Bias Voltage", 
                                            100, -0.1, 0.1);
    
    // Create ntuple for detailed event data
    fAnalysisManager->CreateNtuple("SdBitEvents", "Event Data");
    fAnalysisManager->CreateNtupleDColumn("EventID");
    fAnalysisManager->CreateNtupleDColumn("EnergyDeposited");
    fAnalysisManager->CreateNtupleDColumn("FlipProbability");
    fAnalysisManager->CreateNtupleIColumn("StateFlip");
    fAnalysisManager->FinishNtuple();
}

void SdBitRunAction::EndOfRunAction(const G4Run* run)
{
    G4int nofEvents = run->GetNumberOfEvent();
    if (nofEvents == 0) return;

    G4cout << "### Run " << run->GetRunID() << " ended with " 
           << nofEvents << " events." << G4endl;

    // Print run summary
    G4cout << "=== Run Summary ===" << G4endl;
    G4cout << "Total events processed: " << nofEvents << G4endl;

    // Save histograms and ntuples
    fAnalysisManager->Write();
    fAnalysisManager->CloseFile();

    G4cout << "Analysis output saved to sd_bit_simulation.root" << G4endl;
}