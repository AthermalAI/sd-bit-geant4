/**
 * @file SdBitEventAction.cc
 * @brief Implementation of event action
 */

#include "SdBitEventAction.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4AnalysisManager.hh"

SdBitEventAction::SdBitEventAction()
    : G4UserEventAction(),
      fPrintModulo(10000)
{
}

SdBitEventAction::~SdBitEventAction()
{
}

void SdBitEventAction::BeginOfEventAction(const G4Event*)
{
    // Initialize event-level variables if needed
}

void SdBitEventAction::EndOfEventAction(const G4Event* event)
{
    G4int eventID = event->GetEventID();
    
    // Print progress
    if (eventID % fPrintModulo == 0) {
        G4cout << ">>> Event " << eventID << G4endl;
    }
}