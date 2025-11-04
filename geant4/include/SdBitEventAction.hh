/**
 * @file SdBitEventAction.hh
 * @brief Event action for per-event processing
 */

#ifndef SDBIT_EVENT_ACTION_HH
#define SDBIT_EVENT_ACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"

class G4Event;

/**
 * @class SdBitEventAction
 * @brief Process events and collect statistics
 */
class SdBitEventAction : public G4UserEventAction
{
public:
    SdBitEventAction();
    ~SdBitEventAction() override;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;

private:
    G4int fPrintModulo;
};

#endif