/**
 * @file SdBitRunAction.hh
 * @brief Run action for analysis management
 */

#ifndef SDBIT_RUN_ACTION_HH
#define SDBIT_RUN_ACTION_HH

#include "G4UserRunAction.hh"
#include "globals.hh"

class G4Run;
class G4AnalysisManager;

/**
 * @class SdBitRunAction
 * @brief Manage analysis output and run-level statistics
 */
class SdBitRunAction : public G4UserRunAction
{
public:
    SdBitRunAction();
    ~SdBitRunAction() override;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run) override;

private:
    G4AnalysisManager* fAnalysisManager;
};

#endif