/**
 * @file SdBitSteppingAction.hh
 * @brief Stepping action for detailed particle tracking
 */

#ifndef SDBIT_STEPPING_ACTION_HH
#define SDBIT_STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class G4Step;

/**
 * @class SdBitSteppingAction
 * @brief Track particle steps for detailed analysis
 */
class SdBitSteppingAction : public G4UserSteppingAction
{
public:
    SdBitSteppingAction();
    ~SdBitSteppingAction() override;

    void UserSteppingAction(const G4Step* step) override;

private:
    G4bool fTrackParticles;
};

#endif