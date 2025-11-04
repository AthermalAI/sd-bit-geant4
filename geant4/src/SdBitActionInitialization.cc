/**
 * @file SdBitActionInitialization.cc
 * @brief Implementation of action initialization
 */

#include "SdBitActionInitialization.hh"
#include "SdBitPrimaryGeneratorAction.hh"
#include "SdBitRunAction.hh"
#include "SdBitEventAction.hh"
#include "SdBitSteppingAction.hh"

SdBitActionInitialization::SdBitActionInitialization()
    : G4VUserActionInitialization()
{
}

SdBitActionInitialization::~SdBitActionInitialization()
{
}

void SdBitActionInitialization::Build() const
{
    SetUserAction(new SdBitPrimaryGeneratorAction);
    SetUserAction(new SdBitRunAction);
    SetUserAction(new SdBitEventAction);
    SetUserAction(new SdBitSteppingAction);
}

void SdBitActionInitialization::BuildForMaster() const
{
    SetUserAction(new SdBitRunAction);
}