/**
 * @file SdBitActionInitialization.hh
 * @brief Action initialization for sd-bit simulation
 */

#ifndef SDBIT_ACTION_INITIALIZATION_HH
#define SDBIT_ACTION_INITIALIZATION_HH

#include "G4VUserActionInitialization.hh"

/**
 * @class SdBitActionInitialization
 * @brief Initialize user actions for sd-bit simulation
 */
class SdBitActionInitialization : public G4VUserActionInitialization
{
public:
    SdBitActionInitialization();
    ~SdBitActionInitialization() override;

    void Build() const override;
    void BuildForMaster() const override;
};

#endif