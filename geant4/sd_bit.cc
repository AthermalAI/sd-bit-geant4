/**
 * @file sd_bit.cc
 * @brief Main application for sd-bit GEANT4 simulation
 * 
 * This application simulates the nanoscale physics of a stochastic-decay bit
 * using GEANT4's particle transport capabilities. It models radioactive decay
 * events in a microscopic isotope source and their interaction with a silicon
 * bistable memory core.
 */

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4SteppingVerbose.hh"

#include "SdBitDetectorConstruction.hh"
#include "SdBitPhysicsList.hh"
#include "SdBitActionInitialization.hh"

int main(int argc, char** argv)
{
    // Detect interactive mode (if no arguments) and define UI session
    G4UIExecutive* ui = nullptr;
    if (argc == 1) {
        ui = new G4UIExecutive(argc, argv);
    }

    // Optionally: choose a different Random engine
    // G4Random::setTheEngine(new CLHEP::MTwistEngine);

    // Use G4SteppingVerboseWithUnits
    G4int precision = 4;
    G4SteppingVerbose::UseBestUnit(precision);

    // Construct the default run manager
    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

    // Set mandatory initialization classes
    runManager->SetUserInitialization(new SdBitDetectorConstruction());
    runManager->SetUserInitialization(new SdBitPhysicsList());
    runManager->SetUserInitialization(new SdBitActionInitialization());

    // Initialize visualization
    G4VisManager* visManager = new G4VisExecutive;
    visManager->Initialize();

    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    // Process macro or start UI session
    if (!ui) {
        // batch mode
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command + fileName);
    } else {
        // interactive mode
        UImanager->ApplyCommand("/control/execute macros/init_vis.mac");
        ui->SessionStart();
        delete ui;
    }

    // Job termination
    delete visManager;
    delete runManager;

    return 0;
}