/**
 * @file SdBitDetectorConstruction.hh
 * @brief Detector construction for sd-bit nanoscale geometry
 * 
 * Defines the nanoscale geometry of an sd-bit including:
 * - Silicon fluctuation core (bistable memory element)
 * - Isotope source (Am-241 or Tritium)
 * - Tungsten collimator for particle direction
 * - Vacuum environment for isolation
 */

#ifndef SDBIT_DETECTOR_CONSTRUCTION_HH
#define SDBIT_DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "globals.hh"

class G4Box;
class G4VSensitiveDetector;

/**
 * @class SdBitDetectorConstruction
 * @brief Constructs the nanoscale sd-bit geometry
 */
class SdBitDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    SdBitDetectorConstruction();
    ~SdBitDetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

private:
    // Logical volumes
    G4LogicalVolume* fFluctuationCoreLog;
    G4LogicalVolume* fIsotopeSourceLog;
    G4LogicalVolume* fCollimatorLog;
    G4LogicalVolume* fWorldLog;

    // Materials
    G4Material* fSiliconMat;
    G4Material* fTungstenMat;
    G4Material* fAmericiumMat;
    G4Material* fTritiumMat;
    G4Material* fVacuumMat;

    // Geometry parameters
    static constexpr G4double kWorldSize = 1.0; // micrometers
    static constexpr G4double kCoreSize = 50.0; // nanometers (half-size)
    static constexpr G4double kSourceSize = 25.0; // nanometers
    static constexpr G4double kCollimatorRadius = 10.0; // nanometers
    static constexpr G4double kCollimatorLength = 100.0; // nanometers

    // Helper methods
    void DefineMaterials();
    G4Material* CreateAmericium241();
    G4Material* CreateTritium();
};

#endif