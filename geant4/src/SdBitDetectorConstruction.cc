/**
 * @file SdBitDetectorConstruction.cc
 * @brief Implementation of sd-bit detector construction
 */

#include "SdBitDetectorConstruction.hh"
#include "SdBitSensitiveDetector.hh"

#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4SDManager.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

SdBitDetectorConstruction::SdBitDetectorConstruction()
    : G4VUserDetectorConstruction(),
      fFluctuationCoreLog(nullptr),
      fIsotopeSourceLog(nullptr),
      fCollimatorLog(nullptr),
      fWorldLog(nullptr),
      fSiliconMat(nullptr),
      fTungstenMat(nullptr),
      fAmericiumMat(nullptr),
      fTritiumMat(nullptr),
      fVacuumMat(nullptr)
{
}

SdBitDetectorConstruction::~SdBitDetectorConstruction()
{
}

G4VPhysicalVolume* SdBitDetectorConstruction::Construct()
{
    // Define materials
    DefineMaterials();

    // World volume (vacuum environment)
    G4Box* worldBox = new G4Box("World", 
                                kWorldSize * micrometer, 
                                kWorldSize * micrometer, 
                                kWorldSize * micrometer);
    
    fWorldLog = new G4LogicalVolume(worldBox, fVacuumMat, "WorldLog");
    
    G4VPhysicalVolume* worldPhys = new G4PVPlacement(0,                // no rotation
                                                     G4ThreeVector(),  // at origin
                                                     fWorldLog,        // logical volume
                                                     "World",          // name
                                                     0,                // mother volume
                                                     false,            // no boolean operation
                                                     0,                // copy number
                                                     true);            // check overlaps

    // Isotope source (Am-241 or Tritium)
    G4Box* sourceBox = new G4Box("Source", 
                                 kSourceSize * nanometer,
                                 kSourceSize * nanometer,
                                 kSourceSize * nanometer);
    
    fIsotopeSourceLog = new G4LogicalVolume(sourceBox, fAmericiumMat, "SourceLog");
    
    new G4PVPlacement(0,
                      G4ThreeVector(0, 0, -75 * nanometer), // Below core
                      fIsotopeSourceLog,
                      "IsotopeSource",
                      fWorldLog,
                      false,
                      0,
                      true);

    // Tungsten collimator (cylindrical tunnel)
    G4Tubs* collimatorTube = new G4Tubs("Collimator",
                                        0,                              // inner radius
                                        kCollimatorRadius * nanometer,  // outer radius
                                        kCollimatorLength * nanometer,  // half-length
                                        0,                              // start angle
                                        2 * pi);                       // spanning angle
    
    fCollimatorLog = new G4LogicalVolume(collimatorTube, fTungstenMat, "CollimatorLog");
    
    new G4PVPlacement(0,
                      G4ThreeVector(0, 0, -25 * nanometer), // Between source and core
                      fCollimatorLog,
                      "Collimator",
                      fWorldLog,
                      false,
                      0,
                      true);

    // Fluctuation core (silicon bistable element)
    G4Box* coreBox = new G4Box("Core",
                               kCoreSize * nanometer,
                               kCoreSize * nanometer,
                               kCoreSize * nanometer);
    
    fFluctuationCoreLog = new G4LogicalVolume(coreBox, fSiliconMat, "CoreLog");
    
    new G4PVPlacement(0,
                      G4ThreeVector(0, 0, 75 * nanometer), // Above collimator
                      fFluctuationCoreLog,
                      "FluctuationCore",
                      fWorldLog,
                      false,
                      0,
                      true);

    // Visualization attributes
    G4VisAttributes* worldVisAtt = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0, 0.1));
    worldVisAtt->SetVisibility(false);
    fWorldLog->SetVisAttributes(worldVisAtt);

    G4VisAttributes* sourceVisAtt = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 0.8));
    fIsotopeSourceLog->SetVisAttributes(sourceVisAtt);

    G4VisAttributes* collimatorVisAtt = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.9));
    fCollimatorLog->SetVisAttributes(collimatorVisAtt);

    G4VisAttributes* coreVisAtt = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0, 0.7));
    fFluctuationCoreLog->SetVisAttributes(coreVisAtt);

    return worldPhys;
}

void SdBitDetectorConstruction::ConstructSDandField()
{
    // Create sensitive detector for the fluctuation core
    SdBitSensitiveDetector* coreSD = new SdBitSensitiveDetector("CoreSD");
    G4SDManager::GetSDMpointer()->AddNewDetector(coreSD);
    fFluctuationCoreLog->SetSensitiveDetector(coreSD);
}

void SdBitDetectorConstruction::DefineMaterials()
{
    G4NistManager* nist = G4NistManager::Instance();

    // Standard materials
    fSiliconMat = nist->FindOrBuildMaterial("G4_Si");
    fTungstenMat = nist->FindOrBuildMaterial("G4_W");
    fVacuumMat = nist->FindOrBuildMaterial("G4_Galactic");

    // Custom isotope materials
    fAmericiumMat = CreateAmericium241();
    fTritiumMat = CreateTritium();
}

G4Material* SdBitDetectorConstruction::CreateAmericium241()
{
    // Create Am-241 isotope
    G4Isotope* Am241 = new G4Isotope("Am241", 95, 241, 241.0568 * g/mole);
    G4Element* Am = new G4Element("Americium", "Am", 1);
    Am->AddIsotope(Am241, 100.0 * perCent);

    // Create Am-241 material (metallic americium density)
    G4Material* americiumMat = new G4Material("Americium241", 
                                              13.67 * g/cm3, 
                                              1,
                                              kStateSolid,
                                              293.15 * kelvin,
                                              1.0 * atmosphere);
    americiumMat->AddElement(Am, 1);

    return americiumMat;
}

G4Material* SdBitDetectorConstruction::CreateTritium()
{
    // Create H-3 (Tritium) isotope
    G4Isotope* H3 = new G4Isotope("H3", 1, 3, 3.016049 * g/mole);
    G4Element* tritiumElement = new G4Element("Tritium", "T", 1);
    tritiumElement->AddIsotope(H3, 100.0 * perCent);

    // Create tritium gas material
    G4Material* tritiumMat = new G4Material("TritiumGas",
                                            0.000134 * g/cm3, // Tritium gas density at STP
                                            1,
                                            kStateGas,
                                            293.15 * kelvin,
                                            1.0 * atmosphere);
    tritiumMat->AddElement(tritiumElement, 2); // H2 molecule

    return tritiumMat;
}