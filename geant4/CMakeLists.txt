cmake_minimum_required(VERSION 3.16)
project(sd_bit)

# Find required packages
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collect source files
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Create executable
add_executable(sd_bit sd_bit.cc ${sources} ${headers})

# Link libraries
target_link_libraries(sd_bit ${Geant4_LIBRARIES})

# Set C++ standard
set_property(TARGET sd_bit PROPERTY CXX_STANDARD 17)

# Copy macros to build directory
file(COPY ${PROJECT_SOURCE_DIR}/macros DESTINATION ${PROJECT_BINARY_DIR})

# Installation
install(TARGETS sd_bit DESTINATION bin)