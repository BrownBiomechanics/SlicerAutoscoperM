cmake_minimum_required(VERSION 3.13.4)

project(SlicerAutoscoperM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://autoscoperm.slicer.org/")
set(EXTENSION_CATEGORY "Tracking")
set(EXTENSION_CONTRIBUTORS "Anthony Lombardi (Kitware), Amy Morton (Brown University), Bardiya Akhbari (Brown University), Beatriz Paniagua (Kitware), Jean-Christophe Fillion-Robin (Kitware)")
set(EXTENSION_DESCRIPTION "SlicerAutoscoperM is a 2D to 3D image registration multi-modal software developed as a tool to investigate intra-articular joint motion during dynamic tasks.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/BrownBiomechanics/SlicerAutoscoperM/main/SlicerAutoscoperM.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/SlicerAutoscoperM/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AutoscoperM)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
