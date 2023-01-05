cmake_minimum_required(VERSION 3.13.4)

project(SlicerAutoscoperM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerAutoscoperM")
set(EXTENSION_CATEGORY "Tracking")
set(EXTENSION_CONTRIBUTORS "Bardiya Akhbari (Brown University), Amy Morton (Brown University)")
set(EXTENSION_DESCRIPTION "SlicerAutoscoperM is a 2D to 3D image registration multi-modal software developed as a tool to investigate intra-articular joint motion during dynamic tasks.")
set(EXTENSION_ICONURL "./SlicerAutoscoperM.png")
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
