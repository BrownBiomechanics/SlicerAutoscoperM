cmake_minimum_required(VERSION 3.13.4)

project(SlicerAutoscoperM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://autoscoperm.slicer.org/")
set(EXTENSION_CATEGORY "Tracking")
set(EXTENSION_CONTRIBUTORS "Anthony Lombardi (Kitware), Amy Morton (Brown University), Bardiya Akhbari (Brown University), Beatriz Paniagua (Kitware), Jean-Christophe Fillion-Robin (Kitware)")
set(EXTENSION_DESCRIPTION "SlicerAutoscoperM is an extension for 3D Slicer for 2D-3D image registration integrating with Autoscoper for image-based 3D motion tracking of skeletal structures.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/BrownBiomechanics/SlicerAutoscoperM/main/SlicerAutoscoperM.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/BrownBiomechanics/SlicerAutoscoperM/releases/download/docs-resources/AutoscoperMainWindow.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies
set(EXTENSION_BUILD_SUBDIRECTORY inner-build)

set(SUPERBUILD_TOPLEVEL_PROJECT inner)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})
mark_as_superbuild(Slicer_DIR)

find_package(Git REQUIRED)
mark_as_superbuild(GIT_EXECUTABLE)

#-----------------------------------------------------------------------------
# SuperBuild setup
option(${EXTENSION_NAME}_SUPERBUILD "Build ${EXTENSION_NAME} and the projects it depends on." ON)
mark_as_advanced(${EXTENSION_NAME}_SUPERBUILD)
if(${EXTENSION_NAME}_SUPERBUILD)
  include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
  return()
endif()

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AutoscoperM)
add_subdirectory(TrackingEvaluation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
set(EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS)
if(DEFINED Autoscoper_DIR)
  list(APPEND EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS "${Autoscoper_DIR};Autoscoper;ALL;/")
endif()
set(${EXTENSION_NAME}_CPACK_INSTALL_CMAKE_PROJECTS "${EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS}" CACHE STRING "List of external projects to install" FORCE)

if(APPLE)
  set(EXTENSION_FIXUP_BUNDLE_LIBRARY_DIRECTORIES)
  set(GLEW_RUNTIME_LIBRARY_DIR "${Autoscoper_DIR}/../Autoscoper-build/GLEW-install/${Slicer_INSTALL_THIRDPARTY_LIB_DIR}")
  list(APPEND EXTENSION_FIXUP_BUNDLE_LIBRARY_DIRECTORIES ${GLEW_RUNTIME_LIBRARY_DIR})
  set(TIFF_RUNTIME_LIBRARY_DIR "${Autoscoper_DIR}/../Autoscoper-build/TIFF-install/${Slicer_INSTALL_THIRDPARTY_LIB_DIR}")
  list(APPEND EXTENSION_FIXUP_BUNDLE_LIBRARY_DIRECTORIES ${TIFF_RUNTIME_LIBRARY_DIR})
  set(${EXTENSION_NAME}_FIXUP_BUNDLE_LIBRARY_DIRECTORIES "${EXTENSION_FIXUP_BUNDLE_LIBRARY_DIRECTORIES}" CACHE STRING "List of directories to look up libraries to copy into the application package" FORCE)
endif()

#-----------------------------------------------------------------------------
list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${EXTENSION_NAME};ALL;/")
list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${${EXTENSION_NAME}_CPACK_INSTALL_CMAKE_PROJECTS}")
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
