#-----------------------------------------------------------------------------
# External project common settings
#-----------------------------------------------------------------------------

set(ep_common_c_flags "${CMAKE_C_FLAGS_INIT} ${ADDITIONAL_C_FLAGS}")
set(ep_common_cxx_flags "${CMAKE_CXX_FLAGS_INIT} ${ADDITIONAL_CXX_FLAGS}")

#-----------------------------------------------------------------------------
# Top-level "external" project
#-----------------------------------------------------------------------------

# Extension dependencies
foreach(dep ${EXTENSION_DEPENDS})
  mark_as_superbuild(${dep}_DIR)
endforeach()

set(proj ${SUPERBUILD_TOPLEVEL_PROJECT})

# Project dependencies
set(${proj}_DEPENDS
  Autoscoper-OpenCL
  )

if(DEFINED ENV{SlicerAutoscoperM_CUDA_PATH})
  file(TO_CMAKE_PATH "$ENV{SlicerAutoscoperM_CUDA_PATH}" _cuda_path)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # https://cmake.org/cmake/help/latest/variable/CMAKE_GENERATOR_TOOLSET.html#visual-studio-toolset-selection
    if(NOT "${CMAKE_GENERATOR_TOOLSET}" MATCHES "cuda=")
      set(_generator_toolset_key_value "cuda=${_cuda_path}")
      if(NOT "${CMAKE_GENERATOR_TOOLSET}" STREQUAL "")
        set(CMAKE_GENERATOR_TOOLSET "${CMAKE_GENERATOR_TOOLSET},${_generator_toolset_key_value}")
      else()
        set(CMAKE_GENERATOR_TOOLSET "${_generator_toolset_key_value}")
      endif()
    endif()
  else()
    # See https://cmake.org/cmake/help/latest/envvar/CUDACXX.html#cudacxx
    set(ENV{CUDACXX} ${_cuda_path}/bin/nvcc)
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Variable expected by CUDA language also used in Autoscoper
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    mark_as_superbuild(VARS CMAKE_GENERATOR_TOOLSET PROJECTS Autoscoper-CUDA)
  else()
    mark_as_superbuild(VARS CMAKE_CUDA_COMPILER PROJECTS Autoscoper-CUDA)
  endif()
  list(APPEND ${proj}_DEPENDS
    Autoscoper-CUDA
    )
endif()

ExternalProject_Include_Dependencies(${proj}
  PROJECT_VAR proj
  SUPERBUILD_VAR ${EXTENSION_NAME}_SUPERBUILD
  )

ExternalProject_Add(${proj}
  ${${proj}_EP_ARGS}
  DOWNLOAD_COMMAND ""
  INSTALL_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${EXTENSION_BUILD_SUBDIRECTORY}
  CMAKE_CACHE_ARGS
    # Compiler settings
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    # Output directories
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
    # Superbuild
    -D${EXTENSION_NAME}_SUPERBUILD:BOOL=OFF
    -DEXTENSION_SUPERBUILD_BINARY_DIR:PATH=${${EXTENSION_NAME}_BINARY_DIR}
  DEPENDS
    ${${proj}_DEPENDS}
  )

ExternalProject_AlwaysConfigure(${proj})
