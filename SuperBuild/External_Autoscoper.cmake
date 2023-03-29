
set(proj Autoscoper)

# Set dependency list
set(${proj}_DEPENDS
  ""
  )

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj)

if(${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED Autoscoper_DIR AND NOT EXISTS ${Autoscoper_DIR})
  message(FATAL_ERROR "Autoscoper_DIR [${Autoscoper_DIR}] variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED ${proj}_DIR AND NOT ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)

  ExternalProject_SetIfNotDefined(
    Slicer_${proj}_GIT_REPOSITORY
    "https://github.com/BrownBiomechanics/Autoscoper.git"
    QUIET
  )

  ExternalProject_SetIfNotDefined(
    Slicer_${proj}_GIT_TAG
    "60b3d6ec0a5e4b796ef97baaa7a3a9b0351280e7"
    QUIET
  )

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  # Workaround improper generation of target file when destination associated
  # with "install(EXPORT <export-name> DESTINATION <dir>)" start with "./"
  if(NOT APPLE)
    set(Slicer_INSTALL_THIRDPARTY_LIB_DIR ${Slicer_THIRDPARTY_LIB_DIR})
  endif()

  if(UNIX AND NOT APPLE)
    if(NOT DEFINED OpenGL_GL_PREFERENCE OR "${OpenGL_GL_PREFERENCE}" STREQUAL "")
      set(OpenGL_GL_PREFERENCE "LEGACY")
    endif()
    if(NOT "${OpenGL_GL_PREFERENCE}" MATCHES "^(LEGACY|GLVND)$")
      message(FATAL_ERROR "OpenGL_GL_PREFERENCE variable is expected to be set to LEGACY or GLVND")
    endif()
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DOpenGL_GL_PREFERENCE:STRING=${OpenGL_GL_PREFERENCE}
      )
  endif()

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY "${Slicer_${proj}_GIT_REPOSITORY}"
    GIT_TAG "${Slicer_${proj}_GIT_TAG}"
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
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
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_BIN_DIR}
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_LIB_DIR}
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
      # Install directories
      -DAutoscoper_BIN_DIR:STRING=${Slicer_INSTALL_THIRDPARTY_BIN_DIR}
      -DAutoscoper_LIB_DIR:STRING=${Slicer_INSTALL_THIRDPARTY_LIB_DIR}
      # Options
      -DAutoscoper_SUPERBUILD:BOOL=ON
      -DAutoscoper_INSTALL_Qt_LIBRARIES:BOOL=OFF
      -DAutoscoper_INSTALL_SAMPLE_DATA:BOOL=OFF
      -DAutoscoper_RENDERING_BACKEND:STRING=OpenCL
      -DQt5_DIR:PATH=${Qt5_DIR}
      # Dependencies
      # NA
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    INSTALL_COMMAND ""
    DEPENDS
      ${${proj}_DEPENDS}
    )

  set(${proj}_DIR ${EP_BINARY_DIR}/Autoscoper-build)

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDS})
endif()

mark_as_superbuild(${proj}_DIR:PATH)
