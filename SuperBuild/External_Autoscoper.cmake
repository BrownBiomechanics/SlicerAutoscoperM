
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
    "4bdedd802034a1c341099d735010690105a106ad"
    QUIET
  )

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  # Workaround improper generation of target file when destination associated
  # with "install(EXPORT <export-name> DESTINATION <dir>)" start with "./"
  if(NOT APPLE)
    set(Slicer_INSTALL_THIRDPARTY_LIB_DIR ${Slicer_THIRDPARTY_LIB_DIR})
  endif()

  set(Autoscoper_INSTALL_DEPENDENCIES TRUE)
  if(APPLE)
    # Dependency libraries (e.g Glew) will be installed leveraging
    # the macOS "fix-up" script.
    set(Autoscoper_INSTALL_DEPENDENCIES FALSE)
  endif()

  if(APPLE)
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DAutoscoper_EXECUTABLE_LINK_FLAGS:STRING=${Slicer_INSTALL_THIRDPARTY_EXECUTABLE_LINK_FLAGS}
      -DAutoscoper_MACOSX_BUNDLE:STRING=OFF
      )
  endif()

  if(NOT "${Autoscoper_RENDERING_BACKEND}" MATCHES "^(CUDA|OpenCL)$")
    message(FATAL_ERROR "Autoscoper_RENDERING_BACKEND must be set to CUDA or OpenCL")
  endif()

  if(Autoscoper_RENDERING_BACKEND STREQUAL "OpenCL")
    set(Autoscoper_OPENCL_USE_ICD_LOADER TRUE)
    if(APPLE)
      set(Autoscoper_OPENCL_USE_ICD_LOADER FALSE)
    endif()
  else()
    set(Autoscoper_OPENCL_USE_ICD_LOADER FALSE)
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
      -DAutoscoper_CONFIGURE_LAUCHER_SCRIPT:BOOL=OFF
      -DAutoscoper_OPENCL_USE_ICD_LOADER:BOOL=${Autoscoper_OPENCL_USE_ICD_LOADER}
      -DAutoscoper_INSTALL_DEPENDENCIES:BOOL=${Autoscoper_INSTALL_DEPENDENCIES}
      -DAutoscoper_INSTALL_Qt_LIBRARIES:BOOL=OFF
      -DAutoscoper_INSTALL_SAMPLE_DATA:BOOL=OFF
      -DAutoscoper_RENDERING_BACKEND:STRING=${Autoscoper_RENDERING_BACKEND}
      -DQt5_DIR:PATH=${Qt5_DIR}
      # Dependencies
      # NA
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    INSTALL_COMMAND ""
    DEPENDS
      ${${proj}_DEPENDS}
    )

  if(APPLE)
    set(glew_library ${EP_BINARY_DIR}/GLEW-install/${Slicer_INSTALL_THIRDPARTY_LIB_DIR}/libGLEW$<$<CONFIG:Debug>:d>.2.2.0.dylib)
    ExternalProject_Add_Step(${proj} fix_glew_rpath
      COMMAND install_name_tool -id ${glew_library} ${glew_library}
      DEPENDEES build
      )
    set(tiff_library ${EP_BINARY_DIR}/TIFF-install/${Slicer_INSTALL_THIRDPARTY_LIB_DIR}/libtiff.5.8.0.dylib)
    ExternalProject_Add_Step(${proj} fix_tiff_rpath
      COMMAND install_name_tool -id ${tiff_library} ${tiff_library}
      DEPENDEES build
      )
    set(autoscoper_executable ${EP_BINARY_DIR}/Autoscoper-build/${Slicer_INSTALL_THIRDPARTY_BIN_DIR}/autoscoper)
    ExternalProject_Add_Step(${proj} fix_autoscoper_rpath
      COMMAND install_name_tool
        -change "@rpath/libGLEW$<$<CONFIG:Debug>:d>.2.2.dylib" ${glew_library}
        -change "@rpath/libtiff.5.dylib" ${tiff_library}
        ${autoscoper_executable}
      DEPENDEES fix_glew_rpath fix_tiff_rpath
      )
  endif()

  set(${proj}_DIR ${EP_BINARY_DIR}/Autoscoper-build)

  #-----------------------------------------------------------------------------
  # Launcher setting specific to build tree

  # library paths
  set(${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
    ${EP_BINARY_DIR}/GLEW-Install/${Slicer_THIRDPARTY_BIN_DIR} # Glew library
    ${EP_BINARY_DIR}/TIFF-Install/${Slicer_THIRDPARTY_BIN_DIR} # TIFF library
    )
  if(Autoscoper_OPENCL_USE_ICD_LOADER)
    list(APPEND ${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
      ${EP_BINARY_DIR}/OpenCL-ICD-Loader-build/${Slicer_THIRDPARTY_BIN_DIR}/${CMAKE_CFG_INTDIR} # OpenCL library
      )
  endif()
  mark_as_superbuild(
    VARS ${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
    LABELS "LIBRARY_PATHS_LAUNCHER_BUILD"
    )

  # paths
  set(${proj}_PATHS_LAUNCHER_BUILD
    ${EP_BINARY_DIR}/Autoscoper-build/${Slicer_THIRDPARTY_BIN_DIR}/${CMAKE_CFG_INTDIR} # Autoscoper executable
    )
  mark_as_superbuild(
    VARS ${proj}_PATHS_LAUNCHER_BUILD
    LABELS "PATHS_LAUNCHER_BUILD"
    )

  #-----------------------------------------------------------------------------
  # Launcher setting specific to install tree

  # NA

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDS})
endif()

mark_as_superbuild(${proj}_DIR:PATH)
