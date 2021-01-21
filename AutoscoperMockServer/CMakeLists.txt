cmake_minimum_required(VERSION 3.13.4)

project(AutoscoperMockServer)

find_package(Qt5
  COMPONENTS
    Core
    Network
  REQUIRED
  )

set(CMAKE_AUTOMOC ON)

set(application_name ${PROJECT_NAME})

add_executable(${application_name}
  main.cpp
  AutoscoperMockMainWindow.cpp
  Socket.cpp
  )

target_link_libraries(${application_name}
  Qt5::Core
  Qt5::Network
  )
