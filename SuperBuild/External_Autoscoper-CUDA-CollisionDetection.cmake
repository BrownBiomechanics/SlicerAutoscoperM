set(proj Autoscoper-CUDA-CollisionDetection)
set(${proj}_RENDERING_BACKEND CUDA)
set(${proj}_COLLISION_DETECTION 1)
include(${CMAKE_CURRENT_LIST_DIR}/External_Autoscoper.cmake)
