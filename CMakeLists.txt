cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(ppo)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(testPPO testPPO.cpp) 
target_link_libraries(testPPO ${TORCH_LIBRARIES})


