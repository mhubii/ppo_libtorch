cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(ppo)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(train_ppo TrainPPO.cpp) 
target_link_libraries(train_ppo ${TORCH_LIBRARIES})

add_executable(test_ppo TestPPO.cpp) 
target_link_libraries(test_ppo ${TORCH_LIBRARIES})
