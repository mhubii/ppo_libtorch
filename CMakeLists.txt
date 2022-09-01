cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ppo)

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(train_ppo TrainPPO.cpp) 
target_link_libraries(train_ppo ${TORCH_LIBRARIES})
target_include_directories(train_ppo PRIVATE ${CMAKE_CURRENT_SOURCE})
set_property(TARGET train_ppo PROPERTY CXX_STANDARD_14)

add_executable(test_ppo TestPPO.cpp) 
target_link_libraries(test_ppo ${TORCH_LIBRARIES})
target_include_directories(test_ppo PRIVATE ${CMAKE_CURRENT_SOURCE})
set_property(TARGET test_ppo PROPERTY CXX_STANDARD_14)
