cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(PROJECT_NAME particle_sim)
project(${PROJECT_NAME} CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##########################################################
# Project files, packages, libraries and so on
##########################################################

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/src/main.cu
    ${CMAKE_SOURCE_DIR}/src/particle.cu
    ${CMAKE_SOURCE_DIR}/include/patch.h
    ${CMAKE_SOURCE_DIR}/include/particle.h
    ${CMAKE_SOURCE_DIR}/include/vector.h
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
