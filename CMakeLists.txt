cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
set(WARNINGS_AS_ERRORS ON)

##########################################################
# Project name, files, packages, libraries and so on
##########################################################

set(PROJECT_NAME particle_sim)
project(${PROJECT_NAME} CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include/)

add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/src/main.cu
    ${CMAKE_SOURCE_DIR}/src/particle.cu
    ${CMAKE_SOURCE_DIR}/src/particle_box.cu
    ${CMAKE_SOURCE_DIR}/include/patch.h
    ${CMAKE_SOURCE_DIR}/include/particle.h
    ${CMAKE_SOURCE_DIR}/include/particle_box.h
    ${CMAKE_SOURCE_DIR}/include/cell_view.h
    ${CMAKE_SOURCE_DIR}/include/vec.h
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

set_target_properties(${PROJECT_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
