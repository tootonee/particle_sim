cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
set(WARNINGS_AS_ERRORS ON)

##########################################################
# Project name, files, packages, libraries and so on
##########################################################

set(PROJECT_NAME particle_sim_fast)
project(${PROJECT_NAME} CXX CUDA)

file(GLOB HEADERS ${CMAKE_SOURCE_DIR}/include/*.h)


add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/src/main.cu
    ${CMAKE_SOURCE_DIR}/src/particle.cu
    ${CMAKE_SOURCE_DIR}/src/particle_box.cu
    ${CMAKE_SOURCE_DIR}/src/pdb_export.cu
    ${CMAKE_SOURCE_DIR}/src/cell_view.cu
    ${HEADERS}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
set_target_properties(${PROJECT_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#[[ target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11) ]]
