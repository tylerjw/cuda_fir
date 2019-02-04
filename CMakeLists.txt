cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

project(fir LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

# use scripts/cudaComputeVersion.bash to determine arch
set(CMAKE_CUDA_FLAGS "-arch=sm_61 -m64 --ptxas-options=-v")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)