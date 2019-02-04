cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

project(fir LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CUDA_NVCC_FLAGS "--ptxas-options=-v")

if (NOT CUDA_TARGET_COMPUTE_CAPABILITY)
    if("$ENV{CUDA_SM}" STREQUAL "")
        set(ENV{CUDA_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}")
        set(ENV{CUDA_CUDART_LIBRARY} "${CUDA_CUDART_LIBRARY}")
        set(ENV{CMAKE_CXX_COMPILER} "${CMAKE_CXX_COMPILER}")
        execute_process(COMMAND 
            bash -c "${CMAKE_CURRENT_SOURCE_DIR}/scripts/get_cuda_sm.sh" 
            OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY_)
    else()
        set(CUDA_TARGET_COMPUTE_CAPABILITY_ $ENV{CUDA_SM})
    endif()

    set(CUDA_TARGET_COMPUTE_CAPABILITY "${CUDA_TARGET_COMPUTE_CAPABILITY_}")

    execute_process(COMMAND 
        bash -c "echo -n $(echo ${CUDA_TARGET_COMPUTE_CAPABILITY})" 
        OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY)
    execute_process(COMMAND 
        bash -c "echo ${CUDA_TARGET_COMPUTE_CAPABILITY} | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n" 
        OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY)

    message(STATUS 
        "CUDA device-side code will assume compute capability \
        ${FORMATTED_COMPUTE_CAPABILITY}")
endif()
set(CUDA_GENCODE
    "-arch=sm_${CUDA_TARGET_COMPUTE_CAPABILITY}")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_GENCODE}" )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)