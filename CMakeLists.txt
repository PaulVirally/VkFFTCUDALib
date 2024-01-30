cmake_minimum_required(VERSION 3.18)
project(VkFFTCUDA
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Shared library for Julia bindings for VkFFT for the CUDA backend")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release as none was specified.")
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

include(GNUInstallDirs)

set(VKFFT_BACKEND 1)
add_library(VkFFTCUDA SHARED src/vkfft_cuda.cu)

find_package(CUDA 9.0 REQUIRED)
set_target_properties(VkFFTCUDA PROPERTIES CUDA_ARCHITECTURES 60 70 75 80 86)
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
-std=c++11
-DVKFFT_BACKEND=${VKFFT_BACKEND}
-gencode arch=compute_60,code=compute_60
-gencode arch=compute_70,code=compute_70 
-gencode arch=compute_75,code=compute_75 
-gencode arch=compute_80,code=compute_80 
-gencode arch=compute_86,code=compute_86>")
set_target_properties(VkFFTCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(VkFFTCUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_SOURCE_DIR}/lib/VkFFT/vkFFT/)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND})

add_library(half INTERFACE)
target_include_directories(half INTERFACE ${CMAKE_SOURCE_DIR}/lib/VkFFT/half_lib/)

find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
add_definitions(-DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}")
target_include_directories(VkFFTCUDA PUBLIC ${CUDA_INCLUDE_DIRS})

set_target_properties(VkFFTCUDA PROPERTIES 
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/vkfft_cuda.cuh)

target_include_directories(VkFFTCUDA PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(VkFFTCUDA PRIVATE ${CMAKE_SOURCE_DIR}/lib/VkFFT/vkFFT/)

target_link_libraries(VkFFTCUDA PUBLIC ${CUDA_LIBRARIES} cuda ${CUDA_NVRTC_LIB} VkFFT half)

install(TARGETS VkFFTCUDA
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})