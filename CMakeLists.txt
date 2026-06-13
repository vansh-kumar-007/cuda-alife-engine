cmake_minimum_required(VERSION 3.18)
project(CudaALifeEngine CUDA CXX)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Tell CMake your GPU architecture
# 86 = Ampere (RTX 3050) - this is YOUR specific GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Phase 1: hello_cuda executable
add_executable(hello_cuda src/hello_cuda.cu)
target_compile_options(hello_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

add_executable(performance_test src/performance_test.cu)
target_compile_options(performance_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)