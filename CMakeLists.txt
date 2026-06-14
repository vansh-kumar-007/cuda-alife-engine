cmake_minimum_required(VERSION 3.18)
project(CudaALifeEngine CUDA CXX C)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 1. Require CUDA
find_package(CUDAToolkit REQUIRED)

# 2. Fetch GLFW automatically
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# 3. GLAD (We will download the GLAD source files in the next step, but we define the library here)
# For now, we just keep your executables as they are to make sure CMake still configures correctly.

add_executable(hello_cuda src/hello_cuda.cu)
target_compile_options(hello_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(performance_test src/performance_test.cu)
target_compile_options(performance_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(particle_sim src/particle_sim.cu)
target_compile_options(particle_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(spatial_hash src/spatial_hash.cu)
target_compile_options(spatial_hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)