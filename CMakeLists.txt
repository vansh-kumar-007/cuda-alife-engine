cmake_minimum_required(VERSION 3.18)
project(CudaALifeEngine CUDA CXX C)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# External library paths
set(GLFW_INC  ${CMAKE_SOURCE_DIR}/external/glfw/include)
set(GLFW_LIB  ${CMAKE_SOURCE_DIR}/external/glfw/lib-vc2022/glfw3.lib)
set(GLAD_INC  ${CMAKE_SOURCE_DIR}/external/glad/include)
set(GLAD_SRC  ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c)

# Reusable compile options
set(CUDA_OPTS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Phase 1
add_executable(hello_cuda src/hello_cuda.cu)
target_compile_options(hello_cuda PRIVATE ${CUDA_OPTS})

add_executable(performance_test src/performance_test.cu)
target_compile_options(performance_test PRIVATE ${CUDA_OPTS})

# Phase 2
add_executable(particle_sim src/particle_sim.cu)
target_compile_options(particle_sim PRIVATE ${CUDA_OPTS})

add_executable(spatial_hash src/spatial_hash.cu)
target_compile_options(spatial_hash PRIVATE ${CUDA_OPTS})

# Phase 3
add_executable(window_test
    src/window_test.cpp
    ${GLAD_SRC}
)
target_include_directories(window_test PRIVATE ${GLFW_INC} ${GLAD_INC} ${GLAD_INC}/KHR)
target_link_libraries(window_test PRIVATE ${GLFW_LIB} opengl32)


add_executable(particle_render
    src/particle_render.cpp
    ${GLAD_SRC}
)
target_include_directories(particle_render PRIVATE ${GLFW_INC} ${GLAD_INC})
target_link_libraries(particle_render PRIVATE ${GLFW_LIB} opengl32)