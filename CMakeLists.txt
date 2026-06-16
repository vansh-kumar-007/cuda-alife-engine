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

add_executable(cuda_gl_interop
    src/cuda_gl_interop.cu
    ${GLAD_SRC}
)
target_include_directories(cuda_gl_interop PRIVATE ${GLFW_INC} ${GLAD_INC})
target_link_libraries(cuda_gl_interop PRIVATE
    ${GLFW_LIB} opengl32 CUDA::cudart
)

add_executable(creature_test
    src/creature_test.cu
    src/creatures.cu
)
target_include_directories(creature_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(creature_test PRIVATE ${CUDA_OPTS})

add_executable(creature_viz
    src/creature_viz.cu
    src/creatures.cu
    ${GLAD_SRC}
)
target_include_directories(creature_viz PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${GLFW_INC} ${GLAD_INC})
target_link_libraries(creature_viz PRIVATE
    ${GLFW_LIB} opengl32 CUDA::cudart)
target_compile_options(creature_viz PRIVATE ${CUDA_OPTS})

add_executable(food_system
    src/food_system.cu
    src/creatures.cu
)
target_include_directories(food_system PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(food_system PRIVATE ${CUDA_OPTS})