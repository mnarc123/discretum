cmake_minimum_required(VERSION 3.24)
project(discretum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DISCRETUM_ENABLE_GPU "Enable HIP/ROCm GPU support" OFF)
set(GPU_TARGETS "gfx1100" CACHE STRING "GPU architectures (RX 7900 XTX)")

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# OpenMP for parallel fitness evaluations
find_package(OpenMP REQUIRED)

# Compilation options
add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Core library (CPU only)
add_library(discretum_core
    src/core/graph.cpp
    src/automaton/rule_parametric.cpp
    src/automaton/evolution.cpp
    src/geometry/ollivier_ricci.cpp
    src/geometry/spectral_dimension.cpp
    src/geometry/geodesic.cpp
    src/geometry/metric_tensor.cpp
    src/search/fitness.cpp
    src/search/cmaes.cpp
    src/search/genetic.cpp
    src/search/ensemble.cpp
)
target_link_libraries(discretum_core PUBLIC 
    Eigen3::Eigen 
    fmt::fmt 
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

# GPU library (optional)
if(DISCRETUM_ENABLE_GPU)
    set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    enable_language(HIP)
    find_package(hip REQUIRED)
    
    add_library(discretum_gpu
        src/gpu/hip_graph_evolve.hip
        src/gpu/hip_ollivier.hip
        src/gpu/hip_random_walk.hip
        src/gpu/hip_utils.hip
    )
    set_source_files_properties(
        src/gpu/hip_graph_evolve.hip
        src/gpu/hip_ollivier.hip
        src/gpu/hip_random_walk.hip
        src/gpu/hip_utils.hip
        PROPERTIES LANGUAGE HIP
    )
    set_target_properties(discretum_gpu PROPERTIES HIP_ARCHITECTURES ${GPU_TARGETS})
    target_link_libraries(discretum_gpu PRIVATE hip::device)
    target_link_libraries(discretum_gpu PUBLIC discretum_core)
    target_include_directories(discretum_gpu PRIVATE /opt/rocm-7.2.0/include)
    target_compile_definitions(discretum_core PUBLIC DISCRETUM_HAS_GPU=1)
    
    set(GPU_LINK_LIB discretum_gpu)
else()
    set(GPU_LINK_LIB "")
endif()

# Main search executable
add_executable(discretum_search src/main_search.cpp)
target_link_libraries(discretum_search PRIVATE discretum_core ${GPU_LINK_LIB})

# Validation executable
add_executable(discretum_validate src/main_validate.cpp)
target_link_libraries(discretum_validate PRIVATE discretum_core ${GPU_LINK_LIB})

# Diagnostics executable
add_executable(discretum_diagnose src/main_diagnose.cpp)
target_link_libraries(discretum_diagnose PRIVATE discretum_core ${GPU_LINK_LIB})

# Calibration executable
add_executable(discretum_calibrate src/main_calibrate.cpp)
target_link_libraries(discretum_calibrate PRIVATE discretum_core ${GPU_LINK_LIB})

# Tests (CPU tests always, GPU tests only if enabled)
set(TEST_SOURCES
    tests/test_graph.cpp
    tests/test_ollivier_ricci.cpp
    tests/test_spectral_dimension.cpp
    tests/test_evolution.cpp
    tests/test_fitness.cpp
    tests/test_geometry.cpp
)
if(DISCRETUM_ENABLE_GPU)
    list(APPEND TEST_SOURCES tests/test_gpu_kernels.cpp)
endif()

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE discretum_core ${GPU_LINK_LIB} Catch2::Catch2WithMain)

# Benchmarks
add_executable(bench_evolution benchmarks/bench_evolution.cpp)
target_link_libraries(bench_evolution PRIVATE discretum_core ${GPU_LINK_LIB})

add_executable(bench_ollivier benchmarks/bench_ollivier.cpp)
target_link_libraries(bench_ollivier PRIVATE discretum_core ${GPU_LINK_LIB})

add_executable(bench_gpu_vs_cpu benchmarks/bench_gpu_vs_cpu.cpp)
target_link_libraries(bench_gpu_vs_cpu PRIVATE discretum_core ${GPU_LINK_LIB})

# Enable testing
enable_testing()
add_test(NAME unit_tests COMMAND tests)
