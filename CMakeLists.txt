cmake_minimum_required(VERSION 3.20)
project(turboquant VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(TQ_BUILD_TESTS "Build tests" OFF)
option(TQ_BUILD_BENCH "Build benchmarks" OFF)
option(TQ_BUILD_CUDA "Build CUDA backend" OFF)
option(TQ_BUILD_METAL "Build Metal backend" OFF)

# Threads (pthread)
find_package(Threads REQUIRED)

# Core library
file(GLOB TQ_CORE_SOURCES src/core/*.c)
file(GLOB TQ_CACHE_SOURCES src/cache/*.c)
file(GLOB TQ_CPU_SOURCES src/backend/cpu/*.c)
file(GLOB TQ_ENGINE_SOURCES src/engine/*.c)

add_library(turboquant STATIC
    ${TQ_CORE_SOURCES}
    ${TQ_CACHE_SOURCES}
    ${TQ_CPU_SOURCES}
    ${TQ_ENGINE_SOURCES}
)
target_include_directories(turboquant PUBLIC include)
target_link_libraries(turboquant PRIVATE m Threads::Threads)

# Shared library for Python bindings
add_library(turboquant_shared SHARED
    ${TQ_CORE_SOURCES}
    ${TQ_CACHE_SOURCES}
    ${TQ_CPU_SOURCES}
    ${TQ_ENGINE_SOURCES}
)
target_include_directories(turboquant_shared PUBLIC include)
target_link_libraries(turboquant_shared PRIVATE m Threads::Threads)
set_target_properties(turboquant_shared PROPERTIES
    OUTPUT_NAME turboquant
    POSITION_INDEPENDENT_CODE ON)

# Metal backend (Apple Silicon GPU)
if(TQ_BUILD_METAL AND APPLE)
    enable_language(OBJC)

    # Compile .metal shaders to .metallib (requires Xcode metal compiler)
    execute_process(
        COMMAND xcrun --find metal
        OUTPUT_VARIABLE TQ_METAL_COMPILER
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TQ_METAL_COMPILER_FOUND
    )

    file(GLOB TQ_METAL_SHADERS src/backend/metal/*.metal)
    set(TQ_METALLIB "${CMAKE_BINARY_DIR}/turboquant.metallib")

    if(TQ_METAL_COMPILER_FOUND EQUAL 0 AND TQ_METAL_COMPILER)
        # Build individual .air files then link into .metallib
        set(TQ_METAL_AIR_FILES "")
        foreach(shader ${TQ_METAL_SHADERS})
            get_filename_component(shader_name ${shader} NAME_WE)
            set(air_file "${CMAKE_BINARY_DIR}/${shader_name}.air")
            add_custom_command(
                OUTPUT ${air_file}
                COMMAND xcrun -sdk macosx metal -c ${shader} -o ${air_file}
                        -std=metal3.0 -O2
                DEPENDS ${shader}
                COMMENT "Compiling Metal shader: ${shader_name}.metal"
            )
            list(APPEND TQ_METAL_AIR_FILES ${air_file})
        endforeach()

        add_custom_command(
            OUTPUT ${TQ_METALLIB}
            COMMAND xcrun -sdk macosx metallib ${TQ_METAL_AIR_FILES} -o ${TQ_METALLIB}
            DEPENDS ${TQ_METAL_AIR_FILES}
            COMMENT "Linking Metal library: turboquant.metallib"
        )
        add_custom_target(turboquant_metallib ALL DEPENDS ${TQ_METALLIB})
    else()
        message(WARNING "TurboQuant: Metal shader compiler not found (need Xcode). "
                        "Shaders will not be compiled — runtime .metallib loading will fail. "
                        "Obj-C dispatch code will still be built.")
        add_custom_target(turboquant_metallib)  # no-op target
    endif()

    # Compile Objective-C dispatch files
    file(GLOB TQ_METAL_OBJ_SOURCES src/backend/metal/*.m)
    target_sources(turboquant PRIVATE ${TQ_METAL_OBJ_SOURCES})
    target_sources(turboquant_shared PRIVATE ${TQ_METAL_OBJ_SOURCES})

    # Set Objective-C ARC for .m files
    set_source_files_properties(${TQ_METAL_OBJ_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # Link Metal and Foundation frameworks
    target_link_libraries(turboquant PRIVATE
        "-framework Metal" "-framework Foundation"
    )
    target_link_libraries(turboquant_shared PRIVATE
        "-framework Metal" "-framework Foundation"
    )

    # Define TQ_HAS_METAL for conditional compilation
    target_compile_definitions(turboquant PRIVATE TQ_HAS_METAL=1)
    target_compile_definitions(turboquant_shared PRIVATE TQ_HAS_METAL=1)

    # Ensure metallib is built before the library
    add_dependencies(turboquant turboquant_metallib)
    add_dependencies(turboquant_shared turboquant_metallib)

    message(STATUS "TurboQuant: Metal backend enabled — shaders: ${TQ_METALLIB}")
endif()

# Compiler warnings
target_compile_options(turboquant PRIVATE
    -Wall -Wextra -Wpedantic -Wno-unused-parameter
)
target_compile_options(turboquant_shared PRIVATE
    -Wall -Wextra -Wpedantic -Wno-unused-parameter
)

# Tests
if(TQ_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    file(GLOB TEST_SOURCES tests/*.cpp)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} turboquant GTest::gtest_main)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # llama.cpp integration test
    add_executable(test_llamacpp_integration
        integrations/llamacpp/test_integration.cpp)
    target_include_directories(test_llamacpp_integration PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/integrations/llamacpp)
    target_link_libraries(test_llamacpp_integration turboquant GTest::gtest_main)
    add_test(NAME test_llamacpp_integration COMMAND test_llamacpp_integration)
endif()

# Benchmarks
if(TQ_BUILD_BENCH)
    file(GLOB BENCH_SOURCES bench/*.cpp)
    file(GLOB BENCH_PERF_SOURCES bench/performance/*.cpp)
    list(APPEND BENCH_SOURCES ${BENCH_PERF_SOURCES})
    foreach(bench_src ${BENCH_SOURCES})
        get_filename_component(bench_name ${bench_src} NAME_WE)
        add_executable(${bench_name} ${bench_src})
        target_link_libraries(${bench_name} turboquant)
    endforeach()
endif()

# CLI inference tool
add_executable(tq_run tools/tq_run.c)
target_link_libraries(tq_run turboquant)

# Debug comparison tool
add_executable(debug_compare tools/debug_compare.c)
target_link_libraries(debug_compare turboquant)

# TQM converter tool
add_executable(tq_convert tools/tq_convert.c)
target_link_libraries(tq_convert turboquant)

# Examples (always built)
file(GLOB EXAMPLE_C_SOURCES examples/*.c)
file(GLOB EXAMPLE_CXX_SOURCES examples/*.cpp)
foreach(ex_src ${EXAMPLE_C_SOURCES} ${EXAMPLE_CXX_SOURCES})
    get_filename_component(ex_name ${ex_src} NAME_WE)
    add_executable(${ex_name} ${ex_src})
    target_link_libraries(${ex_name} turboquant)
endforeach()
