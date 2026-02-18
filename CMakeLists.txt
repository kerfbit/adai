cmake_minimum_required(VERSION 3.10)
project(adai VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy

# ============================================================================
# Optimization Flags for Maximum Performance
# ============================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release build optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
            -O3                    # Maximum optimization
            -march=native          # Optimize for current CPU architecture
            -mtune=native          # Tune for current CPU
            -ffast-math            # Fast floating point math
            -funroll-loops         # Unroll loops for performance
            -ftree-vectorize       # Auto-vectorization
        )
        message(STATUS "Release mode: Enabled aggressive optimizations (-O3 -march=native)")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/O2 /arch:AVX2)
        message(STATUS "Release mode: Enabled MSVC optimizations (/O2 /arch:AVX2)")
    endif()
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE} (use -DCMAKE_BUILD_TYPE=Release for full optimizations)")
endif()

# Windows-specific settings
if(WIN32 OR MINGW)
    message(STATUS "Configuring for Windows build")
    # Ensure we have proper Windows API version (Windows 7+)
    add_compile_definitions(_WIN32_WINNT=0x0601)
    
    # For MinGW cross-compilation: static linking to avoid DLL dependencies
    if(MINGW)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -static")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")
        message(STATUS "  Enabling static linking for MinGW")
    endif()
endif()

# Options
option(BUILD_TESTING "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_GPU "Enable GPU acceleration with CUDA" OFF)
# TODO: See TD-007 in TECHNICAL_DEBT.md - Add ENABLE_BLAS option for optimized matrix operations
# TODO: Add FindBLAS.cmake module to detect OpenBLAS, Intel MKL, or Apple Accelerate
# TODO: Add conditional linking: target_link_libraries(adai PRIVATE ${BLAS_LIBRARIES})
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)
option(ENABLE_CCACHE "Use ccache for compilation caching" ON)

# GPU/CUDA support
if(ENABLE_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    
    # Set CUDA architectures (compute capability)
    # Can be overridden with -DCMAKE_CUDA_ARCHITECTURES
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
    endif()
    
    message(STATUS "GPU acceleration enabled")
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    add_compile_definitions(ADAI_ENABLE_GPU)
else()
    message(STATUS "GPU acceleration disabled (enable with -DENABLE_GPU=ON)")
endif()

# Enable ccache if available and requested
if(ENABLE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
    else()
        message(STATUS "ccache requested but not found")
    endif()
endif()

# Compiler options
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer enabled")
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
endif()

if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
    message(STATUS "ThreadSanitizer enabled")
endif()

if(ENABLE_COVERAGE)
    add_compile_options(--coverage)
    add_link_options(--coverage)
    message(STATUS "Code coverage enabled")
endif()

# Clang-tidy integration
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endif()

# Include directories
include_directories(src)

# Subdirectories
add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    
    # Use FetchContent to automatically download and build Google Test
    include(FetchContent)
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    # Make Google Test available
    FetchContent_MakeAvailable(googletest)
    
    # Add tests subdirectory
    add_subdirectory(tests)
    
    message(STATUS "Google Test v1.14.0 fetched and configured")
endif()