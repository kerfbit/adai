cmake_minimum_required(VERSION 3.10)
project(adai VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy

# Options
option(BUILD_TESTING "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)
option(ENABLE_CCACHE "Use ccache for compilation caching" ON)

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