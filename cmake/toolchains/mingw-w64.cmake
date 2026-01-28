# CMake Toolchain File for Cross-Compiling to Windows using MinGW-w64
# 
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/mingw-w64.cmake \
#         -DCMAKE_BUILD_TYPE=Release \
#         -B build-windows
#   cmake --build build-windows

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Specify the cross compiler
# MinGW-w64 variants:
# - x86_64-w64-mingw32 for 64-bit Windows
# - i686-w64-mingw32 for 32-bit Windows
set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)

# Find the compiler
find_program(CMAKE_C_COMPILER NAMES ${TOOLCHAIN_PREFIX}-gcc)
find_program(CMAKE_CXX_COMPILER NAMES ${TOOLCHAIN_PREFIX}-g++)
find_program(CMAKE_RC_COMPILER NAMES ${TOOLCHAIN_PREFIX}-windres)
find_program(CMAKE_AR NAMES ${TOOLCHAIN_PREFIX}-ar)
find_program(CMAKE_RANLIB NAMES ${TOOLCHAIN_PREFIX}-ranlib)

# Set the system root for finding libraries
set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_PREFIX})

# Adjust the default behavior of the FIND_XXX() commands:
# - Search programs in the host environment
# - Search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Windows-specific settings
set(CMAKE_EXECUTABLE_SUFFIX .exe)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX .dll)
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_SUFFIX .a)

# Ensure static linking for portability
# This avoids dependency on MinGW runtime DLLs
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-static-libgcc -static-libstdc++" CACHE STRING "" FORCE)

# Windows-specific compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32_WINNT=0x0601" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_WIN32_WINNT=0x0601" CACHE STRING "" FORCE)

# Disable features that may cause issues in cross-compilation
set(BUILD_API_SERVER OFF CACHE BOOL "Disable API server for Windows build" FORCE)
set(ENABLE_GPU OFF CACHE BOOL "Disable GPU for Windows build" FORCE)

message(STATUS "Cross-compiling for Windows using MinGW-w64")
message(STATUS "  Toolchain prefix: ${TOOLCHAIN_PREFIX}")
message(STATUS "  C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Find root path: ${CMAKE_FIND_ROOT_PATH}")
