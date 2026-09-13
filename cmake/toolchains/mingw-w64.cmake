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

# TD-159: cpp-httplib's socket layer (used by chatbot, and by every other current/future
# httplib-consuming binary this toolchain builds) calls straight into Winsock2 APIs
# (socket/select/shutdown/closesocket/recv/WSACleanup), confirmed via real undefined-reference
# link errors (__imp_socket etc.) before this was added. ws2_32 is Windows' own system import
# library (ships with every Windows install), not something to vendor.
#
# Deliberately NOT folded into CMAKE_EXE_LINKER_FLAGS above: that variable is placed by the
# Makefiles generator *before* a target's own object files on the link command line (confirmed
# via the actual generated link.txt), so a library listed there is invisible to ld's single-pass,
# left-to-right symbol resolution by the time it reaches the objects that need it — ld had
# already moved past `-lws2_32` before discovering chatbot.exe's own undefined winsock
# references, so the link still failed with this in CMAKE_EXE_LINKER_FLAGS. link_libraries()
# instead appends to each target's own LINK_LIBRARIES list, which the same generator places
# *after* the object files (see linkLibs.rsp in that same link.txt) — the position that actually
# resolves symbols referenced by those objects.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_libraries(ws2_32)
endif()

# Windows-specific compiler flags
#
# TD-159: NOGDI/NOMINMAX/WIN32_LEAN_AND_MEAN, applied globally rather than at each windows.h
# include site (there's no single one to patch — httplib.h pulls it in transitively via
# winsock2.h, and any future includer would reopen the same problem). Confirmed with a real
# repro: without NOGDI specifically, <wingdi.h> (pulled in by <windows.h> unconditionally on this
# MinGW headers version — WIN32_LEAN_AND_MEAN does not gate that particular include) #defines
# ERROR as a plain macro, which silently rewrites this codebase's own
# `enum class Level { ..., ERROR }` (src/Logger.hpp) into `enum class Level { ..., 0 }` wherever
# any file transitively includes windows.h before Logger.hpp — confirmed via a real cross-compile
# of ModelNameService.cpp failing with "expected identifier before numeric constant" pointing at
# that exact enumerator. NOMINMAX (suppresses windows.h's own min/max macros, which would
# otherwise shadow std::min/std::max at every call site across this codebase) is the standard
# companion define for the same class of problem and is included pre-emptively.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32_WINNT=0x0601 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -DNOGDI" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_WIN32_WINNT=0x0601 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -DNOGDI" CACHE STRING "" FORCE)

# Disable features that may cause issues in cross-compilation
set(BUILD_API_SERVER OFF CACHE BOOL "Disable API server for Windows build" FORCE)
set(ENABLE_GPU OFF CACHE BOOL "Disable GPU for Windows build" FORCE)

message(STATUS "Cross-compiling for Windows using MinGW-w64")
message(STATUS "  Toolchain prefix: ${TOOLCHAIN_PREFIX}")
message(STATUS "  C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Find root path: ${CMAKE_FIND_ROOT_PATH}")
