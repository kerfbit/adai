# Building ADAI

This guide covers everything you need to know about building the ADAI project from source.

## Table of Contents

- [System Requirements](#system-requirements)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Build Options](#build-options)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Build Targets](#build-targets)
- [Troubleshooting](#troubleshooting)
- [Advanced Build Configurations](#advanced-build-configurations)

## System Requirements

### Minimum Requirements

- **CPU**: x86_64 architecture (64-bit)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk**: 500 MB for source and build files
- **OS**: Linux, macOS, or Windows

### Supported Compilers

|Compiler|Minimum Version|Recommended|
|----------|----------------|-------------|
|GCC|7.0|11.0+|
|Clang|5.0|13.0+|
|MSVC|2017 (19.10)|2022|
|AppleClang|10.0|Latest|

## Dependencies

### Required Dependencies

1. **CMake** (≥ 3.10, **3.23+ recommended**)
   - Build system generator
   - CMake 3.23+ required for CMake presets support
   - [https://cmake.org/download/](https://cmake.org/download/)

2. **C++17 Compatible Compiler**
   - See supported compilers above

3. **Git**
   - Version control system
   - Required for cloning repository

### Optional Dependencies

1. **ccache**
   - Compilation caching tool
   - Automatically detected and used when available
   - Dramatically speeds up recompilation (70-80% faster)

2. **clang-format** (≥ 10.0)
   - Code formatting tool
   - Ensures consistent code style

3. **clang-tidy** (≥ 10.0)
   - Static analysis tool
   - Catches common bugs and issues

4. **lcov**
   - Coverage report generation
   - For development and testing

5. **CUDA Toolkit** (≥ 11.0, **for GPU acceleration**)
   - NVIDIA GPU acceleration support
   - Optional - enables GPU-accelerated matrix operations
   - Required only if building with `-DENABLE_GPU=ON`
   - Download: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Requires NVIDIA GPU with compute capability 6.0+

**Note**: Google Test is no longer a system dependency. It's automatically downloaded and built using CMake FetchContent.

## Quick Start

### Clone Repository

```bash
git clone https://github.com/yourusername/adai.git
cd adai
```

### Modern Build (Recommended)

Using CMake presets (CMake 3.23+):

```bash
# Configure and build in one step
cmake --preset debug
cmake --build --preset debug

# Run tests
ctest --preset debug
```

### Traditional Build

For older CMake versions or custom configurations:

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build (parallel builds recommended)
make -j$(nproc)  # Linux/macOS
# OR
make -j%NUMBER_OF_PROCESSORS%  # Windows

# Verify build by running tests
ctest --output-on-failure
```

### Quick Test Run

```bash
# From build directory
./tests/matrixTests
./tests/optimizerTests
./tests/integrationTests
```

## Modern Build Optimizations

ADAI uses several modern CMake features to optimize build times and simplify workflows.

### CMake Presets (Recommended)

CMake presets provide pre-configured build configurations for common scenarios. This is the easiest way to build ADAI.

```bash
# List available presets
cmake --list-presets

# Configure using a preset
cmake --preset debug          # Debug build
cmake --preset release        # Release build
cmake --preset asan          # AddressSanitizer build
cmake --preset coverage      # Code coverage build

# Build using a preset
cmake --build --preset debug
cmake --build --preset release

# Run tests using a preset
ctest --preset debug
ctest --preset asan
```

#### Available Presets

|Preset|Description|Use Case|
|--------|-------------|----------|
|**debug**|Debug build with all symbols|Development and debugging|
|**release**|Optimized release build|Production and performance|
|**relwithdebinfo**|Optimized with debug info|Performance debugging|
|**asan**|AddressSanitizer enabled|Memory error detection|
|**ubsan**|UndefinedBehaviorSanitizer|Undefined behavior detection|
|**tsan**|ThreadSanitizer|Race condition detection|
|**coverage**|Code coverage enabled|Test coverage analysis|
|**clang-tidy**|Static analysis enabled|Code quality checks|
|**ci**|CI/CD configuration|Automated builds|

### Compilation Caching with ccache

ADAI automatically uses ccache when available to significantly speed up recompilation:

```bash
# Install ccache (if not already installed)
# Ubuntu/Debian
sudo apt-get install ccache

# Fedora/RHEL
sudo dnf install ccache

# macOS
brew install ccache

# Verify ccache is working
ccache -s  # Show statistics

# Configure will automatically detect and use ccache
cmake --preset debug

# After building, check cache statistics
ccache -s
```

**Performance Impact**: ccache can reduce recompilation time by 70-80% for incremental builds.

**Disable ccache** (if needed):

```bash
cmake -DENABLE_CCACHE=OFF --preset debug
```

### Precompiled Headers

ADAI uses precompiled headers for commonly included STL headers to reduce compilation time:

- **adai_core**: `<vector>`, `<string>`, `<memory>`, `<algorithm>`, `<cmath>`, etc.
- **adai_nlp**: `<fstream>`, `<sstream>`, `<regex>`, `<queue>`, etc.

Precompiled headers are automatically used when building libraries. No configuration needed.

**Benefits**:

- Faster clean builds (20-30% reduction)
- Automatic reuse across translation units
- No manual header management

### FetchContent Dependencies

Google Test is automatically downloaded and built using CMake's FetchContent:

```cmake
# No need to manually install Google Test
# CMake handles it automatically during configuration
```

**Benefits**:

- No system-wide Google Test installation required
- Consistent test framework version across all builds
- Automatic setup in CI/CD environments

**Offline builds**: If you need to build offline, download Google Test manually:

```bash
# Download Google Test v1.14.0 once
git clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git _deps/googletest-src
```

## Build Options

**Note**: For CMake 3.23+, using [CMake Presets](#cmake-presets-recommended) is recommended instead of manually specifying options.

### CMake Configuration Options

Configure the build with various options (traditional method):

```bash
# Disable building examples
cmake -DBUILD_EXAMPLES=OFF ..

# Disable building tests
cmake -DBUILD_TESTING=OFF ..

# Set build type
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Enable Address Sanitizer (debug builds)
cmake -DENABLE_ASAN=ON ..

# Enable UndefinedBehavior Sanitizer
cmake -DENABLE_UBSAN=ON ..

# Enable Thread Sanitizer
cmake -DENABLE_TSAN=ON ..

# Enable code coverage
cmake -DENABLE_COVERAGE=ON ..

# Enable clang-tidy static analysis
cmake -DENABLE_CLANG_TIDY=ON ..

# Disable ccache (enabled by default)
cmake -DENABLE_CCACHE=OFF ..

# Specify compiler
cmake -DCMAKE_CXX_COMPILER=clang++ ..
cmake -DCMAKE_CXX_COMPILER=g++-11 ..

# Combine multiple options
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_EXAMPLES=ON \
      -DCMAKE_CXX_COMPILER=g++ \
      -DENABLE_CCACHE=ON \
      ..
```

### Available CMake Options

|Option|Default|Description|
|--------|---------|-------------|
|`BUILD_TESTING`|ON|Build test suite|
|`BUILD_EXAMPLES`|ON|Build example programs|
|`ENABLE_GPU`|OFF|Enable GPU acceleration with CUDA|
|`ENABLE_ASAN`|OFF|Enable AddressSanitizer|
|`ENABLE_UBSAN`|OFF|Enable UndefinedBehaviorSanitizer|
|`ENABLE_TSAN`|OFF|Enable ThreadSanitizer|
|`ENABLE_COVERAGE`|OFF|Enable code coverage|
|`ENABLE_CLANG_TIDY`|OFF|Run clang-tidy during build|
|`ENABLE_CCACHE`|ON|Use ccache for caching|

### Build Types

|Build Type|Optimization|Debug Info|Use Case|
|------------|-------------|------------|----------|
|**Debug**|-O0|Full|Development, debugging|
|**Release**|-O3|None|Production, performance testing|
|**RelWithDebInfo**|-O2|Full|Performance testing with debugging|
|**MinSizeRel**|-Os|None|Size-constrained environments|

### GPU Acceleration (Optional)

ADAI supports optional GPU acceleration using NVIDIA CUDA for faster matrix operations.

#### Requirements

- NVIDIA GPU with compute capability 6.0 or higher
- CUDA Toolkit 11.0 or later
- NVIDIA drivers properly installed

#### Installing CUDA

Ubuntu/Debian:

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install -y cuda-toolkit-11-8
```

Fedora/RHEL:

```bash
# Install CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA Toolkit
sudo dnf install -y cuda-toolkit-11-8
```

#### Building with GPU Support

```bash
# Configure with GPU support
cmake -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release ..

# Optionally specify CUDA architectures (compute capabilities)
cmake -DENABLE_GPU=ON \
      -DCMAKE_CUDA_ARCHITECTURES="60;70;80" \
      -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)

# Test GPU functionality
./gpu_example
```

#### GPU Architecture Targets

By default, ADAI compiles for multiple GPU architectures. You can customize this:

|Compute Capability|GPU Examples|CMake Flag|
|-------------------|--------------|------------|
|6.0|Pascal (GTX 10xx, P100)|60|
|6.1|Pascal (GTX 10xx Ti, P4)|61|
|7.0|Volta (V100)|70|
|7.5|Turing (RTX 20xx, T4)|75|
|8.0|Ampere (A100)|80|
|8.6|Ampere (RTX 30xx, A10)|86|

Example - Build only for RTX 30xx series:

```bash
cmake -DENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES="86" ..
```

#### Verifying GPU Support

After building with GPU support:

```bash
# Run the GPU example
./gpu_example

# Output will show:
# - GPU device information
# - Performance comparisons (CPU vs GPU)
# - Speedup metrics
```

#### GPU API Usage

When GPU support is enabled, the Matrix class gains additional methods:

```cpp
#include "Matrix.hpp"

// Initialize GPU (do this once at startup)
Matrix::gpu_initialize();

// Create matrices
Matrix A(1000, 1000);
Matrix B(1000, 1000);
A.randomize();
B.randomize();

// Use GPU-accelerated operations
Matrix C = A.multiply_gpu(B);      // Fast matrix multiplication
Matrix D = A.add_gpu(B);           // Fast element-wise addition
Matrix E = A.transpose_gpu();      // Fast transpose
Matrix F = A.scale_gpu(2.5f);      // Fast scalar multiplication
Matrix G = A.hadamard_gpu(B);      // Fast element-wise multiply

// Cleanup (do this before exit)
Matrix::gpu_cleanup();
```

**Note**: GPU operations transfer data to/from GPU memory. For best performance:

- Use GPU operations on large matrices (500x500 or larger)
- Batch multiple operations together when possible
- Keep data on GPU between operations (future enhancement)

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    ccache \
    clang-format \
    clang-tidy \
    lcov

# Build using presets (CMake 3.23+)
cmake --preset release
cmake --build --preset release
ctest --preset release

# OR traditional build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest
```

### Linux (Fedora/RHEL/CentOS)

```bash
# Install dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    git \
    ccache \
    clang-tools-extra \
    lcov

# Build using presets (CMake 3.23+)
cmake --preset release
cmake --build --preset release
ctest --preset release

# OR traditional build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake ccache clang-format cppcheck lcov

# Build using presets (CMake 3.23+)
cmake --preset release
cmake --build --preset release
ctest --preset release

# OR traditional build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
ctest
```

### Windows (Visual Studio)

```powershell
# Install Visual Studio 2019 or later with C++ workload
# Install CMake from https://cmake.org/download/
# Install Git from https://git-scm.com/download/win

# Open Visual Studio Developer Command Prompt
# Navigate to project directory

# Create build directory
mkdir build
cd build

# Generate Visual Studio project
cmake -G "Visual Studio 16 2019" ..
# OR for VS 2022
cmake -G "Visual Studio 17 2022" ..

# Build
cmake --build . --config Release

# Run tests
ctest -C Release --output-on-failure
```

### Windows (MinGW)

```bash
# Install MinGW-w64
# Add MinGW bin directory to PATH

# Build
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make -j4
ctest
```

## Build Targets

### Main Executables

```bash
# From build directory

# Production Applications
make chatbot              # Chatbot CLI application
make chatbot_trainer      # Chatbot training tool

# Examples (if BUILD_EXAMPLES=ON)
make tokenizer_example           # BPE tokenizer demo
make encoder_example             # Encoder demo
make mha_example                 # Multi-head attention demo
make feedforward_example         # Feed-forward network demo
make encoderblock_example        # Encoder block demo
make decoderblock_example        # Decoder block demo
make textgenerator_example       # Text generation demo
make encoder_decoder_example     # Full transformer demo
make optimizer_example           # Optimizer demo
```

### Test Targets

```bash
# Build all tests
make

# Run all tests
ctest

# Run specific test suites
make matrixTests && ./tests/matrixTests
make optimizerTests && ./tests/optimizerTests
make integrationTests && ./tests/integrationTests

# Run tests with verbose output
ctest -V

# Run tests matching pattern
ctest -R "Matrix.*"
```

### Library Targets

The project builds modular libraries:

```bash
make adai_core           # Matrix, Activation, Optimizer
make adai_layers         # LayerNorm, Embeddings, PositionalEncoding
make adai_attention      # MultiHeadAttention, CrossAttention
make adai_feedforward    # FeedForward networks
make adai_transformer    # EncoderBlock, DecoderBlock
make adai_models         # Complete models (Encoder, Decoder, etc.)
make adai_nlp            # Tokenizer, TextGenerator
```

## Troubleshooting

### Common Issues

#### CMake Cannot Find Compiler

```bash
# Specify compiler explicitly
cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 ..

# Or set environment variable
export CXX=/usr/bin/clang++
cmake ..
```

#### Google Test Not Found

Google Test is automatically fetched by CMake. If you encounter issues:

```bash
# Clear CMake cache and reconfigure
rm -rf build/*
cd build
cmake ..
```

#### Out of Memory During Build

```bash
# Reduce parallel jobs
make -j2  # Use only 2 jobs instead of all cores

# Or build sequentially
make
```

#### Tests Fail with Segmentation Fault

```bash
# Build with debug symbols and run under debugger
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./tests/failingTest
(gdb) run
(gdb) backtrace
```

#### Undefined References at Link Time

```bash
# Clean rebuild
rm -rf build/*
cd build
cmake ..
make clean
make
```

### Platform-Specific Issues

#### Linux: Missing C++17 Support

```bash
# Install newer GCC
sudo apt-get install g++-11
cmake -DCMAKE_CXX_COMPILER=g++-11 ..
```

#### macOS: Missing Xcode Tools

```bash
# Install Xcode Command Line Tools
xcode-select --install
```

#### Windows: CMake Cannot Find Visual Studio

```powershell
# List available generators
cmake --help

# Specify generator explicitly
cmake -G "Visual Studio 17 2022" -A x64 ..
```

## Advanced Build Configurations

### Static Analysis

```bash
# Build with clang-tidy
cmake -DENABLE_CLANG_TIDY=ON ..
make
```

### Code Coverage

```bash
# Configure with coverage enabled
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make

# Run tests
ctest

# Generate coverage report
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/gtest/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage_html

# View report
xdg-open coverage_html/index.html  # Linux
open coverage_html/index.html      # macOS
```

### Memory Checking (AddressSanitizer)

```bash
# Build with ASAN
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON ..
make

# Run tests (ASAN will detect memory issues)
ctest

# For detailed output
export ASAN_OPTIONS=verbosity=1
./tests/matrixTests
```

### Cross-Compilation

#### Windows from Linux (MinGW-w64)

Build native Windows executables from Linux using the automated scripts:

```bash
# Quick build
./scripts/build_windows.sh

# Create distribution package with all files
./scripts/package_windows.sh
```

For detailed instructions, see [Windows Cross-Compilation Guide](windows-cross-compilation.md).

Manual cross-compilation:

```bash
# Install MinGW-w64
sudo apt-get install mingw-w64 g++-mingw-w64  # Ubuntu/Debian

# Configure
mkdir build-windows && cd build-windows
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
      -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . -j$(nproc)

# Executables will be in src/*.exe
```

#### ARM Cross-Compilation

```bash
# Example: Cross-compile for ARM
cmake -DCMAKE_TOOLCHAIN_FILE=arm-toolchain.cmake ..
make
```

### Installation

```bash
# Install to system directories
sudo make install

# Install to custom prefix
cmake -DCMAKE_INSTALL_PREFIX=/opt/adai ..
make
sudo make install
```

### Clean Build

```bash
# Clean build artifacts
cd build
make clean

# Complete clean (remove all generated files)
cd ..
rm -rf build
mkdir build && cd build
cmake ..
make
```

## Build Performance Tips

### Parallel Builds

```bash
# Use all CPU cores
make -j$(nproc)          # Linux
make -j$(sysctl -n hw.ncpu)  # macOS

# Limit cores (reduce memory usage)
make -j4
```

### Incremental Builds

```bash
# Only rebuild changed files
make

# Force rebuild of specific target
make clean
make targetName
```

### Using Ninja (Faster than Make)

```bash
# Install ninja
sudo apt-get install ninja-build  # Linux
brew install ninja                # macOS

# Configure with Ninja
cmake -G Ninja ..

# Build
ninja

# Run specific target
ninja matrixTests
```

### CCache (Speed Up Recompilation)

```bash
# Install ccache
sudo apt-get install ccache

# Configure CMake to use ccache
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make
```

## Continuous Integration

The project uses GitHub Actions for CI. See `.github/workflows/ci.yml` for configuration.

Local CI simulation:

```bash
# Run the same checks as CI
scripts/format_code.sh
scripts/analyze_code.sh
scripts/run_tests.sh
```

## Getting Help

If you encounter build issues:

1. Check this troubleshooting section
2. Search existing GitHub issues
3. Check `PROCESS_IMPROVEMENT_PLAN.md` Section 3 for CMake details
4. Create a new issue with:
   - Your OS and version
   - Compiler and version
   - CMake version
   - Full error output
   - Steps to reproduce

---

**Happy Building!** 🔨
