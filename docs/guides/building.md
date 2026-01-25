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

| Compiler | Minimum Version | Recommended |
|----------|----------------|-------------|
| GCC      | 7.0            | 11.0+       |
| Clang    | 5.0            | 13.0+       |
| MSVC     | 2017 (19.10)   | 2022        |
| AppleClang | 10.0         | Latest      |

## Dependencies

### Required Dependencies

1. **CMake** (≥ 3.10)
   - Build system generator
   - https://cmake.org/download/

2. **C++17 Compatible Compiler**
   - See supported compilers above

3. **Git**
   - Version control system
   - Required for cloning repository

### Optional Dependencies

1. **Google Test** (≥ 1.10.0)
   - Automatically fetched by CMake
   - For running tests

2. **clang-format** (≥ 10.0)
   - Code formatting tool
   - Ensures consistent code style

3. **clang-tidy** (≥ 10.0)
   - Static analysis tool
   - Catches common bugs and issues

4. **lcov**
   - Coverage report generation
   - For development and testing

## Quick Start

### Clone Repository

```bash
git clone https://github.com/yourusername/adai.git
cd adai
```

### Basic Build

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

## Build Options

### CMake Configuration Options

Configure the build with various options:

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

# Enable code coverage
cmake -DENABLE_COVERAGE=ON ..

# Specify compiler
cmake -DCMAKE_CXX_COMPILER=clang++ ..
cmake -DCMAKE_CXX_COMPILER=g++-11 ..

# Combine multiple options
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_EXAMPLES=ON \
      -DCMAKE_CXX_COMPILER=g++ \
      ..
```

### Build Types

| Build Type | Optimization | Debug Info | Use Case |
|------------|-------------|------------|----------|
| **Debug** | -O0 | Full | Development, debugging |
| **Release** | -O3 | None | Production, performance testing |
| **RelWithDebInfo** | -O2 | Full | Performance testing with debugging |
| **MinSizeRel** | -Os | None | Size-constrained environments |

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    clang-format \
    clang-tidy \
    lcov

# Build
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
    clang-tools-extra \
    lcov

# Build
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
brew install cmake clang-format cppcheck lcov

# Build
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
