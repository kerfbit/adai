# Windows Cross-Compilation Guide

This guide explains how to build Windows native executables for the ADAI chatbot from a Linux development environment using MinGW-w64 cross-compilation.

## Overview

Cross-compilation allows you to build Windows `.exe` files on a Linux machine without needing a Windows environment. The executables are statically linked and portable - they can run on any Windows 7+ system without additional dependencies.

## Prerequisites

### Required Tools

1. **MinGW-w64 Toolchain** - Cross-compiler for Windows targets
2. **CMake** (version 3.10 or later)
3. **Make** or **Ninja** build system

### Installing MinGW-w64

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install mingw-w64 g++-mingw-w64
```

#### Fedora/RHEL
```bash
sudo dnf install mingw64-gcc-c++
```

#### Arch Linux
```bash
sudo pacman -S mingw-w64-gcc
```

#### Verify Installation
```bash
x86_64-w64-mingw32-g++ --version
```

You should see output indicating the MinGW-w64 compiler version.

## Quick Start

### One-Command Build

```bash
./scripts/build_windows.sh
```

This script will:
1. Check for required tools
2. Configure CMake with the MinGW-w64 toolchain
3. Build Windows executables
4. Report build status

### Create Distribution Package

```bash
./scripts/package_windows.sh
```

This creates a complete Windows distribution in `dist-windows/` with:
- All `.exe` files
- Required data files (vocab.txt, models, etc.)
- README.txt for Windows users
- Batch launcher script
- ZIP archive ready for distribution

## Manual Build Process

### Step 1: Configure Build

```bash
mkdir build-windows
cd build-windows

cmake \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  ..
```

### Step 2: Build

```bash
cmake --build . --config Release -j$(nproc)
```

### Step 3: Verify Executables

```bash
find . -name "*.exe"
```

You should see:
- `src/chatbot.exe` - Interactive chatbot CLI
- `src/chatbot_trainer.exe` - Model training tool

## Toolchain Configuration

The MinGW-w64 toolchain file (`cmake/toolchains/mingw-w64.cmake`) configures:

### Compiler Settings
- **Target**: Windows x86_64 (64-bit)
- **Compiler**: `x86_64-w64-mingw32-g++`
- **Standard**: C++17
- **Linking**: Static (no external DLL dependencies)

### Static Linking
```cmake
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")
```

This ensures the executables are self-contained and don't require:
- `libgcc_s_seh-1.dll`
- `libstdc++-6.dll`
- `libwinpthread-1.dll`

### Windows API Version
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32_WINNT=0x0601")
```

Targets Windows 7+ API level (0x0601).

## Build Configuration

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTING` | OFF | Build test suite (disabled for Windows) |
| `BUILD_EXAMPLES` | OFF | Build example programs |
| `BUILD_API_SERVER` | OFF | Build REST API server (requires cpp-httplib) |
| `ENABLE_GPU` | OFF | Enable CUDA support (not available for cross-compilation) |

### Custom Configuration

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  ..
```

## Testing on Windows

### Method 1: Copy to Windows Machine

1. Build and package:
   ```bash
   ./scripts/build_windows.sh
   ./scripts/package_windows.sh
   ```

2. Transfer `dist-windows/adai-chatbot-windows-x64.zip` to Windows

3. Extract and run:
   ```cmd
   cd adai-chatbot-windows-x64
   run_chatbot.bat
   ```

### Method 2: Wine (Linux)

Test Windows executables on Linux using Wine:

```bash
# Install Wine
sudo apt-get install wine64

# Run Windows executable on Linux
wine build-windows/src/chatbot.exe --help
```

**Note**: Wine testing is not a substitute for real Windows testing.

## Distribution

### Package Contents

The `package_windows.sh` script creates:

```
dist-windows/adai-chatbot-windows-x64/
├── chatbot.exe              # Main chatbot application
├── chatbot_trainer.exe      # Training tool
├── vocab.txt                # Vocabulary file
├── sample_training_data.txt # Sample data
├── chatbot_model.bin*       # Model files (if available)
├── README.txt               # Windows user guide
└── run_chatbot.bat          # Launcher script
```

### Distribution Methods

1. **ZIP Archive**: `adai-chatbot-windows-x64.zip` (created by package script)
2. **Direct Copy**: Copy entire folder to USB drive or network share
3. **Installer**: Use NSIS or Inno Setup to create Windows installer

## Architecture Details

### Cross-Compilation Workflow

```
Linux Development Environment
         ↓
    CMake + MinGW-w64
         ↓
   Windows PE Executables
         ↓
    Transfer to Windows
         ↓
    Run on Windows
```

### Static vs Dynamic Linking

**Static Linking (Used)**:
- ✅ No external DLL dependencies
- ✅ Single executable file
- ✅ Portable across Windows systems
- ❌ Larger executable size

**Dynamic Linking (Not Used)**:
- ✅ Smaller executable size
- ❌ Requires MinGW runtime DLLs
- ❌ Compatibility issues

### Windows API Compatibility

The build targets **Windows 7** as minimum version (`_WIN32_WINNT=0x0601`):

| Windows Version | Supported |
|-----------------|-----------|
| Windows 11      | ✅ Yes    |
| Windows 10      | ✅ Yes    |
| Windows 8/8.1   | ✅ Yes    |
| Windows 7       | ✅ Yes    |
| Windows Vista   | ⚠️ Untested |
| Windows XP      | ❌ No     |

## Troubleshooting

### MinGW Compiler Not Found

**Error**:
```
ERROR: MinGW-w64 toolchain not found!
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install mingw-w64 g++-mingw-w64

# Verify
which x86_64-w64-mingw32-g++
```

### CMake Configuration Failed

**Error**:
```
CMake Error: Could not create named generator MinGW Makefiles
```

**Solution**:
1. Ensure CMake version 3.10+
2. Verify toolchain file path
3. Check MinGW installation

### Executable Won't Run on Windows

**Symptoms**:
- "Not a valid Win32 application"
- Missing DLL errors
- Immediate crash

**Solutions**:

1. **Check Architecture**:
   ```bash
   file build-windows/src/chatbot.exe
   # Should show: PE32+ executable (console) x86-64
   ```

2. **Verify Static Linking**:
   ```bash
   x86_64-w64-mingw32-objdump -p build-windows/src/chatbot.exe | grep DLL
   # Should only show Windows system DLLs
   ```

3. **Test with Wine**:
   ```bash
   wine64 build-windows/src/chatbot.exe --help
   ```

### Build Errors

**Error**: Undefined reference to `WinMain`

**Solution**: Check that main() function is properly defined in .cpp files

**Error**: Cannot find `-lpthread`

**Solution**: Windows doesn't use pthread - the toolchain file should handle this

## Advanced Topics

### 32-bit Windows Build

To build for 32-bit Windows, modify `cmake/toolchains/mingw-w64.cmake`:

```cmake
set(TOOLCHAIN_PREFIX i686-w64-mingw32)
```

Then rebuild:
```bash
./scripts/build_windows.sh clean
./scripts/build_windows.sh
```

### Debug Build

```bash
mkdir build-windows-debug
cd build-windows-debug

cmake \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  ..

cmake --build . -j$(nproc)
```

### Optimized Release Build

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=x86-64" \
  ..
```

### Cross-Compiling with API Server

**Note**: cpp-httplib support for Windows cross-compilation requires additional setup:

1. Install Windows headers for cpp-httplib
2. Update toolchain to find Windows libraries
3. Enable in build:
   ```bash
   cmake -DBUILD_API_SERVER=ON ...
   ```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Windows Build

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install MinGW
      run: |
        sudo apt-get update
        sudo apt-get install -y mingw-w64 g++-mingw-w64
    
    - name: Build Windows Executables
      run: |
        ./scripts/build_windows.sh
    
    - name: Package Windows Distribution
      run: |
        ./scripts/package_windows.sh
    
    - name: Upload Artifacts
      uses: actions/upload-artifact@v3
      with:
        name: windows-build
        path: dist-windows/adai-chatbot-windows-x64.zip
```

### GitLab CI Example

```yaml
build-windows:
  stage: build
  image: ubuntu:22.04
  
  before_script:
    - apt-get update
    - apt-get install -y mingw-w64 g++-mingw-w64 cmake make zip
  
  script:
    - ./scripts/build_windows.sh
    - ./scripts/package_windows.sh
  
  artifacts:
    paths:
      - dist-windows/adai-chatbot-windows-x64.zip
    expire_in: 1 week
```

## Performance Considerations

### Binary Size

Typical sizes with static linking:
- `chatbot.exe`: 2-4 MB
- `chatbot_trainer.exe`: 2-4 MB

### Optimization Flags

The default Release build uses `-O3` optimization. Additional flags:

```cmake
-march=x86-64        # Generic x86-64 (maximum compatibility)
-march=native        # Optimize for build machine (may not run elsewhere)
-flto                # Link-time optimization (slower build, smaller binary)
```

### Runtime Performance

Cross-compiled executables have the same performance as native Windows builds:
- No emulation layer
- Native Windows PE format
- Direct system calls

## Security Considerations

### Code Signing

Windows executables should be code-signed for distribution:

1. Obtain code signing certificate
2. Use `osslsigncode` on Linux:
   ```bash
   osslsigncode sign \
     -certs cert.pem \
     -key key.pem \
     -in chatbot.exe \
     -out chatbot-signed.exe
   ```

### Static Analysis

Run security checks before distribution:

```bash
# Check for hardcoded credentials
grep -r "password\|secret\|key" src/

# Verify no debug symbols in release
x86_64-w64-mingw32-objdump -h chatbot.exe | grep debug
```

## License Compliance

The ADAI chatbot uses:
- **MinGW-w64**: Permissive licensing (runtime exception)
- **Standard Library**: Static linking allowed
- **No external dependencies**: Self-contained executable

Distributing statically-linked executables does not impose additional licensing requirements.

## References

- [MinGW-w64 Project](http://mingw-w64.org/)
- [CMake Cross-Compiling](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)
- [Windows API Documentation](https://docs.microsoft.com/en-us/windows/win32/)

## Support

For issues with Windows cross-compilation:

1. Check build logs in `build-windows/`
2. Verify MinGW installation
3. Test with Wine on Linux
4. Create GitHub issue with:
   - Build output
   - CMake version
   - MinGW version
   - Host OS details
