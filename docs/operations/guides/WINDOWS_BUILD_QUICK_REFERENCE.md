# Windows Cross-Compilation Quick Reference

## One-Command Build

```bash
./scripts/build_windows.sh
```

## One-Command Package

```bash
./scripts/package_windows.sh
```

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install mingw-w64 g++-mingw-w64

# Fedora/RHEL
sudo dnf install mingw64-gcc-c++

# Arch Linux
sudo pacman -S mingw-w64-gcc
```

## Output Locations

- **Build**: `build-windows/src/*.exe`
- **Package**: `dist-windows/adai-chatbot-windows-x64/`
- **Archive**: `dist-windows/adai-chatbot-windows-x64.zip`

## Executables Built

- `chatbot.exe` - Interactive chatbot CLI (3.1 MB)
- `chatbot_trainer.exe` - Model training tool (3.1 MB)

## Distribution Package Contents

```text
adai-chatbot-windows-x64/
├── chatbot.exe              # Main application
├── chatbot_trainer.exe      # Training tool
├── vocab.txt                # Vocabulary
├── sample_training_data.txt # Sample data
├── chatbot_model.bin*       # Models (if available)
├── README.txt               # User guide
└── run_chatbot.bat          # Launcher
```

## Manual Build

```bash
# Configure
mkdir build-windows && cd build-windows
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/mingw-w64.cmake \
      -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . -j$(nproc)
```

## Verification

```bash
# Check file format
file build-windows/src/chatbot.exe
# Should show: PE32+ executable (console) x86-64

# Check dependencies
x86_64-w64-mingw32-objdump -p build-windows/src/chatbot.exe | grep "DLL Name:"
# Should show only: KERNEL32.dll, msvcrt.dll
```

## Test with Wine

```bash
wine64 build-windows/src/chatbot.exe --help
```

## Deployment

1. Copy `adai-chatbot-windows-x64.zip` to Windows
2. Extract archive
3. Run `run_chatbot.bat` or `chatbot.exe`

## Full Documentation

- [Windows Cross-Compilation Guide](docs/guides/windows-cross-compilation.md)
- [Implementation Summary](WINDOWS_CROSS_COMPILATION_SUMMARY.md)

## Key Features

✅ Static linking (no DLL dependencies)
✅ Windows 7+ compatibility
✅ Self-contained executables
✅ 1.9 MB compressed package
✅ Single-command workflow

## Troubleshooting

MinGW not found?

```bash
sudo apt-get install mingw-w64 g++-mingw-w64
which x86_64-w64-mingw32-g++
```

Build errors?

```bash
./scripts/build_windows.sh clean  # Clean build
cmake --version  # Check CMake >= 3.10
```

Won't run on Windows?

- Requires Windows 7+ 64-bit
- Extract all files from ZIP
- Run from Command Prompt to see errors
