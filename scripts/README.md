# Development Scripts

This directory contains utility scripts for ADAI development.

## Available Scripts

### 🚀 install_incremental_trainer.sh

Installs the `incremental_trainer` sub-system (trainer, dataset manager, and optionally the registry server) to a local or remote host.

**Usage:**

```bash
# Local install (requires sudo)
sudo ./scripts/install_incremental_trainer.sh

# Include the distributed registry server binary
sudo ./scripts/install_incremental_trainer.sh --with-registry-server

# Custom build directory and install path
sudo ./scripts/install_incremental_trainer.sh --build-dir build-release --install-path /usr/local/adai

# Coordinator-only node (registry_server + systemd unit; no trainer required)
sudo ./scripts/install_incremental_trainer.sh --coordinator

# Remote install via SSH + rsync
sudo ./scripts/install_incremental_trainer.sh --remote user@192.168.1.7

# Remote install with custom SSH key and session checkpoint sync
sudo ./scripts/install_incremental_trainer.sh \
  --remote user@192.168.1.7 --ssh-key ~/.ssh/id_adai --sync-sessions
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--install-path PATH` | `/opt/adai` | Installation root directory |
| `--user USER` | `adai` | Service user to own installed files |
| `--group GROUP` | `adai` | Service group |
| `--build-dir DIR` | `build-gpu-clang` | CMake build directory containing `bin/` |
| `--config-src PATH` | `<repo-root>/config.conf` | Source `config.conf` to install |
| `--vocab-src PATH` | `<repo-root>/vocab.txt` | Source `vocab.txt` to install |
| `--with-registry-server` | off | Also install the `registry_server` binary |
| `--coordinator` | off | Coordinator-only mode (implies `--with-registry-server`) |
| `--remote HOST` | — | Install to a remote host via SSH + rsync |
| `--sync-sessions` | off | (with `--remote`) Rsync `training_sessions/` to remote |
| `--ssh-key PATH` | — | SSH identity file forwarded to all ssh/rsync calls |

**What it does (local install):**

1. Creates a `adai` system user (idempotent)
2. Creates the full directory layout under `<install-path>`, including `training_data/gutenberg_data/` and `training_data/huggingface_data/`
3. Copies binaries from `<build-dir>/bin/` with mode 755
4. Copies `config.conf` (mode 640) and `vocab.txt` (mode 644) to `<install-path>/config/`
5. Appends commented-out distributed-registry stubs (`REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID`, `REGISTRY_TIMEOUT_MS`) to the installed config
6. Sets ownership to `<user>:<group>` recursively
7. Verifies `incremental_trainer` and `dataset_manager` execute correctly

**Coordinator mode** (`--coordinator`) installs only `registry_server` and writes an `adai-registry.service` systemd unit. Worker nodes then set `REGISTRY_SERVER_URL` in their `config.conf` to point at the coordinator.

**Requirements:**

- Root privileges for local/coordinator install
- Built binaries (see `--build-dir`); build with `-DBUILD_METRICS_API_SERVER=ON` for `registry_server`
- `rsync` and `ssh` for remote installs

---

### 🛠️ install_systemd_service.sh

Installs the ADAI chatbot API server as a systemd service.

**Usage:**

```bash
sudo ./scripts/install_systemd_service.sh [--install-path PATH] [--user USER] [--port PORT]
```

---

### 📈 install_metrics_service.sh

Installs the ADAI metrics API server as a systemd service.

**Usage:**

```bash
sudo ./scripts/install_metrics_service.sh [--install-path PATH] [--port PORT]
```

---

### 📊 check_tech_debt.sh

Scans codebase for technical debt markers and verifies tracking.

**Usage:**

```bash
./scripts/check_tech_debt.sh
```

**What it does:**

- Scans for TODO, FIXME, HACK, and XXX markers in code
- Verifies all markers are tracked in TECHNICAL_DEBT.md
- Reports summary of tracked technical debt items
- Highlights high-priority items

**Output:**

- Lists any untracked technical debt markers
- Shows count of active tracked items
- Displays high-priority debt items
- Exit code 0 if all debt is tracked, 1 if untracked markers found

**Best Practices:**

- Run before committing code
- All new debt markers must be tracked in TECHNICAL_DEBT.md
- Reference debt items in code: `// See TD-XXX in TECHNICAL_DEBT.md`
- Create GitHub issues for new debt using `.github/ISSUE_TEMPLATE/technical-debt.md`

---

### 🎨 format_code.sh

Formats all C++ source files using clang-format.

**Usage:**

```bash
./scripts/format_code.sh
```

**Requirements:** clang-format

```bash
sudo apt-get install clang-format
```

---

### 🔬 analyze_code.sh

Runs static analysis on C++ source files using clang-tidy.

**Usage:**

```bash
# Analyze all source files
./scripts/analyze_code.sh

# Analyze specific files
./scripts/analyze_code.sh src/Matrix.cpp src/Optimizer.cpp
```

**Requirements:** clang-tidy

```bash
sudo apt-get install clang-tidy
```

---

### 🧪 run_tests.sh

Runs the test suite with optional sanitizers and coverage.

**Usage:**

```bash
# Run tests normally
./scripts/run_tests.sh

# Run with AddressSanitizer
./scripts/run_tests.sh --asan

# Run with UndefinedBehaviorSanitizer
./scripts/run_tests.sh --ubsan

# Run with ThreadSanitizer
./scripts/run_tests.sh --tsan

# Run with coverage analysis
./scripts/run_tests.sh --coverage

# Verbose output
./scripts/run_tests.sh --verbose

# Combine options
./scripts/run_tests.sh --asan --verbose
```

**Requirements:**

- AddressSanitizer/UndefinedBehaviorSanitizer: GCC/Clang with sanitizer support
- Coverage: lcov

```bash
sudo apt-get install lcov
```

---

## Quick Start

```bash
# Format code before committing
./scripts/format_code.sh

# Check for code quality issues
./scripts/analyze_code.sh

# Run all tests
./scripts/run_tests.sh

# Run tests with memory leak detection
./scripts/run_tests.sh --asan
```

---

## Pre-commit Hook (Optional)

To automatically format code before commits:

```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
./scripts/format_code.sh
git add -u
EOF

chmod +x .git/hooks/pre-commit
```
