# Development Scripts

This directory contains utility scripts for building, deploying, testing, and maintaining the ADAI project.

## Installation and Deployment

### install_server_bundle.sh

Installs `metrics_api_server`, `registry_server`, and `mns_server` as a co-located set of systemd services on a single machine. Supports SQLite (default) and PostgreSQL persistence backends.

```bash
# SQLite (default)
sudo ./scripts/install_server_bundle.sh --build-dir build/portable --yes

# PostgreSQL (install packages, create DB, apply schema)
sudo ./scripts/install_server_bundle.sh --build-dir build/portable --setup-postgres --yes
```

Key options: `--storage-backend`, `--db-path`, `--db-url`, `--setup-postgres`, `--pg-db-name`, `--pg-db-user`. Run `--help` for the full list.

See [Server Bundle Deployment](../docs/operations/deployment/SERVER_BUNDLE_DEPLOYMENT.md) for the full guide.

### install_incremental_trainer.sh

Installs the `incremental_trainer` sub-system (trainer, dataset manager, and optionally the registry server) to a local or remote host. Supports local, remote (SSH+rsync), and coordinator-only modes.

```bash
# Local install (requires sudo)
sudo ./scripts/install_incremental_trainer.sh

# Include the distributed registry server binary
sudo ./scripts/install_incremental_trainer.sh --with-registry-server

# Coordinator-only node (registry_server + systemd unit; no trainer)
sudo ./scripts/install_incremental_trainer.sh --coordinator

# Remote install via SSH + rsync
sudo ./scripts/install_incremental_trainer.sh --remote user@192.168.1.7
```

| Flag | Default | Description |
|------|---------|-------------|
| `--install-path PATH` | `/opt/adai` | Installation root directory |
| `--user USER` | `adai` | Service user |
| `--build-dir DIR` | `build-gpu-clang` | CMake build directory containing `bin/` |
| `--with-registry-server` | off | Also install the `registry_server` binary |
| `--coordinator` | off | Coordinator-only mode (implies `--with-registry-server`) |
| `--remote HOST` | -- | Install to a remote host via SSH + rsync |
| `--sync-sessions` | off | Rsync `training_sessions/` to remote |
| `--ssh-key PATH` | -- | SSH identity file for ssh/rsync calls |

### install_chatbot_API.sh

Installs the ADAI chatbot API server as a systemd service. Creates necessary directories, users, and configuration files.

```bash
sudo ./scripts/install_chatbot_API.sh [--install-path PATH] [--user USER] [--port PORT]
```

### install_metrics_service.sh

Installs the ADAI metrics API server as a standalone systemd service. For co-located deployment of all three servers, use `install_server_bundle.sh` instead.

```bash
sudo ./scripts/install_metrics_service.sh [--install-path PATH] [--port PORT]
```

### install_mns_server.sh

Installs the Model Name Service (`mns_server`) as a systemd service.

```bash
sudo ./scripts/install_mns_server.sh [--install-path PATH] [--port PORT]
```

### adai.service

Systemd service unit file for the ADAI chatbot API server. Includes security hardening (filesystem protection, capability restrictions, syscall filtering) and resource limits (4G memory, 50% CPU). Copy to `/etc/systemd/system/adai.service` and enable with `systemctl`.

### cloudflared/install_cloudflared.sh

Installs a Cloudflare Tunnel connector as a systemd service, exposing metrics/MNS/registry/chatbot/trainer-admin to the Android tablet apps under `kerfbit.dev` subdomains when off the home LAN. Runs as a dedicated `cloudflared` system user (not `adai`). Assumes the `cloudflared` binary and the tunnel itself already exist — see [Cloudflare Tunnel Relay](../docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md) for the full setup, including the ingress config templates (`config-storage.yml.template`, `config-chat.yml.template`) in the same directory. `config-chat.yml.template` fronts both `chat.kerfbit.dev` and `trainer.kerfbit.dev` on the one `adai-chat-tunnel` connector (same machine, two Cloudflare Access applications with separate service tokens — see the doc).

```bash
sudo ./scripts/cloudflared/install_cloudflared.sh \
  --tunnel-name adai-storage-tunnel \
  --config-src ./config-storage.yml \
  --credentials-src ~/.cloudflared/<uuid>.json --yes
```

### setup_postgres.sql

PostgreSQL schema for the metrics API server and model name service. Creates all tables (sessions, metrics_history, generation_quality, abnormal_samples, models, training_history, roles), indexes, and schema_version tracking. Idempotent -- safe to re-run.

```bash
sudo -u postgres createdb -O adai adai
sudo -u postgres psql -d adai -f scripts/setup_postgres.sql
```

This file is run automatically by `install_server_bundle.sh --setup-postgres`.

## Packaging

### package_server_bundle.sh

Packages all server binaries, install scripts, config templates, SQL schema, and dashboard into a self-contained tarball for distribution to target hosts.

```bash
# Auto-detect build dir, version from git tag
./scripts/package_server_bundle.sh

# Include trainer binaries
./scripts/package_server_bundle.sh --include-trainer

# Custom output directory and version
./scripts/package_server_bundle.sh --output-dir dist/ --version v1.2.0
```

### package_windows.sh

Packages Windows cross-compiled executables with all MinGW-w64 runtime DLLs into a distributable ZIP. Requires a prior `build_windows.sh` run.

```bash
./scripts/package_windows.sh
```

## Build

### build_windows.sh

Cross-compiles the ADAI project for Windows from Linux using the MinGW-w64 toolchain.

```bash
# Standard build
./scripts/build_windows.sh

# Clean build
./scripts/build_windows.sh clean
```

Requires: `mingw-w64` (`sudo apt-get install mingw-w64 g++-mingw-w64`).

## Running

### model_service.sh

Full-featured model service manager. Loads and manages the `chatbot_api_server` as a foreground or background daemon. Handles build, start, stop, restart, status, health checks, and log tailing without requiring systemd or root privileges.

```bash
./scripts/model_service.sh start
./scripts/model_service.sh start --model models/model.bin --port 9000 --foreground
./scripts/model_service.sh stop
./scripts/model_service.sh restart --log-level DEBUG
./scripts/model_service.sh health
./scripts/model_service.sh logs
```

### run_chatbot.sh

Launches the chatbot client, automatically starting the API server in the background if it is not already running. Waits for the server to become healthy before connecting the CLI client.

```bash
./scripts/run_chatbot.sh
```

### run_chatbot_gui.sh

Quick launcher for the Qt-based chatbot GUI. Checks for graphical display, binary, vocab file, and model availability. Fixes snap/system library conflicts by resetting library paths.

```bash
./scripts/run_chatbot_gui.sh
```

### chatbot_gui_fixed.sh

Minimal wrapper to run `chatbot_gui` with correct system library paths, working around snap/system library conflicts.

```bash
./scripts/chatbot_gui_fixed.sh
```

### serve_dashboard.py

Simple HTTP server that serves the training metrics dashboard with CORS support on port 8082. Designed to work alongside the metrics API server on port 8081.

```bash
python3 scripts/serve_dashboard.py
```

### check_ports.sh

Checks whether ports 8080, 8081, and 8082 have active listeners, identifying the owning process. Uses `ss`, `netstat`, or `lsof` depending on availability.

```bash
./scripts/check_ports.sh
```

## Code Quality and Formatting

### format_code.sh

Formats all C++ source files in `src/` and `tests/` using clang-format. Prefers `clang-format-18` for CI consistency.

```bash
./scripts/format_code.sh
```

### analyze_code.sh

Runs clang-tidy static analysis on C++ source files. Generates `compile_commands.json` if missing.

```bash
# Analyze all source files
./scripts/analyze_code.sh

# Analyze specific files
./scripts/analyze_code.sh src/Matrix.cpp src/Optimizer.cpp
```

### apply_narrowing_fixes.py

Reads a clang-tidy warnings file and automatically applies `static_cast` fixes for `cppcoreguidelines-narrowing-conversions` and related `bugprone` warnings.

```bash
python3 scripts/apply_narrowing_fixes.py <warnings_file>
```

### fix_markdown_lint.py

Finds and fixes common markdownlint violations across all `.md` files in the repository. Fixes MD009 (trailing whitespace), MD022 (blank lines around headings), MD029 (ordered list numbering), MD031 (blank lines around code blocks), MD032 (blank lines around lists), MD036 (emphasis as heading), MD040 (code block language), and MD060 (table formatting).

```bash
# Check all .md files in the repo (exit 1 if issues found)
python3 scripts/fix_markdown_lint.py --check

# Fix all .md files in the repo
python3 scripts/fix_markdown_lint.py

# Fix files under a specific directory
python3 scripts/fix_markdown_lint.py --dir docs/

# Fix specific files
python3 scripts/fix_markdown_lint.py docs/README.md docs/guides/*.md

# Also run markdownlint --fix if installed
python3 scripts/fix_markdown_lint.py --markdownlint
```

### check_tech_debt.sh

Scans source code for TODO, FIXME, HACK, and XXX markers and verifies they are tracked in `TECHNICAL_DEBT.md`. Reports high-priority items and untracked debt.

```bash
./scripts/check_tech_debt.sh
```

### scan_todos.sh

Scans `src/`, `tests/`, and `include/` for TODO comments and cross-references them against `TECHNICAL_DEBT.md`. Generates a timestamped report file.

```bash
./scripts/scan_todos.sh
```

## Testing

### run_tests.sh

Runs the test suite via CTest with optional sanitizers and coverage.

```bash
./scripts/run_tests.sh                   # Normal run
./scripts/run_tests.sh --asan            # AddressSanitizer
./scripts/run_tests.sh --ubsan           # UndefinedBehaviorSanitizer
./scripts/run_tests.sh --tsan            # ThreadSanitizer
./scripts/run_tests.sh --coverage        # Code coverage (requires lcov)
./scripts/run_tests.sh --verbose         # Verbose output
```

### test_config_reload.sh

Tests the configuration hot-reloading feature (SIGHUP signal handling). Starts the server, modifies the config file, sends SIGHUP, and verifies the server picks up changes.

```bash
./scripts/test_config_reload.sh
```

### test_log_rotation.sh

Tests log file creation, rotation, and size limit enforcement.

```bash
./scripts/test_log_rotation.sh
```

### test_signal_handling.sh

Verifies the server handles SIGTERM gracefully for clean shutdown.

```bash
./scripts/test_signal_handling.sh
```

### test_sigint.sh

Tests SIGINT (Ctrl+C) signal handling for graceful server shutdown.

```bash
./scripts/test_sigint.sh
```

### manual_test_reload.sh

Interactive helper for manually testing configuration hot-reload. Creates a test config, starts the server, and instructs the user to edit the config and send SIGHUP.

```bash
./scripts/manual_test_reload.sh
```

### test_chatbot_gui.sh

Verifies the chatbot GUI executable is properly built and linked (checks existence, permissions, file size, library dependencies).

```bash
./scripts/test_chatbot_gui.sh
```

### test_chatbot_gui_comprehensive.sh

Comprehensive test suite for the chatbot GUI covering build verification, dependency checks, integration tests, and code quality.

```bash
./scripts/test_chatbot_gui_comprehensive.sh
```

### verify_cli_parallel.sh

Verifies OpenMP parallel processing is correctly linked in the CLI chatbot binary.

```bash
./scripts/verify_cli_parallel.sh
```

### verify_gui_parallel.sh

Verifies parallel processing support in the chatbot GUI binary (checks OpenMP linkage and wrapper script).

```bash
./scripts/verify_gui_parallel.sh
```

### verify_special_token_fixes.py

Runs tokenizer tests and API server tests to verify special token handling is correct after vocabulary fixes.

```bash
python3 scripts/verify_special_token_fixes.py
```

### batch_api_client.py

Example/test client demonstrating the batch processing API. Includes five examples: basic batch chat, batch sessions, performance comparison (single vs. batch), variable-length efficiency analysis, and a customer support simulation.

```bash
python3 scripts/batch_api_client.py
```

## Diagnostics

### monitor_training.py

Real-time CLI dashboard for training metrics. Polls the metrics summary JSON file and displays a live updating dashboard with loss curves, progress bars, and timing estimates.

```bash
python3 scripts/monitor_training.py [--summary-file PATH] [--refresh-rate N] [--format full|compact|minimal]
```

## Docker

### docker_build.sh

Builds the Docker image for the ADAI chatbot API server.

```bash
./scripts/docker_build.sh                        # Build with defaults
./scripts/docker_build.sh -t v1.0.0              # Specific tag
./scripts/docker_build.sh --no-cache             # Build without cache
./scripts/docker_build.sh --platform linux/amd64 # Specific platform
```

### docker_deploy.sh

Manages Docker container lifecycle for the ADAI chatbot API server (start, stop, restart, logs, status, shell, clean).

```bash
./scripts/docker_deploy.sh start                 # Start with defaults
./scripts/docker_deploy.sh start -p 9090         # Custom port
./scripts/docker_deploy.sh stop                  # Stop container
./scripts/docker_deploy.sh status                # Check status + health
./scripts/docker_deploy.sh logs                  # Tail container logs
./scripts/docker_deploy.sh shell                 # Open shell in container
./scripts/docker_deploy.sh clean                 # Remove container and image
```

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

# Start the model service
./scripts/model_service.sh start
```

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
