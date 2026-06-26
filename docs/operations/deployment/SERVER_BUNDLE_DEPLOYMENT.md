# Server Bundle Deployment Guide

Deploying the ADAI server infrastructure: `metrics_api_server`, `registry_server`, and `mns_server` as a co-located set of systemd services with SQL-backed persistence.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Building](#4-building)
5. [Packaging for Distribution](#5-packaging-for-distribution)
6. [Installation](#6-installation)
   - 6.1 [Quick Start (SQLite)](#61-quick-start-sqlite)
   - 6.2 [Quick Start (PostgreSQL)](#62-quick-start-postgresql)
   - 6.3 [Install Script Reference](#63-install-script-reference)
7. [Database Backends](#7-database-backends)
   - 7.1 [SQLite (Default)](#71-sqlite-default)
   - 7.2 [PostgreSQL](#72-postgresql)
   - 7.3 [Dual-Write Mode](#73-dual-write-mode)
   - 7.4 [File-Only Mode](#74-file-only-mode)
8. [PostgreSQL Setup](#8-postgresql-setup)
   - 8.1 [Automated Setup](#81-automated-setup)
   - 8.2 [Manual Setup](#82-manual-setup)
   - 8.3 [Schema Reference](#83-schema-reference)
9. [Services Reference](#9-services-reference)
   - 9.1 [Model Name Service (mns_server)](#91-model-name-service-mns_server)
   - 9.2 [Dataset Registry Server (registry_server)](#92-dataset-registry-server-registry_server)
   - 9.3 [Training Metrics API Server (metrics_api_server)](#93-training-metrics-api-server-metrics_api_server)
10. [Configuration](#10-configuration)
11. [Service Management](#11-service-management)
12. [Connecting Trainers](#12-connecting-trainers)
13. [Monitoring and Health Checks](#13-monitoring-and-health-checks)
14. [Backup and Recovery](#14-backup-and-recovery)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Overview

The **server bundle** is the infrastructure layer that supports training and model management. It runs on a single host (typically the training machine or a dedicated coordinator) and provides three services:

| Service | Port | Purpose |
|---------|------|---------|
| `mns_server` | 8083 | Model identity registry (names, roles, artifacts, lifecycle) |
| `registry_server` | 8082 | Dataset queue coordination for distributed training |
| `metrics_api_server` | 8081 | Real-time training metrics REST API and dashboard |

All three communicate via localhost and are installed, started, and managed as a single unit by `install_server_bundle.sh`.

---

## 2. Architecture

```
              Trainer Machines                      Server Bundle (one host)
  ┌──────────────────────────────┐     ┌──────────────────────────────────────┐
  │  incremental_trainer         │     │                                      │
  │    ├─ push metrics ──────────┼────>│  metrics_api_server  :8081           │
  │    ├─ acquire/release data ──┼────>│  registry_server     :8082           │
  │    └─ register model ────────┼────>│  mns_server          :8083           │
  │                              │     │                                      │
  │  dataset_manager             │     │  Persistence:                        │
  │    └─ queue files ───────────┼────>│    SQLite  ─ metrics.db / models.db  │
  └──────────────────────────────┘     │    and/or                            │
                                       │    PostgreSQL ─ adai database        │
  ┌──────────────────────────────┐     │    and/or                            │
  │  dashboard.html (browser)    │     │    JSONL flat files                  │
  │    └─ poll /api/sessions ────┼────>│                                      │
  └──────────────────────────────┘     └──────────────────────────────────────┘
```

---

## 3. Prerequisites

- Linux with systemd (Ubuntu 20.04+, Debian 11+, RHEL 8+, Fedora 36+)
- C++17 toolchain (GCC 10+ or Clang 14+) for building, not required on deploy target
- `libsqlite3-dev` (build dependency; runtime library is typically pre-installed)
- `libpq-dev` (only if building with `ENABLE_POSTGRES_METRICS=ON`)
- `cpp-httplib` header (vendored in `external/cpp-httplib/`)

---

## 4. Building

### Portable Release Build

The portable preset produces baseline x86-64 binaries (`-march=x86-64`) safe for deployment to any modern Linux machine:

```bash
cmake --preset portable
cmake --build --preset portable
```

Binaries are placed in `build/portable/bin/`.

### Standard Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make metrics_api_server registry_server mns_server mns_cli -j$(nproc)
```

### With PostgreSQL Support

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_POSTGRES_METRICS=ON
```

This links `libpq` and compiles the PostgreSQL metrics backend. Without this flag, only SQLite and file backends are available at runtime.

---

## 5. Packaging for Distribution

The `package_server_bundle.sh` script creates a self-contained tarball that can be copied to any target host and installed without access to the source repository.

### Create Package

```bash
# Auto-detect build directory, version from git
./scripts/package_server_bundle.sh

# Specify build dir and version explicitly
./scripts/package_server_bundle.sh --build-dir build/portable --version v1.2.0

# Include trainer binaries in the bundle
./scripts/package_server_bundle.sh --include-trainer

# Keep debug symbols (skip stripping)
./scripts/package_server_bundle.sh --no-strip
```

### Package Contents

```
adai-server-bundle-<version>/
  bin/
    metrics_api_server          Training metrics REST API
    registry_server             Dataset queue coordinator
    mns_server                  Model Name Service
    mns_cli                     MNS command-line client
    incremental_trainer         (only with --include-trainer)
    dataset_manager             (only with --include-trainer)
    vocab_builder               (included if built)
  scripts/
    install_server_bundle.sh    Bundle installer
    install_incremental_trainer.sh
    install_metrics_service.sh
    install_mns_server.sh
    setup_postgres.sql          PostgreSQL schema (metrics + MNS tables)
  config.conf                   Default configuration template
  config-remote.conf            Remote/distributed configuration template
  vocab.txt                     Default BPE vocabulary
  dashboard.html                Training metrics web dashboard
  README.txt                    Quick-start instructions
```

### Deploy to Target Host

```bash
scp adai-server-bundle-v1.2.0.tar.gz deploy@training-server:/tmp/
ssh deploy@training-server
cd /tmp && tar xzf adai-server-bundle-v1.2.0.tar.gz
cd adai-server-bundle-v1.2.0
sudo scripts/install_server_bundle.sh --build-dir . --yes
```

---

## 6. Installation

### 6.1 Quick Start (SQLite)

SQLite is the default backend. No additional software is required on the target host.

```bash
sudo ./scripts/install_server_bundle.sh --build-dir . --yes
```

This creates:
- System user `adai` and directories under `/opt/adai/`
- Config file at `/opt/adai/etc/config.conf`
- SQLite database at `/opt/adai/training_sessions/metrics.db`
- Three systemd services: `adai-mns`, `adai-registry`, `adai-metrics`

### 6.2 Quick Start (PostgreSQL)

The `--setup-postgres` flag handles the entire PostgreSQL lifecycle: package installation, database/role creation, and schema bootstrap.

```bash
sudo ./scripts/install_server_bundle.sh --build-dir . --setup-postgres --yes
```

This does everything from 6.1 plus:
- Installs `postgresql`, `postgresql-client`, `libpq-dev` via the system package manager
- Creates PostgreSQL role `adai` and database `adai`
- Runs `setup_postgres.sql` to create all tables and indexes
- Sets `METRICS_STORAGE_BACKEND=postgres+file` and auto-derives the connection URL

### 6.3 Install Script Reference

```
install_server_bundle.sh [OPTIONS]

Service options:
  --install-path PATH         Install root (default: /opt/adai)
  --user USER                 Service user (default: adai)
  --group GROUP               Service group (default: adai)
  --build-dir DIR             Directory containing bin/ (default: build/portable)
  --mns-port PORT             MNS port (default: 8083)
  --registry-port PORT        Registry port (default: 8082)
  --metrics-port PORT         Metrics API port (default: 8081)

Directory options:
  --metrics-dir DIR           Metrics/sessions directory
  --registry-data-dir DIR     Registry data directory
  --mns-data-dir DIR          MNS data directory

Database options:
  --storage-backend BACKEND   file | sqlite | postgres | sqlite+file | postgres+file
  --db-path PATH              SQLite file path (default: <metrics-dir>/metrics.db)
  --db-url URL                PostgreSQL connection URL
  --db-pool-size N            PostgreSQL pool size (default: 4)

PostgreSQL options:
  --setup-postgres            Install PostgreSQL packages, create DB/role, run schema
  --pg-db-name NAME           Database name (default: adai)
  --pg-db-user USER           PostgreSQL role (default: same as --user)

General:
  --yes                       Skip confirmation prompts
  --help                      Show full help
```

---

## 7. Database Backends

The `metrics_api_server` supports four storage modes, controlled by the `METRICS_STORAGE_BACKEND` config key or the `--storage-backend` CLI flag.

### 7.1 SQLite (Default)

```
METRICS_STORAGE_BACKEND=sqlite+file
METRICS_DB_PATH=training_sessions/metrics.db
```

- Single-file database, zero external dependencies
- WAL mode for concurrent reads during training
- Suitable for single-server deployments

The MNS always uses its own SQLite database (`models.db`) regardless of this setting.

### 7.2 PostgreSQL

```
METRICS_STORAGE_BACKEND=postgres+file
METRICS_DB_URL=postgresql://adai@localhost/adai
METRICS_DB_POOL_SIZE=4
```

- Connection pool with configurable size
- Automatic retry with backoff on connection loss (100ms, 400ms, 1600ms)
- Shared database for metrics and MNS tables (via `setup_postgres.sql`)
- Requires the binary to be built with `-DENABLE_POSTGRES_METRICS=ON`

### 7.3 Dual-Write Mode

Backends ending in `+file` (e.g., `sqlite+file`, `postgres+file`) write to both the SQL database and the traditional JSONL/JSON flat files. This allows:

- Validating SQL output against known-good JSONL files during migration
- Operators to inspect raw JSONL with standard tools (`jq`, `grep`)
- Safe rollback to file-only mode if needed

Once confidence is established, switch to `sqlite` or `postgres` (without `+file`) to stop writing flat files.

### 7.4 File-Only Mode

```
METRICS_STORAGE_BACKEND=file
```

Original behavior: append-only JSONL and JSON summary files. No SQL database is created. History queries are limited to the in-memory ring buffer (default 10,000 records). The DB-backed API endpoints (`/api/sessions/{key}/metrics/db-history`, `/api/metrics/compare`, `/api/sessions/{key}/metrics/export`) return errors in this mode.

---

## 8. PostgreSQL Setup

### 8.1 Automated Setup

```bash
sudo ./scripts/install_server_bundle.sh --setup-postgres --yes
```

The installer automatically:

1. Detects the package manager (`apt`, `dnf`, or `yum`)
2. Installs PostgreSQL server and client packages
3. Starts and enables the `postgresql` systemd service
4. Creates a PostgreSQL role matching the service user (default: `adai`)
5. Creates the database (default: `adai`) owned by that role
6. Applies `scripts/setup_postgres.sql` to create all tables
7. Verifies peer authentication works for the service user

### 8.2 Manual Setup

If you prefer to manage PostgreSQL yourself:

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-client libpq-dev   # Debian/Ubuntu
sudo dnf install postgresql-server postgresql postgresql-devel  # RHEL/Fedora

# Initialize and start (RHEL/Fedora only)
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql

# Create role and database
sudo -u postgres createuser adai
sudo -u postgres createdb -O adai adai

# Apply schema
sudo -u postgres psql -d adai -f scripts/setup_postgres.sql

# Verify connectivity
sudo -u adai psql -d adai -c 'SELECT version();'
```

Then install the server bundle with the connection URL:

```bash
sudo ./scripts/install_server_bundle.sh --build-dir . \
  --storage-backend postgres+file \
  --db-url "postgresql://adai@localhost/adai" \
  --yes
```

### 8.3 Schema Reference

`setup_postgres.sql` creates these tables (all idempotent with `IF NOT EXISTS`):

**Metrics tables** (used by `metrics_api_server`):

| Table | Purpose |
|-------|---------|
| `schema_version` | Migration tracking (shared across subsystems) |
| `sessions` | One row per training session with summary fields |
| `metrics_history` | Per-sample metrics records (loss, lr, grad norm, etc.) |
| `generation_quality` | BLEU/ROUGE scores per epoch |
| `abnormal_samples` | Flagged outlier training samples |

**Name service tables** (used by `mns_server`):

| Table | Purpose |
|-------|---------|
| `models` | Model identity, architecture, artifact location |
| `training_history` | Training run records linked to metrics sessions |
| `roles` | Role-to-model mapping (e.g., `chatbot` -> model name) |

---

## 9. Services Reference

### 9.1 Model Name Service (`mns_server`)

Central identity registry for trained models.

| | |
|---|---|
| Port | 8083 |
| Storage | SQLite (`models.db`) in data directory |
| Systemd unit | `adai-mns.service` |
| Health check | `curl http://localhost:8083/health` |

Key endpoints:
- `POST /api/models` — register a model
- `GET /api/models` — list all models
- `PUT /api/models/:name/state` — transition lifecycle state
- `PUT /api/models/:name/role` — assign production role
- `GET /api/roles/:role` — resolve role to model

Full API reference: [MODEL_NAME_SERVICE.md](../guides/MODEL_NAME_SERVICE.md)

### 9.2 Dataset Registry Server (`registry_server`)

Coordinates dataset queue access across distributed trainers.

| | |
|---|---|
| Port | 8082 |
| Storage | Flat files with advisory locking |
| Systemd unit | `adai-registry.service` |
| Health check | `curl http://localhost:8082/health` |

Key endpoints:
- `POST /api/acquire` — atomically acquire pending files for a run
- `POST /api/release` — return files to the pool on failure
- `POST /api/mark-trained` — mark files as trained
- `GET /api/status` — queue summary

### 9.3 Training Metrics API Server (`metrics_api_server`)

Receives and stores training metrics pushed by `incremental_trainer`. Serves dashboards and monitoring queries.

| | |
|---|---|
| Port | 8081 |
| Storage | SQLite and/or PostgreSQL and/or JSONL files |
| Systemd unit | `adai-metrics.service` |
| Health check | `curl http://localhost:8081/health` |
| Dashboard | `dashboard.html` (static file, polls the API) |

Key consumer endpoints:
- `GET /api/sessions` — list all sessions (with optional `?status=completed&from=...` filters)
- `GET /api/sessions/{key}/metrics/current` — real-time snapshot
- `GET /api/sessions/{key}/metrics/db-history?from=...&to=...&limit=N` — DB-backed time-range query
- `GET /api/sessions/{key}/metrics/export?format=csv` — full history export
- `GET /api/metrics/compare?keys=k1,k2&metric=loss` — cross-session comparison
- `GET /api/metrics/prometheus/aggregate` — Prometheus scrape target

Full API reference: [TRAINING_METRICS_API.md](../../development/TRAINING_METRICS_API.md)

---

## 10. Configuration

The install script writes `/opt/adai/etc/config.conf`. Key settings:

```ini
# Service discovery
NAME_SERVICE_URL=http://localhost:8083
REGISTRY_SERVER_URL=http://localhost:8082
METRICS_SERVER_URL=http://localhost:8081

# Database persistence
METRICS_STORAGE_BACKEND=sqlite+file    # or postgres+file, sqlite, postgres, file
METRICS_DB_PATH=training_sessions/metrics.db
# METRICS_DB_URL=postgresql://adai@localhost/adai
# METRICS_DB_POOL_SIZE=4

# Session management
METRICS_MAX_LIVE_SESSIONS=16
METRICS_COMPLETED_TTL_SECONDS=3600
METRICS_SWEEP_INTERVAL_SECONDS=60
METRICS_STALENESS_THRESHOLD_SECONDS=60
```

All keys can be overridden via environment variables of the same name, or via `systemctl edit adai-metrics`:

```ini
[Service]
Environment="METRICS_STORAGE_BACKEND=postgres"
Environment="METRICS_DB_URL=postgresql://adai@dbhost/adai"
```

---

## 11. Service Management

```bash
# Status
systemctl status adai-mns adai-registry adai-metrics

# Start / stop / restart all
for svc in adai-mns adai-registry adai-metrics; do
  sudo systemctl restart "$svc"
done

# View logs
journalctl -u adai-metrics -f          # follow live
journalctl -u adai-mns -n 50           # last 50 lines
journalctl -u adai-registry --since "1 hour ago"

# Disable auto-start
sudo systemctl disable adai-registry
```

**Start order:** `adai-mns` starts first (no dependencies). `adai-metrics` declares `Wants=adai-mns.service` so it starts after MNS and can query the model registry at `/api/models`.

---

## 12. Connecting Trainers

Configure trainers to push metrics and coordinate datasets via the server bundle:

```ini
# In the trainer's config.conf or environment
METRICS_SERVER_URL=http://training-server:8081
REGISTRY_SERVER_URL=http://training-server:8082
NAME_SERVICE_URL=http://training-server:8083
```

Or pass via command line:

```bash
./incremental_trainer train \
  --config config-remote.conf
```

The trainer auto-derives a session key, pushes metrics via HTTP, and registers the model with the MNS on completion.

---

## 13. Monitoring and Health Checks

```bash
# All three services
for port in 8081 8082 8083; do
  echo -n "Port ${port}: "
  curl -s "http://localhost:${port}/health" | head -c 80
  echo
done

# Active training sessions
curl -s http://localhost:8081/api/sessions | python3 -m json.tool

# Prometheus scrape (all sessions, labelled)
curl -s http://localhost:8081/api/metrics/prometheus/aggregate

# Database row counts (PostgreSQL)
sudo -u adai psql -d adai -c "
  SELECT 'sessions' AS tbl, COUNT(*) FROM sessions
  UNION ALL
  SELECT 'metrics_history', COUNT(*) FROM metrics_history
  UNION ALL
  SELECT 'models', COUNT(*) FROM models;
"

# Database row counts (SQLite)
sqlite3 /opt/adai/training_sessions/metrics.db \
  "SELECT 'sessions', COUNT(*) FROM sessions;
   SELECT 'metrics_history', COUNT(*) FROM metrics_history;"
```

---

## 14. Backup and Recovery

### SQLite

```bash
# Online backup (safe while server is running due to WAL mode)
sqlite3 /opt/adai/training_sessions/metrics.db ".backup '/backup/metrics.db'"

# Restore
sudo systemctl stop adai-metrics
cp /backup/metrics.db /opt/adai/training_sessions/metrics.db
chown adai:adai /opt/adai/training_sessions/metrics.db
sudo systemctl start adai-metrics
```

### PostgreSQL

```bash
# Dump
sudo -u adai pg_dump adai > /backup/adai_$(date +%Y%m%d).sql

# Restore
sudo -u postgres psql -c "DROP DATABASE IF EXISTS adai;"
sudo -u postgres createdb -O adai adai
sudo -u adai psql -d adai < /backup/adai_20260625.sql
```

### JSONL Files

If running in dual-write mode (`sqlite+file` or `postgres+file`), the JSONL files in `training_sessions/` are a complete secondary copy and can serve as a backup source independently.

---

## 15. Troubleshooting

### Service fails to start

```bash
journalctl -u adai-metrics -n 50 --no-pager
# Common causes:
#   - Port already in use (another instance or stale PID)
#   - Database file permissions (chown adai:adai)
#   - PostgreSQL not running or role missing
```

### PostgreSQL connection refused

```bash
# Check server is running
systemctl status postgresql

# Check role exists
sudo -u postgres psql -c "SELECT rolname FROM pg_roles WHERE rolname = 'adai';"

# Check pg_hba.conf allows peer auth for the role
sudo grep -v '^#' /etc/postgresql/*/main/pg_hba.conf | grep -v '^$'
```

### SQLite "database is locked"

This should not occur under normal operation (WAL mode allows concurrent reads). If it does:

```bash
# Check for stale lock files
ls -la /opt/adai/training_sessions/metrics.db*

# Verify only one metrics_api_server is running
pgrep -a metrics_api_server
```

### Metrics not appearing in dashboard

```bash
# Verify trainer is pushing
curl -s http://localhost:8081/api/sessions | python3 -m json.tool

# Check trainer config
grep METRICS_SERVER_URL config.conf

# Test push manually
curl -X POST http://localhost:8081/api/sessions/test/start \
  -H 'Content-Type: application/json' \
  -d '{"session_id":1,"total_epochs":1}'
```

### Reinstalling / upgrading

The install script is idempotent. To upgrade, build new binaries and re-run:

```bash
sudo ./scripts/install_server_bundle.sh --build-dir . --yes
```

Existing databases, config files, and training data are preserved. Only binaries and systemd units are overwritten.
