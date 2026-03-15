# Adai Log File Location Standards

This document outlines the standard and best-practice locations for storing application log files across supported platforms (Linux and Windows) for the Adai project. Following these standards ensures consistency, security, and proper integration with native OS logging and monitoring tools.

## 1. Linux Logging Standards

On Linux systems, the appropriate log directory depends tightly on whether the application runs as a system-wide daemon (service) or as an individual user's process.

### System-Wide Services (Daemon/Root)
* **Standard Path:** `/var/log/adai/`
* **Examples:**
  * `/var/log/adai/chatbot_server.log`
  * `/var/log/adai/metrics_api.log`
* **Permissions:** The directory must be owned by the dedicated service user and group (e.g., `adai:adai`) with `755` permissions. The individual log files should restrict read access where appropriate (e.g., `640` or `600`).
* **Management:** Writing to `/var/log/` natively integrates with standard log maintenance utilities like `logrotate`.

### User-Level Execution (Local User)
* **Standard Path:** `~/.local/state/adai/log/` (following the XDG Base Directory specification) or `~/.cache/adai/log/`
* **Examples:**
  * `~/.local/state/adai/log/incremental_trainer.log`
  * `~/.local/state/adai/log/dev_server.log`
* **Usage:** Use this when a developer or unprivileged user runs the application manually. Attempting to run as a standard user with a configuration pointing to `/var/log/adai/` will result in `Permission denied` errors.

## 2. Windows Logging Standards

Windows logging conventions separate system-wide application data from user-specific application data.

### System-Wide Services
* **Standard Path:** `%PROGRAMDATA%\adai\Logs\`
* **Examples:** 
  * `C:\ProgramData\adai\Logs\chatbot_server.log`
* **Permissions:** Must grant write access to the specific service account (e.g., `LOCAL SYSTEM`, `NETWORK SERVICE`, or a custom service account). Standard user accounts should typically only have read/execute access.

### User-Level Execution (Local User)
* **Standard Path:** `%LOCALAPPDATA%\adai\Logs\`
* **Examples:** 
  * `C:\Users\<Username>\AppData\Local\adai\Logs\incremental_trainer.log`
* **Usage:** For command-line executions, local development instances, or testing by individual users.

## 3. Implementation and Configuration in Adai

In the Adai project, logging destinations are controlled using the `LOG_FILE_PATH` parameter.

* **Configuration setting (`config.conf`):** `LOG_FILE_PATH=/var/log/adai/chatbot.log`
* **Environment Variable:** `export LOG_FILE_PATH=/tmp/adai.log` (Overrides standard configuration)

### Best Practices to Follow

1. **Default to STDERR/STDOUT:** 
   Applications like `incremental_trainer` or containerized components (like Docker/Kubernetes deployments) should default heavily to stdout/stderr. System orchestrators (`systemd`/`journalctl` or Kubernetes) will correctly capture and manage these standard streams without needing file setups.
2. **Opt-in File Logging:** 
   Writing to a fixed log file should be an *opt-in* feature explicitly specified by supplying a `LOG_FILE_PATH`. If the path is empty, fall back to console-only output. 
3. **Directory Creation:** 
   When a `LOG_FILE_PATH` is specified, the application (or its initialization scripts) should ensure the parent directories exist before initializing the spdlog file sink.
4. **Log File Rotation:** 
   Adai uses built-in log rotation (via `spdlog` functionality like `LOG_MAX_SIZE_MB` and `LOG_MAX_FILES`). When writing to standard system paths (like `/var/log`), avoid conflicts between Adai's internal rotation and external `logrotate` configurations. Prefer external compression tools for archiving rotated files.
