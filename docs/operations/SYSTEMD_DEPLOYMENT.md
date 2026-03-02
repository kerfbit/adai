# systemd Deployment Guide

## Overview

This guide covers deploying the ADAI Chatbot API Server as a systemd service on Linux systems. This enables the chatbot to run as a managed system daemon with automatic startup, graceful shutdown, resource limits, and monitoring capabilities.

## Quick Start

### Automated Installation

The simplest way to install the service:

```bash
# 1. Build the project
cd /path/to/adai
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_API_SERVER=ON
make chatbot_api_server

# 2. Run the installation script
cd ..
sudo ./scripts/install_systemd_service.sh

# 3. Verify the service is running
systemctl status adai
curl http://localhost:8080/health
```

The installation script automatically:
- Creates a system user (`adai`)
- Sets up directory structure in `/opt/adai`
- Copies executables and configuration
- Installs and starts the systemd service

### Manual Installation

For custom installations or understanding the process:

```bash
# See Manual Installation section below
```

## Prerequisites

### System Requirements

- **Linux Distribution**: Any systemd-based distribution
  - Ubuntu 18.04+
  - Debian 10+
  - CentOS/RHEL 7+
  - Fedora 30+
  - Arch Linux
  
- **systemd Version**: 230+ (check with `systemctl --version`)

- **Root Access**: Required for service installation

- **Build Tools**: g++, cmake, make (if building from source)

### Building the Executable

```bash
# Clone repository
git clone https://github.com/rjv717/adai.git
cd adai

# Build with Release optimizations
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_API_SERVER=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF
make chatbot_api_server -j$(nproc)

# Verify executable
./chatbot_api_server --help
```

## Automated Installation

### Using the Installation Script

The `install_systemd_service.sh` script automates the entire installation process.

#### Default Installation

```bash
sudo ./scripts/install_systemd_service.sh
```

**Installs to:**
- Executable: `/opt/adai/bin/chatbot_api_server`
- Vocabulary: `/opt/adai/vocab/vocab.txt`
- Models: `/opt/adai/models/`
- Logs: `/var/log/adai/`
- Config: `/etc/adai/config.conf`
- Service: `/etc/systemd/system/adai.service`

**Creates:**
- User: `adai` (system user, no login)
- Group: `adai`
- Port: 8080

#### Custom Installation

```bash
# Install to custom location
sudo ./scripts/install_systemd_service.sh \
  --install-path /usr/local/adai \
  --port 9000

# Use custom user and group
sudo ./scripts/install_systemd_service.sh \
  --user mychatbot \
  --group mychatbot \
  --log-level DEBUG

# View all options
./scripts/install_systemd_service.sh --help
```

#### Installation Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--install-path PATH` | Installation directory | `/opt/adai` |
| `--user USER` | Service user | `adai` |
| `--group GROUP` | Service group | `adai` |
| `--port PORT` | Server port | 8080 |
| `--log-level LEVEL` | Log level (DEBUG/INFO/WARN/ERROR) | INFO |
| `--help` | Show help message | - |

### What the Script Does

1. **Creates System User**: Non-login system user for security
2. **Creates Directories**: `/opt/adai/{bin,vocab,models}`, `/var/log/adai`, `/etc/adai`
3. **Copies Files**: Executable, vocabulary, configuration
4. **Sets Permissions**: Proper ownership and access controls
5. **Installs Service**: systemd unit file in `/etc/systemd/system/`
6. **Enables Service**: Sets service to start on boot
7. **Starts Service**: Immediately starts the chatbot
8. **Verifies**: Checks service status and listening port

## Manual Installation

For advanced users or custom setups:

### Step 1: Create Service User

```bash
# Create system user (no login)
sudo useradd -r -s /bin/false -d /opt/adai -c "ADAI Chatbot Service" adai
```

### Step 2: Create Directory Structure

```bash
# Create directories
sudo mkdir -p /opt/adai/{bin,vocab,models}
sudo mkdir -p /var/log/adai
sudo mkdir -p /etc/adai

# Set ownership
sudo chown -R adai:adai /opt/adai
sudo chown -R adai:adai /var/log/adai
```

### Step 3: Install Files

```bash
# Copy executable
sudo cp build/chatbot_api_server /opt/adai/bin/
sudo chmod 755 /opt/adai/bin/chatbot_api_server

# Copy vocabulary
sudo cp vocab.txt /opt/adai/vocab/
sudo chmod 644 /opt/adai/vocab/vocab.txt

# Copy models (if available)
sudo cp models/* /opt/adai/models/ 2>/dev/null || true
```

### Step 4: Create Configuration File

```bash
sudo tee /etc/adai/config.conf > /dev/null <<'EOF'
# ADAI Chatbot Service Configuration

VOCAB_PATH=/opt/adai/vocab/vocab.txt
PORT=8080
SESSION_TIMEOUT=30
LOG_LEVEL=INFO

# Model Architecture
D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
MAX_SEQ_LENGTH=1024

# Text Generation
MAX_LENGTH=100
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=50
BEAM_WIDTH=4
STRATEGY=nucleus
EOF

sudo chmod 644 /etc/adai/config.conf
```

### Step 5: Install systemd Service File

```bash
# Copy service file
sudo cp scripts/adai.service /etc/systemd/system/

# Update paths if using custom installation
sudo nano /etc/systemd/system/adai.service

# Reload systemd
sudo systemctl daemon-reload
```

### Step 6: Enable and Start Service

```bash
# Enable (start on boot)
sudo systemctl enable adai.service

# Start service
sudo systemctl start adai.service

# Check status
sudo systemctl status adai.service
```

## Service Management

### Basic Commands

```bash
# Start service
sudo systemctl start adai

# Stop service (graceful shutdown)
sudo systemctl stop adai

# Restart service
sudo systemctl restart adai

# Reload configuration (if supported)
sudo systemctl reload adai

# Check status
systemctl status adai

# Enable (start on boot)
sudo systemctl enable adai

# Disable (don't start on boot)
sudo systemctl disable adai

# Check if service is active
systemctl is-active adai

# Check if service is enabled
systemctl is-enabled adai
```

### Viewing Logs

```bash
# View all logs
journalctl -u adai

# Follow logs (tail -f equivalent)
journalctl -u adai -f

# View last 50 lines
journalctl -u adai -n 50

# View logs since last boot
journalctl -u adai -b

# View logs for specific time range
journalctl -u adai --since "2026-03-01 10:00:00"
journalctl -u adai --since "1 hour ago"
journalctl -u adai --since today

# View only errors
journalctl -u adai -p err

# View with timestamps
journalctl -u adai -o short-precise

# Export logs
journalctl -u adai > adai-logs.txt
```

### Log Filtering

```bash
# Filter by log level
journalctl -u adai | grep "\[error\]"
journalctl -u adai | grep "\[warn\]"
journalctl -u adai | grep "\[info\]"

# Filter by keyword
journalctl -u adai | grep "Graceful Shutdown"
journalctl -u adai | grep "Server starting"

# Combine filters
journalctl -u adai -n 100 | grep "\[error\]"
```

## Configuration

### Configuration File

**Location:** `/etc/adai/config.conf`

```ini
# Server Configuration
VOCAB_PATH=/opt/adai/vocab/vocab.txt
MODEL_PATH=/opt/adai/models/model.bin  # Optional
PORT=8080
SESSION_TIMEOUT=30
LOG_LEVEL=INFO

# Model Architecture
D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
MAX_SEQ_LENGTH=1024

# Text Generation
MAX_LENGTH=100
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=50
BEAM_WIDTH=4
STRATEGY=nucleus
```

**After editing:**
```bash
sudo systemctl restart adai
```

### Environment Variables

Override configuration via environment variables in the service file:

**Edit service file:**
```bash
sudo systemctl edit adai
```

**Add overrides:**
```ini
[Service]
Environment="LOG_LEVEL=DEBUG"
Environment="PORT=9000"
Environment="TEMPERATURE=0.7"
```

**Reload and restart:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart adai
```

### Using Environment File

Create `/etc/adai/environment`:

```bash
# /etc/adai/environment
LOG_LEVEL=DEBUG
TEMPERATURE=0.7
```

Update service file:
```bash
sudo systemctl edit adai
```

Add:
```ini
[Service]
EnvironmentFile=/etc/adai/environment
```

## Security and Hardening

### Security Features Enabled

The systemd service file includes extensive security hardening:

#### Filesystem Protection
- ✅ **ProtectSystem=strict**: System directories read-only
- ✅ **ProtectHome=yes**: Home directories inaccessible
- ✅ **ReadWritePaths**: Only `/var/log/adai` and `/opt/adai/models` writable
- ✅ **PrivateTmp=yes**: Private `/tmp` and `/var/tmp`
- ✅ **PrivateDevices=yes**: Limited device access

#### Kernel Protection
- ✅ **ProtectKernelLogs=yes**: No access to kernel logs
- ✅ **ProtectKernelModules=yes**: Cannot load kernel modules
- ✅ **ProtectKernelTunables=yes**: Cannot modify kernel parameters
- ✅ **ProtectControlGroups=yes**: cgroups read-only

#### Privilege Restrictions
- ✅ **NoNewPrivileges=true**: Cannot gain new privileges
- ✅ **CapabilityBoundingSet**: No Linux capabilities
- ✅ **RestrictSUIDSGID=yes**: No setuid/setgid
- ✅ **LockPersonality=yes**: Prevent personality changes

#### Network and System Call Restrictions
- ✅ **RestrictAddressFamilies**: Only IPv4/IPv6
- ✅ **SystemCallFilter**: Limited to essential syscalls
- ✅ **RestrictRealtime=yes**: No real-time scheduling

### Customizing Security

If the service fails with permission errors, you can adjust security settings:

```bash
sudo systemctl edit adai
```

**Loosen filesystem protection:**
```ini
[Service]
ProtectSystem=full
ReadWritePaths=/opt/adai /var/log/adai
```

**Allow more system calls:**
```ini
[Service]
# Comment out or remove SystemCallFilter if needed
SystemCallFilter=
```

**Check which system calls are blocked:**
```bash
journalctl -u adai | grep "SECCOMP"
```

## Resource Management

### Resource Limits in Service File

```ini
[Service]
# Memory limits
MemoryMax=4G          # Hard limit
MemoryHigh=3G         # Soft limit (throttle when exceeded)

# CPU limits
CPUQuota=50%          # 50% of one CPU core

# File and process limits
LimitNOFILE=65536     # Open files
LimitNPROC=512        # Processes/threads
LimitFSIZE=2G         # File size
LimitCORE=0           # Core dumps disabled
```

### Monitoring Resource Usage

```bash
# Real-time monitoring
systemctl status adai

# Detailed resource info
systemd-cgtop

# Memory usage
systemd-cgls /system.slice/adai.service

# Get service property
systemctl show adai -p MemoryCurrent
systemctl show adai -p CPUUsageNSec
```

### Adjusting Limits

```bash
sudo systemctl edit adai
```

```ini
[Service]
MemoryMax=8G
CPUQuota=100%
LimitNOFILE=131072
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart adai
```

## Troubleshooting

### Service Won't Start

**Check service status:**
```bash
systemctl status adai -l
```

**Check logs:**
```bash
journalctl -u adai -n 50 --no-pager
```

**Common issues:**

1. **Missing vocabulary file:**
   ```
   Error: Could not open vocabulary file
   ```
   Solution: Ensure `/opt/adai/vocab/vocab.txt` exists

2. **Permission denied:**
   ```
   Permission denied: /opt/adai/...
   ```
   Solution: Check ownership
   ```bash
   sudo chown -R adai:adai /opt/adai
   ```

3. **Port already in use:**
   ```
   Error: Address already in use
   ```
   Solution: Change port in config or stop conflicting service
   ```bash
   sudo lsof -i :8080
   ```

4. **Executable not found:**
   ```
   Failed to execute command: No such file or directory
   ```
   Solution: Verify executable path in service file
   ```bash
   ls -l /opt/adai/bin/chatbot_api_server
   ```

### Service Crashes

**Check crash logs:**
```bash
journalctl -u adai -p err --since "1 hour ago"
```

**Collect core dump (if enabled):**
```bash
coredumpctl list
coredumpctl info adai
```

**Increase logging verbosity:**
```bash
sudo systemctl edit adai
```

```ini
[Service]
Environment="LOG_LEVEL=DEBUG"
```

**Check resource limits:**
```bash
systemctl show adai -p MemoryCurrent
systemctl show adai -p LimitNOFILE
```

### Service Won't Stop

**Force stop:**
```bash
sudo systemctl kill adai
```

**Check timeout:**
```bash
systemctl show adai -p TimeoutStopSec
```

**Increase stop timeout:**
```bash
sudo systemctl edit adai
```

```ini
[Service]
TimeoutStopSec=60
```

### High Resource Usage

**Check memory:**
```bash
systemctl status adai | grep Memory
```

**Check CPU:**
```bash
top -p $(systemctl show -p MainPID --value adai)
```

**Reduce model size:**
Edit `/etc/adai/config.conf`:
```ini
D_MODEL=256
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
MAX_SEQ_LENGTH=512
```

### Permission Errors

**SELinux issues (CentOS/RHEL):**
```bash
# Check if SELinux is blocking
sudo ausearch -m avc -ts recent

# Set appropriate context
sudo semanage fcontext -a -t bin_t "/opt/adai/bin/chatbot_api_server"
sudo restorecon -v /opt/adai/bin/chatbot_api_server
```

**AppArmor issues (Ubuntu/Debian):**
```bash
# Check AppArmor status
sudo aa-status

# Create profile or set to complain mode
sudo aa-complain /opt/adai/bin/chatbot_api_server
```

## Maintenance

### Updating the Service

```bash
# 1. Stop service
sudo systemctl stop adai

# 2. Backup current installation
sudo cp /opt/adai/bin/chatbot_api_server /opt/adai/bin/chatbot_api_server.backup

# 3. Build new version
cd /path/to/adai
git pull
cd build
make chatbot_api_server

# 4. Install new binary
sudo cp chatbot_api_server /opt/adai/bin/

# 5. Restart service
sudo systemctl start adai

# 6. Verify
systemctl status adai
```

### Updating Configuration

```bash
# Edit config
sudo nano /etc/adai/config.conf

# Restart service
sudo systemctl restart adai

# Or reload if supported
sudo systemctl reload adai
```

### Backup

**Backup script:**
```bash
#!/bin/bash
BACKUP_DIR="/backup/adai/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup executable
cp /opt/adai/bin/chatbot_api_server "$BACKUP_DIR/"

# Backup configuration
cp /etc/adai/config.conf "$BACKUP_DIR/"

# Backup models
cp -r /opt/adai/models "$BACKUP_DIR/"

# Backup service file
cp /etc/systemd/system/adai.service "$BACKUP_DIR/"

echo "Backup complete: $BACKUP_DIR"
```

### Log Rotation

systemd automatically rotates journal logs, but you can configure limits:

```bash
# Edit journald configuration
sudo nano /etc/systemd/journald.conf
```

```ini
[Journal]
SystemMaxUse=1G
SystemMaxFileSize=100M
MaxRetentionSec=1month
```

```bash
# Restart journald
sudo systemctl restart systemd-journald
```

**Manual log cleanup:**
```bash
# Remove logs older than 7 days
journalctl --vacuum-time=7d

# Remove logs until disk usage is below 500M
journalctl --vacuum-size=500M
```

## Monitoring and Alerting

### Health Checks

**HTTP endpoint:**
```bash
curl http://localhost:8080/health
```

**systemd watchdog (optional):**

Edit service file:
```ini
[Service]
WatchdogSec=30
```

Application must send `sd_notify(0, "WATCHDOG=1")` periodically.

### Integration with Monitoring Tools

#### Prometheus

Export systemd metrics:
```bash
# Install node_exporter with systemd collector
# /metrics endpoint will include systemd_unit_state{name="adai.service"}
```

#### Nagios/Icinga

Check script:
```bash
#!/bin/bash
if systemctl is-active --quiet adai; then
    echo "OK - ADAI service is running"
    exit 0
else
    echo "CRITICAL - ADAI service is not running"
    exit 2
fi
```

#### Alerting on Failures

Create systemd override to send alerts:

```bash
sudo systemctl edit adai
```

```ini
[Unit]
OnFailure=alert-admin@%n.service
```

Create alert service:
```bash
sudo nano /etc/systemd/system/alert-admin@.service
```

```ini
[Unit]
Description=Send alert for %i

[Service]
Type=oneshot
ExecStart=/usr/local/bin/send-alert.sh %i
```

## Advanced Topics

### Multiple Instances

Run multiple chatbot instances on different ports:

```bash
# Copy service file
sudo cp /etc/systemd/system/adai.service /etc/systemd/system/adai@.service

# Edit template
sudo nano /etc/systemd/system/adai@.service
```

```ini
[Unit]
Description=ADAI Chatbot Instance %i

[Service]
Environment="PORT=%i"
Environment="CONFIG_FILE=/etc/adai/config-%i.conf"
```

**Start instances:**
```bash
sudo systemctl start adai@8080
sudo systemctl start adai@8081
sudo systemctl start adai@8082
```

### Custom Start/Stop Scripts

Use ExecStartPre and ExecStopPost:

```ini
[Service]
ExecStartPre=/usr/local/bin/adai-pre-start.sh
ExecStopPost=/usr/local/bin/adai-post-stop.sh
```

### Socket Activation

For on-demand startup:

```bash
sudo nano /etc/systemd/system/adai.socket
```

```ini
[Unit]
Description=ADAI Chatbot Socket

[Socket]
ListenStream=8080
Accept=no

[Install]
WantedBy=sockets.target
```

Update service:
```ini
[Service]
# Remove Environment="PORT=..."
# Application reads socket from systemd
```

## Migration from Docker

Converting Docker deployment to systemd:

1. **Export Docker volumes:**
   ```bash
   docker cp adai-chatbot-api:/app/models ./models
   docker cp adai-chatbot-api:/app/vocab ./vocab
   ```

2. **Stop Docker container:**
   ```bash
   docker-compose down
   ```

3. **Install systemd service:**
   ```bash
   sudo ./scripts/install_systemd_service.sh
   ```

4. **Verify same configuration:**
   Compare environment variables in docker-compose.yml with `/etc/adai/config.conf`

## Performance Tuning

### CPU Affinity

Pin service to specific CPU cores:

```bash
sudo systemctl edit adai
```

```ini
[Service]
CPUAffinity=0-3  # Use cores 0, 1, 2, 3
```

### I/O Priority

```ini
[Service]
IOSchedulingClass=best-effort
IOSchedulingPriority=4
```

### Nice Level

```ini
[Service]
Nice=-10  # Higher priority (-20 to 19)
```

## References

- [systemd.service Documentation](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [systemd.exec Documentation](https://www.freedesktop.org/software/systemd/man/systemd.exec.html)
- [Service Files](../../scripts/adai.service)
- [Configuration Guide](../development/STEP1_COMPLETE.md)
- [Signal Handling](../development/STEP2_COMPLETE.md)
- [Structured Logging](../development/STEP3_COMPLETE.md)
- [Docker Deployment](DOCKER_DEPLOYMENT.md)

---

**Step 5: systemd Service File - COMPLETE ✅**
