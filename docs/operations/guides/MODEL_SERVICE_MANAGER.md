# ADAI Model Service Manager

The `model_service.sh` script is a comprehensive management utility for the ADAI Chatbot API Server. It simplifies the process of building, starting, stopping, and monitoring the service without requiring `systemd`, root privileges, or complex manual commands.

This script is ideal for:

- **Development**: Quickly start/stop the server while coding.
- **Testing**: Run integration tests against a local instance.
- **Docker**: Serve as the entrypoint for containerized deployments.
- **Simple Deployment**: Run as a background daemon on a server.

## Location

```bash
scripts/model_service.sh
```

## Quick Start

```bash
# Start the service in the background (will build if necessary)
./scripts/model_service.sh start

# Check status
./scripts/model_service.sh status

# Tail logs
./scripts/model_service.sh logs

# Stop the service
./scripts/model_service.sh stop
```

## Commands

The script follows the pattern: `./model_service.sh <command> [options]`

|Command|Description|
|---|---|
|`start`|Builds the binary (if missing), validates configuration, and starts the service. By default, it runs in the background (daemon mode) and waits for the HTTP health check to pass.|
|`stop`|Stops the running service using the PID file. It sends `SIGTERM`, waits for graceful shutdown, and force-kills if necessary.|
|`restart`|Convenience command that runs `stop` followed by `start`.|
|`status`|Checks if the process is running. Displays PID, port, memory usage, and health status.|
|`health`|Make a request to the `/health` endpoint and prints the response.|
|`logs`|Tails the service log file (`tail -f`). Press `Ctrl+C` to exit.|
|`build`|Compiles the `chatbot_api_server` binary using CMake. Options like `--build-type` and `--jobs` apply here.|
|`help`|Displays the help message with usage examples.|

## Configuration

Configuration is layered. Command-line flags override environment variables, which override `config.conf` values.

### Command-Line Arguments

|Flag|Description|Default|
|---|---|---|
|`--config <path>`|Path to the configuration file|`./config.conf`|
|`--vocab <path>`|Path to the vocabulary file|From config|
|`--model <path>`|Path to model weights file|From config/None (random init)|
|`--port <number>`|Port to listen on|8080 (or from config)|
|`--log-level <lvl>`|Logging verbosity (`DEBUG`, `INFO`, `WARN`, `ERROR`)|`INFO`|
|`--build-type <type>`|Build configuration (`debug` or `release`)|`release`|
|`--foreground`|Run in current terminal (do not daemonize)|`false`|
|`--jobs <n>`|Parallel build jobs for CMake|`nproc`|
|`--pidfile <path>`|Location of the PID file|`/tmp/adai_model_service.pid`|
|`--logfile <path>`|Location of the log file|`/tmp/adai_model_service.log`|

### Configuration File (`config.conf`)

The script reads default values from the `config.conf` file in the project root (or as specified by `--config`).

Key variables used by the script:

- `VOCAB_PATH`: Required for server startup.
- `MODEL_PATH`: Optional model weights.
- `PORT`: Service port.

## Examples

### 1. Development Mode

Run in the foreground to see logs immediately and use a debug build.

```bash
./scripts/model_service.sh start \
  --foreground \
  --build-type debug \
  --log-level DEBUG
```

### 2. Custom Port and Model

Start a background service with a specific model file on port 9090.

```bash
./scripts/model_service.sh start \
  --model ./models/epoch_20.bin \
  --port 9090
```

### 3. Check Health

Verify the service is responsive.

```bash
./scripts/model_service.sh health
# Output: {"status":"ok","uptime":123}
```

### 4. Restart with New Configuration

Reload the service with a different log level.

```bash
./scripts/model_service.sh restart --log-level WARN
```

## Troubleshooting

### "Service exited immediately"

If `start` fails, check the logs for the immediate error:

```bash
cat /tmp/adai_model_service.log
```

Common causes:

- Missing vocabulary file (`--vocab` or `VOCAB_PATH` in config).
- Port already in use (the script tries to detect this, but race conditions exist).
- Invalid model file format.

### "Binary not found"

The script attempts to build automatically. If that fails, try building manually to see compilation errors:

```bash
./scripts/model_service.sh build
```

### "Stale PID file"

If the server crashed hard, a PID file might remain. The script usually detects if the process is dead and removes the file, but you can manually remove it if needed:

```bash
rm /tmp/adai_model_service.pid
```

## Systemd vs. Script

- **Use `model_service.sh`** for ad-hoc usage, development, CI/CD pipelines, and unprivileged execution.
- **Use Systemd** (via `scripts/install_chatbot_API.sh`) for production deployments where the service must auto-start on boot and restart on failure.
