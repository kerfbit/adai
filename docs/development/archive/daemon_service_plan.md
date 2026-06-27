# Plan: Transition to a Robust Daemon Service

The goal is to evolve the existing `adai_chatbot` executable into a production-ready daemon. We will leverage the existing web API and Docker setup, adding robust configuration, signal handling, and structured logging. This will ensure the service can be managed reliably by systemd or container orchestration platforms.

> **Note**: An existing daemon service (`metrics-api-server`) is already available for capturing training metrics. The steps below integrate with that service rather than building new metrics infrastructure.

## Step-by-Step Process

### 1. Externalize Configuration

* **Objective**: Decouple configuration from the source code to allow for easier management across different environments.
* **Action**:
  * Modify `src/main.cpp` to read configuration from a file (e.g., `/etc/adai/config.json`) or environment variables.
  * Configuration parameters to externalize:
    * Model path
    * Vocabulary path
    * Server port
    * Logging level
    * Metrics daemon URL (`METRICS_SERVER_URL`, default `http://localhost:8081`)
    * Metrics push enabled flag (`METRICS_PUSH_ENABLED`)
  * The `Training Metrics Service` configuration block in `config.conf` already documents all metrics-related parameters. Use those settings to connect to the existing `metrics-api-server` daemon.
* **Rationale**: This change will make it easier to manage the service in development vs. production without requiring a rebuild of the application. Externalizing metrics settings allows the application to point at the running `metrics-api-server` daemon without recompiling.

### 2. Implement Signal Handling

* **Objective**: Ensure the service can shut down gracefully.
* **Action**:
  * In `src/main.cpp`, implement signal handlers for `SIGTERM` and `SIGINT`.
  * The handler should trigger a clean shutdown sequence:
    1. Stop the web server.
    2. Signal the `IncrementalTrainer` to finish its current training cycle and save the model.
    3. Flush any pending metrics to the `metrics-api-server` daemon (via `POST /api/control/flush`) before disconnecting.
    4. Join the trainer thread.
    5. Exit the application.
* **Rationale**: Prevents data corruption and ensures that the model state is saved correctly when the service is stopped or restarted. Flushing metrics on shutdown guarantees no in-flight metrics are lost when the daemon terminates.

### 3. Introduce Structured Logging

* **Objective**: Improve monitoring and debugging capabilities.
* **Action**:
  * Integrate a logging library like `spdlog` (header-only, easy to integrate with CMake).
  * Replace all instances of `std::cout` and `std::cerr` with structured log messages.
  * Configure the logger in `src/main.cpp` to output JSON-formatted logs to `stdout`/`stderr`.
* **Rationale**: Structured logs are machine-readable and essential for effective log analysis, especially in containerized environments.

### 4. Refine Docker Configuration

* **Objective**: Align the Docker setup with best practices for running a service, including the existing metrics daemon.
* **Action**:
  * Update the `Dockerfile` to document the required environment variables for configuration.
  * Ensure the `CMD` in the `Dockerfile` runs the `adai_chatbot` executable directly. The container runtime will manage the process.
  * Add a `metrics-api-server` service to `docker-compose.yml` using the existing `metrics_api_server` binary, exposing it on port `8081`. Configure `chatbot-api` with `depends_on: metrics-api-server` and set `METRICS_PUSH_ENABLED=true` and `METRICS_SERVER_URL=http://metrics-api-server:8081` so metrics flow automatically to the daemon.
* **Rationale**: Running the metrics daemon as a separate container follows the single-responsibility principle and allows the metrics API to be queried independently of the chatbot service.

### 5. Wire the systemd Service File to the Metrics Daemon (for non-Docker deployments)

* **Objective**: Enable proper management of the chatbot as a system service on Linux hosts, with the metrics daemon as a prerequisite.
* **Action**:
  * The `scripts/adai.service` systemd unit already exists. Update its `[Unit]` section to declare:

    ```ini
    After=network-online.target metrics-api-server.service
    Wants=metrics-api-server.service
    ```

  * Ensure `metrics-api-server.service` (located at the repository root) is installed alongside `adai.service`:

```text
    sudo cp metrics-api-server.service /etc/systemd/system/
    sudo cp scripts/adai.service       /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now metrics-api-server adai
    ```

  * Set `METRICS_PUSH_ENABLED=true` and `METRICS_SERVER_URL=http://localhost:8081` in the `[Service]` environment block of `scripts/adai.service`.
* **Rationale**: Declaring `Wants=metrics-api-server.service` ensures systemd starts the metrics daemon before the chatbot, so metrics are never dropped at startup. Using the existing daemon avoids duplicating metrics infrastructure.

## Verification Plan

* **Configuration**: Verify that the service starts correctly using settings from both a configuration file and environment variables, including `METRICS_SERVER_URL` and `METRICS_PUSH_ENABLED`.
* **Metrics Daemon Connectivity**: After starting both services, confirm the chatbot is pushing metrics by querying `GET http://localhost:8081/api/metrics/current` and verifying a non-empty response.
* **Graceful Shutdown**: Test that sending `SIGTERM` to the process (e.g., `kill <pid>`) results in a clean shutdown, that the model is saved, and that `POST /api/control/flush` is called before exit.
* **Logging**: Confirm that the service outputs JSON-formatted logs to standard output.
* **Docker**: Use `docker-compose up -d` to launch both the chatbot and the metrics daemon, then check `docker-compose logs chatbot-api` and query `http://localhost:8081/health` to confirm the metrics daemon is healthy.
* **systemd**: On a bare-metal or VM deployment, use `sudo systemctl start metrics-api-server adai` and verify both are active with `systemctl status`. Inspect metrics via `curl http://localhost:8081/api/session/status` and chatbot logs via `journalctl -u adai`.
