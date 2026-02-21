# Plan: Transition to a Robust Daemon Service

The goal is to evolve the existing `adai_chatbot` executable into a production-ready daemon. We will leverage the existing web API and Docker setup, adding robust configuration, signal handling, and structured logging. This will ensure the service can be managed reliably by systemd or container orchestration platforms.

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
* **Rationale**: This change will make it easier to manage the service in development vs. production without requiring a rebuild of the application.

### 2. Implement Signal Handling

* **Objective**: Ensure the service can shut down gracefully.
* **Action**:
  * In `src/main.cpp`, implement signal handlers for `SIGTERM` and `SIGINT`.
  * The handler should trigger a clean shutdown sequence:
    1. Stop the web server.
    2. Signal the `IncrementalTrainer` to finish its current training cycle and save the model.
    3. Join the trainer thread.
    4. Exit the application.
* **Rationale**: Prevents data corruption and ensures that the model state is saved correctly when the service is stopped or restarted.

### 3. Introduce Structured Logging

* **Objective**: Improve monitoring and debugging capabilities.
* **Action**:
  * Integrate a logging library like `spdlog` (header-only, easy to integrate with CMake).
  * Replace all instances of `std::cout` and `std::cerr` with structured log messages.
  * Configure the logger in `src/main.cpp` to output JSON-formatted logs to `stdout`/`stderr`.
* **Rationale**: Structured logs are machine-readable and essential for effective log analysis, especially in containerized environments.

### 4. Refine Docker Configuration

* **Objective**: Align the Docker setup with best practices for running a service.
* **Action**:
  * Update the `Dockerfile` to document the required environment variables for configuration.
  * Ensure the `CMD` in the `Dockerfile` runs the `adai_chatbot` executable directly. The container runtime will manage the process.
* **Rationale**: Simplifies the container's responsibility and relies on standard container orchestration features for process management.

### 5. Create a systemd Service File (for non-Docker deployments)

* **Objective**: Enable proper management of the chatbot as a system service on Linux hosts.
* **Action**:
  * Create a `adai.service` file in `scripts/`.
  * The service file should define:
    * The user and group to run the service.
    * The path to the `adai_chatbot` executable.
    * Configuration for automatic restarts on failure.
* **Rationale**: This makes the chatbot a true Linux daemon that can be managed with standard system administration tools like `systemctl`.

## Verification Plan

* **Configuration**: Verify that the service starts correctly using settings from both a configuration file and environment variables.
* **Graceful Shutdown**: Test that sending `SIGTERM` to the process (e.g., `kill <pid>`) results in a clean shutdown and that the model is saved.
* **Logging**: Confirm that the service outputs JSON-formatted logs to standard output.
* **Docker**: Use `docker-compose up -d` to launch the service and check its status and logs (`docker-compose logs adai`).
* **systemd**: On a bare-metal or VM deployment, use `sudo systemctl start adai` to run the service and `journalctl -u adai` to inspect its logs.
