# Deployment Documentation

Deployment guides for the ADAI Chatbot API Server.

## Available Guides

- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** — Full Docker and Docker Compose deployment guide: multi-stage builds, volume management, Nginx reverse proxy, SSL/TLS, health checks, monitoring, and troubleshooting
- **[docker.md](docker.md)** — Concise Docker deployment reference
- **[SYSTEMD_DEPLOYMENT.md](SYSTEMD_DEPLOYMENT.md)** — Deploying as a managed Linux system service: automated installation, service management, security hardening, resource limits, and log management via journald

## Choosing a Deployment Method

| Scenario | Recommended Approach | Guide |
| --- | --- | --- |
| Local development | Docker Compose | [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) |
| Single server (VPS) | Docker + Nginx reverse proxy | [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) |
| Managed Linux service | systemd | [SYSTEMD_DEPLOYMENT.md](SYSTEMD_DEPLOYMENT.md) |
| Cloud (AWS/GCP/Azure) | Docker + cloud container service | [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) |
| Kubernetes | Docker image + K8s manifests | [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) |

## Quick Start

### Docker

```bash
# Build image
./scripts/docker_build.sh

# Deploy
docker-compose up -d

# Verify
curl http://localhost:8080/health
```

### systemd

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_API_SERVER=ON
make chatbot_api_server -j$(nproc)
cd ..

# Install and start service
sudo ./scripts/install_systemd_service.sh

# Verify
systemctl status adai
curl http://localhost:8080/health
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Single-turn chat |
| `POST` | `/chat/session` | Session-based chat |
| `POST` | `/clear-session` | Clear session history |

## Default Ports

| Service | Port |
| --- | --- |
| API Server | 8080 |
| Nginx HTTP | 80 |
| Nginx HTTPS | 443 |

## Related Documentation

- **[Operations Manual](../OPERATIONS_MANUAL.md)** — Consolidated reference covering all deployment topics
- **[Operations Documentation](../README.md)** — Top-level operations index
- **[Configuration Reference](../OPERATIONS_MANUAL.md#7-configuration-reference)** — All `config.conf` keys
- **[RAG Configuration](../OPERATIONS_MANUAL.md#8-rag-configuration--activation)** — Enabling RAG in deployment
