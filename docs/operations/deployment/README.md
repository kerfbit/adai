# Deployment Documentation

Deployment guides for the ADAI system.

## Available Guides

- **[SERVER_BUNDLE_DEPLOYMENT.md](SERVER_BUNDLE_DEPLOYMENT.md)** — Deploying the server infrastructure bundle: metrics API, model name service, and dataset registry as co-located systemd services with SQLite or PostgreSQL persistence, packaging for distribution, and connecting trainers
- **[docker.md](docker.md)** — Docker and Docker Compose deployment guide: multi-stage builds, volume management, Nginx reverse proxy, SSL/TLS, health checks, monitoring, and troubleshooting
- **[SYSTEMD_DEPLOYMENT.md](SYSTEMD_DEPLOYMENT.md)** — Deploying the chatbot API server as a managed Linux system service: automated installation, service management, security hardening, resource limits, and log management via journald
- **[CLOUDFLARE_TUNNEL_RELAY.md](CLOUDFLARE_TUNNEL_RELAY.md)** — Exposing metrics/MNS/registry/chatbot to the Android tablet apps when off the home LAN, via Cloudflare Tunnel + Cloudflare Access service tokens fronted by kerfbit.dev

## Choosing a Deployment Method

| Scenario | Recommended Approach | Guide |
| --- | --- | --- |
| Training infrastructure (metrics, MNS, registry) | Server bundle | [SERVER_BUNDLE_DEPLOYMENT.md](SERVER_BUNDLE_DEPLOYMENT.md) |
| Training infra with PostgreSQL | Server bundle + `--setup-postgres` | [SERVER_BUNDLE_DEPLOYMENT.md](SERVER_BUNDLE_DEPLOYMENT.md) |
| Local development | Docker Compose | [docker.md](docker.md) |
| Single server (VPS) | Docker + Nginx reverse proxy | [docker.md](docker.md) |
| Chatbot API service | systemd | [SYSTEMD_DEPLOYMENT.md](SYSTEMD_DEPLOYMENT.md) |
| Cloud (AWS/GCP/Azure) | Docker + cloud container service | [docker.md](docker.md) |
| Kubernetes | Docker image + K8s manifests | [docker.md](docker.md) |
| Remote/off-LAN access from the Android tablet | Cloudflare Tunnel + Access | [CLOUDFLARE_TUNNEL_RELAY.md](CLOUDFLARE_TUNNEL_RELAY.md) |

## Quick Start

### Server Bundle (Training Infrastructure)

```bash
# Build portable binaries
cmake --preset portable && cmake --build --preset portable

# Package for distribution
./scripts/package_server_bundle.sh --output-dir dist/

# Install on target (SQLite default)
sudo ./scripts/install_server_bundle.sh --build-dir . --yes

# Install with PostgreSQL
sudo ./scripts/install_server_bundle.sh --build-dir . --setup-postgres --yes

# Verify
curl http://localhost:8081/health   # metrics
curl http://localhost:8082/health   # registry
curl http://localhost:8083/health   # MNS
```

### Docker (Chatbot API)

```bash
# Build image
./scripts/docker_build.sh

# Deploy
docker-compose up -d

# Verify
curl http://localhost:8080/health
```

### systemd (Chatbot API)

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_API_SERVER=ON
make chatbot_api_server -j$(nproc)
cd ..

# Install and start service
sudo ./scripts/install_chatbot_API.sh

# Verify
systemctl status adai
curl http://localhost:8080/health
```

## Default Ports

| Service | Port | Install Script |
| --- | --- | --- |
| Chatbot API Server | 8080 | `install_chatbot_API.sh` |
| Metrics API Server | 8081 | `install_server_bundle.sh` |
| Dataset Registry Server | 8082 | `install_server_bundle.sh` |
| Model Name Service | 8083 | `install_server_bundle.sh` |
| Nginx HTTP | 80 | Docker Compose |
| Nginx HTTPS | 443 | Docker Compose |

## Related Documentation

- **[Operations Manual](../OPERATIONS_MANUAL.md)** — Consolidated reference covering all deployment topics
- **[Operations Documentation](../README.md)** — Top-level operations index
- **[Configuration Reference](../OPERATIONS_MANUAL.md#7-configuration-reference)** — All `config.conf` keys
- **[Training Metrics API](../../development/TRAINING_METRICS_API.md)** — Full endpoint reference for metrics_api_server
- **[Model Name Service](../guides/MODEL_NAME_SERVICE.md)** — Full MNS operational manual
- **[RAG Configuration](../OPERATIONS_MANUAL.md#8-rag-configuration--activation)** — Enabling RAG in deployment
