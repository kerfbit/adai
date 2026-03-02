# Docker Deployment Guide

## Overview

This guide covers deploying the ADAI Chatbot API Server using Docker and Docker Compose. The containerized deployment provides isolation, reproducibility, and simplified management for both development and production environments.

## Quick Start

### Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 1.29+ (v2 recommended)
- At least 4GB RAM available for the container
- Vocabulary file: `vocab.txt`

### Basic Deployment

```bash
# 1. Build the Docker image
docker-compose build

# 2. Start the service
docker-compose up -d

# 3. Check logs
docker-compose logs -f chatbot-api

# 4. Test the API
curl http://localhost:8080/health
```

## Configuration

### Configuration Priority

The chatbot service loads configuration in the following order (highest to lowest priority):

1. **Command-line arguments** (passed via `command:` in docker-compose.yml)
2. **Environment variables** (set in docker-compose.yml or via `docker run -e`)
3. **Configuration file** (mounted at `/etc/adai/config.conf`)
4. **Default values** (defined in the application)

### Environment Variables

All configuration can be controlled via environment variables. See `docker-compose.yml` for the complete list.

#### Server Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VOCAB_PATH` | **Yes** | - | Path to BPE vocabulary file |
| `MODEL_PATH` | No | - | Path to pretrained model weights |
| `PORT` | No | 8080 | HTTP server listening port |
| `SESSION_TIMEOUT` | No | 30 | Session timeout in minutes |
| `LOG_LEVEL` | No | INFO | Logging verbosity: DEBUG, INFO, WARN, ERROR |

#### Model Architecture

| Variable | Default | Description |
|----------|---------|-------------|
| `D_MODEL` | 512 | Model embedding dimension |
| `NUM_HEADS` | 8 | Number of attention heads (must divide D_MODEL) |
| `D_FF` | 2048 | Feed-forward network dimension |
| `NUM_ENCODER_LAYERS` | 6 | Number of encoder layers |
| `NUM_DECODER_LAYERS` | 6 | Number of decoder layers |
| `MAX_SEQ_LENGTH` | 1024 | Maximum sequence length in tokens |

#### Text Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_LENGTH` | 100 | Maximum tokens to generate per response |
| `TEMPERATURE` | 1.0 | Sampling temperature (0.1-2.0) |
| `TOP_P` | 0.9 | Nucleus sampling threshold (0.0-1.0) |
| `TOP_K` | 50 | Top-k sampling candidates |
| `BEAM_WIDTH` | 4 | Beam search width |
| `STRATEGY` | nucleus | Generation strategy: greedy, beam, temperature, top_k, nucleus |

### Using a Configuration File

Instead of environment variables, you can mount a configuration file:

**docker-compose.yml:**
```yaml
volumes:
  - ./config.conf:/etc/adai/config.conf:ro
```

**config.conf:**
```ini
VOCAB_PATH=/app/vocab/vocab.txt
LOG_LEVEL=INFO
PORT=8080
TEMPERATURE=0.7
STRATEGY=nucleus
```

See [config.conf.example](../../config.conf.example) for a complete template.

## Volume Mounts

### Required Volumes

```yaml
volumes:
  # Vocabulary file (REQUIRED)
  - ./vocab:/app/vocab:ro
```

Ensure `vocab.txt` exists in the local `./vocab` directory.

### Optional Volumes

```yaml
volumes:
  # Pretrained model weights
  - ./models:/app/models:ro
  
  # Application logs (if file logging is enabled)
  - ./logs:/app/logs:rw
  
  # Custom configuration file
  - ./config.conf:/etc/adai/config.conf:ro
```

## Deployment Scenarios

### Development Environment

For local development with debug logging:

**docker-compose.override.yml:**
```yaml
version: '3.8'

services:
  chatbot-api:
    environment:
      - LOG_LEVEL=DEBUG
    # Mount source code for live development (requires rebuild)
    volumes:
      - ./src:/app/src:ro
```

```bash
docker-compose up --build
```

### Production Environment

Production deployment with resource limits:

**docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  chatbot-api:
    restart: always
    environment:
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### With Nginx Reverse Proxy

The included docker-compose.yml has an nginx service (disabled by default):

```bash
# Enable the nginx profile
docker-compose --profile production up -d
```

This provides:
- SSL/TLS termination
- Rate limiting
- Request buffering
- Static file serving

Configure SSL certificates in `docker/nginx/ssl/`.

## Container Management

### Starting and Stopping

```bash
# Start in background
docker-compose up -d

# Start in foreground (see logs)
docker-compose up

# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Viewing Logs

```bash
# Follow all logs
docker-compose logs -f

# Follow chatbot-api logs only
docker-compose logs -f chatbot-api

# View last 100 lines
docker-compose logs --tail=100 chatbot-api

# Filter by log level
docker-compose logs chatbot-api | grep "\[error\]"
```

### Restarting the Service

```bash
# Restart all services
docker-compose restart

# Restart chatbot-api only
docker-compose restart chatbot-api

# Rebuild and restart
docker-compose up -d --build
```

### Graceful Shutdown

The container handles SIGTERM gracefully:

```bash
# Graceful shutdown (sends SIGTERM)
docker-compose stop

# Force shutdown after 10 seconds
docker-compose stop -t 10
```

Expected log output:
```
[2026-03-01 16:15:17.862] [info] ==================================================
[2026-03-01 16:15:17.862] [info]          Initiating Graceful Shutdown
[2026-03-01 16:15:17.862] [info] ==================================================
[2026-03-01 16:15:17.862] [info] [1/3] API server stopped
[2026-03-01 16:15:17.862] [info] [2/3] Model state: not persisted
[2026-03-01 16:15:17.862] [info] [3/3] Cleaning up resources...
[2026-03-01 16:15:17.862] [info] Graceful shutdown complete
```

## Health Checks

### Container Health Check

Docker automatically monitors container health:

```bash
# View health status
docker ps
# Look for "healthy" or "unhealthy" in STATUS column

# Inspect health check details
docker inspect adai-chatbot-api | jq '.[0].State.Health'
```

### Manual Health Check

```bash
# HTTP health endpoint
curl http://localhost:8080/health

# Expected response
{"status": "healthy"}
```

## Monitoring

### Container Stats

```bash
# Real-time resource usage
docker stats adai-chatbot-api

# Single snapshot
docker stats --no-stream adai-chatbot-api
```

### Log Aggregation

For production deployments, integrate with log aggregation:

**Using Docker logging driver:**
```yaml
services:
  chatbot-api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Using external logging (e.g., Fluentd):**
```yaml
services:
  chatbot-api:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "adai.chatbot"
```

## Networking

### Port Mapping

By default, the API is exposed on `http://localhost:8080`.

To change the host port:

**docker-compose.yml:**
```yaml
ports:
  - "9000:8080"  # Access on http://localhost:9000
```

### Custom Network

```yaml
services:
  chatbot-api:
    networks:
      - adai-network

networks:
  adai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16
```

### Connecting External Services

```bash
# Connect another container to the adai network
docker network connect adai_adai-network my-other-container
```

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker-compose logs chatbot-api
```

**Common issues:**
- Missing vocabulary file: Ensure `vocab.txt` exists in `./vocab`
- Port conflict: Change port mapping if 8080 is in use
- Insufficient memory: Increase Docker memory limit

### Health Check Failing

```bash
# Check if server is listening
docker-compose exec chatbot-api curl -f localhost:8080/health

# Check logs for errors
docker-compose logs chatbot-api | grep "\[error\]"
```

### High Memory Usage

```bash
# Check memory stats
docker stats adai-chatbot-api

# Adjust model parameters to reduce memory:
# - Decrease D_MODEL
# - Decrease NUM_ENCODER_LAYERS/NUM_DECODER_LAYERS
# - Decrease MAX_SEQ_LENGTH
```

### Slow Response Times

- Enable DEBUG logging: `LOG_LEVEL=DEBUG`
- Check generation parameters: Reduce `MAX_LENGTH`
- Monitor CPU usage: Ensure adequate CPU allocation

### Viewing Container Filesystem

```bash
# Execute shell in running container
docker-compose exec chatbot-api /bin/bash

# View files
ls -la /app
cat /etc/adai/config.conf
```

## Building Custom Images

### Development Build

```bash
# Build with development settings
docker build -t adai-chatbot:dev .

# Run with custom settings
docker run -d \
  --name adai-dev \
  -p 8080:8080 \
  -v $(pwd)/vocab:/app/vocab:ro \
  -e LOG_LEVEL=DEBUG \
  adai-chatbot:dev
```

### Multi-Architecture Build

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t adai-chatbot:latest \
  --push \
  .
```

### Optimized Production Build

```bash
# Build with specific version tag
docker build \
  -t adai-chatbot:1.0.0 \
  --build-arg CMAKE_BUILD_TYPE=Release \
  .
```

## Security Best Practices

### Run as Non-Root User

The container already runs as user `adai` (UID 1000). Verify:

```bash
docker-compose exec chatbot-api whoami
# Should output: adai
```

### Read-Only Filesystem

For enhanced security:

```yaml
services:
  chatbot-api:
    read_only: true
    tmpfs:
      - /tmp
```

### Resource Limits

Always set resource limits in production:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Secrets Management

For sensitive configuration:

```bash
# Use Docker secrets (Swarm mode)
echo "secret_value" | docker secret create vocab_path -

# Or environment file
docker-compose --env-file .env.production up -d
```

## Maintenance

### Updating the Service

```bash
# 1. Pull latest changes
git pull

# 2. Rebuild image
docker-compose build

# 3. Recreate containers
docker-compose up -d --force-recreate
```

### Backup and Restore

**Backup model weights:**
```bash
# Copy from container
docker cp adai-chatbot-api:/app/models/model.bin ./backup/

# Or use volume backup
docker run --rm \
  -v adai_model-data:/data \
  -v $(pwd)/backup:/backup \
  ubuntu tar czf /backup/model-backup.tar.gz /data
```

**Restore model weights:**
```bash
# Copy to container
docker cp ./backup/model.bin adai-chatbot-api:/app/models/

# Or restore volume
docker run --rm \
  -v adai_model-data:/data \
  -v $(pwd)/backup:/backup \
  ubuntu tar xzf /backup/model-backup.tar.gz -C /
```

### Cleaning Up

```bash
# Remove stopped containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Prune unused images
docker image prune -a

# Full cleanup (careful!)
docker system prune -a --volumes
```

## Advanced Configuration

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml adai

# Scale service
docker service scale adai_chatbot-api=3

# View services
docker service ls
docker service ps adai_chatbot-api
```

### Using Kubernetes

See [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) for Kubernetes manifests.

### CI/CD Integration

**Example GitLab CI:**
```yaml
build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - docker-compose pull
    - docker-compose up -d
```

## Performance Tuning

### CPU Optimization

The image is built with `-march=native` for the build host. For portability:

```dockerfile
# In Dockerfile, change:
-DCMAKE_CXX_FLAGS="-march=native"
# To:
-DCMAKE_CXX_FLAGS="-march=x86-64-v2"
```

### Memory Optimization

Reduce memory footprint:
- Use smaller model dimensions (`D_MODEL=256`)
- Reduce sequence length (`MAX_SEQ_LENGTH=512`)
- Use greedy decoding (`STRATEGY=greedy`)

### Caching

Enable BuildKit for faster builds:

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with cache
docker build --cache-from adai-chatbot:latest .
```

## Next Steps

- **Production Monitoring**: Set up Prometheus/Grafana for metrics
- **High Availability**: Deploy multiple replicas behind a load balancer  
- **Auto-Scaling**: Configure based on CPU/memory metrics
- **Logging**: Integrate with ELK/Splunk for centralized logging

## References

- [Dockerfile](../../Dockerfile)
- [docker-compose.yml](../../docker-compose.yml)
- [Configuration Guide](CONFIG.md)
- [API Documentation](../README.md)
- [Signal Handling](../development/STEP2_COMPLETE.md)
- [Structured Logging](../development/STEP3_COMPLETE.md)

---

**Step 4: Docker Configuration - COMPLETE ✅**
