# Docker Deployment Guide

**ADAI Chatbot API Server - Containerization**
Version 1.0.0
Date: January 25, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Docker Image](#docker-image)
5. [Docker Compose](#docker-compose)
6. [Configuration](#configuration)
7. [Volume Management](#volume-management)
8. [Networking](#networking)
9. [Production Deployment](#production-deployment)
10. [Monitoring and Logging](#monitoring-and-logging)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Overview

The ADAI Chatbot API Server is containerized using Docker for easy deployment, scalability, and portability. This guide covers everything from basic local development to production deployment.

### Key Features

- ✅ **Multi-stage build** - Optimized image size (~200MB runtime)
- ✅ **Non-root user** - Enhanced security
- ✅ **Health checks** - Automatic container health monitoring
- ✅ **Volume mounts** - Persistent model and log storage
- ✅ **Resource limits** - CPU and memory constraints
- ✅ **Reverse proxy** - Nginx integration for production
- ✅ **Security hardening** - SSL/TLS, rate limiting, CORS

### Architecture

```text
┌─────────────────────────────────────────┐
│          Docker Host                    │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │     Nginx (Optional)              │  │
│  │  - Reverse Proxy                  │  │
│  │  - SSL/TLS Termination            │  │
│  │  - Rate Limiting                  │  │
│  └────────────┬──────────────────────┘  │
│               │                         │
│  ┌────────────▼──────────────────────┐  │
│  │   ADAI Chatbot API Container      │  │
│  │  - API Server (Port 8080)         │  │
│  │  - Session Management             │  │
│  │  - Model Inference                │  │
│  └────────────┬──────────────────────┘  │
│               │                         │
│  ┌────────────▼──────────────────────┐  │
│  │      Volume Mounts                │  │
│  │  /app/models  - Model artifacts   │  │
│  │  /app/vocab   - Vocabulary        │  │
│  │  /app/logs    - Application logs  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## Prerequisites

### Required

- **Docker:** Version 20.10 or higher
- **Docker Compose:** Version 1.29 or higher (optional, but recommended)
- **Disk Space:** At least 2GB free space
- **Memory:** Minimum 2GB RAM, recommended 4GB+

### Optional

- **Docker Hub Account:** For pushing images to a registry
- **SSL Certificates:** For HTTPS in production

### Installation

**Ubuntu/Debian:**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add user to docker group (logout and login to take effect)
sudo usermod -aG docker $USER
```

**macOS:**

```bash
# Install Docker Desktop
brew install --cask docker
```

**Verify Installation:**

```bash
docker --version
docker-compose --version
```

---

## Quick Start

### 1. Build the Docker Image

```bash
# Navigate to project root
cd /path/to/adai

# Build using script (recommended)
chmod +x scripts/docker_build.sh
./scripts/docker_build.sh

# Or build directly
docker build -t adai-chatbot:latest .
```

**Build Time:** ~5-10 minutes (first build)

### 2. Run with Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f chatbot-api
```

### 3. Run with Docker Command

```bash
# Create required directories
mkdir -p models logs

# Run container
docker run -d \
  --name adai-chatbot-api \
  -p 8080:8080 \
  -v $(pwd)/vocab.txt:/app/vocab/vocab.txt:ro \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs:rw \
  adai-chatbot:latest
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Send a message
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, how are you?"}'
```

### 5. Stop the Container

```bash
# Using Docker Compose
docker-compose down

# Using Docker command
docker stop adai-chatbot-api
docker rm adai-chatbot-api
```

---

## Docker Image

### Image Structure

The Docker image uses a **multi-stage build** to minimize size:

#### Stage 1: Builder

- Base: `ubuntu:22.04`
- Installs build dependencies (cmake, g++, make)
- Downloads cpp-httplib
- Compiles the API server
- Size: ~1.5GB

#### Stage 2: Runtime

- Base: `ubuntu:22.04`
- Copies only the compiled binary
- Installs minimal runtime dependencies
- Creates non-root user (`adai`)
- Size: ~200MB

### Building the Image

**Using the Build Script:**

```bash
# Build with default settings
./scripts/docker_build.sh

# Build with custom tag
./scripts/docker_build.sh -t v1.0.0

# Build without cache
./scripts/docker_build.sh --no-cache

# Build for specific platform
./scripts/docker_build.sh --platform linux/amd64
```

**Manual Build:**

```bash
# Default build
docker build -t adai-chatbot:latest .

# Build with specific tag
docker build -t adai-chatbot:v1.0.0 .

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t adai-chatbot:latest \
  --push .
```

### Inspecting the Image

```bash
# View image details
docker images adai-chatbot:latest

# Inspect image
docker inspect adai-chatbot:latest

# View image layers
docker history adai-chatbot:latest
```

---

## Docker Compose

### Configuration File

The `docker-compose.yml` file defines the complete deployment stack:

```yaml
version: '3.8'

services:
  chatbot-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: adai-chatbot:latest
    container_name: adai-chatbot-api
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
      - ./vocab:/app/vocab:ro
      - ./logs:/app/logs:rw
    environment:
      - PORT=8080
      - MAX_LENGTH=100
      - TEMPERATURE=1.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Common Commands

```bash
# Start services
docker-compose up -d

# Start with build
docker-compose up -d --build

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f chatbot-api

# Check status
docker-compose ps

# Restart service
docker-compose restart chatbot-api

# Scale service (if configured)
docker-compose up -d --scale chatbot-api=3
```

### Production Profile

Enable Nginx reverse proxy for production:

```bash
# Start with production profile
docker-compose --profile production up -d

# This starts both chatbot-api and nginx services
```

---

## Configuration

### Environment Variables

Configure the container using environment variables:

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `PORT` | `8080` | API server port |
| `MAX_LENGTH` | `100` | Maximum generation length |
| `TEMPERATURE` | `1.0` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling threshold |
| `TOP_K` | `50` | Top-k sampling parameter |
| `BEAM_WIDTH` | `4` | Beam search width |
| `SESSION_TIMEOUT` | `1800` | Session timeout (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |

**Example:**

```bash
docker run -d \
  -e PORT=9090 \
  -e MAX_LENGTH=200 \
  -e TEMPERATURE=0.8 \
  -e SESSION_TIMEOUT=3600 \
  adai-chatbot:latest
```

### Command-Line Arguments

Override default behavior with command-line arguments:

```bash
docker run -d adai-chatbot:latest \
  ./chatbot_api_server \
  --vocab /app/vocab/vocab.txt \
  --port 8080 \
  --max-length 150 \
  --temperature 0.7 \
  --session-timeout 2400
```

### Configuration Files

Mount custom configuration files:

```bash
docker run -d \
  -v /path/to/config.json:/app/config.json:ro \
  adai-chatbot:latest
```

---

## Volume Management

### Volume Types

**1. Read-Only Volumes (Model Artifacts):**

```bash
-v /path/to/models:/app/models:ro
-v /path/to/vocab.txt:/app/vocab/vocab.txt:ro
```

**2. Read-Write Volumes (Logs):**

```bash
-v /path/to/logs:/app/logs:rw
```

**3. Named Volumes (Docker-managed):**

```yaml
volumes:
  model-data:
  logs-data:
```

### Directory Structure

```text
/app/
├── chatbot_api_server    # Binary executable
├── models/               # Model artifacts (mounted)
│   ├── model_v1.bin
│   └── model_v2.bin
├── vocab/                # Vocabulary files (mounted)
│   └── vocab.txt
└── logs/                 # Application logs (mounted)
    ├── api.log
    └── errors.log
```

### Managing Volumes

```bash
# Create named volume
docker volume create adai-models

# List volumes
docker volume ls

# Inspect volume
docker volume inspect adai-models

# Remove unused volumes
docker volume prune

# Backup volume
docker run --rm \
  -v adai-models:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/models-backup.tar.gz /data
```

---

## Networking

### Port Mapping

**Default Port:**

```bash
-p 8080:8080
# Host:Container
```

**Custom Host Port:**

```bash
-p 9090:8080
# Access at http://localhost:9090
```

**Multiple Instances:**

```bash
docker run -d -p 8081:8080 --name api-1 adai-chatbot:latest
docker run -d -p 8082:8080 --name api-2 adai-chatbot:latest
docker run -d -p 8083:8080 --name api-3 adai-chatbot:latest
```

### Docker Networks

**Default Bridge Network:**

```bash
docker network ls
# Containers can communicate using container names
```

**Custom Network:**

```bash
# Create network
docker network create adai-network

# Run container on custom network
docker run -d \
  --network adai-network \
  --name chatbot-api \
  adai-chatbot:latest
```

**Connect Multiple Containers:**

```bash
# Start API server
docker run -d --network adai-network --name chatbot-api adai-chatbot:latest

# Start Nginx on same network
docker run -d --network adai-network --name nginx nginx:alpine
```

### Network Troubleshooting

```bash
# Inspect network
docker network inspect adai-network

# Check container IP
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' chatbot-api

# Test connectivity
docker exec chatbot-api ping nginx
```

---

## Production Deployment

### Production Prerequisites

- **SSL Certificates:** For HTTPS
- **Domain Name:** With DNS configured
- **Load Balancer:** (Optional) For high availability
- **Monitoring Tools:** Prometheus, Grafana, etc.

### SSL Certificate Setup

**Generate Self-Signed Certificate (Testing):**

```bash
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem
```

**Use Let's Encrypt (Production):**

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem docker/nginx/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem docker/nginx/ssl/key.pem
```

### Production Deployment Steps

**1. Prepare Environment:**

```bash
# Create production directory
mkdir -p /opt/adai
cd /opt/adai

# Clone repository
git clone https://github.com/rjv717/adai.git
cd adai

# Set up SSL certificates
mkdir -p docker/nginx/ssl
# Copy your SSL certificates here
```

**2. Build Production Image:**

```bash
# Build with production tag
./scripts/docker_build.sh -t production

# Or with version tag
./scripts/docker_build.sh -t v1.0.0
```

**3. Configure Production Settings:**

```bash
# Edit docker-compose.yml
# Update environment variables
# Configure resource limits
# Enable restart policies
```

**4. Start with Production Profile:**

```bash
# Start all services including Nginx
docker-compose --profile production up -d

# Verify services
docker-compose ps
```

**5. Verify Deployment:**

```bash
# Check health
curl https://yourdomain.com/health

# Test API
curl -X POST https://yourdomain.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'
```

### Load Balancing

**Using Docker Swarm:**

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml adai

# Scale service
docker service scale adai_chatbot-api=3
```

**Using External Load Balancer (AWS ELB, etc.):**

```text
Internet → Load Balancer → Multiple Docker Hosts → Containers
```

### High Availability Setup

```yaml
# docker-compose.yml for HA
version: '3.8'

services:
  chatbot-api:
    image: adai-chatbot:production
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

---

## Monitoring and Logging

### Container Logs

**View Real-time Logs:**

```bash
# All logs
docker logs -f adai-chatbot-api

# Last 100 lines
docker logs --tail 100 adai-chatbot-api

# Since timestamp
docker logs --since 2026-01-25T00:00:00 adai-chatbot-api
```

**Log Rotation:**

```bash
docker run -d \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  adai-chatbot:latest
```

### Application Logs

Logs are stored in the mounted volume:

```bash
# View application logs
tail -f logs/api.log

# Search for errors
grep ERROR logs/api.log

# Analyze access patterns
cat logs/api.log | grep POST | wc -l
```

### Health Monitoring

**Health Check Endpoint:**

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' adai-chatbot-api

# Manual health check
curl http://localhost:8080/health
```

**Automated Monitoring:**

```bash
# Simple monitoring script
#!/bin/bash
while true; do
  if ! curl -sf http://localhost:8080/health > /dev/null; then
    echo "Health check failed!"
    docker restart adai-chatbot-api
  fi
  sleep 60
done
```

### Performance Metrics

**Container Stats:**

```bash
# Real-time stats
docker stats adai-chatbot-api

# One-time stats
docker stats --no-stream adai-chatbot-api
```

**Resource Usage:**

```bash
# CPU and memory usage
docker inspect -f '{{.State.Pid}}' adai-chatbot-api | xargs ps -o %cpu,%mem,cmd -p
```

### Integration with Monitoring Tools

**Prometheus:**

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

**Grafana:**

```yaml
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker logs adai-chatbot-api

# Common causes:
# - Port already in use
# - Missing vocabulary file
# - Insufficient resources
```

**Solution:**

```bash
# Check port availability
sudo netstat -tulpn | grep 8080

# Verify volume mounts
docker inspect adai-chatbot-api | grep Mounts -A 20

# Check resource limits
docker stats
```

#### 2. Health Check Failing

```bash
# Check health status
docker inspect --format='{{json .State.Health}}' adai-chatbot-api

# Test endpoint manually
docker exec adai-chatbot-api curl localhost:8080/health
```

**Solution:**

```bash
# Increase health check timeout
docker run -d \
  --health-timeout=30s \
  adai-chatbot:latest
```

#### 3. Permission Denied

```bash
# Error: Permission denied accessing /app/logs
```

**Solution:**

```bash
# Fix permissions on host
chmod 755 logs/
chown -R 1000:1000 logs/

# Or run with correct user
docker run -d \
  --user 1000:1000 \
  adai-chatbot:latest
```

#### 4. Out of Memory

```bash
# Container killed: OOM
```

**Solution:**

```bash
# Increase memory limit
docker run -d \
  --memory=4g \
  --memory-swap=4g \
  adai-chatbot:latest
```

#### 5. Network Issues

```bash
# Cannot connect to container
```

**Solution:**

```bash
# Check container is running
docker ps | grep adai-chatbot

# Check port mapping
docker port adai-chatbot-api

# Test from inside container
docker exec adai-chatbot-api curl localhost:8080/health
```

### Debugging Commands

```bash
# Enter container shell
docker exec -it adai-chatbot-api /bin/bash

# Check process status
docker exec adai-chatbot-api ps aux

# Check network connectivity
docker exec adai-chatbot-api netstat -tulpn

# View environment variables
docker exec adai-chatbot-api printenv

# Check disk usage
docker exec adai-chatbot-api df -h
```

### Log Analysis

```bash
# Search for specific errors
docker logs adai-chatbot-api 2>&1 | grep -i error

# Count error occurrences
docker logs adai-chatbot-api 2>&1 | grep -c "error"

# Export logs for analysis
docker logs adai-chatbot-api > debug.log 2>&1
```

---

## Best Practices

### Security

1. **Run as Non-Root User**
   - ✅ Already implemented in Dockerfile
   - User: `adai` (UID 1000)

2. **Limit Container Capabilities**

   ```bash
   docker run -d \
     --cap-drop=ALL \
     --cap-add=NET_BIND_SERVICE \
     adai-chatbot:latest
   ```

3. **Use Read-Only Root Filesystem**

   ```bash
   docker run -d \
     --read-only \
     --tmpfs /tmp \
     adai-chatbot:latest
   ```

4. **Scan Images for Vulnerabilities**

   ```bash
   docker scan adai-chatbot:latest
   ```

5. **Keep Base Images Updated**

   ```bash
   # Rebuild regularly
   ./scripts/docker_build.sh --no-cache
   ```

### Performance

1. **Use Multi-Stage Builds**
   - ✅ Already implemented
   - Reduces image size by ~85%

2. **Optimize Layer Caching**
   - Copy dependency files first
   - Copy source code last

3. **Set Resource Limits**

   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 4G
   ```

4. **Enable Health Checks**
   - ✅ Already configured
   - Ensures automatic recovery

5. **Use Volume Mounts for Data**
   - Don't store data in containers
   - Use volumes for persistence

### Operations

1. **Use Docker Compose for Orchestration**
   - Easier management
   - Reproducible deployments

2. **Implement Logging Strategy**
   - Centralized logging
   - Log rotation
   - Structured logs (JSON)

3. **Tag Images Properly**

   ```bash
   # Use semantic versioning
   docker tag adai-chatbot:latest adai-chatbot:v1.0.0
   docker tag adai-chatbot:latest adai-chatbot:production
   ```

4. **Regular Backups**

   ```bash
   # Backup volumes
   docker run --rm \
     -v adai-models:/data \
     -v $(pwd):/backup \
     ubuntu tar czf /backup/backup.tar.gz /data
   ```

5. **Monitor Resource Usage**
   - Set up alerts
   - Track trends
   - Plan capacity

### Development Workflow

1. **Local Development**

   ```bash
   # Use volume mounts for live code updates
   docker run -d \
     -v $(pwd)/src:/app/src:ro \
     adai-chatbot:dev
   ```

2. **Testing**

   ```bash
   # Run tests in container
   docker run --rm \
     adai-chatbot:latest \
     ./run_tests.sh
   ```

3. **CI/CD Integration**

   ```yaml
   # .github/workflows/docker.yml
   - name: Build Docker image
     run: docker build -t adai-chatbot:${{ github.sha }} .

   - name: Push to registry
     run: docker push adai-chatbot:${{ github.sha }}
   ```

---

## Additional Resources

### Documentation

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [ADAI REST API Documentation](../api/rest-api.md)
- [ADAI Quick Start Guide](../api/README.md)

### Tools

- **Docker Desktop:** GUI for Docker management
- **Portainer:** Web-based Docker management UI
- **Watchtower:** Automatic container updates
- **ctop:** Top-like interface for containers

### Community

- [Docker Community Forums](https://forums.docker.com/)
- [Stack Overflow - Docker](https://stackoverflow.com/questions/tagged/docker)

---

## Summary

This guide covered comprehensive Docker deployment for the ADAI Chatbot API Server:

- ✅ **Quick Start:** Get running in minutes
- ✅ **Image Management:** Build, inspect, optimize
- ✅ **Docker Compose:** Orchestrate multi-container deployments
- ✅ **Configuration:** Environment variables, volumes, networking
- ✅ **Production:** SSL, load balancing, high availability
- ✅ **Monitoring:** Logs, health checks, metrics
- ✅ **Troubleshooting:** Common issues and solutions
- ✅ **Best Practices:** Security, performance, operations

**Next Steps:**

1. Build your Docker image: `./scripts/docker_build.sh`
2. Test locally: `docker-compose up -d`
3. Configure for production: Update `docker-compose.yml`
4. Deploy: Follow production deployment steps
5. Monitor: Set up logging and health checks

---

**Document Version:** 1.0.0
**Last Updated:** January 25, 2026
**Maintainer:** ADAI Project Team
