# Multi-stage Dockerfile for ADAI Chatbot API Server
# Stage 1: Build environment
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    g++ \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy only necessary files for dependency installation
COPY scripts/install_httplib.sh scripts/
RUN chmod +x scripts/install_httplib.sh && \
    ./scripts/install_httplib.sh

# Copy source code and build configuration
COPY CMakeLists.txt CMakePresets.json ./
COPY src/ src/
COPY external/ external/
COPY gtest/ gtest/

# Build the API server
RUN mkdir -p build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_API_SERVER=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DENABLE_ASAN=OFF \
    -DENABLE_UBSAN=OFF \
    -DENABLE_TSAN=OFF && \
    make chatbot_api_server -j$(nproc)

# Stage 2: Runtime environment
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create application user for security
RUN useradd -m -u 1000 -s /bin/bash adai

# Set working directory
WORKDIR /app

# Copy built binary from builder stage
COPY --from=builder /build/build/chatbot_api_server /app/

# Create directories for model artifacts
RUN mkdir -p /app/models /app/vocab /app/logs && \
    chown -R adai:adai /app

# Copy sample vocabulary (if exists)
COPY --chown=adai:adai vocab.txt /app/vocab/ 2>/dev/null || true

# Switch to non-root user
USER adai

# Expose default API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - can be overridden via docker run or docker-compose
CMD ["./chatbot_api_server", \
     "--vocab", "/app/vocab/vocab.txt", \
     "--port", "8080", \
     "--max-length", "100", \
     "--temperature", "1.0"]

# Metadata
LABEL maintainer="ADAI Project"
LABEL version="1.0.0"
LABEL description="ADAI Transformer-based Chatbot API Server"
LABEL org.opencontainers.image.source="https://github.com/rjv717/adai"
