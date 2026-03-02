# Multi-stage Dockerfile for ADAI Chatbot API Server
#
# TODO: See TECHNICAL_DEBT.md Future Enhancement #12 - Multi-architecture builds (AMD64, ARM64)
# TODO: See TECHNICAL_DEBT.md Future Enhancement #15 - Distroless runtime image
#
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
RUN mkdir -p /app/models /app/vocab /app/logs /etc/adai && \
    chown -R adai:adai /app

# Copy sample vocabulary and example config
COPY --chown=adai:adai vocab.txt /app/vocab/ 2>/dev/null || true
COPY --chown=adai:adai config.conf.example /etc/adai/config.conf.example

# Switch to non-root user
USER adai

# Expose default API port
EXPOSE 8080

# ============================================================
# ENVIRONMENT VARIABLE CONFIGURATION
# ============================================================
# 
# Configuration Priority (highest to lowest):
# 1. Command-line arguments (--port, --log-level, etc.)
# 2. Environment variables (set below or via docker run -e / docker-compose)
# 3. Configuration file (mounted at /etc/adai/config.conf)
# 4. Default values (shown below)
#
# Override these with docker run: docker run -e LOG_LEVEL=DEBUG adai-chatbot
# Override with docker-compose: See environment section in docker-compose.yml
# ============================================================

# ------------------------------------------------------------
# Server Configuration
# ------------------------------------------------------------

# VOCAB_PATH: Path to vocabulary file (REQUIRED)
# Must point to a valid BPE vocabulary file
ENV VOCAB_PATH=/app/vocab/vocab.txt

# MODEL_PATH: Path to saved model weights (OPTIONAL)
# If not set, model starts with random initialization
# Example: /app/models/trained_model.bin
# ENV MODEL_PATH=

# PORT: HTTP server port (default: 8080)
# The port on which the API server listens
ENV PORT=8080

# SESSION_TIMEOUT: Session timeout in minutes (default: 30)
# How long to keep conversation context in memory
ENV SESSION_TIMEOUT=30

# LOG_LEVEL: Logging verbosity (default: INFO)
# Options: DEBUG, INFO, WARN, ERROR
# DEBUG = most verbose, ERROR = only errors
ENV LOG_LEVEL=INFO

# ------------------------------------------------------------
# Model Architecture Parameters
# ------------------------------------------------------------
# These define the transformer model structure.
# Change only if building a custom model architecture.

# D_MODEL: Model dimension (default: 512)
# Size of the embedding and hidden layers
ENV D_MODEL=512

# NUM_HEADS: Number of attention heads (default: 8)
# Must evenly divide D_MODEL
ENV NUM_HEADS=8

# D_FF: Feed-forward dimension (default: 2048)
# Size of the intermediate feed-forward layer
ENV D_FF=2048

# NUM_ENCODER_LAYERS: Number of encoder layers (default: 6)
ENV NUM_ENCODER_LAYERS=6

# NUM_DECODER_LAYERS: Number of decoder layers (default: 6)
ENV NUM_DECODER_LAYERS=6

# MAX_SEQ_LENGTH: Maximum sequence length (default: 1024)
# Maximum tokens the model can process
ENV MAX_SEQ_LENGTH=1024

# ------------------------------------------------------------
# Text Generation Parameters
# ------------------------------------------------------------
# Control how the model generates responses

# MAX_LENGTH: Maximum generation length (default: 100)
# Maximum number of tokens to generate in a response
ENV MAX_LENGTH=100

# TEMPERATURE: Sampling temperature (default: 1.0)
# Higher = more random, Lower = more deterministic
# Range: 0.1 to 2.0 (typically)
ENV TEMPERATURE=1.0

# TOP_P: Nucleus sampling threshold (default: 0.9)
# Cumulative probability threshold for nucleus sampling
# Range: 0.0 to 1.0
ENV TOP_P=0.9

# TOP_K: Top-k sampling parameter (default: 50)
# Number of top candidates to consider
# Range: 1 to vocabulary size
ENV TOP_K=50

# BEAM_WIDTH: Beam search width (default: 4)
# Number of beams for beam search strategy
ENV BEAM_WIDTH=4

# STRATEGY: Text generation strategy (default: nucleus)
# Options: greedy, beam, temperature, top_k, nucleus
# - greedy: Always pick highest probability token
# - beam: Beam search with BEAM_WIDTH beams
# - temperature: Sample with temperature scaling
# - top_k: Sample from top K tokens
# - nucleus: Sample from top P probability mass
ENV STRATEGY=nucleus

# ============================================================
# END CONFIGURATION
# ============================================================

# Health check endpoint
# Verifies the server is responsive and ready to handle requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the chatbot API server
# The server reads configuration from environment variables automatically
# No configuration file is required, but /etc/adai/config.conf can be mounted
# for file-based configuration if preferred
CMD ["./chatbot_api_server"]

# Metadata
LABEL maintainer="ADAI Project"
LABEL version="1.0.0"
LABEL description="ADAI Transformer-based Chatbot API Server"
LABEL org.opencontainers.image.source="https://github.com/rjv717/adai"
