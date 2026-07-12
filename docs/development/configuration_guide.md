# Configuration Guide

The ADAI chatbot service supports multiple configuration methods, allowing flexible deployment in different environments.

## Configuration Priority

Configuration values are loaded with the following priority (highest to lowest):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration file**
4. **Default values** (lowest priority)

This means command-line arguments override everything, environment variables override the config file, and so on.

## Configuration Methods

### 1. Environment Variables

The easiest way to configure the service, especially in Docker/container environments.

```bash
export VOCAB_PATH=/path/to/vocab.txt
export MODEL_PATH=/path/to/model.bin
export PORT=8080
export LOG_LEVEL=INFO
export TEMPERATURE=0.8
./chatbot_api_server
```

Available environment variables:

Server Configuration:

- `VOCAB_PATH` - Path to vocabulary file (required)
- `MODEL_PATH` - Path to model weights file
- `PORT` - Server port (default: 8080)
- `SESSION_TIMEOUT` - Session timeout in minutes (default: 30)
- `LOG_LEVEL` - Logging level: DEBUG, INFO, WARN, ERROR (default: INFO)

Model Architecture:

- `D_MODEL` - Model dimension (default: 512)
- `NUM_HEADS` - Number of attention heads (default: 8)
- `D_FF` - Feed-forward dimension (default: 2048)
- `NUM_ENCODER_LAYERS` - Number of encoder layers (default: 6)
- `NUM_DECODER_LAYERS` - Number of decoder layers (default: 6)
- `MAX_SEQ_LENGTH` - Maximum sequence length (default: 1024)

Generation Parameters:

- `MAX_LENGTH` or `MAX_GEN_LENGTH` - Maximum generation length (default: 100)
- `TEMPERATURE` - Sampling temperature (default: 1.0)
- `TOP_P` - Nucleus sampling threshold (default: 0.9)
- `TOP_K` - Top-k sampling parameter (default: 50)
- `BEAM_WIDTH` - Beam search width (default: 4)
- `STRATEGY` - Generation strategy: greedy, beam, temperature, top_k, nucleus (default: nucleus)

### 2. Configuration File

Use a configuration file for persistent settings. The default location is `/etc/adai/config.conf`.

**Format:** Simple key=value pairs (like .env files)

```bash
# Example: /etc/adai/config.conf
VOCAB_PATH=/app/vocab/vocab.txt
MODEL_PATH=/app/models/model.bin
PORT=8080
LOG_LEVEL=INFO
TEMPERATURE=0.8
```

See `config.conf.example` in the project root for a complete example.

Using a custom config file location:

```bash
./chatbot_api_server --config /path/to/my-config.conf
```

### 3. Command-Line Arguments

Override any configuration with command-line arguments:

```bash
./chatbot_api_server \
  --vocab /path/to/vocab.txt \
  --model /path/to/model.bin \
  --port 8080 \
  --log-level INFO \
  --temperature 0.8 \
  --strategy nucleus
```

Run `./chatbot_api_server --help` for a complete list of available arguments.

## Docker Deployment

### Using docker-compose (Recommended)

The `docker-compose.yml` file is pre-configured with environment variables:

```yaml
environment:
  - VOCAB_PATH=/app/vocab/vocab.txt
  - PORT=8080
  - LOG_LEVEL=INFO
  # ... other settings
```

Edit `docker-compose.yml` to customize, then:

```bash
docker-compose up -d
```

### Using docker run

```bash
docker run -d \
  -e VOCAB_PATH=/app/vocab/vocab.txt \
  -e PORT=8080 \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/vocab:/app/vocab:ro \
  -p 8080:8080 \
  adai-chatbot:latest
```

## Configuration Examples

### Development Environment

Use command-line arguments for quick iteration:

```bash
./chatbot_api_server \
  --vocab ./vocab.txt \
  --port 8080 \
  --log-level DEBUG \
  --temperature 1.2
```

### Production Environment (Bare Metal)

Use a configuration file for stability:

1. Create `/etc/adai/config.conf`:

```text
   VOCAB_PATH=/opt/adai/vocab.txt
   MODEL_PATH=/opt/adai/models/production.bin
   PORT=8080
   LOG_LEVEL=INFO
   SESSION_TIMEOUT=30
   ```

1. Run the service:

   ```bash
   ./chatbot_api_server
   ```

### Production Environment (Docker)

Use environment variables in `docker-compose.yml`:

```yaml
services:
  chatbot-api:
    image: adai-chatbot:latest
    environment:
      - VOCAB_PATH=/app/vocab/vocab.txt
      - MODEL_PATH=/app/models/production.bin
      - PORT=8080
      - LOG_LEVEL=WARN
      - TEMPERATURE=0.7
    volumes:
      - ./models:/app/models:ro
      - ./vocab:/app/vocab:ro
    ports:
      - "8080:8080"
```

## Using the Service Script

The easiest way to run the server is using `model_service.sh`, which automatically uses `config.conf`:

```bash
# Start in background
./scripts/model_service.sh start

# Start in foreground with custom config
./scripts/model_service.sh start --foreground --config my_custom_config.conf
```

See [operations/guides/MODEL_SERVICE_MANAGER.md](../operations/guides/MODEL_SERVICE_MANAGER.md) for full details.

## Constructor Parameter Order

The `EncoderDecoderModel` constructor receives parameters in this order:

```cpp
EncoderDecoderModel(
    vocab_size,          // Vocabulary size
    d_model,             // Model dimension (512)
    encoder_layers,      // Number of encoder layers (6)
    decoder_layers,      // Number of decoder layers (6)
    num_heads,           // Number of attention heads (8)
    d_ff,                // Feed-forward dimension (2048)
    max_seq_length       // Maximum sequence length (1024)
)
```

## Migration from Old Configuration

If you were using command-line arguments exclusively, you can:

1. **Keep using them** - Command-line arguments still work and have the highest priority
2. **Migrate to environment variables** - More suitable for containers
3. **Use a config file** - Better for traditional deployments

## Troubleshooting

### Configuration not loading

Check the startup logs. The service prints the loaded configuration at startup, showing which values are being used.

### Environment variables not working in Docker

Make sure they're defined in `docker-compose.yml` under the `environment:` section or passed with `docker run -e`.

### Config file not found

The default location is `/etc/adai/config.conf`. If the file doesn't exist, it's silently skipped (not an error). You can specify a different location with `--config`.

### Values not being overridden

Remember the priority order: CLI args > env vars > config file > defaults. Check which method has higher priority.
