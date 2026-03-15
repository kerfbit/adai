# Server Configuration Quick Start

## The Parameter Order Issue - FIXED ✅

The parameter order issue in `chatbot_api_server` has been **corrected**. The `EncoderDecoderModel` constructor now receives parameters in the correct order:

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

## Using the Configuration File

### Quick Start

1. **Use the provided config file:**

   ```bash
   ./build/src/chatbot_api_server --config config.conf
   ```

2. **Or copy to system location:**

   ```bash
   sudo mkdir -p /etc/adai
   sudo cp config.conf /etc/adai/config.conf
   ./build/src/chatbot_api_server
   ```

### Using the Service Script (Recommended)

The easiest way to run the server is using the `model_service.sh` helper script, which automatically uses `config.conf` and handles background execution:

```bash
# Start in background
./scripts/model_service.sh start

# Start in foreground with custom config
./scripts/model_service.sh start --foreground --config my_custom_config.conf
```

See [docs/operations/MODEL_SERVICE_MANAGER.md](docs/operations/MODEL_SERVICE_MANAGER.md) for full details.

### Configuration File Locations

The server looks for configuration in this order:

1. Custom path via `--config` flag
2. Default path: `/etc/adai/config.conf`
3. Environment variables
4. Built-in defaults

### Provided Configuration Files

- **`config.conf`** - Ready-to-use production configuration
- **`config.conf.example`** - Documented example with all options

### Verifying the Configuration

Start the server and check the configuration output:

```bash
./build/src/chatbot_api_server --config config.conf
```

You should see:

```text
Loading configuration from: config.conf
==================================================
         ADAI Chatbot Service Configuration
==================================================
Server Settings:
  Model path:       <new model>
  Vocabulary:       /home/rodney/Repos/adai/vocab.txt
  Port:             8080
  Session timeout:  30 minutes
  Log level:        INFO

Model Architecture:
  d_model:          512      ← Correct
  num_heads:        8         ← Correct
  d_ff:             2048
  encoder_layers:   6
  decoder_layers:   6
  max_seq_length:   1024
```

### Customizing the Configuration

Edit `config.conf` to adjust settings:

```bash
# Change port
PORT=9090

# Adjust generation parameters
TEMPERATURE=0.7
STRATEGY=greedy

# Set model path
MODEL_PATH=/path/to/trained/model.bin
```

### Environment Variable Override

You can override any config file setting with environment variables:

```bash
export PORT=9090
export LOG_LEVEL=DEBUG
./build/src/chatbot_api_server --config config.conf
```

Priority order: **CLI args > Env vars > Config file > Defaults**

### Docker Usage

With Docker, use environment variables in `docker-compose.yml` or mount the config file:

```yaml
services:
  chatbot-api:
    volumes:
      - ./config.conf:/etc/adai/config.conf:ro
```

## Verification

✅ **Parameter order:** FIXED
✅ **Config file:** `config.conf` created
✅ **Server startup:** Verified working
✅ **Model initialization:** Parameters correct (d_model=512, num_heads=8)

## Next Steps

1. **Start the server:**

   ```bash
   ./build/src/chatbot_api_server --config config.conf
   ```

2. **Test with curl:**

   ```bash
   curl -X POST http://localhost:8080/health
   ```

3. **For production, adjust:**
   - `VOCAB_PATH` to your vocabulary location
   - `MODEL_PATH` to your trained model
   - `PORT` if needed
   - `LOG_LEVEL` (INFO for production, DEBUG for development)
