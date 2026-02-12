# ADAI API Documentation

This directory contains documentation for the ADAI Chatbot API.

## Quick Links

- **[REST API Documentation](rest-api.md)** - Complete API reference with examples
- **[Batch Processing Guide](batch-processing.md)** - Batch processing for API endpoints ✨
- **[Dataset Batch Processing](data/dataset-batch-processing.md)** - Dataset batch processing integration ✨ NEW
- [Build Instructions](#build-instructions)
- [Quick Start](#quick-start)

---

## Build Instructions

### 1. Install cpp-httplib

The API server requires cpp-httplib (header-only HTTP library):

```bash
# Option A: Use provided script (recommended)
./scripts/install_httplib.sh

# Option B: Manual installation (Ubuntu/Debian)
sudo apt-get install libhttplib-dev

# Option C: Download manually
mkdir -p external/cpp-httplib
cd external/cpp-httplib
wget https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.15.3/httplib.h
```

### 2. Build the API Server

```bash
# From project root
mkdir -p build
cd build

# Configure with API server enabled
cmake .. -DBUILD_API_SERVER=ON -DCMAKE_BUILD_TYPE=Release

# Build
make chatbot_api_server

# Verify
./src/chatbot_api_server --help
```

---

## Quick Start

### Start the Server

```bash
# Navigate to build directory
cd build/src

# Start server with vocabulary (creates new model)
./chatbot_api_server --vocab ../../vocab.txt --port 8080

# Or with pre-trained model
./chatbot_api_server \
    --model ../../models/chatbot.bin \
    --vocab ../../vocab.txt \
    --port 8080
```

### Test the API

**Health Check:**

```bash
curl http://localhost:8080/health
```

**Single-Turn Chat:**

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, how are you?"}'
```

**Multi-Turn Chat:**

```bash
# First message (creates session)
curl -X POST http://localhost:8080/chat/session \
  -H "Content-Type: application/json" \
  -d '{"message":"My name is Alice."}'

# Note the session_id in the response, then:
curl -X POST http://localhost:8080/chat/session \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<SESSION_ID>","message":"What is my name?"}'
```

---

## Configuration

### Command-Line Options

**Server:**

- `--port <n>` - Port number (default: 8080)
- `--timeout <n>` - Session timeout in minutes (default: 30)

**Model:**

- `--model <path>` - Pre-trained model file (optional)
- `--vocab <path>` - Vocabulary file (required)
- `--d-model <n>` - Model dimension (default: 512)
- `--num-heads <n>` - Attention heads (default: 8)
- `--d-ff <n>` - Feed-forward dimension (default: 2048)
- `--enc-layers <n>` - Encoder layers (default: 6)
- `--dec-layers <n>` - Decoder layers (default: 6)

**Generation:**

- `--max-gen-len <n>` - Max response length (default: 100)
- `--temperature <f>` - Sampling temperature (default: 1.0)
- `--top-p <f>` - Nucleus sampling threshold (default: 0.9)
- `--strategy <str>` - Strategy: greedy, beam, temperature, top_k, nucleus (default: nucleus)

### Example Configurations

**Development (fast):**

```bash
./chatbot_api_server \
    --vocab vocab.txt \
    --d-model 256 \
    --num-heads 4 \
    --enc-layers 3 \
    --dec-layers 3 \
    --max-gen-len 50 \
    --strategy greedy
```

**Production (quality):**

```bash
./chatbot_api_server \
    --model trained_model.bin \
    --vocab vocab.txt \
    --d-model 768 \
    --num-heads 12 \
    --enc-layers 12 \
    --dec-layers 12 \
    --max-gen-len 200 \
    --temperature 0.9 \
    --top-p 0.95 \
    --strategy nucleus
```

---

## Available Endpoints

| Method | Endpoint | Description |
| -------- | ---------- | ------------- |
| GET | `/health` | Server health check |
| POST | `/chat` | Single-turn conversation |
| POST | `/chat/session` | Multi-turn conversation |
| POST | `/clear-session` | Clear session history |

See [REST API Documentation](rest-api.md) for detailed information.

---

## Client Examples

### Python

```python
import requests

def chat(message, session_id=None):
    url = "http://localhost:8080/chat/session"
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(url, json=payload)
    data = response.json()
    return data["response"], data.get("session_id")

# Usage
response, sid = chat("Hi!")
print(f"Bot: {response}")
response, sid = chat("Tell me a joke.", sid)
print(f"Bot: {response}")
```

### JavaScript

```javascript
const axios = require('axios');

async function chat(message, sessionId = null) {
    const payload = { message };
    if (sessionId) payload.session_id = sessionId;

    const response = await axios.post('http://localhost:8080/chat/session', payload);
    return [response.data.response, response.data.session_id];
}

// Usage
let [response, sid] = await chat("Hi!");
console.log("Bot:", response);
[response, sid] = await chat("Tell me a joke.", sid);
console.log("Bot:", response);
```

### cURL

```bash
# Store session ID
SESSION_ID=$(curl -s -X POST http://localhost:8080/chat/session \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!"}' | jq -r '.session_id')

# Continue conversation
curl -s -X POST http://localhost:8080/chat/session \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"How are you?\"}" | jq
```

---

## Troubleshooting

### Build Issues

**Problem:** cpp-httplib not found

```text
cpp-httplib not found. API server will not be built.
```

**Solution:** Run `./scripts/install_httplib.sh` and rebuild

**Problem:** Linking errors
**Solution:** Ensure all dependencies are built:

```bash
cd build
cmake .. -DBUILD_API_SERVER=ON
make clean
make chatbot_api_server
```

### Runtime Issues

**Problem:** Port already in use
**Solution:** Use different port: `--port 8081`

**Problem:** Slow responses
**Solutions:**

- Reduce `--max-gen-len`
- Use `--strategy greedy`
- Reduce model size (`--d-model`, layers)

**Problem:** High memory usage
**Solutions:**

- Reduce `--timeout` (shorter session lifetime)
- Reduce `--max-seq-len`
- Monitor with `/health` endpoint

---

## Performance Tips

1. **Use greedy decoding** for faster (but less diverse) responses
2. **Reduce model size** for development/testing
3. **Limit session timeout** to free memory
4. **Run multiple instances** for horizontal scaling
5. **Use reverse proxy** (nginx) for load balancing

---

## Next Steps

- Read the [full REST API documentation](rest-api.md)
- Train a custom model (see `chatbot_trainer`)
- Deploy with systemd or Docker (see deployment section in REST API docs)
- Add authentication/rate limiting for production

---

**Related Documentation:**

- [Chatbot Completeness Analysis](../reference/chatbot-completeness.md)
- [Training Guide](../guides/) (TODO)
- [Architecture Overview](../architecture/) (TODO)
