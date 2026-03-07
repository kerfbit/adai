# Chatbot REST API Documentation

ADAI Chatbot API Server
Version 1.0.0
Date: January 24, 2026

---

## Overview

The ADAI Chatbot API provides a REST interface for interacting with transformer-based language models. It supports both single-turn and multi-turn conversations with session management, multiple text generation strategies, and configurable parameters.

**NEW in v1.1:** Batch processing endpoints for high-throughput applications! Process multiple requests 2-4x faster. See [Batch Processing Documentation](batch-processing.md).

**Base URL:** `http://localhost:8080` (default)

**Content-Type:** `application/json`

---

## Quick Start

### 1. Install Dependencies

```bash
# Install cpp-httplib (header-only library)
cd /path/to/adai
./scripts/install_httplib.sh
```

### 2. Build the API Server

```bash
cd build
cmake .. -DBUILD_API_SERVER=ON
make chatbot_api_server
```

### 3. Start the Server

```bash
# With a pre-trained model
./chatbot_api_server --model path/to/model.bin --vocab path/to/vocab.txt --port 8080

# With a new model (requires training)
./chatbot_api_server --vocab path/to/vocab.txt --port 8080
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Single-turn chat
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, how are you?"}'
```

---

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check server status and get active session count.

**Request:** None

Response:

```json
{
  "status": "ok",
  "active_sessions": 5
}
```

Status Codes:

- `200 OK` - Server is running

Example:

```bash
curl http://localhost:8080/health
```

---

### 2. Single-Turn Chat

**Endpoint:** `POST /chat`

**Description:** Send a single message and receive a response. No conversation history is maintained.

Request Body:

```json
{
  "message": "What is the capital of France?"
}
```

Response:

```json
{
  "success": true,
  "response": "The capital of France is Paris."
}
```

Error Response:

```json
{
  "success": false,
  "error": "Missing 'message' field in request"
}
```

Status Codes:

- `200 OK` - Success
- `500 Internal Server Error` - Generation failed

Example:

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is machine learning?"}'
```

**Use Case:** Simple Q&A, one-off queries, stateless interactions

---

### 3. Multi-Turn Chat (Session-Based)

**Endpoint:** `POST /chat/session`

**Description:** Send a message within a conversation session. Maintains conversation history across multiple requests.

Request Body:

```json
{
  "session_id": "abc123...",
  "message": "Tell me about transformers."
}
```

Parameters:

- `session_id` (optional): Session identifier. If empty or not provided, a new session is created.
- `message` (required): User message

Response:

```json
{
  "success": true,
  "response": "Transformers are a type of neural network architecture...",
  "session_id": "abc123..."
}
```

Error Response:

```json
{
  "success": false,
  "error": "Missing 'message' field in request"
}
```

Status Codes:

- `200 OK` - Success
- `500 Internal Server Error` - Generation failed

Example (New Session):

```bash
# First message - creates new session
curl -X POST http://localhost:8080/chat/session \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, my name is Alice."}'

# Response includes session_id
# {"success":true,"response":"Hello Alice! How can I help you?","session_id":"a1b2c3d4..."}
```

Example (Continuing Session):

```bash
# Subsequent message - use session_id from previous response
curl -X POST http://localhost:8080/chat/session \
  -H "Content-Type: application/json" \
  -d '{"session_id":"a1b2c3d4...","message":"What is my name?"}'

# Response remembers context
# {"success":true,"response":"Your name is Alice.","session_id":"a1b2c3d4..."}
```

**Use Case:** Chatbots, conversational agents, context-aware interactions

Session Management:

- Sessions automatically expire after 30 minutes of inactivity (configurable with `--timeout` flag)
- Conversation history is limited to 10 messages and 2048 tokens (configurable in code)
- Expired sessions are cleaned up during health checks

---

### 4. Clear Session

**Endpoint:** `POST /clear-session`

**Description:** Clear conversation history for a specific session.

Request Body:

```json
{
  "session_id": "abc123..."
}
```

Response:

```json
{
  "success": true,
  "message": "Session cleared"
}
```

Error Response:

```json
{
  "success": false,
  "error": "Session not found"
}
```

Status Codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid session ID

Example:

```bash
curl -X POST http://localhost:8080/clear-session \
  -H "Content-Type: application/json" \
  -d '{"session_id":"a1b2c3d4..."}'
```

**Use Case:** Reset conversation, start fresh with same session ID

---

### 5. Batch Chat (NEW in v1.1) 🚀

**Endpoint:** `POST /chat/batch`

**Description:** Process multiple messages in a single request for higher throughput. Uses dynamic batching to minimize padding overhead.

Request Body:

```json
{
  "messages": [
    "What is AI?",
    "What is ML?",
    "What is NLP?"
  ]
}
```

Response:

```json
{
  "success": true,
  "responses": [
    "AI is artificial intelligence...",
    "ML is machine learning...",
    "NLP is natural language processing..."
  ],
  "stats": {
    "total_tokens": 150,
    "actual_tokens": 120,
    "padding_ratio": 0.2,
    "num_batches": 1,
    "avg_batch_size": 3.0,
    "efficiency": 80.0
  }
}
```

Benefits:

- 2-4x faster than sequential single requests
- 20-60% reduction in padding overhead
- Automatic dynamic batching by sequence length

**See:** [Batch Processing Documentation](batch-processing.md) for detailed usage and examples.

---

### 6. Batch Session Chat (NEW in v1.1) 🚀

**Endpoint:** `POST /chat/batch-session`

**Description:** Process multiple messages with session support. Maintains conversation history for each session.

Request Body:

```json
{
  "messages": ["Hello", "How are you?"],
  "session_ids": ["user_1", "user_2"]
}
```

Response:

```json
{
  "success": true,
  "responses": ["Hi! How can I help?", "I'm doing well!"],
  "session_ids": ["user_1", "user_2"],
  "stats": {
    "total_tokens": 100,
    "actual_tokens": 85,
    "padding_ratio": 0.15,
    "num_batches": 1,
    "avg_batch_size": 2.0,
    "efficiency": 85.0
  }
}
```

**Note:** If `session_ids` is omitted, new sessions are created automatically.

**See:** [Batch Processing Documentation](batch-processing.md) for detailed usage and examples.

---

## Configuration

### Command-Line Options

```bash
./chatbot_api_server [OPTIONS]

Server Options:
  --port <number>          Port number (default: 8080)
  --timeout <minutes>      Session timeout in minutes (default: 30)

Model Options:
  --model <path>           Path to pre-trained model file (optional)
  --vocab <path>           Path to vocabulary file (required)
  --d-model <number>       Model dimension (default: 512)
  --num-heads <number>     Number of attention heads (default: 8)
  --d-ff <number>          Feed-forward dimension (default: 2048)
  --enc-layers <n>         Number of encoder layers (default: 6)
  --dec-layers <n>         Number of decoder layers (default: 6)
  --max-seq-len <n>        Maximum sequence length (default: 1024)

Generation Options:
  --max-gen-len <n>        Maximum generation length (default: 100)
  --temperature <f>        Generation temperature (default: 1.0)
  --top-p <f>              Nucleus sampling threshold (default: 0.9)
  --strategy <str>         Generation strategy (default: nucleus)
                           Options: greedy, beam, temperature, top_k, nucleus

Help:
  --help, -h               Show help message
```

### Generation Strategies

The API supports multiple text generation strategies (configured at server startup):

1. **Greedy Decoding** (`--strategy greedy`)
   - Selects the most likely token at each step
   - Fast, deterministic
   - May produce repetitive text

2. **Beam Search** (`--strategy beam`)
   - Explores multiple hypotheses
   - Balances quality and diversity
   - Configurable beam width (default: 4)

3. **Temperature Sampling** (`--strategy temperature`)
   - Controls randomness via temperature parameter
   - Higher temperature = more random
   - Lower temperature = more deterministic

4. **Top-k Sampling** (`--strategy top_k`)
   - Samples from k most likely tokens
   - Reduces low-probability tokens
   - Configurable k (default: 50)

5. **Nucleus (Top-p) Sampling** (`--strategy nucleus`) **[DEFAULT]**
   - Samples from smallest set of tokens with cumulative probability >= p
   - Adaptive vocabulary size
   - Configurable p (default: 0.9)

---

## Examples

### Example 1: Simple Q&A Bot

```python
import requests
import json

BASE_URL = "http://localhost:8080"

def ask_question(question):
    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"message": question})
    )
    return response.json()["response"]

# Usage
answer = ask_question("What is the speed of light?")
print(answer)
```

### Example 2: Multi-Turn Conversation

```python
import requests
import json

BASE_URL = "http://localhost:8080"

class ChatSession:
    def __init__(self):
        self.session_id = None

    def send_message(self, message):
        payload = {"message": message}
        if self.session_id:
            payload["session_id"] = self.session_id

        response = requests.post(
            f"{BASE_URL}/chat/session",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        data = response.json()
        self.session_id = data.get("session_id")
        return data["response"]

    def clear(self):
        if self.session_id:
            requests.post(
                f"{BASE_URL}/clear-session",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"session_id": self.session_id})
            )

# Usage
chat = ChatSession()
print(chat.send_message("Hi, I'm working on a Python project."))
print(chat.send_message("Can you help me with error handling?"))
print(chat.send_message("What did I say I was working on?"))
```

### Example 3: Bash Script

```bash
#!/bin/bash

BASE_URL="http://localhost:8080"

# Health check
echo "Checking server health..."
curl -s $BASE_URL/health | jq

# Single-turn chat
echo -e "\nAsking a question..."
curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is AI?"}' | jq

# Multi-turn conversation
echo -e "\nStarting conversation..."
RESPONSE=$(curl -s -X POST $BASE_URL/chat/session \
  -H "Content-Type: application/json" \
  -d '{"message":"My favorite color is blue."}')

echo $RESPONSE | jq

# Extract session_id
SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')

echo -e "\nContinuing conversation..."
curl -s -X POST $BASE_URL/chat/session \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"What is my favorite color?\"}" | jq
```

### Example 4: JavaScript (Node.js)

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8080';

async function singleTurnChat(message) {
    const response = await axios.post(`${BASE_URL}/chat`, {
        message: message
    });
    return response.data.response;
}

async function multiTurnChat() {
    let sessionId = null;

    const sendMessage = async (message) => {
        const payload = { message };
        if (sessionId) payload.session_id = sessionId;

        const response = await axios.post(`${BASE_URL}/chat/session`, payload);
        sessionId = response.data.session_id;
        return response.data.response;
    };

    console.log(await sendMessage("Hello!"));
    console.log(await sendMessage("Tell me a joke."));
    console.log(await sendMessage("Can you explain the previous joke?"));
}

// Run examples
(async () => {
    console.log("Single-turn:", await singleTurnChat("What is 2+2?"));
    await multiTurnChat();
})();
```

---

## Error Handling

### Common Error Responses

Missing Required Field:

```json
{
  "success": false,
  "error": "Missing 'message' field in request"
}
```

Generation Failed:

```json
{
  "success": false,
  "error": "Generation failed: <error details>"
}
```

Session Not Found:

```json
{
  "success": false,
  "error": "Session not found"
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request (missing fields, invalid session)
- `500 Internal Server Error` - Server error (generation failed, model error)

---

## Performance Considerations

### Latency

- **Single-turn chat:** ~100-500ms (depends on generation length and model size)
- **Multi-turn chat:** ~100-500ms + context processing overhead
- **Health check:** <1ms

### Throughput

- The server handles requests sequentially (single-threaded generation)
- For high-throughput scenarios, consider:
  - Running multiple server instances (horizontal scaling)
  - Load balancing across instances
  - Implementing request queuing

### Memory Usage

- Each session stores conversation history (limited to 2048 tokens)
- Active sessions are kept in memory
- Expired sessions are cleaned up during health checks
- Monitor memory usage with large numbers of concurrent sessions

---

## Security Considerations

⚠️ **Important:** This is a development/research implementation. For production deployment:

1. **Authentication/Authorization:** Add API keys, OAuth, or JWT tokens
2. **Rate Limiting:** Implement per-IP or per-user rate limits
3. **Input Validation:** Add stricter input sanitization
4. **HTTPS:** Use TLS/SSL for encrypted communication
5. **CORS:** Configure Cross-Origin Resource Sharing appropriately
6. **Logging:** Add request logging and monitoring
7. **Timeouts:** Implement request timeouts to prevent hanging

---

## Monitoring

### Health Check Endpoint

Use `/health` for monitoring:

```bash
# Add to monitoring script
while true; do
    STATUS=$(curl -s http://localhost:8080/health | jq -r '.status')
    if [ "$STATUS" != "ok" ]; then
        echo "Server unhealthy!"
        # Send alert
    fi
    sleep 60
done
```

### Metrics to Track

- Active sessions (`/health` response)
- Request latency (application-level logging)
- Error rate (failed requests)
- Memory usage (`top`, `htop`, system monitoring)
- CPU usage (generation is CPU-intensive)

---

## Troubleshooting

### Server Won't Start

**Problem:** Port already in use

```text
Failed to start server
```

**Solution:** Use a different port

```bash
./chatbot_api_server --vocab vocab.txt --port 8081
```

**Problem:** cpp-httplib not found

```text
cpp-httplib not found. API server will not be built.
```

**Solution:** Install cpp-httplib

```bash
./scripts/install_httplib.sh
# Then rebuild
cd build && cmake .. -DBUILD_API_SERVER=ON && make chatbot_api_server
```

### Slow Response Times

**Problem:** Generation takes too long

Solutions:

- Reduce `--max-gen-len` (shorter responses)
- Use faster generation strategy (`--strategy greedy`)
- Reduce model size (`--d-model`, `--enc-layers`, `--dec-layers`)
- Consider model optimization (quantization, pruning)

### Out of Memory

**Problem:** Server crashes or high memory usage

Solutions:

- Reduce session timeout (`--timeout 10`)
- Reduce max sequence length (`--max-seq-len 512`)
- Limit concurrent sessions (application-level)
- Reduce model size

### Poor Response Quality

**Problem:** Responses are incoherent or repetitive

Solutions:

- Train model on domain-specific data
- Adjust generation parameters:
  - Increase temperature (`--temperature 1.2`)
  - Use nucleus sampling (`--strategy nucleus --top-p 0.9`)
  - Increase top-p for more diversity (`--top-p 0.95`)
- Increase model capacity (more layers, larger dimension)

---

## Development

### Adding Custom Endpoints

Edit `src/ChatbotAPI.cpp` and `src/ChatbotAPI.hpp`:

```cpp
// In ChatbotAPI.hpp
std::string handle_custom_endpoint(const std::string& request_body);

// In ChatbotAPI.cpp constructor
server_impl_->server.Post("/custom", [this](const httplib::Request& req, httplib::Response& res) {
    std::string response = handle_custom_endpoint(req.body);
    res.set_content(response, "application/json");
    res.status = 200;
});

// Implement handler
std::string ChatbotAPI::handle_custom_endpoint(const std::string& request_body) {
    // Your implementation
    return create_json_response("Custom response");
}
```

### Testing

```bash
# Unit tests (if available)
cd build
ctest

# Manual testing
./chatbot_api_server --vocab ../vocab.txt --port 8080 &
sleep 2
curl http://localhost:8080/health
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message":"test"}'
pkill chatbot_api_server
```

---

## Deployment

### Local Development

```bash
./chatbot_api_server --vocab vocab.txt --port 8080
```

### Production (systemd service)

Create `/etc/systemd/system/chatbot-api.service`:

```ini
[Unit]
Description=ADAI Chatbot API Server
After=network.target

[Service]
Type=simple
User=chatbot
WorkingDirectory=/opt/adai
ExecStart=/opt/adai/chatbot_api_server --model /opt/adai/model.bin --vocab /opt/adai/vocab.txt --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable chatbot-api
sudo systemctl start chatbot-api
sudo systemctl status chatbot-api
```

### Docker

Create `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN ./scripts/install_httplib.sh
RUN mkdir -p build && cd build && \
    cmake .. -DBUILD_API_SERVER=ON -DCMAKE_BUILD_TYPE=Release && \
    make chatbot_api_server

EXPOSE 8080
CMD ["./build/chatbot_api_server", "--vocab", "vocab.txt", "--port", "8080"]
```

Build and run:

```bash
docker build -t adai-chatbot .
docker run -p 8080:8080 -v $(pwd)/model.bin:/app/model.bin adai-chatbot \
    ./build/chatbot_api_server --model model.bin --vocab vocab.txt --port 8080
```

---

## References

- [ADAI Project Repository](https://github.com/rjv717/adai)
- [cpp-httplib Documentation](https://github.com/yhirose/cpp-httplib)
- [Chatbot Completeness Analysis](../reference/chatbot-completeness.md)
- [REST API Best Practices](https://restfulapi.net/)

---

**Last Updated:** January 24, 2026
**Version:** 1.0.0
**Contact:** See project repository for issues and contributions
