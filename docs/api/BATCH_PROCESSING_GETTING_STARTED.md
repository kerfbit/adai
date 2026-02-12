# Batch Processing - Getting Started

This guide will get you up and running with batch processing in under 5 minutes.

## Quick Start

### 1. Build the API Server (if not already built)

```bash
cd build
cmake .. -DBUILD_API_SERVER=ON
make chatbot_api_server
```

### 2. Start the Server

```bash
./chatbot_api_server --vocab ../vocab.txt --port 8080
```

### 3. Test Batch Processing

**Using Python:**

```bash
cd scripts
./batch_api_client.py
```

**Using cURL:**

```bash
curl -X POST http://localhost:8080/chat/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "What is AI?",
      "What is machine learning?",
      "What is deep learning?"
    ]
  }'
```

## What You Get

- **2-4x faster** processing than sequential requests
- **20-60% less padding** with intelligent batching
- **Full session support** for multi-turn conversations
- **Real-time statistics** on efficiency

## Examples Included

1. **Python Client** (`scripts/batch_api_client.py`)
   - 5 comprehensive examples
   - Performance comparisons
   - Best practices demonstrations

2. **C++ Client** (`scripts/batch_api_example.cpp`)
   - Low-level API usage
   - Performance benchmarking

## Documentation

- **Complete Guide:** [`docs/api/batch-processing.md`](../docs/api/batch-processing.md)
- **Quick Reference:** [`docs/api/batch-processing-quickref.md`](../docs/api/batch-processing-quickref.md)
- **REST API Docs:** [`docs/api/rest-api.md`](../docs/api/rest-api.md)

## Common Use Cases

### Process Multiple User Requests
```python
import requests

messages = [
    "User 1 question",
    "User 2 question",
    "User 3 question"
]

response = requests.post('http://localhost:8080/chat/batch',
                        json={'messages': messages})

for i, resp in enumerate(response.json()['responses'], 1):
    print(f"User {i}: {resp}")
```

### Maintain Conversation Context
```python
# First batch
result = requests.post('http://localhost:8080/chat/batch-session',
    json={'messages': ['Hi!', 'Hello!']})

session_ids = result.json()['session_ids']

# Follow-up batch
result = requests.post('http://localhost:8080/chat/batch-session',
    json={
        'messages': ['How are you?', 'What can you do?'],
        'session_ids': session_ids
    })
```

## Performance Tips

1. **Batch 10-50 requests** for optimal throughput
2. **Group similar-length messages** for better efficiency
3. **Monitor the stats field** to track efficiency
4. **Aim for >60% efficiency** by optimizing batch composition

## Next Steps

1. ✅ Run the example client: `./scripts/batch_api_client.py`
2. ✅ Read the full guide: `docs/api/batch-processing.md`
3. ✅ Integrate into your application
4. ✅ Monitor performance metrics
5. ✅ Optimize based on stats

## Need Help?

- Check the troubleshooting section in [`docs/api/batch-processing.md`](../docs/api/batch-processing.md#troubleshooting)
- Review the examples in `scripts/`
- Examine the implementation in `src/ChatbotAPI.cpp`

---

**Happy Batching! 🚀**
