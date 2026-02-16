# Batch Processing Quick Reference & Getting Started

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

## Endpoints

### POST /chat/batch

Process multiple messages (stateless)

```bash
curl -X POST http://localhost:8080/chat/batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Q1", "Q2", "Q3"]}'
```

### POST /chat/batch-session

Process multiple messages with sessions (stateful)

```bash
curl -X POST http://localhost:8080/chat/batch-session \
  -H "Content-Type: application/json" \
  -d '{
    "messages": ["Hello", "How are you?"],
    "session_ids": ["sid_1", "sid_2"]
  }'
```

## Python Quick Start

```python
import requests

# Batch chat
response = requests.post('http://localhost:8080/chat/batch', json={
    'messages': ['Question 1?', 'Question 2?', 'Question 3?']
})

result = response.json()
for i, resp in enumerate(result['responses'], 1):
    print(f"{i}. {resp}")

# Check efficiency
print(f"Efficiency: {result['stats']['efficiency']:.1f}%")
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

- **Complete Guide:** [batch-processing.md](batch-processing.md)
- **REST API Docs:** [rest-api.md](rest-api.md)

## Response Format

```json
{
  "success": true,
  "responses": ["Answer 1", "Answer 2", "Answer 3"],
  "session_ids": ["sid_1", "sid_2", "sid_3"],
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

## Best Practices

1. **Batch size:** 10-50 messages optimal
2. **Length uniformity:** Group similar-length messages
3. **Monitor efficiency:** Aim for >60% efficiency
4. **Error handling:** Check `success` field

## Common Patterns

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

### Process Multiple Users with Sessions

```python
messages = [user1_msg, user2_msg, user3_msg]
session_ids = [user1_id, user2_id, user3_id]

result = requests.post('http://localhost:8080/chat/batch-session', json={
    'messages': messages,
    'session_ids': session_ids
})
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

### Handle New Conversations

```python
# Omit session_ids for new conversations
result = requests.post('http://localhost:8080/chat/batch-session', json={
    'messages': ['Hello!', 'Hi there!', 'Good morning!']
})

# Get auto-generated session IDs
new_session_ids = result.json()['session_ids']
```

## Performance Tips

1. **Batch 10-50 requests** for optimal throughput
2. **Group similar-length messages** for better efficiency
3. **Monitor the stats field** to track efficiency
4. **Aim for >60% efficiency** by optimizing batch composition
5. **Reuse sessions** to maintain context

## Next Steps

1. ✅ Run the example client: `./scripts/batch_api_client.py`
2. ✅ Read the full guide: [batch-processing.md](batch-processing.md)
3. ✅ Integrate into your application
4. ✅ Monitor performance metrics
5. ✅ Optimize based on stats

## Need Help?

- Check the troubleshooting section in [batch-processing.md](batch-processing.md#troubleshooting)
- Review the examples in `scripts/`
- Examine the implementation in `src/ChatbotAPI.cpp`

## See Also

- [Full Batch Processing Documentation](batch-processing.md)
- [REST API Reference](rest-api.md)
- [Example Client Code](../../scripts/batch_api_client.py)
