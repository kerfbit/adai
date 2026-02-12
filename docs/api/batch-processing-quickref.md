# Batch Processing Quick Reference

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

## Key Benefits

- **2-4x faster** than sequential requests
- **20-60% less padding** with dynamic batching
- **Same results** as single requests
- **Session support** for multi-turn conversations

## Best Practices

1. **Batch size:** 10-50 messages optimal
2. **Length uniformity:** Group similar-length messages
3. **Monitor efficiency:** Aim for >60% efficiency
4. **Error handling:** Check `success` field

## Common Patterns

### Process Multiple Users
```python
messages = [user1_msg, user2_msg, user3_msg]
session_ids = [user1_id, user2_id, user3_id]

result = requests.post('http://localhost:8080/chat/batch-session', json={
    'messages': messages,
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

1. **Batch 10-50 requests** for best throughput
2. **Group by length** for better efficiency
3. **Reuse sessions** to maintain context
4. **Monitor stats** to optimize batching

## See Also

- [Full Batch Processing Documentation](batch-processing.md)
- [REST API Reference](rest-api.md)
- [Example Client Code](../../scripts/batch_api_client.py)
