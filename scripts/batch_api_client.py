#!/usr/bin/env python3

# @adai-status: beta        (documented example client, not a maintained production tool)
# @adai-version: 0.7.0
# @adai-reviewed: 2026-09-07

"""
Batch Processing API Client Example

Demonstrates batch processing capabilities of the ChatbotAPI using Python.
Provides examples of /chat/batch and /chat/batch-session endpoints with
performance comparisons and efficiency analysis.
"""

import requests
import time
import json
from typing import List, Dict, Any


class BatchChatbotClient:
    """Client for interacting with the Chatbot API with batch processing support"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def chat(self, message: str) -> Dict[str, Any]:
        """Single message chat (stateless)"""
        url = f"{self.base_url}/chat"
        payload = {"message": message}
        response = self.session.post(url, json=payload)
        return response.json()
    
    def chat_session(self, message: str, session_id: str = "") -> Dict[str, Any]:
        """Single message with session (stateful)"""
        url = f"{self.base_url}/chat/session"
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        response = self.session.post(url, json=payload)
        return response.json()
    
    def batch_chat(self, messages: List[str]) -> Dict[str, Any]:
        """Batch chat processing (stateless)"""
        url = f"{self.base_url}/chat/batch"
        payload = {"messages": messages}
        response = self.session.post(url, json=payload)
        return response.json()
    
    def batch_chat_session(self, messages: List[str], session_ids: List[str] = None) -> Dict[str, Any]:
        """Batch chat with sessions (stateful)"""
        url = f"{self.base_url}/chat/batch-session"
        payload = {"messages": messages}
        if session_ids:
            payload["session_ids"] = session_ids
        response = self.session.post(url, json=payload)
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check server health"""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        return response.json()


def example_1_basic_batch():
    """Example 1: Basic batch processing"""
    print("\n" + "="*60)
    print("Example 1: Basic Batch Chat Processing")
    print("="*60)
    
    client = BatchChatbotClient()
    
    messages = [
        "What is the capital of France?",
        "Explain quantum computing briefly.",
        "What's the weather like today?",
        "Tell me a joke.",
        "How do I make coffee?"
    ]
    
    print(f"\nSending {len(messages)} messages in one batch request...")
    
    start_time = time.time()
    result = client.batch_chat(messages)
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.3f} seconds")
    
    if result.get("success"):
        print(f"\nReceived {len(result['responses'])} responses:")
        for i, (msg, resp) in enumerate(zip(messages, result['responses']), 1):
            print(f"\n{i}. Q: {msg}")
            print(f"   A: {resp}")
        
        # Show batch statistics
        if 'stats' in result:
            stats = result['stats']
            print(f"\nBatch Efficiency Statistics:")
            print(f"  Total tokens (with padding): {stats['total_tokens']}")
            print(f"  Actual tokens: {stats['actual_tokens']}")
            print(f"  Padding ratio: {stats['padding_ratio']:.2%}")
            print(f"  Number of batches: {stats['num_batches']}")
            print(f"  Average batch size: {stats['avg_batch_size']:.1f}")
            print(f"  Efficiency: {stats['efficiency']:.1f}%")
    else:
        print(f"Error: {result.get('error')}")


def example_2_batch_sessions():
    """Example 2: Batch processing with sessions"""
    print("\n" + "="*60)
    print("Example 2: Batch Chat with Session Management")
    print("="*60)
    
    client = BatchChatbotClient()
    
    # First batch: Initial greetings
    print("\nBatch 1: Initial greetings")
    messages_1 = [
        "Hi, my name is Alice.",
        "Hi, my name is Bob.",
        "Hi, my name is Charlie."
    ]
    
    result_1 = client.batch_chat_session(messages_1)
    
    if result_1.get("success"):
        print(f"Created {len(result_1['session_ids'])} sessions:")
        for msg, resp, sid in zip(messages_1, result_1['responses'], result_1['session_ids']):
            print(f"\n  User: {msg}")
            print(f"  Bot: {resp}")
            print(f"  Session ID: {sid}")
        
        # Second batch: Follow-up questions using same sessions
        print("\n\nBatch 2: Follow-up questions (with context)")
        messages_2 = [
            "What's my name?",
            "What did I just tell you?",
            "Do you remember my name?"
        ]
        
        session_ids = result_1['session_ids']
        result_2 = client.batch_chat_session(messages_2, session_ids)
        
        if result_2.get("success"):
            for msg, resp in zip(messages_2, result_2['responses']):
                print(f"\n  User: {msg}")
                print(f"  Bot: {resp}")
    else:
        print(f"Error: {result_1.get('error')}")


def example_3_performance_comparison():
    """Example 3: Performance comparison between single and batch requests"""
    print("\n" + "="*60)
    print("Example 3: Performance Comparison")
    print("="*60)
    
    client = BatchChatbotClient()
    
    messages = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is natural language processing?",
        "What is computer vision?",
        "What is reinforcement learning?",
        "What is deep learning?",
        "What is neural networks?",
        "What is transformers?",
        "What is attention mechanism?",
        "What is BERT?"
    ]
    
    # Test 1: Sequential single requests
    print(f"\nTest 1: Processing {len(messages)} messages sequentially...")
    start_time = time.time()
    
    single_responses = []
    for msg in messages:
        result = client.chat(msg)
        if result.get("success"):
            single_responses.append(result["response"])
    
    single_elapsed = time.time() - start_time
    print(f"Total time: {single_elapsed:.3f}s")
    print(f"Average per request: {single_elapsed/len(messages):.3f}s")
    
    # Test 2: Batch request
    print(f"\nTest 2: Processing {len(messages)} messages in one batch...")
    start_time = time.time()
    
    batch_result = client.batch_chat(messages)
    
    batch_elapsed = time.time() - start_time
    print(f"Total time: {batch_elapsed:.3f}s")
    print(f"Average per request: {batch_elapsed/len(messages):.3f}s")
    
    # Calculate speedup
    speedup = single_elapsed / batch_elapsed if batch_elapsed > 0 else 0
    print(f"\n{'='*40}")
    print(f"Speedup with batch processing: {speedup:.2f}x")
    print(f"Time saved: {single_elapsed - batch_elapsed:.3f}s ({((single_elapsed - batch_elapsed)/single_elapsed)*100:.1f}%)")
    print(f"{'='*40}")


def example_4_variable_length_efficiency():
    """Example 4: Efficiency with variable length messages"""
    print("\n" + "="*60)
    print("Example 4: Batch Efficiency with Variable Length Messages")
    print("="*60)
    
    client = BatchChatbotClient()
    
    # Messages of varying lengths to demonstrate dynamic batching
    messages = [
        "Hi",
        "Hello there",
        "Good morning, how are you doing today?",
        "I have a question about your batch processing capabilities",
        "Can you explain in detail how the dynamic batching algorithm groups similar length sequences together to minimize padding overhead and improve computational efficiency?",
        "This is an extremely long message that should demonstrate how the system handles very long inputs efficiently by grouping them with other long messages, thereby reducing the amount of padding needed and maximizing the utilization of compute resources during batch processing operations.",
        "Short",
        "Medium length message",
        "Another long message that will be grouped with similar length sequences for optimal processing efficiency"
    ]
    
    print(f"\nProcessing {len(messages)} messages of varying lengths:")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. Length {len(msg):3d}: {msg[:50]}{'...' if len(msg) > 50 else ''}")
    
    result = client.batch_chat(messages)
    
    if result.get("success"):
        print(f"\n✓ Successfully processed all messages")
        
        if 'stats' in result:
            stats = result['stats']
            print(f"\nBatch Efficiency Metrics:")
            print(f"  {'Total tokens (with padding):':<30} {stats['total_tokens']:>10,}")
            print(f"  {'Actual tokens:':<30} {stats['actual_tokens']:>10,}")
            print(f"  {'Padding tokens:':<30} {stats['total_tokens'] - stats['actual_tokens']:>10,}")
            print(f"  {'Padding ratio:':<30} {stats['padding_ratio']:>10.1%}")
            print(f"  {'Number of batches created:':<30} {stats['num_batches']:>10}")
            print(f"  {'Average batch size:':<30} {stats['avg_batch_size']:>10.1f}")
            print(f"  {'Overall efficiency:':<30} {stats['efficiency']:>10.1f}%")
            
            print(f"\n💡 Insights:")
            if stats['padding_ratio'] < 0.2:
                print("   Excellent batching efficiency! Low padding overhead.")
            elif stats['padding_ratio'] < 0.4:
                print("   Good batching efficiency. Reasonable padding overhead.")
            else:
                print("   High padding ratio. Consider more uniform message lengths.")
    else:
        print(f"Error: {result.get('error')}")


def example_5_real_world_scenario():
    """Example 5: Real-world customer support scenario"""
    print("\n" + "="*60)
    print("Example 5: Real-World Customer Support Batch Processing")
    print("="*60)
    
    client = BatchChatbotClient()
    
    # Simulate batch of customer queries arriving at once
    customer_queries = [
        "How do I reset my password?",
        "What are your business hours?",
        "I need help with my order #12345",
        "Do you offer international shipping?",
        "Can I return an item after 30 days?",
        "What payment methods do you accept?",
        "Is there a warranty on your products?",
        "How do I track my shipment?",
    ]
    
    print(f"\nProcessing {len(customer_queries)} customer queries...")
    print("This simulates handling multiple customer requests simultaneously\n")
    
    start_time = time.time()
    result = client.batch_chat(customer_queries)
    elapsed = time.time() - start_time
    
    if result.get("success"):
        print(f"✓ All queries processed in {elapsed:.3f}s\n")
        
        for i, (query, response) in enumerate(zip(customer_queries, result['responses']), 1):
            print(f"{'─'*60}")
            print(f"Query #{i}: {query}")
            print(f"Response: {response}")
        
        print(f"{'─'*60}")
        print(f"\nCustomer Support Metrics:")
        print(f"  Total queries handled: {len(customer_queries)}")
        print(f"  Average response time: {elapsed/len(customer_queries):.3f}s per query")
        print(f"  Throughput: {len(customer_queries)/elapsed:.1f} queries/second")
    else:
        print(f"Error: {result.get('error')}")


def main():
    """Run all examples"""
    print("="*60)
    print("Batch Processing API Client Examples")
    print("="*60)
    print("\nMake sure the ChatbotAPI server is running:")
    print("  ./chatbot_api_server --vocab vocab.txt --port 8080\n")
    
    client = BatchChatbotClient()
    
    # Check server health
    try:
        health = client.health()
        print(f"✓ Server Status: {health.get('status', 'unknown')}")
        print(f"  Active sessions: {health.get('active_sessions', 0)}\n")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to server. Is it running?")
        return
    
    try:
        # Run examples
        example_1_basic_batch()
        example_2_batch_sessions()
        example_3_performance_comparison()
        example_4_variable_length_efficiency()
        example_5_real_world_scenario()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
