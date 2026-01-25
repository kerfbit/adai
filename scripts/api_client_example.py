#!/usr/bin/env python3
"""
Simple Python client for ADAI Chatbot API

Usage:
    python client_example.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8080"

def health_check():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is healthy")
            print(f"  Active sessions: {data.get('active_sessions', 0)}")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {BASE_URL}")
        print("  Make sure the server is running:")
        print("  ./build/src/chatbot_api_server --vocab vocab.txt --port 8080")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def single_turn_chat(message):
    """Send a single message (no conversation history)"""
    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        json={"message": message}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            return data["response"]
        else:
            return f"Error: {data.get('error', 'Unknown error')}"
    else:
        return f"HTTP Error {response.status_code}"

def multi_turn_chat():
    """Interactive multi-turn conversation"""
    session_id = None
    
    print("\n" + "="*60)
    print("ADAI Chatbot - Multi-Turn Conversation")
    print("="*60)
    print("Type 'exit' to quit, 'clear' to reset conversation\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                if session_id:
                    response = requests.post(
                        f"{BASE_URL}/clear-session",
                        json={"session_id": session_id}
                    )
                    if response.status_code == 200:
                        print("Conversation cleared!\n")
                    session_id = None
                else:
                    print("No active session to clear.\n")
                continue
            
            # Send message
            payload = {"message": user_input}
            if session_id:
                payload["session_id"] = session_id
            
            response = requests.post(
                f"{BASE_URL}/chat/session",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    session_id = data.get("session_id")
                    print(f"Bot: {data['response']}\n")
                else:
                    print(f"Error: {data.get('error', 'Unknown error')}\n")
            else:
                print(f"HTTP Error {response.status_code}\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

def demo_api_calls():
    """Demonstrate various API calls"""
    print("\n" + "="*60)
    print("API Examples")
    print("="*60 + "\n")
    
    # Single-turn examples
    print("1. Single-Turn Chat Examples:")
    print("-" * 40)
    
    questions = [
        "What is machine learning?",
        "Explain transformers in one sentence.",
        "What is 2+2?"
    ]
    
    for q in questions:
        print(f"Q: {q}")
        answer = single_turn_chat(q)
        print(f"A: {answer}\n")
    
    # Multi-turn example
    print("\n2. Multi-Turn Conversation Example:")
    print("-" * 40)
    
    session_id = None
    conversation = [
        "Hi, my name is Alice.",
        "What is my name?",
        "I like Python programming.",
        "What programming language do I like?"
    ]
    
    for msg in conversation:
        payload = {"message": msg}
        if session_id:
            payload["session_id"] = session_id
        
        response = requests.post(
            f"{BASE_URL}/chat/session",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            print(f"You: {msg}")
            print(f"Bot: {data.get('response', 'No response')}\n")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("ADAI Chatbot API Client")
    print("="*60)
    
    # Check server health
    print("\nChecking server health...")
    if not health_check():
        sys.exit(1)
    
    # Show menu
    print("\n" + "="*60)
    print("What would you like to do?")
    print("="*60)
    print("1. Interactive multi-turn chat")
    print("2. Run API examples")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            multi_turn_chat()
        elif choice == "2":
            demo_api_calls()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")

if __name__ == "__main__":
    main()
