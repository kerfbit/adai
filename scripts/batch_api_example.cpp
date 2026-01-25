/**
 * Batch Processing API Example
 * 
 * Demonstrates the batch processing capabilities of the ChatbotAPI,
 * showing how to process multiple requests efficiently using the
 * /chat/batch and /chat/batch-session endpoints.
 */

#include <iostream>
#include <chrono>
#include <curl/curl.h>
#include <sstream>

// Callback for handling HTTP response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Helper function to make HTTP POST request
std::string make_post_request(const std::string& url, const std::string& json_data) {
    CURL* curl;
    CURLcode res;
    std::string response;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        
        res = curl_easy_perform(curl);
        
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    
    return response;
}

void example_batch_chat() {
    std::cout << "\n=== Example 1: Batch Chat (Stateless) ===" << std::endl;
    std::cout << "Processing 5 independent messages in one batch request..." << std::endl;
    
    std::string url = "http://localhost:8080/chat/batch";
    
    std::string json_request = R"({
        "messages": [
            "What is the capital of France?",
            "Explain quantum computing briefly.",
            "What's the weather like today?",
            "Tell me a joke.",
            "How do I make coffee?"
        ]
    })";
    
    auto start = std::chrono::high_resolution_clock::now();
    std::string response = make_post_request(url, json_request);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Response: " << response << std::endl;
    std::cout << "Time taken: " << duration.count() << "ms" << std::endl;
}

void example_batch_session() {
    std::cout << "\n=== Example 2: Batch Session (Stateful) ===" << std::endl;
    std::cout << "Processing multiple conversations with session tracking..." << std::endl;
    
    std::string url = "http://localhost:8080/chat/batch-session";
    
    // First batch: Initial messages (sessions will be created)
    std::string json_request_1 = R"({
        "messages": [
            "Hi, my name is Alice.",
            "Hi, my name is Bob.",
            "Hi, my name is Charlie."
        ]
    })";
    
    std::cout << "\nBatch 1: Initial greetings" << std::endl;
    std::string response_1 = make_post_request(url, json_request_1);
    std::cout << "Response 1: " << response_1 << std::endl;
    
    // Parse session IDs from response (simplified - in production use proper JSON parser)
    // For demonstration, we'll assume we extracted the session IDs
    
    // Second batch: Follow-up messages using the same sessions
    std::string json_request_2 = R"({
        "messages": [
            "What's my name?",
            "What did I just tell you?",
            "Do you remember my name?"
        ],
        "session_ids": [
            "session_alice_123",
            "session_bob_456", 
            "session_charlie_789"
        ]
    })";
    
    std::cout << "\nBatch 2: Follow-up questions (with session context)" << std::endl;
    std::string response_2 = make_post_request(url, json_request_2);
    std::cout << "Response 2: " << response_2 << std::endl;
}

void example_performance_comparison() {
    std::cout << "\n=== Example 3: Performance Comparison ===" << std::endl;
    std::cout << "Comparing single requests vs batch processing..." << std::endl;
    
    std::vector<std::string> messages = {
        "Question 1: What is AI?",
        "Question 2: What is ML?",
        "Question 3: What is NLP?",
        "Question 4: What is CV?",
        "Question 5: What is RL?"
    };
    
    // Test 1: Sequential single requests
    std::cout << "\nTest 1: Sequential single requests" << std::endl;
    auto start_single = std::chrono::high_resolution_clock::now();
    
    for (const auto& msg : messages) {
        std::string json = "{\"message\":\"" + msg + "\"}";
        make_post_request("http://localhost:8080/chat", json);
    }
    
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single);
    std::cout << "Total time: " << duration_single.count() << "ms" << std::endl;
    std::cout << "Average per request: " << (duration_single.count() / messages.size()) << "ms" << std::endl;
    
    // Test 2: Batch request
    std::cout << "\nTest 2: Batch request" << std::endl;
    auto start_batch = std::chrono::high_resolution_clock::now();
    
    std::ostringstream batch_json;
    batch_json << "{\"messages\":[";
    for (size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) batch_json << ",";
        batch_json << "\"" << messages[i] << "\"";
    }
    batch_json << "]}";
    
    std::string response = make_post_request("http://localhost:8080/chat/batch", batch_json.str());
    
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch);
    std::cout << "Total time: " << duration_batch.count() << "ms" << std::endl;
    std::cout << "Average per request: " << (duration_batch.count() / messages.size()) << "ms" << std::endl;
    
    // Calculate speedup
    float speedup = static_cast<float>(duration_single.count()) / duration_batch.count();
    std::cout << "\nSpeedup with batch processing: " << speedup << "x" << std::endl;
    
    // Show batch statistics from response
    std::cout << "\nBatch statistics: " << response << std::endl;
}

void example_batch_efficiency() {
    std::cout << "\n=== Example 4: Batch Efficiency Metrics ===" << std::endl;
    std::cout << "Demonstrating padding efficiency with dynamic batching..." << std::endl;
    
    std::string url = "http://localhost:8080/chat/batch";
    
    // Messages of varying lengths
    std::string json_request = R"({
        "messages": [
            "Hi",
            "Hello there",
            "Good morning, how are you today?",
            "I have a question about your service",
            "Can you help me understand how batch processing works in this API?",
            "This is a much longer message that will demonstrate how the dynamic batching groups similar length sequences together to minimize padding overhead",
            "Short",
            "Medium length message here",
            "Another very long message that should be grouped with other long messages for efficient processing without too much padding waste"
        ]
    })";
    
    std::string response = make_post_request(url, json_request);
    
    std::cout << "Response with efficiency stats:" << std::endl;
    std::cout << response << std::endl;
    
    std::cout << "\nNote: The 'stats' field shows:" << std::endl;
    std::cout << "  - total_tokens: Total tokens including padding" << std::endl;
    std::cout << "  - actual_tokens: Real tokens (excluding padding)" << std::endl;
    std::cout << "  - padding_ratio: Fraction of tokens that are padding" << std::endl;
    std::cout << "  - num_batches: Number of batches created" << std::endl;
    std::cout << "  - avg_batch_size: Average sequences per batch" << std::endl;
    std::cout << "  - efficiency: (1 - padding_ratio) * 100%" << std::endl;
}

int main() {
    std::cout << "Batch Processing API Examples" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "\nMake sure the ChatbotAPI server is running on port 8080" << std::endl;
    std::cout << "Start with: ./chatbot_api_server --vocab vocab.txt --port 8080" << std::endl;
    
    try {
        // Example 1: Basic batch chat
        example_batch_chat();
        
        // Example 2: Batch with sessions
        example_batch_session();
        
        // Example 3: Performance comparison
        example_performance_comparison();
        
        // Example 4: Efficiency metrics
        example_batch_efficiency();
        
        std::cout << "\n=== All examples completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
