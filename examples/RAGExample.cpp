/**
 * RAG (Retrieval-Augmented Generation) Example
 * 
 * Demonstrates how to use the RAG system for question answering
 * with factual grounding from a document knowledge base.
 * 
 * This example:
 * 1. Loads a pre-trained encoder-decoder model
 * 2. Creates a document store with sample knowledge
 * 3. Sets up RAG inference
 * 4. Answers questions using retrieved context
 * 5. Shows retrieval scores and source documents
 */

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "DocumentStore.hpp"
#include "EncoderDecoderModel.hpp"
#include "RAGInference.hpp"
#include "encoder.hpp"

// ANSI color codes for pretty output
#define RESET "\033[0m"
#define BOLD "\033[1m"
#define CYAN "\033[36m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"

void printHeader(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN << "═══════════════════════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "═══════════════════════════════════════════════════════" << RESET << "\n\n";
}

void printSection(const std::string& title) {
    std::cout << "\n" << BOLD << BLUE << ">>> " << title << RESET << "\n";
}

int main(int argc, char* argv[]) {
    try {
        printHeader("RAG (Retrieval-Augmented Generation) Demo");

        // Configuration
        std::string model_path = "chatbot_model.bin";
        std::string vocab_path = "vocab.txt";
        
        if (argc >= 2) {
            model_path = argv[1];
        }
        if (argc >= 3) {
            vocab_path = argv[2];
        }

        // ====================================================================
        // Step 1: Initialize Model and Encoder
        // ====================================================================
        printSection("Initializing Model and Encoder");
        
        std::cout << "Loading model from: " << model_path << "\n";
        std::cout << "Loading vocabulary from: " << vocab_path << "\n";

        // Create encoder for embeddings
        const int vocab_size = 5000;
        const int d_model = 512;
        const int num_encoder_layers = 6;
        const int num_heads = 8;
        const int d_ff = 2048;
        const int max_seq_len = 512;

        auto encoder = std::make_shared<LLMEncoder>(
            vocab_size, d_model, num_encoder_layers, num_heads, d_ff, max_seq_len
        );
        
        // Load encoder from model file
        try {
            encoder->load_weights(model_path);
            std::cout << GREEN << "✓ Encoder loaded successfully" << RESET << "\n";
        } catch (const std::exception& e) {
            std::cout << YELLOW << "⚠ Could not load encoder, using random initialization" << RESET << "\n";
        }

        // Create full encoder-decoder model
        auto model = std::make_shared<EncoderDecoderModel>(
            vocab_size, d_model, num_encoder_layers, 6, num_heads, d_ff, max_seq_len
        );
        
        try {
            model->load_model(model_path);
            std::cout << GREEN << "✓ Model loaded successfully" << RESET << "\n";
        } catch (const std::exception& e) {
            std::cout << YELLOW << "⚠ Could not load model, using random initialization" << RESET << "\n";
        }

        // ====================================================================
        // Step 2: Create Document Store
        // ====================================================================
        printSection("Building Knowledge Base");

        auto doc_store = std::make_shared<DocumentStore>(encoder);

        // Add sample documents about various topics
        std::vector<std::pair<std::string, std::string>> knowledge_docs = {
            {"ml_definition", 
             "Machine learning is a subset of artificial intelligence that enables "
             "systems to learn and improve from experience without being explicitly "
             "programmed. It focuses on developing computer programs that can access "
             "data and use it to learn for themselves."},
            
            {"dl_definition",
             "Deep learning is a subset of machine learning that uses neural networks "
             "with multiple layers. These deep neural networks can automatically learn "
             "hierarchical representations of data, making them particularly effective "
             "for tasks like image recognition, natural language processing, and speech recognition."},
            
            {"transformer_definition",
             "Transformers are a type of deep learning architecture introduced in 2017 "
             "that relies entirely on attention mechanisms. Unlike recurrent neural networks, "
             "transformers can process all input data in parallel, making them highly efficient "
             "for sequence-to-sequence tasks like translation and text generation."},
            
            {"python_definition",
             "Python is a high-level, interpreted programming language known for its "
             "simplicity and readability. It supports multiple programming paradigms "
             "including procedural, object-oriented, and functional programming. Python "
             "is widely used in web development, data science, artificial intelligence, "
             "and scientific computing."},
            
            {"rag_definition",
             "Retrieval-Augmented Generation (RAG) is a technique that combines information "
             "retrieval with text generation. RAG systems first retrieve relevant documents "
             "from a knowledge base, then use those documents as context to generate more "
             "accurate and factually-grounded responses. This approach reduces hallucination "
             "and allows models to access up-to-date information without retraining."},
            
            {"bert_definition",
             "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained "
             "language model developed by Google. Unlike traditional language models that read "
             "text left-to-right or right-to-left, BERT reads text bidirectionally. This allows "
             "it to better understand context and has led to significant improvements in tasks "
             "like question answering, sentiment analysis, and named entity recognition."},
            
            {"cpp_definition",
             "C++ is a general-purpose programming language created as an extension of C. "
             "It supports procedural, object-oriented, and generic programming. C++ is known "
             "for its performance and is widely used in systems programming, game development, "
             "embedded systems, and high-performance computing applications."},
        };

        std::cout << "Adding documents to knowledge base...\n";
        for (const auto& [id, text] : knowledge_docs) {
            doc_store->addDocument(id, text);
            std::cout << "  ✓ Added: " << id << "\n";
        }
        std::cout << GREEN << "\n✓ Knowledge base created with " 
                  << doc_store->size() << " documents" << RESET << "\n";

        // ====================================================================
        // Step 3: Configure RAG System
        // ====================================================================
        printSection("Configuring RAG System");

        RAGInference::RAGConfig rag_config;
        rag_config.num_retrieved_docs = 2;      // Retrieve top-2 most relevant docs
        rag_config.retrieval_threshold = 0.1f;  // Minimum similarity score
        rag_config.include_scores = true;       // Show similarity scores in context
        rag_config.max_context_length = 256;    // Max tokens for context
        
        // Configure text generation
        rag_config.gen_config.max_length = 100;
        rag_config.gen_config.temperature = 0.7f;
        rag_config.gen_config.top_p = 0.9f;

        auto rag = std::make_unique<RAGInference>(model, doc_store, rag_config);
        
        std::cout << "RAG Configuration:\n";
        std::cout << "  • Retrieved documents: " << rag_config.num_retrieved_docs << "\n";
        std::cout << "  • Retrieval threshold: " << rag_config.retrieval_threshold << "\n";
        std::cout << "  • Max context length: " << rag_config.max_context_length << " tokens\n";
        std::cout << "  • Generation max length: " << rag_config.gen_config.max_length << "\n";
        std::cout << "  • Temperature: " << rag_config.gen_config.temperature << "\n";

        // ====================================================================
        // Step 4: Demonstrate RAG with Sample Queries
        // ====================================================================
        printSection("RAG Question Answering Demo");

        std::vector<std::string> queries = {
            "What is machine learning?",
            "How do transformers work?",
            "What is RAG?",
            "Tell me about Python programming language",
            "What is the difference between BERT and transformers?"
        };

        for (const auto& query : queries) {
            std::cout << "\n" << MAGENTA << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << RESET << "\n";
            std::cout << BOLD << "Query: " << RESET << query << "\n";
            std::cout << MAGENTA << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << RESET << "\n\n";

            // Retrieve documents with scores
            std::vector<std::pair<float, const Document*>> retrieved_docs;
            
            try {
                // Generate response using RAG
                std::string response = rag->generateWithRetrieval(query, retrieved_docs);

                // Display retrieved documents
                std::cout << YELLOW << "Retrieved Documents (" << retrieved_docs.size() << "):" << RESET << "\n";
                for (size_t i = 0; i < retrieved_docs.size(); ++i) {
                    const auto& [score, doc] = retrieved_docs[i];
                    std::cout << "\n" << BOLD << "  [" << (i+1) << "] " << doc->id 
                              << RESET << " (similarity: " << std::fixed << std::setprecision(4) 
                              << score << ")\n";
                    std::cout << "      " << doc->text.substr(0, 100) << "...\n";
                }

                // Display generated response
                std::cout << "\n" << GREEN << BOLD << "Response:" << RESET << "\n";
                std::cout << response << "\n";

            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        }

        // ====================================================================
        // Step 5: Interactive Mode
        // ====================================================================
        printSection("Interactive RAG Mode");
        
        std::cout << "Enter queries to get RAG-enhanced responses.\n";
        std::cout << "Commands:\n";
        std::cout << "  'exit' or 'quit' - Exit the program\n";
        std::cout << "  'add' - Add a new document to knowledge base\n";
        std::cout << "  'list' - List all documents in knowledge base\n";
        std::cout << "  'stats' - Show knowledge base statistics\n\n";

        std::string input;
        while (true) {
            std::cout << CYAN << "\n>>> " << RESET;
            std::getline(std::cin, input);

            if (input.empty()) continue;
            
            if (input == "exit" || input == "quit") {
                std::cout << "Goodbye!\n";
                break;
            }
            
            if (input == "list") {
                auto ids = doc_store->getAllDocumentIds();
                std::cout << "\nDocuments in knowledge base (" << ids.size() << "):\n";
                for (const auto& id : ids) {
                    std::cout << "  • " << id << "\n";
                }
                continue;
            }
            
            if (input == "stats") {
                std::cout << "\nKnowledge Base Statistics:\n";
                std::cout << "  Total documents: " << rag->getNumDocuments() << "\n";
                std::cout << "  Retrieval config: top-" << rag->getConfig().num_retrieved_docs << "\n";
                continue;
            }
            
            if (input == "add") {
                std::cout << "Enter document ID: ";
                std::string doc_id;
                std::getline(std::cin, doc_id);
                
                std::cout << "Enter document text: ";
                std::string doc_text;
                std::getline(std::cin, doc_text);
                
                try {
                    rag->addDocument(doc_id, doc_text);
                    std::cout << GREEN << "✓ Document added successfully" << RESET << "\n";
                } catch (const std::exception& e) {
                    std::cout << "Error: " << e.what() << "\n";
                }
                continue;
            }

            // Process as query
            try {
                std::vector<std::pair<float, const Document*>> retrieved_docs;
                std::string response = rag->generateWithRetrieval(input, retrieved_docs);
                
                std::cout << YELLOW << "\nRetrieved: ";
                for (const auto& [score, doc] : retrieved_docs) {
                    std::cout << doc->id << " (" << std::fixed << std::setprecision(2) 
                              << score << ") ";
                }
                std::cout << RESET << "\n";
                
                std::cout << GREEN << "\nResponse: " << RESET << response << "\n";
                
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        }

        printHeader("RAG Demo Complete");
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
