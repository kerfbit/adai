#pragma once

// @adai-status: beta
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07


#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "DocumentStore.hpp"
#include "EncoderDecoderModel.hpp"
#include "Matrix.hpp"
#include "TextGenerator.hpp"

/**
 * @brief RAG (Retrieval-Augmented Generation) Inference Engine
 *
 * Combines document retrieval with generative language models to produce
 * factually-grounded, context-aware responses. The system:
 * 1. Retrieves relevant documents for a given query
 * 2. Augments the query with retrieved context
 * 3. Generates a response conditioned on both query and context
 *
 * Benefits:
 * - Factual grounding: Responses based on retrieved evidence
 * - Up-to-date knowledge: Knowledge can be updated by adding documents
 * - Reduced hallucination: Model has access to real information
 * - Explainability: Can cite source documents
 * - Scalable knowledge: No need to retrain for new information
 *
 * Architecture:
 *   Query → DocumentStore.retrieve() → Top-k Documents
 *       ↓
 *   [Context: doc1, doc2, ...] + Query → Encoder-Decoder → Response
 *
 * Usage:
 *   RAGInference rag(model, document_store, config);
 *   rag.addDocument("doc1", "Python is a programming language...");
 *   std::string response = rag.generate("What is Python?");
 */
class RAGInference {
   public:
    /**
     * @brief Configuration for RAG inference
     */
    struct RAGConfig {
        int num_retrieved_docs{3};                   // Number of documents to retrieve
        float retrieval_threshold{0.0f};             // Minimum similarity score (0 = no filter)
        bool include_scores{false};                  // Include similarity scores in context
        std::string context_separator;               // Separator between documents
        std::string context_prefix;                  // Prefix before context
        std::string query_prefix;                    // Prefix before query
        int max_context_length{512};                 // Maximum tokens for context
        TextGenerator::GenerationConfig gen_config;  // Generation parameters

        RAGConfig()
            : context_separator("\n\n"),
              context_prefix("Context:\n"),
              query_prefix("\n\nQuestion: ") {}
    };

   private:
    std::shared_ptr<EncoderDecoderModel> model;  // Seq2seq model for generation
    std::shared_ptr<DocumentStore> doc_store;    // Document store for retrieval
    RAGConfig config;                            // RAG configuration

    /**
     * @brief Format retrieved documents into context string
     *
     * @param retrieved_docs Vector of (similarity, document) pairs
     * @return std::string Formatted context string
     */
    std::string formatContext(
        const std::vector<std::pair<float, const Document*>>& retrieved_docs) const;

    /**
     * @brief Build augmented prompt with retrieved context
     *
     * @param query User query
     * @param context Retrieved context string
     * @return std::string Complete prompt for generation
     */
    std::string buildAugmentedPrompt(const std::string& query, const std::string& context) const;

    /**
     * @brief Truncate context to fit within token limit
     *
     * @param context Context string
     * @param max_tokens Maximum number of tokens
     * @return std::string Truncated context
     */
    static std::string truncateContext(const std::string& context, int max_tokens);

   public:
    /**
     * @brief Constructor for RAG inference engine
     *
     * @param model Shared pointer to EncoderDecoderModel
     * @param doc_store Shared pointer to DocumentStore
     * @param config RAG configuration
     */
    RAGInference(std::shared_ptr<EncoderDecoderModel> model,
                 std::shared_ptr<DocumentStore> doc_store, RAGConfig config = RAGConfig());

    /**
     * @brief Generate response using RAG
     *
     * Retrieves relevant documents and generates a response
     * conditioned on both the query and retrieved context.
     *
     * @param query User query/question
     * @return std::string Generated response
     */
    std::string generate(const std::string& query);

    /**
     * @brief Generate response with detailed retrieval info
     *
     * @param query User query
     * @param retrieved_docs Output: Retrieved documents with scores
     * @return std::string Generated response
     */
    std::string generateWithRetrieval(
        const std::string& query, std::vector<std::pair<float, const Document*>>& retrieved_docs);

    /**
     * @brief Add document to the knowledge base
     *
     * Convenience method that forwards to DocumentStore.
     *
     * @param id Document ID
     * @param text Document text
     * @param metadata Optional metadata
     */
    void addDocument(const std::string& id, const std::string& text,
                     const std::unordered_map<std::string, std::string>& metadata = {});

    /**
     * @brief Remove document from knowledge base
     *
     * @param id Document ID
     * @return bool True if removed, false if not found
     */
    bool removeDocument(const std::string& id);

    /**
     * @brief Get document by ID
     *
     * @param id Document ID
     * @return const Document* Pointer to document or nullptr
     */
    const Document* getDocument(const std::string& id) const;

    /**
     * @brief Get number of documents in knowledge base
     *
     * @return size_t Number of documents
     */
    size_t getNumDocuments() const;

    /**
     * @brief Update RAG configuration
     *
     * @param new_config New configuration
     */
    void setConfig(const RAGConfig& new_config);

    /**
     * @brief Get current configuration
     *
     * @return const RAGConfig& Current configuration
     */
    const RAGConfig& getConfig() const {
        return config;
    }

    /**
     * @brief Retrieve documents without generating response
     *
     * Useful for inspecting retrieval quality.
     *
     * @param query Query string
     * @param k Number of documents to retrieve
     * @return std::vector<std::pair<float, const Document*>> Retrieved documents with scores
     */
    std::vector<std::pair<float, const Document*>> retrieveOnly(const std::string& query,
                                                                int k = -1) const;
};
