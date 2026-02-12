#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "Matrix.hpp"
#include "encoder.hpp"

/**
 * @brief Document structure for RAG (Retrieval-Augmented Generation)
 * 
 * Contains text content and its corresponding dense embedding vector
 * produced by the encoder model for semantic similarity search.
 */
struct Document {
    std::string id;           // Unique document identifier
    std::string text;         // Document text content
    Matrix embedding;         // Dense embedding vector [d_model]
    std::unordered_map<std::string, std::string> metadata;  // Optional metadata

    Document() = default;
    Document(const std::string& id_, const std::string& text_)
        : id(id_), text(text_) {}
};

/**
 * @brief Document Store for RAG system
 * 
 * Manages a collection of documents with their embeddings for
 * efficient semantic similarity search and retrieval.
 * 
 * Features:
 * - Document indexing with encoder-based embeddings
 * - Cosine similarity search for top-k retrieval
 * - Document management (add, remove, update)
 * - Metadata storage for each document
 * 
 * Usage:
 *   DocumentStore store(encoder_ptr);
 *   store.addDocument("doc1", "Machine learning is...");
 *   auto results = store.retrieve("What is ML?", 3);
 */
class DocumentStore {
private:
    std::shared_ptr<LLMEncoder> encoder;  // Encoder for generating embeddings
    std::vector<Document> documents;       // Stored documents with embeddings
    std::unordered_map<std::string, size_t> id_to_index;  // Fast lookup by ID

    /**
     * @brief Compute cosine similarity between two embedding vectors
     * 
     * @param emb1 First embedding matrix [d_model]
     * @param emb2 Second embedding matrix [d_model]
     * @return float Cosine similarity score in [-1, 1]
     */
    float cosineSimilarity(const Matrix& emb1, const Matrix& emb2) const;

    /**
     * @brief Generate sentence embedding from encoder output
     * 
     * Uses mean pooling over the sequence dimension to produce
     * a fixed-size sentence representation.
     * 
     * @param encoder_output Matrix of shape [seq_len, d_model]
     * @return Matrix Single vector of shape [1, d_model]
     */
    Matrix getSentenceEmbedding(const Matrix& encoder_output) const;

public:
    /**
     * @brief Constructor for DocumentStore
     * 
     * @param encoder Shared pointer to LLMEncoder for generating embeddings
     */
    explicit DocumentStore(std::shared_ptr<LLMEncoder> encoder);

    /**
     * @brief Add a document to the store
     * 
     * Generates embedding for the document text using the encoder
     * and stores it for later retrieval.
     * 
     * @param id Unique document identifier
     * @param text Document text content
     * @param metadata Optional key-value metadata
     * @throws std::invalid_argument if id already exists
     */
    void addDocument(const std::string& id, const std::string& text,
                     const std::unordered_map<std::string, std::string>& metadata = {});

    /**
     * @brief Remove a document from the store
     * 
     * @param id Document identifier to remove
     * @return bool True if document was removed, false if not found
     */
    bool removeDocument(const std::string& id);

    /**
     * @brief Retrieve top-k most similar documents to query
     * 
     * Encodes the query text and performs cosine similarity search
     * against all stored documents, returning the k most similar ones.
     * 
     * @param query Query text
     * @param k Number of documents to retrieve
     * @return std::vector<std::pair<float, const Document*>> Pairs of (similarity_score, document)
     *         sorted by descending similarity
     */
    std::vector<std::pair<float, const Document*>> retrieve(const std::string& query, int k = 3) const;

    /**
     * @brief Get document by ID
     * 
     * @param id Document identifier
     * @return const Document* Pointer to document, or nullptr if not found
     */
    const Document* getDocument(const std::string& id) const;

    /**
     * @brief Get total number of documents in store
     * 
     * @return size_t Number of documents
     */
    size_t size() const { return documents.size(); }

    /**
     * @brief Check if store is empty
     * 
     * @return bool True if no documents stored
     */
    bool empty() const { return documents.empty(); }

    /**
     * @brief Clear all documents from store
     */
    void clear();

    /**
     * @brief Get all document IDs
     * 
     * @return std::vector<std::string> List of all document IDs
     */
    std::vector<std::string> getAllDocumentIds() const;
};
