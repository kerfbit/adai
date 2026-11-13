#pragma once

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-12

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "Matrix.hpp"
#include "encoder.hpp"

/**
 * @brief Document structure for RAG (Retrieval-Augmented Generation)
 *
 * One indexed entry: a whole document, or (TD-349) one chunk of a document longer than the
 * encoder's max_seq_length, with its text and its dense embedding for similarity search.
 */
struct Document {
    std::string id;    // Entry id: the document's, or <id>#<chunk>
    std::string text;  // The entry's text (a chunk's own text)
    Matrix embedding;                                       // Dense embedding vector [d_model]
    // TODO: See TD-351 in TECHNICAL_DEBT.md - stored but never read (no metadata filtering)
    std::unordered_map<std::string, std::string> metadata;  // Optional metadata
    std::string parent_id;  // TD-349: the document this entry belongs to (its own id if unchunked)
    int chunk_index{1};     // 1-based position among the document's chunks
    int chunk_count{1};     // the document's number of chunks

    Document() = default;
    Document(std::string id_, std::string text_) : id(std::move(id_)), text(std::move(text_)) {}
};

/**
 * @brief Document Store for RAG system
 *
 * Manages a collection of documents with their embeddings for
 * efficient semantic similarity search and retrieval.
 *
 * Features:
 * - Document indexing with encoder-based embeddings; a document longer than the encoder's
 *   max_seq_length is indexed as overlapping chunks that each fit (TD-349)
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
    std::shared_ptr<LLMEncoder> encoder;                  // Encoder for generating embeddings
    std::vector<Document> documents;                      // Stored documents with embeddings
    std::unordered_map<std::string, size_t> id_to_index;  // Fast lookup by ID
    // TD-350: guards the encoder (encode() writes its layers' caches) and the document list, so
    // concurrent requests can retrieve while documents are added.
    mutable std::mutex mutex_;

    /**
     * @brief Compute cosine similarity between two embedding vectors
     *
     * @param emb1 First embedding matrix [d_model]
     * @param emb2 Second embedding matrix [d_model]
     * @return float Cosine similarity score in [-1, 1]
     */
    static float cosineSimilarity(const Matrix& emb1, const Matrix& emb2);

    /**
     * @brief Generate sentence embedding from encoder output
     *
     * Uses mean pooling over the sequence dimension to produce
     * a fixed-size sentence representation.
     *
     * @param encoder_output Matrix of shape [seq_len, d_model]
     * @return Matrix Single vector of shape [1, d_model]
     */
    static Matrix getSentenceEmbedding(const Matrix& encoder_output);

    /// TD-349: [begin, end) byte ranges of `text`'s chunks, each at most `window` tokens (one piece
    /// longer than that stays whole), consecutive chunks sharing about a quarter of a window (at
    /// least one piece).
    std::vector<std::pair<std::size_t, std::size_t>> chunkRanges(const std::string& text,
                                                                 std::size_t window) const;
    void removeEntryLocked(const std::string& entry_id);

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
     * Embeds the document with the encoder and stores it for retrieval. TD-349: a document of
     * more than the encoder's max_seq_length tokens is split at whitespace into overlapping chunks
     * that each fit, stored as entries "<id>#1" ... "<id>#N" (parent_id = id, the metadata copied
     * to each); the encoder used to embed only one window of it, so the rest couldn't be retrieved.
     * A document that fits is one entry under its own id.
     *
     * @param id Unique document identifier
     * @param text Document text content
     * @param metadata Optional key-value metadata
     * @throws std::invalid_argument if the id, or one of its chunk ids, already exists
     */
    void addDocument(const std::string& id, const std::string& text,
                     const std::unordered_map<std::string, std::string>& metadata = {});

    /**
     * @brief Remove a document from the store
     *
     * Implemented as swap-with-last-then-pop for O(1) removal: the document
     * previously at the last index is moved into the removed slot. Any
     * `const Document*` obtained from retrieve()/getDocument() before this
     * call may now be dangling (if it pointed at the removed document) or
     * silently refer to a *different* document (if it pointed at the one
     * that got moved into the freed slot) — do not hold such pointers across
     * a removeDocument() call.
     *
     * @param id Document identifier to remove: removes every chunk of a chunked document (TD-349),
     *        or the one entry with that id
     * @return bool True if document was removed, false if not found
     */
    bool removeDocument(const std::string& id);

    /**
     * @brief Retrieve top-k most similar documents to query
     *
     * Encodes the query text and performs cosine similarity search
     * against all stored documents, returning the k most similar ones.
     * Safe to call from several threads at once (TD-350).
     *
     * The returned pointers are valid only until the next call to
     * addDocument()/removeDocument()/clear() — see removeDocument()'s note.
     *
     * @param query Query text
     * @param k Number of documents to retrieve
     * @return std::vector<std::pair<float, const Document*>> Pairs of (similarity_score, entry)
     *         sorted by descending similarity; a chunked document's entries are its chunks
     */
    std::vector<std::pair<float, const Document*>> retrieve(const std::string& query,
                                                            int k = 3) const;

    /**
     * @brief Get document by ID
     *
     * The returned pointer is valid only until the next call to
     * addDocument()/removeDocument()/clear() — see removeDocument()'s note.
     *
     * @param id Entry identifier: a chunked document's entries are "<id>#1" ... (TD-349)
     * @return const Document* Pointer to document, or nullptr if not found
     */
    const Document* getDocument(const std::string& id) const;

    /**
     * @brief Get total number of entries in store
     *
     * @return size_t Number of entries: one per unchunked document, one per chunk (TD-349)
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return documents.size();
    }

    /**
     * @brief Check if store is empty
     *
     * @return bool True if no documents stored
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return documents.empty();
    }

    /**
     * @brief Clear all documents from store
     */
    void clear();

    /**
     * @brief Get all entry IDs
     *
     * @return std::vector<std::string> List of all entry IDs (chunk ids for a chunked document)
     */
    std::vector<std::string> getAllDocumentIds() const;
};
