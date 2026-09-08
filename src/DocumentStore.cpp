// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "DocumentStore.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

DocumentStore::DocumentStore(std::shared_ptr<LLMEncoder> encoder) : encoder(encoder) {
    if (!encoder) {
        throw std::invalid_argument("DocumentStore: encoder cannot be null");
    }
}

float DocumentStore::cosineSimilarity(const Matrix& emb1, const Matrix& emb2) {
    if (emb1.cols != emb2.cols) {
        throw std::invalid_argument("Embeddings must have same dimensions");
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (int i = 0; i < emb1.cols; ++i) {
        float val1 = emb1.data[0][i];
        float val2 = emb2.data[0][i];
        dot_product += val1 * val2;
        norm1 += val1 * val1;
        norm2 += val2 * val2;
    }

    float denom = std::sqrt(norm1) * std::sqrt(norm2);
    if (denom < 1e-9f) {
        return 0.0f;  // Handle zero vectors
    }

    return dot_product / denom;
}

Matrix DocumentStore::getSentenceEmbedding(const Matrix& encoder_output) {
    // Mean pooling over sequence dimension
    // encoder_output shape: [seq_len, d_model]
    // output shape: [1, d_model]

    if (encoder_output.rows == 0 || encoder_output.cols == 0) {
        throw std::invalid_argument("Encoder output cannot be empty");
    }

    Matrix sentence_emb(1, encoder_output.cols);
    sentence_emb.fill(0.0f);

    // Sum over all positions
    for (int i = 0; i < encoder_output.rows; ++i) {
        for (int j = 0; j < encoder_output.cols; ++j) {
            sentence_emb(0, j) += encoder_output(i, j);
        }
    }

    // Average by sequence length
    float scale = 1.0f / static_cast<float>(encoder_output.rows);
    for (int j = 0; j < encoder_output.cols; ++j) {
        sentence_emb(0, j) *= scale;
    }

    return sentence_emb;
}

void DocumentStore::addDocument(const std::string& id, const std::string& text,
                                const std::unordered_map<std::string, std::string>& metadata) {
    // Check if document with this ID already exists
    if (id_to_index.find(id) != id_to_index.end()) {
        throw std::invalid_argument("Document with ID '" + id + "' already exists");
    }

    if (text.empty()) {
        throw std::invalid_argument("Document text cannot be empty");
    }

    // Create document
    Document doc(id, text);
    doc.metadata = metadata;

    // Generate embedding using encoder
    Matrix encoder_output = encoder->encode(text);
    doc.embedding = getSentenceEmbedding(encoder_output);

    // Store document
    size_t index = documents.size();
    documents.push_back(std::move(doc));
    id_to_index[id] = index;
}

bool DocumentStore::removeDocument(const std::string& id) {
    auto it = id_to_index.find(id);
    if (it == id_to_index.end()) {
        return false;  // Document not found
    }

    size_t index = it->second;

    // Remove from vector (swap with last element for efficiency)
    if (index != documents.size() - 1) {
        documents[index] = std::move(documents.back());
        // Update index mapping for swapped document
        id_to_index[documents[index].id] = index;
    }
    documents.pop_back();

    // Remove from index map
    id_to_index.erase(it);

    return true;
}

std::vector<std::pair<float, const Document*>> DocumentStore::retrieve(const std::string& query,
                                                                       int k) const {
    if (documents.empty()) {
        return {};
    }

    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    // Generate query embedding
    Matrix query_output = encoder->encode(query);
    Matrix query_emb = getSentenceEmbedding(query_output);

    // Compute similarities with all documents
    std::vector<std::pair<float, const Document*>> similarities;
    similarities.reserve(documents.size());

    for (const auto& doc : documents) {
        float sim = cosineSimilarity(query_emb, doc.embedding);
        similarities.emplace_back(sim, &doc);
    }

    // Sort by similarity (descending)
    std::partial_sort(similarities.begin(),
                      similarities.begin() + std::min(k, static_cast<int>(similarities.size())),
                      similarities.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Return top-k results
    int num_results = std::min(k, static_cast<int>(similarities.size()));
    similarities.resize(num_results);

    return similarities;
}

const Document* DocumentStore::getDocument(const std::string& id) const {
    auto it = id_to_index.find(id);
    if (it == id_to_index.end()) {
        return nullptr;
    }
    return &documents[it->second];
}

void DocumentStore::clear() {
    documents.clear();
    id_to_index.clear();
}

std::vector<std::string> DocumentStore::getAllDocumentIds() const {
    std::vector<std::string> ids;
    ids.reserve(documents.size());
    for (const auto& doc : documents) {
        ids.push_back(doc.id);
    }
    return ids;
}
