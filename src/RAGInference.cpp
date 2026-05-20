#include "RAGInference.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

RAGInference::RAGInference(std::shared_ptr<EncoderDecoderModel> model,
                           std::shared_ptr<DocumentStore> doc_store, RAGConfig config)
    : model(model), doc_store(doc_store), config(std::move(config)) {
    if (!model) {
        throw std::invalid_argument("RAGInference: model cannot be null");
    }
    if (!doc_store) {
        throw std::invalid_argument("RAGInference: doc_store cannot be null");
    }
}

std::string RAGInference::formatContext(
    const std::vector<std::pair<float, const Document*>>& retrieved_docs) const {
    if (retrieved_docs.empty()) {
        return "";
    }

    std::ostringstream oss;

    for (size_t i = 0; i < retrieved_docs.size(); ++i) {
        const auto& [score, doc] = retrieved_docs[i];

        // Optionally include document ID and similarity score
        if (config.include_scores) {
            oss << "[Doc " << (i + 1) << " (similarity: " << score << ")]: ";
        }

        oss << doc->text;

        // Add separator between documents (but not after the last one)
        if (i < retrieved_docs.size() - 1) {
            oss << config.context_separator;
        }
    }

    return oss.str();
}

std::string RAGInference::buildAugmentedPrompt(const std::string& query,
                                               const std::string& context) const {
    std::ostringstream oss;

    // Add context if available
    if (!context.empty()) {
        oss << config.context_prefix << context;
    }

    // Add query
    oss << config.query_prefix << query;

    return oss.str();
}

std::string RAGInference::truncateContext(const std::string& context, int max_tokens) {
    // Simple truncation by character count
    // In a production system, you'd want to truncate by actual token count
    // For now, we use a rough approximation: ~4 chars per token
    int max_chars = max_tokens * 4;

    if (static_cast<int>(context.length()) <= max_chars) {
        return context;
    }

    // Truncate and add ellipsis
    return context.substr(0, max_chars - 3) + "...";
}

std::string RAGInference::generate(const std::string& query) {
    std::vector<std::pair<float, const Document*>> retrieved_docs;
    return generateWithRetrieval(query, retrieved_docs);
}

std::string RAGInference::generateWithRetrieval(
    const std::string& query, std::vector<std::pair<float, const Document*>>& retrieved_docs) {
    // Step 1: Retrieve relevant documents
    retrieved_docs = doc_store->retrieve(query, config.num_retrieved_docs);

    // Filter by threshold if specified
    if (config.retrieval_threshold > 0.0f) {
        auto it = std::remove_if(
            retrieved_docs.begin(), retrieved_docs.end(),
            [this](const auto& pair) { return pair.first < config.retrieval_threshold; });
        retrieved_docs.erase(it, retrieved_docs.end());
    }

    // Step 2: Format retrieved documents into context
    std::string context = formatContext(retrieved_docs);

    // Step 3: Truncate context if needed
    context = truncateContext(context, config.max_context_length);

    // Step 4: Build augmented prompt
    std::string augmented_prompt = buildAugmentedPrompt(query, context);

    // Step 5: Generate response using the model
    std::string response = model->generate_response(augmented_prompt, config.gen_config.max_length);

    return response;
}

void RAGInference::addDocument(const std::string& id, const std::string& text,
                               const std::unordered_map<std::string, std::string>& metadata) {
    doc_store->addDocument(id, text, metadata);
}

bool RAGInference::removeDocument(const std::string& id) {
    return doc_store->removeDocument(id);
}

const Document* RAGInference::getDocument(const std::string& id) const {
    return doc_store->getDocument(id);
}

size_t RAGInference::getNumDocuments() const {
    return doc_store->size();
}

void RAGInference::setConfig(const RAGConfig& new_config) {
    config = new_config;
}

std::vector<std::pair<float, const Document*>> RAGInference::retrieveOnly(const std::string& query,
                                                                          int k) const {
    int num_docs = (k > 0) ? k : config.num_retrieved_docs;
    return doc_store->retrieve(query, num_docs);
}
