// @adai-status: stable
// @adai-version: 1.4.1
// @adai-reviewed: 2026-10-28

#include "RAGInference.hpp"
#include "Logger.hpp"
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
    validateConfig(this->config);
}

// TD-487: RAG_NUM_DOCS <= 0 used to pass, and then every request failed in DocumentStore::retrieve()
// ("k must be positive") once a document was indexed.
void RAGInference::validateConfig(const RAGConfig& config) {
    if (config.num_retrieved_docs < 1) {
        throw std::invalid_argument("RAGInference: num_retrieved_docs (RAG_NUM_DOCS) must be at least 1, got " +
                                    std::to_string(config.num_retrieved_docs));
    }
    if (config.max_context_length < 1) {
        throw std::invalid_argument(
            "RAGInference: max_context_length (RAG_MAX_CONTEXT_LENGTH) must be at least 1, got " +
            std::to_string(config.max_context_length));
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

std::string RAGInference::buildAugmentedPrompt(const std::string& query, const std::string& context,
                                               const std::string& history) const {
    std::ostringstream oss;

    // Add context if available
    if (!context.empty()) {
        oss << config.context_prefix << context;
    }

    // TD-485: the earlier conversation, between the context and the latest question.
    if (!history.empty()) {
        if (!context.empty()) oss << config.context_separator;
        oss << history;
    }

    // Add query
    oss << config.query_prefix << query;

    return oss.str();
}

// TD-484: the tokens the model will see, counted with its own tokenizer.
size_t RAGInference::countTokens(const std::string& text) const {
    return text.empty() ? 0 : model->get_tokenizer()->encode(text, false).size();
}

// TD-484: the largest position <= pos that doesn't split a UTF-8 sequence.
size_t RAGInference::utf8Boundary(const std::string& s, size_t pos) {
    pos = std::min(pos, s.size());
    while (pos > 0 && pos < s.size() && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) --pos;
    return pos;
}

// TD-484: the whole prompt fits the model's max_seq_length in real tokens, and the context fits
// max_context_length. The question (with its prefix) is kept whole; the context, documents in rank
// order, is cut from the end at a UTF-8 boundary, so lower-ranked documents go first. This used to
// cut the context alone at an estimated 4 characters per token (measured 1.8) and add the question
// after it, so at the shipped 512/512 the question fell past max_seq_length.
// TD-485: the earlier conversation gets the room left after the context, keeping its latest turns
// (cut from the front at a UTF-8 boundary).
std::string RAGInference::buildBudgetedPrompt(const std::string& query, const std::string& context,
                                              const std::string& history) const {
    const std::string question_only = buildAugmentedPrompt(query, "");
    const size_t max_seq = static_cast<size_t>(std::max(model->get_max_seq_length(), 1));
    // A question longer than max_seq_length gets nothing else; the encoder keeps its last tokens.
    if (countTokens(question_only) >= max_seq) {
        return question_only;
    }

    // Longest n in [0, size] with fits(n), on UTF-8 boundaries (fits(0) is taken as true).
    auto longest = [](const std::string& text, auto&& fits_at) -> size_t {
        if (text.empty()) return 0;
        if (fits_at(text.size())) return text.size();
        size_t lo = 0;
        size_t hi = text.size() - 1;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo + 1) / 2;
            if (fits_at(utf8Boundary(text, mid))) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const size_t n = utf8Boundary(text, lo);
        return (n > 0 && fits_at(n)) ? n : 0;
    };

    // The context first, within max_context_length and (without history) max_seq_length.
    std::string ctx;
    if (!context.empty() && config.max_context_length > 0) {
        const size_t max_context = static_cast<size_t>(config.max_context_length);
        auto clipped = [&](size_t n) {
            return n >= context.size() ? context : context.substr(0, n) + "...";
        };
        const size_t n = longest(context, [&](size_t k) {
            const std::string c = clipped(k);
            return countTokens(c) <= max_context && countTokens(buildAugmentedPrompt(query, c)) <= max_seq;
        });
        if (n > 0) ctx = clipped(n);
    }

    // Then as much of the conversation's end as still fits.
    std::string hist;
    if (!history.empty()) {
        auto tail = [&](size_t n) {  // the last n bytes, starting on a UTF-8 boundary
            if (n >= history.size()) return history;
            size_t start = history.size() - n;
            while (start < history.size() &&
                   (static_cast<unsigned char>(history[start]) & 0xC0) == 0x80) {
                ++start;
            }
            return "..." + history.substr(start);
        };
        const size_t n = longest(history, [&](size_t k) {
            return countTokens(buildAugmentedPrompt(query, ctx, tail(k))) <= max_seq;
        });
        if (n > 0) hist = tail(n);
    }
    return buildAugmentedPrompt(query, ctx, hist);
}

std::string RAGInference::generate(const std::string& query, const std::string& history) {
    std::vector<std::pair<float, const Document*>> retrieved_docs;
    return generateWithRetrieval(query, retrieved_docs, history);
}

std::string RAGInference::buildPrompt(const std::string& query, const std::string& history) {
    std::vector<std::pair<float, const Document*>> retrieved_docs;
    return buildPrompt(query, retrieved_docs, history);
}

std::string RAGInference::buildPrompt(const std::string& query,
                                      std::vector<std::pair<float, const Document*>>& retrieved_docs,
                                      const std::string& history) {
    // Step 1: Retrieve relevant documents. TD-485: with the latest question only; a session's whole
    // conversation used to be the query, which the encoder cuts to max_seq_length tokens (the first,
    // before TD-365; the last since), mostly earlier turns.
    retrieved_docs = doc_store->retrieve(query, config.num_retrieved_docs);

    // TD-487: which documents were retrieved, and with what scores.
    for (const auto& [score, doc] : retrieved_docs) {
        adai::Logger::debug("RAG: retrieved '{}' (similarity {:.4f})", doc->id, score);
    }

    // Filter by threshold if specified
    if (config.retrieval_threshold > 0.0f) {
        const size_t before = retrieved_docs.size();
        auto it = std::remove_if(
            retrieved_docs.begin(), retrieved_docs.end(),
            [this](const auto& pair) { return pair.first < config.retrieval_threshold; });
        retrieved_docs.erase(it, retrieved_docs.end());
        // TD-487: this used to leave the prompt without context silently.
        if (before > 0 && retrieved_docs.empty()) {
            adai::Logger::warn("RAG: all {} retrieved documents scored below RAG_THRESHOLD {}; answering without context",
                               before, config.retrieval_threshold);
        }
    }

    // Step 2: Format retrieved documents into context
    std::string context = formatContext(retrieved_docs);

    // Steps 3-4: context + earlier conversation + question, within max_seq_length (TD-484, TD-485)
    return buildBudgetedPrompt(query, context, history);
}

std::string RAGInference::generateWithRetrieval(
    const std::string& query, std::vector<std::pair<float, const Document*>>& retrieved_docs,
    const std::string& history) {
    const std::string augmented_prompt = buildPrompt(query, retrieved_docs, history);

    // Step 5: Generate response using the model. TD-486: this standalone path generates on the CPU
    // with RAGConfig::gen_config; chatbot_api_server instead takes buildPrompt() and generates through
    // ChatbotAPI's own path (the request's strategy and config, reloads, the GPU decode).
    //
    // TD-101 (fixed): this used to call generate_response(), whose max_length
    // parameter is a documented no-op (see its own body / the comment on its
    // gpu_generate_response() twin) — config.gen_config's temperature/top_k/
    // top_p/num_beams were never even passed at all. RAGConfig::gen_config is
    // declared as a full TextGenerator::GenerationConfig specifically to
    // carry generation parameters, but only max_length ever reached the model,
    // and even that was silently dropped once inside generate_response().
    // generate_response_with_strategy() has no unrecognized-strategy error
    // path — passing "" falls through to its own "default to main generate
    // method (uses config)" branch, which since TD-100 syncs generator's
    // config with exactly these arguments first — the same combined
    // temperature+top-k+top-p generation generate_response() always used,
    // now actually configured by this call instead of by whatever generator
    // last happened to hold.
    std::string response = model->generate_response_with_strategy(
        augmented_prompt, config.gen_config.max_length, /*strategy=*/"", config.gen_config.temperature,
        config.gen_config.top_k, config.gen_config.top_p, config.gen_config.num_beams);

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
    validateConfig(new_config);  // TD-487
    config = new_config;
}

std::vector<std::pair<float, const Document*>> RAGInference::retrieveOnly(const std::string& query,
                                                                          int k) const {
    int num_docs = (k > 0) ? k : config.num_retrieved_docs;
    return doc_store->retrieve(query, num_docs);
}
