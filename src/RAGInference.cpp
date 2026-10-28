// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-27

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
std::string RAGInference::buildBudgetedPrompt(const std::string& query,
                                              const std::string& context) const {
    const std::string question_only = buildAugmentedPrompt(query, "");
    if (context.empty() || config.max_context_length <= 0) {
        return question_only;
    }
    const size_t max_seq = static_cast<size_t>(std::max(model->get_max_seq_length(), 1));
    const size_t max_context = static_cast<size_t>(config.max_context_length);
    // A question longer than max_seq_length gets no context; the encoder keeps its last tokens.
    if (countTokens(question_only) >= max_seq) {
        return question_only;
    }

    auto clipped = [&](size_t n) { return n >= context.size() ? context : context.substr(0, n) + "..."; };
    auto fits = [&](size_t n) {
        const std::string ctx = clipped(n);
        return countTokens(ctx) <= max_context &&
               countTokens(buildAugmentedPrompt(query, ctx)) <= max_seq;
    };
    if (fits(context.size())) {
        return buildAugmentedPrompt(query, context);
    }
    size_t lo = 0;
    size_t hi = context.size() - 1;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo + 1) / 2;
        if (fits(utf8Boundary(context, mid))) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const size_t n = utf8Boundary(context, lo);
    if (n == 0 || !fits(n)) {
        return question_only;
    }
    return buildAugmentedPrompt(query, clipped(n));
}

std::string RAGInference::generate(const std::string& query) {
    std::vector<std::pair<float, const Document*>> retrieved_docs;
    return generateWithRetrieval(query, retrieved_docs);
}

std::string RAGInference::generateWithRetrieval(
    const std::string& query, std::vector<std::pair<float, const Document*>>& retrieved_docs) {
    // TODO: See TD-487 in TECHNICAL_DEBT.md - num_retrieved_docs <= 0 throws on every request; a threshold that
    // removes every document is silent; retrieved IDs/scores are never logged.
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

    // TODO: See TD-485 in TECHNICAL_DEBT.md - for /chat/session `query` is the whole conversation, so retrieval
    // embeds only its first max_seq_length tokens (oldest turns), not the latest question.
    // Steps 3-4: the augmented prompt, within max_seq_length and max_context_length tokens (TD-484)
    std::string augmented_prompt = buildBudgetedPrompt(query, context);

    // TODO: See TD-486 in TECHNICAL_DEBT.md - strategy is always "" (STRATEGY/BEAM_WIDTH ignored), gen_config is
    // copied once at startup (no reload/per-request values), and this is the CPU-only path even on a GPU server.
    // Step 5: Generate response using the model
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
    config = new_config;
}

std::vector<std::pair<float, const Document*>> RAGInference::retrieveOnly(const std::string& query,
                                                                          int k) const {
    int num_docs = (k > 0) ? k : config.num_retrieved_docs;
    return doc_store->retrieve(query, num_docs);
}
