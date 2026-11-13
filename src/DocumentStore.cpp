// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-12

#include "DocumentStore.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace {

bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

// [begin, end) of `text` without its leading whitespace.
std::pair<std::size_t, std::size_t> trim_front(const std::string& text, std::size_t begin,
                                               std::size_t end) {
    while (begin < end && is_space(text[begin]))
        ++begin;
    return {begin, end};
}

}  // namespace

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

// TODO: See TD-351 in TECHNICAL_DEBT.md - pools special-token rows too, compressing cosine scores
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

// TD-349: the text is cut into pieces (a whitespace run and the word after it), so a chunk ends
// where the tokenizer's own pre-tokens do. Each piece's token count is taken alone; a chunk is
// filled to the window by those counts, then its real count is checked and pieces dropped until it
// fits.
std::vector<std::pair<std::size_t, std::size_t>> DocumentStore::chunkRanges(
    const std::string& text, std::size_t window) const {
    std::vector<std::pair<std::size_t, std::size_t>> pieces;
    for (std::size_t i = 0; i < text.size();) {
        const std::size_t begin = i;
        while (i < text.size() && is_space(text[i]))
            ++i;
        while (i < text.size() && !is_space(text[i]))
            ++i;
        pieces.emplace_back(begin, i);
    }
    std::vector<std::size_t> counts;
    counts.reserve(pieces.size());
    for (const auto& [b, e] : pieces)
        counts.push_back(encoder->count_tokens(text.substr(b, e - b)));

    const std::size_t overlap = window / 4;
    std::vector<std::pair<std::size_t, std::size_t>> chunks;
    for (std::size_t first = 0; first < pieces.size();) {
        std::size_t last = first;  // one past the chunk's last piece
        std::size_t estimate = 0;
        while (last < pieces.size() && (last == first || estimate + counts[last] <= window))
            estimate += counts[last++];
        auto range = trim_front(text, pieces[first].first, pieces[last - 1].second);
        while (last > first + 1 && encoder->count_tokens(text.substr(
                                       range.first, range.second - range.first)) > window) {
            --last;
            range = trim_front(text, pieces[first].first, pieces[last - 1].second);
        }
        if (range.first < range.second)
            chunks.push_back(range);
        if (last == pieces.size())
            break;
        // The next chunk starts about `overlap` tokens back (at least one piece, so a sentence cut
        // at the boundary appears whole in one of them), but always past this one's start.
        std::size_t next = last;
        std::size_t shared = 0;
        while (next - 1 > first && (next == last || shared + counts[next - 1] <= overlap))
            shared += counts[--next];
        first = next;
    }
    return chunks;
}

void DocumentStore::addDocument(const std::string& id, const std::string& text,
                                const std::unordered_map<std::string, std::string>& metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id_to_index.find(id) != id_to_index.end() ||
        std::any_of(documents.begin(), documents.end(),
                    [&](const Document& d) { return d.parent_id == id; })) {
        throw std::invalid_argument("Document with ID '" + id + "' already exists");
    }

    if (text.empty()) {
        throw std::invalid_argument("Document text cannot be empty");
    }

    const auto window = static_cast<std::size_t>(std::max(1, encoder->get_max_seq_length()));
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    if (encoder->count_tokens(text) > window)
        ranges = chunkRanges(text, window);

    std::vector<Document> entries;
    if (ranges.size() <= 1) {
        // Fits (or is one unbreakable piece): one entry, embedded as encode() truncates it.
        entries.emplace_back(id, text);
    } else {
        for (std::size_t c = 0; c < ranges.size(); ++c) {
            const std::string chunk_id = id + "#" + std::to_string(c + 1);
            if (id_to_index.count(chunk_id) != 0)
                throw std::invalid_argument("Document with ID '" + chunk_id + "' already exists");
            entries.emplace_back(chunk_id,
                                 text.substr(ranges[c].first, ranges[c].second - ranges[c].first));
            entries.back().chunk_index = static_cast<int>(c + 1);
        }
    }
    // Embed every entry before storing any, so a failure leaves the store unchanged.
    for (auto& entry : entries) {
        entry.metadata = metadata;
        entry.parent_id = id;
        entry.chunk_count = static_cast<int>(entries.size());
        entry.embedding = getSentenceEmbedding(encoder->encode(entry.text));
    }
    for (auto& entry : entries) {
        id_to_index[entry.id] = documents.size();
        documents.push_back(std::move(entry));
    }
}

void DocumentStore::removeEntryLocked(const std::string& entry_id) {
    auto it = id_to_index.find(entry_id);
    if (it == id_to_index.end())
        return;
    const size_t index = it->second;
    id_to_index.erase(it);

    // Remove from vector (swap with last element for efficiency)
    if (index != documents.size() - 1) {
        documents[index] = std::move(documents.back());
        // Update index mapping for swapped document
        id_to_index[documents[index].id] = index;
    }
    documents.pop_back();
}

bool DocumentStore::removeDocument(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // TD-349: every entry of the document (its chunks, or the one entry of an unchunked one), else
    // the one entry with this id (a single chunk).
    std::vector<std::string> entry_ids;
    for (const auto& doc : documents)
        if (doc.parent_id == id)
            entry_ids.push_back(doc.id);
    if (entry_ids.empty() && id_to_index.count(id) != 0)
        entry_ids.push_back(id);
    for (const auto& entry_id : entry_ids)
        removeEntryLocked(entry_id);
    return !entry_ids.empty();
}

// TD-350: encode() writes the encoder's layer caches, so the query is encoded and scored under the
// store's lock. Retrieval used to be serialized only by ChatbotAPI holding config_mutex_ (TD-230).
std::vector<std::pair<float, const Document*>> DocumentStore::retrieve(const std::string& query,
                                                                       int k) const {
    std::lock_guard<std::mutex> lock(mutex_);
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
    // TODO: See TD-351 in TECHNICAL_DEBT.md - NaN similarities violate partial_sort's strict weak ordering
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
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = id_to_index.find(id);
    if (it == id_to_index.end()) {
        return nullptr;
    }
    return &documents[it->second];
}

void DocumentStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    documents.clear();
    id_to_index.clear();
}

std::vector<std::string> DocumentStore::getAllDocumentIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(documents.size());
    for (const auto& doc : documents) {
        ids.push_back(doc.id);
    }
    return ids;
}
