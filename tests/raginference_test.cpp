/**
 * RAGInference Unit Tests
 *
 * Comprehensive test suite for the RAGInference class (Retrieval-Augmented Generation).
 * Tests cover document management, retrieval operations, context formatting, prompt building,
 * generation pipeline, and edge cases.
 *
 * Test Categories:
 * 1. Construction and Configuration
 * 2. Document Management
 * 3. Retrieval Operations
 * 4. Context Formatting
 * 5. Prompt Building
 * 6. Generation Pipeline
 * 7. Configuration Management
 * 8. Edge Cases
 */

#include "RAGInference.hpp"
#include <gtest/gtest.h>
#include "../src/Logger.hpp"
#include <algorithm>
#include <sstream>
#include <spdlog/sinks/ostream_sink.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include "DocumentStore.hpp"
#include "EncoderDecoderModel.hpp"
#include "encoder.hpp"

namespace fs = std::filesystem;

class RAGInferenceTest : public ::testing::Test {
   protected:
    std::shared_ptr<LLMEncoder> encoder;
    std::shared_ptr<EncoderDecoderModel> model;
    std::shared_ptr<DocumentStore> doc_store;
    std::unique_ptr<RAGInference> rag;
    fs::path test_dir;
    fs::path vocab_file;

    void SetUp() override {
        // Create temporary directory for test files
        test_dir = fs::temp_directory_path() / "raginference_test";
        fs::create_directories(test_dir);
        vocab_file = test_dir / "vocab.txt";

        // Create a minimal BPE vocabulary file
        create_test_vocabulary();

        // Initialize encoder with small configuration for fast testing
        encoder = std::make_shared<LLMEncoder>(1000,  // vocab_size
                                               128,   // d_model
                                               2,     // num_layers
                                               4,     // num_heads
                                               512,   // d_ff
                                               128    // max_seq_length
        );

        // Load vocabulary
        encoder->load_tokenizer_vocab(vocab_file.string());

        // Initialize encoder-decoder model with small configuration
        model = std::make_shared<EncoderDecoderModel>(1000,  // vocab_size
                                                      128,   // d_model
                                                      2,     // num_encoder_layers
                                                      2,     // num_decoder_layers
                                                      4,     // num_heads
                                                      512,   // d_ff
                                                      128    // max_seq_length
        );

        // Initialize document store
        doc_store = std::make_shared<DocumentStore>(encoder);

        // Create RAG inference with default configuration
        rag = std::make_unique<RAGInference>(model, doc_store);
    }

    void TearDown() override {
        rag.reset();
        doc_store.reset();
        model.reset();
        encoder.reset();

        // Clean up test files
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    void create_test_vocabulary() {
        std::ofstream file(vocab_file);

        // Write special tokens section
        file << "SPECIAL_TOKENS\n";
        file << "pad_token_id 0\n";
        file << "unk_token_id 1\n";
        file << "bos_token_id 2\n";
        file << "eos_token_id 3\n";
        file << "\n";

        // Write vocabulary section
        file << "VOCAB\n";
        file << "<pad>\t0\n";
        file << "<unk>\t1\n";
        file << "<bos>\t2\n";
        file << "<eos>\t3\n";

        // Add common words for testing
        const std::vector<std::string> words = {
            "the",       "machine",   "learning", "is",      "artificial",  "intelligence",
            "algorithm", "data",      "neural",   "network", "model",       "train",
            "test",      "computer",  "science",  "python",  "programming", "language",
            "deep",      "framework", "what",     "how",     "why",         "when",
            "where",     "context",   "question", "answer",  "document",    "retrieval"};

        int token_id = 4;
        for (const auto& word : words) {
            file << word << "\t" << token_id++ << "\n";
        }

        file.close();
    }

    void add_sample_documents() {
        doc_store->addDocument("doc1", "Machine learning is a subset of artificial intelligence.");
        doc_store->addDocument("doc2",
                               "Python is a popular programming language for data science.");
        doc_store->addDocument("doc3", "Deep learning uses neural networks with multiple layers.");
    }
};

// ============================================================================
// Construction and Configuration Tests
// ============================================================================

TEST_F(RAGInferenceTest, ConstructorWithValidParameters) {
    auto test_rag = std::make_unique<RAGInference>(model, doc_store);

    EXPECT_EQ(test_rag->getNumDocuments(), 0);
    EXPECT_NE(&test_rag->getConfig(), nullptr);
}

TEST_F(RAGInferenceTest, ConstructorWithNullModel) {
    EXPECT_THROW({ RAGInference test_rag(nullptr, doc_store); }, std::invalid_argument);
}

TEST_F(RAGInferenceTest, ConstructorWithNullDocumentStore) {
    EXPECT_THROW({ RAGInference test_rag(model, nullptr); }, std::invalid_argument);
}

TEST_F(RAGInferenceTest, ConstructorWithCustomConfig) {
    RAGInference::RAGConfig config;
    config.num_retrieved_docs = 5;
    config.retrieval_threshold = 0.5f;
    config.include_scores = true;

    auto test_rag = std::make_unique<RAGInference>(model, doc_store, config);

    EXPECT_EQ(test_rag->getConfig().num_retrieved_docs, 5);
    EXPECT_FLOAT_EQ(test_rag->getConfig().retrieval_threshold, 0.5f);
    EXPECT_TRUE(test_rag->getConfig().include_scores);
}

TEST_F(RAGInferenceTest, DefaultConfiguration) {
    const auto& config = rag->getConfig();

    EXPECT_EQ(config.num_retrieved_docs, 3);
    EXPECT_FLOAT_EQ(config.retrieval_threshold, 0.0f);
    EXPECT_FALSE(config.include_scores);
    EXPECT_EQ(config.context_separator, "\n\n");
    EXPECT_EQ(config.context_prefix, "Context:\n");
    EXPECT_EQ(config.query_prefix, "\n\nQuestion: ");
    EXPECT_EQ(config.max_context_length, 512);
}

// ============================================================================
// Document Management Tests
// ============================================================================

TEST_F(RAGInferenceTest, AddDocument) {
    rag->addDocument("doc1", "Test document text.");

    EXPECT_EQ(rag->getNumDocuments(), 1);

    const Document* doc = rag->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->id, "doc1");
    EXPECT_EQ(doc->text, "Test document text.");
}

TEST_F(RAGInferenceTest, AddMultipleDocuments) {
    rag->addDocument("doc1", "First document.");
    rag->addDocument("doc2", "Second document.");
    rag->addDocument("doc3", "Third document.");

    EXPECT_EQ(rag->getNumDocuments(), 3);
}

TEST_F(RAGInferenceTest, AddDocumentWithMetadata) {
    std::unordered_map<std::string, std::string> metadata = {{"author", "Test Author"},
                                                             {"year", "2026"}};

    rag->addDocument("doc1", "Document with metadata.", metadata);

    const Document* doc = rag->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->metadata.at("author"), "Test Author");
    EXPECT_EQ(doc->metadata.at("year"), "2026");
}

TEST_F(RAGInferenceTest, RemoveDocument) {
    rag->addDocument("doc1", "First document.");
    rag->addDocument("doc2", "Second document.");

    EXPECT_EQ(rag->getNumDocuments(), 2);

    bool removed = rag->removeDocument("doc1");

    EXPECT_TRUE(removed);
    EXPECT_EQ(rag->getNumDocuments(), 1);
    EXPECT_EQ(rag->getDocument("doc1"), nullptr);
}

TEST_F(RAGInferenceTest, RemoveNonExistentDocument) {
    rag->addDocument("doc1", "Document.");

    bool removed = rag->removeDocument("doc_nonexistent");

    EXPECT_FALSE(removed);
    EXPECT_EQ(rag->getNumDocuments(), 1);
}

TEST_F(RAGInferenceTest, GetDocumentById) {
    rag->addDocument("test_id", "Test content.");

    const Document* doc = rag->getDocument("test_id");

    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->id, "test_id");
    EXPECT_EQ(doc->text, "Test content.");
}

TEST_F(RAGInferenceTest, GetNonExistentDocument) {
    const Document* doc = rag->getDocument("nonexistent");

    EXPECT_EQ(doc, nullptr);
}

TEST_F(RAGInferenceTest, DocumentCountTracking) {
    EXPECT_EQ(rag->getNumDocuments(), 0);

    rag->addDocument("doc1", "First.");
    EXPECT_EQ(rag->getNumDocuments(), 1);

    rag->addDocument("doc2", "Second.");
    EXPECT_EQ(rag->getNumDocuments(), 2);

    rag->removeDocument("doc1");
    EXPECT_EQ(rag->getNumDocuments(), 1);
}

// ============================================================================
// Retrieval Operations Tests
// ============================================================================

TEST_F(RAGInferenceTest, BasicRetrieval) {
    add_sample_documents();

    auto results = rag->retrieveOnly("machine learning");

    // Should retrieve up to num_retrieved_docs (default 3)
    EXPECT_LE(results.size(), 3u);
    EXPECT_GT(results.size(), 0u);

    // Results should be sorted by similarity (descending)
    for (size_t i = 0; i < results.size() - 1; ++i) {
        EXPECT_GE(results[i].first, results[i + 1].first);
    }
}

TEST_F(RAGInferenceTest, RetrievalWithCustomK) {
    add_sample_documents();

    auto results = rag->retrieveOnly("python programming", 2);

    EXPECT_LE(results.size(), 2u);
}

TEST_F(RAGInferenceTest, RetrievalWithKGreaterThanDocumentCount) {
    rag->addDocument("doc1", "Only document.");

    auto results = rag->retrieveOnly("query", 10);

    // Should return all available documents (1), not 10
    EXPECT_EQ(results.size(), 1u);
}

TEST_F(RAGInferenceTest, RetrievalFromEmptyStore) {
    auto results = rag->retrieveOnly("query");

    EXPECT_TRUE(results.empty());
}

TEST_F(RAGInferenceTest, RetrievalDefaultK) {
    // Add more documents than default num_retrieved_docs
    for (int i = 0; i < 5; ++i) {
        rag->addDocument("doc" + std::to_string(i), "Document " + std::to_string(i));
    }

    auto results = rag->retrieveOnly("query");  // Uses config.num_retrieved_docs

    EXPECT_EQ(results.size(), 3u);  // Default is 3
}

// ============================================================================
// Configuration Management Tests
// ============================================================================

TEST_F(RAGInferenceTest, GetConfiguration) {
    const auto& config = rag->getConfig();

    EXPECT_EQ(config.num_retrieved_docs, 3);
    EXPECT_FLOAT_EQ(config.retrieval_threshold, 0.0f);
}

TEST_F(RAGInferenceTest, SetConfiguration) {
    RAGInference::RAGConfig new_config;
    new_config.num_retrieved_docs = 5;
    new_config.retrieval_threshold = 0.7f;
    new_config.include_scores = true;
    new_config.max_context_length = 256;

    rag->setConfig(new_config);

    const auto& config = rag->getConfig();
    EXPECT_EQ(config.num_retrieved_docs, 5);
    EXPECT_FLOAT_EQ(config.retrieval_threshold, 0.7f);
    EXPECT_TRUE(config.include_scores);
    EXPECT_EQ(config.max_context_length, 256);
}

TEST_F(RAGInferenceTest, ConfigurationAffectsRetrieval) {
    // Add 5 documents
    for (int i = 0; i < 5; ++i) {
        rag->addDocument("doc" + std::to_string(i), "Document " + std::to_string(i));
    }

    // Default: retrieve 3 documents
    auto results1 = rag->retrieveOnly("query");
    EXPECT_EQ(results1.size(), 3u);

    // Change config to retrieve 2 documents
    RAGInference::RAGConfig new_config = rag->getConfig();
    new_config.num_retrieved_docs = 2;
    rag->setConfig(new_config);

    auto results2 = rag->retrieveOnly("query");
    EXPECT_EQ(results2.size(), 2u);
}

// ============================================================================
// Generation Pipeline Tests (Basic)
// ============================================================================

TEST_F(RAGInferenceTest, GenerateWithDocuments) {
    add_sample_documents();

    // generate() should not throw (verifies pipeline integration)
    // Note: Untrained model may generate out-of-vocabulary tokens,
    // so we only verify the pipeline doesn't crash
    EXPECT_NO_THROW({ std::string response = rag->generate("What is machine learning?"); });
}

TEST_F(RAGInferenceTest, GenerateWithEmptyDocumentStore) {
    // No documents added - verify generation still works without context
    // Note: Untrained model may generate out-of-vocabulary tokens
    EXPECT_NO_THROW({ std::string response = rag->generate("What is machine learning?"); });
}

// TD-101 regression: generate()/generateWithRetrieval() used to call
// model->generate_response(prompt, config.gen_config.max_length) —
// generate_response()'s max_length parameter is a documented no-op, and
// temperature/top_k/top_p/num_beams were never passed at all despite
// RAGConfig::gen_config being a full TextGenerator::GenerationConfig.
// Verified directly against generator's config state (get_generator() is a
// public accessor), the same technique used for TD-100, rather than
// generated output length/content, which would be unreliable against an
// untrained, randomly-initialized model.
TEST_F(RAGInferenceTest, GenerateSyncsGeneratorConfigFromRAGConfig) {
    RAGInference::RAGConfig custom_config;
    custom_config.gen_config.max_length = 17;
    custom_config.gen_config.temperature = 0.42f;
    custom_config.gen_config.top_k = 7;
    custom_config.gen_config.top_p = 0.55f;
    custom_config.gen_config.num_beams = 1;  // stay off the separate beam-search code path

    RAGInference custom_rag(model, doc_store, custom_config);
    EXPECT_NO_THROW({ custom_rag.generate("What is machine learning?"); });

    auto cfg = model->get_generator()->get_config();
    EXPECT_EQ(cfg.max_length, 17);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.42f);
    EXPECT_EQ(cfg.top_k, 7);
    EXPECT_FLOAT_EQ(cfg.top_p, 0.55f);
}

TEST_F(RAGInferenceTest, GenerateWithRetrievalOutputsDocuments) {
    add_sample_documents();

    std::vector<std::pair<float, const Document*>> retrieved_docs;

    EXPECT_NO_THROW(
        { std::string response = rag->generateWithRetrieval("machine learning", retrieved_docs); });

    // Should have retrieved some documents
    EXPECT_GT(retrieved_docs.size(), 0u);
    EXPECT_LE(retrieved_docs.size(), 3u);  // Default num_retrieved_docs

    // All retrieved docs should be valid
    for (const auto& [score, doc] : retrieved_docs) {
        ASSERT_NE(doc, nullptr);
        EXPECT_GT(score, -2.0f);  // Similarity score range check
        EXPECT_LT(score, 2.0f);
    }
}

TEST_F(RAGInferenceTest, GenerateWithThresholdFiltering) {
    add_sample_documents();

    // Set high retrieval threshold
    RAGInference::RAGConfig config = rag->getConfig();
    config.retrieval_threshold = 0.9f;  // Very high threshold
    rag->setConfig(config);

    std::vector<std::pair<float, const Document*>> retrieved_docs;

    EXPECT_NO_THROW(
        { std::string response = rag->generateWithRetrieval("query", retrieved_docs); });

    // All retrieved docs should meet threshold
    for (const auto& [score, doc] : retrieved_docs) {
        EXPECT_GE(score, 0.9f);
    }
}

TEST_F(RAGInferenceTest, GenerateResponseTypeCorrect) {
    add_sample_documents();

    // Verify generation returns a string without crashing
    std::string response;
    EXPECT_NO_THROW({ response = rag->generate("test query"); });

    // Response type should be std::string (implicit by successful execution)
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(RAGInferenceTest, AddAndRemoveCycles) {
    // Cycle 1: Add and remove
    rag->addDocument("doc1", "First.");
    EXPECT_EQ(rag->getNumDocuments(), 1);
    rag->removeDocument("doc1");
    EXPECT_EQ(rag->getNumDocuments(), 0);

    // Cycle 2: Add same ID again
    rag->addDocument("doc1", "Second version.");
    const Document* doc = rag->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->text, "Second version.");
}

TEST_F(RAGInferenceTest, RetrievalConsistency) {
    add_sample_documents();

    auto results1 = rag->retrieveOnly("machine learning");
    auto results2 = rag->retrieveOnly("machine learning");

    // Same query should produce same results
    EXPECT_EQ(results1.size(), results2.size());

    for (size_t i = 0; i < results1.size(); ++i) {
        EXPECT_FLOAT_EQ(results1[i].first, results2[i].first);
        EXPECT_EQ(results1[i].second->id, results2[i].second->id);
    }
}

TEST_F(RAGInferenceTest, MultipleConfigurationChanges) {
    RAGInference::RAGConfig config1;
    config1.num_retrieved_docs = 2;
    rag->setConfig(config1);
    EXPECT_EQ(rag->getConfig().num_retrieved_docs, 2);

    RAGInference::RAGConfig config2;
    config2.num_retrieved_docs = 5;
    config2.retrieval_threshold = 0.5f;
    rag->setConfig(config2);
    EXPECT_EQ(rag->getConfig().num_retrieved_docs, 5);
    EXPECT_FLOAT_EQ(rag->getConfig().retrieval_threshold, 0.5f);

    RAGInference::RAGConfig config3;
    config3.num_retrieved_docs = 1;
    rag->setConfig(config3);
    EXPECT_EQ(rag->getConfig().num_retrieved_docs, 1);
}

TEST_F(RAGInferenceTest, RetrievalAfterDocumentRemoval) {
    rag->addDocument("doc1", "Machine learning.");
    rag->addDocument("doc2", "Python programming.");
    rag->addDocument("doc3", "Deep learning.");

    auto results_before = rag->retrieveOnly("machine learning");
    size_t count_before = results_before.size();

    // Remove one document
    rag->removeDocument("doc1");

    auto results_after = rag->retrieveOnly("machine learning");

    // Should have fewer or equal documents
    EXPECT_LE(results_after.size(), count_before);
}

TEST_F(RAGInferenceTest, EmptyQueryHandling) {
    add_sample_documents();

    // Empty query should throw TokenizerInputError from BPE tokenizer
    EXPECT_THROW(
        { std::string response = rag->generate(""); },
        std::invalid_argument);  // TokenizerInputError inherits from invalid_argument
}

// TD-487: a config that would fail every request is refused up front.
TEST_F(RAGInferenceTest, ConfigWithNoDocumentsOrNoContextIsRejected) {
    for (int bad : {0, -1}) {
        RAGInference::RAGConfig docs;
        docs.num_retrieved_docs = bad;
        EXPECT_THROW(RAGInference(model, doc_store, docs), std::invalid_argument) << bad;
        EXPECT_THROW(rag->setConfig(docs), std::invalid_argument) << bad;
        RAGInference::RAGConfig context;
        context.max_context_length = bad;
        EXPECT_THROW(RAGInference(model, doc_store, context), std::invalid_argument) << bad;
    }
    // A rejected setConfig() leaves the old config in place, and retrieval keeps working.
    add_sample_documents();
    EXPECT_EQ(rag->getConfig().num_retrieved_docs, 3);
    EXPECT_NO_THROW(rag->generate("machine learning"));
}

// TD-487: retrieval is logged, and a threshold that drops every document says so.
TEST_F(RAGInferenceTest, RetrievalIsLoggedAndAnEmptyingThresholdWarns) {
    add_sample_documents();
    std::ostringstream log;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
    auto logger = adai::Logger::get();
    const auto old_level = logger->level();
    logger->sinks().push_back(sink);
    logger->set_level(spdlog::level::debug);

    RAGInference::RAGConfig strict;
    strict.retrieval_threshold = 2.0f;  // above any cosine similarity
    rag->setConfig(strict);
    std::vector<std::pair<float, const Document*>> hits;
    const std::string prompt = rag->buildPrompt("machine learning", hits, "");

    logger->set_level(old_level);
    auto& sinks = logger->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());

    EXPECT_TRUE(hits.empty());
    EXPECT_EQ(prompt.find(strict.context_prefix), std::string::npos);
    EXPECT_NE(log.str().find("RAG: retrieved 'doc"), std::string::npos) << log.str();
    EXPECT_NE(log.str().find("scored below RAG_THRESHOLD"), std::string::npos) << log.str();
}

// TD-485: in a session longer than max_seq_length, retrieval follows the latest question.
TEST_F(RAGInferenceTest, RetrievalUsesTheLatestQuestionInALongSession) {
    const std::string doc_a = "python programming language data science";
    const std::string doc_b = "deep neural network framework";
    doc_store->addDocument("a", doc_a);
    doc_store->addDocument("b", doc_b);
    std::string history;
    for (int i = 0; i < 60; ++i) history += "User: " + doc_a + "\nAssistant: " + doc_a + "\n";

    // The old query, the whole conversation: the encoder sees only its first max_seq_length
    // tokens, all from the oldest turns, so the latest question takes no part.
    const auto old_hits = rag->retrieveOnly(history + "User: " + doc_b + "\n", 1);
    ASSERT_EQ(old_hits.size(), 1u);
    EXPECT_EQ(old_hits[0].second->id, "a");

    std::vector<std::pair<float, const Document*>> hits;
    rag->generateWithRetrieval(doc_b, hits, history);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].second->id, "b");
}

// ============================================================================
// TD-484: the prompt is budgeted in real tokens
// ============================================================================

class RAGInferencePromptBudgetTest : public ::testing::Test {
   protected:
    static constexpr int kMaxSeq = 48;
    std::shared_ptr<EncoderDecoderModel> model;
    std::unique_ptr<RAGInference> rag;
    BPETokenizer* tok = nullptr;

    void SetUp() override {
        model = std::make_shared<EncoderDecoderModel>(300, 32, 1, 1, 2, 64, kMaxSeq);
        tok = model->get_tokenizer();
        tok->build_vocab({"machine learning is a subset of artificial intelligence",
                          "python is a popular programming language for data science",
                          "deep learning uses neural networks with many layers",
                          "what is machine learning question context"},
                         300);
        auto encoder = std::make_shared<LLMEncoder>(300, 32, 1, 2, 64, kMaxSeq);
        rag = std::make_unique<RAGInference>(model, std::make_shared<DocumentStore>(encoder));
    }

    std::string prompt(const std::string& query, const std::string& context) {
        return rag->buildBudgetedPrompt(query, context);
    }
    std::string prompt_with_history(const std::string& query, const std::string& context,
                                    const std::string& history) {
        return rag->buildBudgetedPrompt(query, context, history);
    }
    size_t tokens(const std::string& text) { return text.empty() ? 0 : tok->encode(text, false).size(); }
    RAGInference::RAGConfig& config() { return rag->config; }

    static bool valid_utf8(const std::string& s) {
        for (size_t i = 0; i < s.size();) {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            const size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
            if (len == 0 || i + len > s.size()) return false;
            for (size_t k = 1; k < len; ++k)
                if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
            i += len;
        }
        return true;
    }
};

TEST_F(RAGInferencePromptBudgetTest, LongContextFitsMaxSeqLengthWithTheQuestionWhole) {
    const std::string doc1 = "machine learning is a subset of artificial intelligence";
    const std::string doc2 = "python is a popular programming language for data science";
    const std::string doc3 = "deep learning uses neural networks with many layers";
    const std::string context = doc1 + "\n\n" + doc2 + "\n\n" + doc3;
    const std::string query = "what is machine learning";
    config().max_context_length = kMaxSeq;  // as shipped: RAG_MAX_CONTEXT_LENGTH == MAX_SEQ_LENGTH

    // The old rule (4 characters per token, context only) overflowed max_seq_length here.
    const std::string old_context =
        context.size() <= static_cast<size_t>(kMaxSeq * 4) ? context : context.substr(0, kMaxSeq * 4 - 3) + "...";
    ASSERT_GT(tokens(config().context_prefix + old_context + config().query_prefix + query),
              static_cast<size_t>(kMaxSeq));

    const std::string p = prompt(query, context);
    EXPECT_LE(tokens(p), static_cast<size_t>(kMaxSeq)) << p;
    const std::string question = config().query_prefix + query;
    ASSERT_GE(p.size(), question.size());
    EXPECT_EQ(p.substr(p.size() - question.size()), question) << "the question must be kept whole";
    EXPECT_NE(p.find(doc1.substr(0, 20)), std::string::npos) << "the top-ranked document goes in first";
    EXPECT_EQ(p.find("neural"), std::string::npos) << "the lowest-ranked document is dropped first";
}

TEST_F(RAGInferencePromptBudgetTest, ContextThatFitsIsUnchanged) {
    const std::string p = prompt("what is it", "machine learning");
    EXPECT_EQ(p, config().context_prefix + "machine learning" + config().query_prefix + "what is it");
}

TEST_F(RAGInferencePromptBudgetTest, ContextIsCappedByMaxContextLength) {
    config().max_context_length = 5;
    const std::string p = prompt("what", "machine learning is a subset of artificial intelligence");
    const std::string ctx = p.substr(config().context_prefix.size(),
                                     p.size() - config().context_prefix.size() - config().query_prefix.size() - 4);
    EXPECT_LE(tokens(ctx), 5u) << ctx;
    EXPECT_FALSE(ctx.empty());
}

TEST_F(RAGInferencePromptBudgetTest, CutsAtUtf8Boundaries) {
    std::string context;
    for (int i = 0; i < 200; ++i) context += "caf\xc3\xa9 \xe2\x82\xac ";  // "café € "
    for (int limit : {3, 4, 5, 7, 11, 20}) {
        config().max_context_length = limit;
        const std::string p = prompt("what", context);
        EXPECT_TRUE(valid_utf8(p)) << "limit " << limit;
        EXPECT_LE(tokens(p), static_cast<size_t>(kMaxSeq));
    }
}

TEST_F(RAGInferencePromptBudgetTest, NoRoomMeansTheQuestionAlone) {
    const std::string question_only = config().query_prefix + "what";
    config().max_context_length = 0;
    EXPECT_EQ(prompt("what", "machine learning"), question_only);
    config().max_context_length = -3;
    EXPECT_EQ(prompt("what", "machine learning"), question_only);

    config().max_context_length = 512;
    std::string long_query;
    for (int i = 0; i < 60; ++i) long_query += "machine learning ";
    EXPECT_EQ(prompt(long_query, "python"), config().query_prefix + long_query)
        << "a question longer than max_seq_length gets no context";
}

// TD-485: context, then the end of the conversation, then the question, within max_seq_length.
TEST_F(RAGInferencePromptBudgetTest, HistoryKeepsItsLatestTurnsWithinTheBudget) {
    std::string history;
    for (int i = 0; i < 30; ++i) history += "User: turn" + std::to_string(i) + " machine learning\n";
    const std::string context = "python is a popular programming language for data science";
    config().max_context_length = 12;

    const std::string p = prompt_with_history("what is deep learning", context, history);
    EXPECT_LE(tokens(p), static_cast<size_t>(kMaxSeq)) << p;
    const std::string question = config().query_prefix + "what is deep learning";
    EXPECT_EQ(p.substr(p.size() - question.size()), question);
    EXPECT_EQ(p.rfind(config().context_prefix, 0), 0u) << "the context comes first";
    // What is kept of the history is its end (token counts vary with TD-222's shared tokenizer cache,
    // so the test doesn't depend on how much fits).
    const size_t cut = p.rfind("...", p.size() - question.size());
    ASSERT_NE(cut, std::string::npos) << p;
    const std::string kept = p.substr(cut + 3, p.size() - question.size() - cut - 3);
    ASSERT_FALSE(kept.empty()) << p;
    EXPECT_EQ(history.substr(history.size() - kept.size()), kept) << "the history's end is kept: " << p;
    EXPECT_EQ(p.find("turn0 "), std::string::npos) << "the oldest turns go first";
}
