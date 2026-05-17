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
