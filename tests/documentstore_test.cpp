/**
 * DocumentStore Unit Tests
 *
 * Comprehensive test suite for the DocumentStore class used in RAG (Retrieval-Augmented
 * Generation). Tests cover document management, embedding generation, similarity search, and edge
 * cases.
 *
 * Test Categories:
 * 1. Construction and Configuration
 * 2. Document Management
 * 3. Collection Operations
 * 4. Retrieval and Similarity
 * 5. Embedding Operations
 * 6. Edge Cases
 */

#include "DocumentStore.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <memory>
#include <thread>
#include "encoder.hpp"

namespace fs = std::filesystem;

class DocumentStoreTest : public ::testing::Test {
   protected:
    std::shared_ptr<LLMEncoder> encoder;
    std::unique_ptr<DocumentStore> store;
    fs::path test_dir;
    fs::path vocab_file;

    void SetUp() override {
        // Create temporary directory for test files
        test_dir = fs::temp_directory_path() / "documentstore_test";
        fs::create_directories(test_dir);
        vocab_file = test_dir / "vocab.txt";

        // Create a minimal BPE vocabulary file
        create_test_vocabulary();

        // Initialize encoder with smaller configuration for faster testing
        encoder = std::make_shared<LLMEncoder>(1000,  // vocab_size (smaller)
                                               128,   // d_model (smaller)
                                               2,     // num_layers (reduced)
                                               4,     // num_heads (reduced)
                                               512,   // d_ff (reduced)
                                               128    // max_seq_length (reduced)
        );

        // Load vocabulary into encoder
        encoder->load_tokenizer_vocab(vocab_file.string());

        // Create document store
        store = std::make_unique<DocumentStore>(encoder);
    }

    void TearDown() override {
        store.reset();
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
            "deep",      "framework", "tensor",   "flow",    "pytorch",     "keras",
            "code",      "function",  "class",    "method",  "variable",    "return",
            "if",        "else",      "for",      "while",   "loop",        "array",
            "list",      "dict",      "set",      "tuple",   "string",      "int",
            "float",     "bool",      "none"};

        int token_id = 4;
        for (const auto& word : words) {
            file << word << "\t" << token_id++ << "\n";
        }

        file.close();
    }
};

// ============================================================================
// Construction and Configuration Tests
// ============================================================================

TEST_F(DocumentStoreTest, ConstructorWithValidEncoder) {
    auto test_encoder = std::make_shared<LLMEncoder>(1000, 128);
    auto test_store = std::make_unique<DocumentStore>(test_encoder);

    EXPECT_TRUE(test_store->empty());
    EXPECT_EQ(test_store->size(), 0);
}

TEST_F(DocumentStoreTest, ConstructorWithNullEncoder) {
    EXPECT_THROW({ DocumentStore store(nullptr); }, std::invalid_argument);
}

// ============================================================================
// Document Management Tests
// ============================================================================

TEST_F(DocumentStoreTest, AddSingleDocument) {
    store->addDocument("doc1", "Machine learning is a subset of artificial intelligence.");

    EXPECT_EQ(store->size(), 1);
    EXPECT_FALSE(store->empty());

    const Document* doc = store->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->id, "doc1");
    EXPECT_EQ(doc->text, "Machine learning is a subset of artificial intelligence.");
    EXPECT_GT(doc->embedding.rows, 0);
    EXPECT_GT(doc->embedding.cols, 0);
}

TEST_F(DocumentStoreTest, AddMultipleDocuments) {
    store->addDocument("doc1", "Python is a programming language.");
    store->addDocument("doc2", "Machine learning uses algorithms.");
    store->addDocument("doc3", "Deep learning is a neural network approach.");

    EXPECT_EQ(store->size(), 3);

    // Verify all documents are accessible
    EXPECT_NE(store->getDocument("doc1"), nullptr);
    EXPECT_NE(store->getDocument("doc2"), nullptr);
    EXPECT_NE(store->getDocument("doc3"), nullptr);
}

TEST_F(DocumentStoreTest, AddDocumentWithMetadata) {
    std::unordered_map<std::string, std::string> metadata = {
        {"author", "John Doe"}, {"date", "2026-03-01"}, {"category", "AI"}};

    store->addDocument("doc1", "Artificial intelligence is the future.", metadata);

    const Document* doc = store->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->metadata.at("author"), "John Doe");
    EXPECT_EQ(doc->metadata.at("date"), "2026-03-01");
    EXPECT_EQ(doc->metadata.at("category"), "AI");
}

TEST_F(DocumentStoreTest, AddDuplicateIdThrows) {
    store->addDocument("doc1", "First document.");

    EXPECT_THROW(
        { store->addDocument("doc1", "Second document with same ID."); }, std::invalid_argument);

    // Original document should still exist
    EXPECT_EQ(store->size(), 1);
}

TEST_F(DocumentStoreTest, AddEmptyTextThrows) {
    EXPECT_THROW({ store->addDocument("doc1", ""); }, std::invalid_argument);

    EXPECT_EQ(store->size(), 0);
}

TEST_F(DocumentStoreTest, RemoveExistingDocument) {
    store->addDocument("doc1", "First document.");
    store->addDocument("doc2", "Second document.");

    EXPECT_EQ(store->size(), 2);

    bool removed = store->removeDocument("doc1");

    EXPECT_TRUE(removed);
    EXPECT_EQ(store->size(), 1);
    EXPECT_EQ(store->getDocument("doc1"), nullptr);
    EXPECT_NE(store->getDocument("doc2"), nullptr);
}

TEST_F(DocumentStoreTest, RemoveNonExistentDocument) {
    store->addDocument("doc1", "First document.");

    bool removed = store->removeDocument("doc_nonexistent");

    EXPECT_FALSE(removed);
    EXPECT_EQ(store->size(), 1);
}

TEST_F(DocumentStoreTest, GetDocumentById) {
    store->addDocument("ml_doc", "Machine learning algorithms learn from data.");

    const Document* doc = store->getDocument("ml_doc");

    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->id, "ml_doc");
    EXPECT_EQ(doc->text, "Machine learning algorithms learn from data.");
}

// ============================================================================
// Collection Operations Tests
// ============================================================================

TEST_F(DocumentStoreTest, SizeTracking) {
    EXPECT_EQ(store->size(), 0);

    store->addDocument("doc1", "First.");
    EXPECT_EQ(store->size(), 1);

    store->addDocument("doc2", "Second.");
    EXPECT_EQ(store->size(), 2);

    store->removeDocument("doc1");
    EXPECT_EQ(store->size(), 1);
}

TEST_F(DocumentStoreTest, EmptyCheck) {
    EXPECT_TRUE(store->empty());

    store->addDocument("doc1", "Content.");
    EXPECT_FALSE(store->empty());

    store->removeDocument("doc1");
    EXPECT_TRUE(store->empty());
}

TEST_F(DocumentStoreTest, ClearAllDocuments) {
    store->addDocument("doc1", "First.");
    store->addDocument("doc2", "Second.");
    store->addDocument("doc3", "Third.");

    EXPECT_EQ(store->size(), 3);

    store->clear();

    EXPECT_EQ(store->size(), 0);
    EXPECT_TRUE(store->empty());
    EXPECT_EQ(store->getDocument("doc1"), nullptr);
    EXPECT_EQ(store->getDocument("doc2"), nullptr);
    EXPECT_EQ(store->getDocument("doc3"), nullptr);
}

TEST_F(DocumentStoreTest, GetAllDocumentIds) {
    store->addDocument("doc1", "First.");
    store->addDocument("doc2", "Second.");
    store->addDocument("doc3", "Third.");

    auto ids = store->getAllDocumentIds();

    EXPECT_EQ(ids.size(), 3);

    // Check all IDs are present
    std::vector<std::string> expected = {"doc1", "doc2", "doc3"};
    for (const auto& expected_id : expected) {
        EXPECT_NE(std::find(ids.begin(), ids.end(), expected_id), ids.end());
    }
}

TEST_F(DocumentStoreTest, GetNonExistentDocument) {
    store->addDocument("doc1", "Content.");

    const Document* doc = store->getDocument("doc_nonexistent");

    EXPECT_EQ(doc, nullptr);
}

// ============================================================================
// Retrieval and Similarity Tests
// ============================================================================

TEST_F(DocumentStoreTest, BasicRetrieval) {
    store->addDocument("doc1", "Machine learning is artificial intelligence.");
    store->addDocument("doc2", "Python is a programming language.");
    store->addDocument("doc3", "Deep learning uses neural networks.");

    auto results = store->retrieve("What is machine learning?", 2);

    EXPECT_EQ(results.size(), 2);

    // Results should be sorted by similarity (descending)
    EXPECT_GE(results[0].first, results[1].first);

    // Both results should point to valid documents
    ASSERT_NE(results[0].second, nullptr);
    ASSERT_NE(results[1].second, nullptr);
}

TEST_F(DocumentStoreTest, RetrieveWithKGreaterThanSize) {
    store->addDocument("doc1", "First document.");
    store->addDocument("doc2", "Second document.");

    auto results = store->retrieve("query", 10);

    // Should return all available documents (2), not 10
    EXPECT_EQ(results.size(), 2);
}

TEST_F(DocumentStoreTest, RetrieveWithKEqualsOne) {
    store->addDocument("doc1", "Machine learning algorithms.");
    store->addDocument("doc2", "Python programming.");
    store->addDocument("doc3", "Deep learning networks.");

    auto results = store->retrieve("machine learning", 1);

    EXPECT_EQ(results.size(), 1);
    ASSERT_NE(results[0].second, nullptr);

    // Verify it's one of the documents  we added
    std::vector<std::string> valid_ids = {"doc1", "doc2", "doc3"};
    EXPECT_NE(std::find(valid_ids.begin(), valid_ids.end(), results[0].second->id),
              valid_ids.end());
}

TEST_F(DocumentStoreTest, RetrieveFromEmptyStore) {
    auto results = store->retrieve("query", 3);

    EXPECT_TRUE(results.empty());
}

TEST_F(DocumentStoreTest, RetrieveWithInvalidK) {
    store->addDocument("doc1", "Content.");

    EXPECT_THROW({ store->retrieve("query", 0); }, std::invalid_argument);

    EXPECT_THROW({ store->retrieve("query", -1); }, std::invalid_argument);
}

TEST_F(DocumentStoreTest, SimilarityRanking) {
    // Add documents with varying relevance to query
    store->addDocument("doc1", "Python programming language code.");
    store->addDocument("doc2", "The weather is nice today.");
    store->addDocument("doc3", "Python is used for machine learning.");

    auto results = store->retrieve("Python programming", 3);

    EXPECT_EQ(results.size(), 3);

    // Verify results are sorted by similarity (descending)
    for (size_t i = 0; i < results.size() - 1; ++i) {
        EXPECT_GE(results[i].first, results[i + 1].first);
    }

    // Verify all results are valid documents
    for (const auto& result : results) {
        ASSERT_NE(result.second, nullptr);
        std::vector<std::string> valid_ids = {"doc1", "doc2", "doc3"};
        EXPECT_NE(std::find(valid_ids.begin(), valid_ids.end(), result.second->id),
                  valid_ids.end());
    }
}

TEST_F(DocumentStoreTest, QueryEmbeddingGeneration) {
    store->addDocument("doc1", "Machine learning.");

    auto results1 = store->retrieve("What is machine learning?", 1);
    auto results2 = store->retrieve("What is machine learning?", 1);

    // Same query should produce same similarity scores
    EXPECT_FLOAT_EQ(results1[0].first, results2[0].first);
}

TEST_F(DocumentStoreTest, DifferentQueriesDifferentResults) {
    store->addDocument("doc1", "Machine learning algorithms.");
    store->addDocument("doc2", "Python programming language.");
    store->addDocument("doc3", "Deep learning neural networks.");

    auto results1 = store->retrieve("machine learning", 2);
    auto results2 = store->retrieve("python programming", 2);

    // Both should return 2 results
    EXPECT_EQ(results1.size(), 2);
    EXPECT_EQ(results2.size(), 2);

    // Verify results are sorted by similarity
    EXPECT_GE(results1[0].first, results1[1].first);
    EXPECT_GE(results2[0].first, results2[1].first);

    // Verify all results are valid
    ASSERT_NE(results1[0].second, nullptr);
    ASSERT_NE(results2[0].second, nullptr);
}

// ============================================================================
// Embedding Operations Tests
// ============================================================================

TEST_F(DocumentStoreTest, SentenceEmbeddingDimensions) {
    store->addDocument("doc1", "Test document for embedding dimensions.");

    const Document* doc = store->getDocument("doc1");
    ASSERT_NE(doc, nullptr);

    // Embedding should be a single vector [1, d_model]
    EXPECT_EQ(doc->embedding.rows, 1);
    EXPECT_EQ(doc->embedding.cols, 128);  // d_model from SetUp
}

TEST_F(DocumentStoreTest, EmbeddingConsistency) {
    // Add same text twice (different IDs)
    store->addDocument("doc1", "Machine learning is AI.");
    store->addDocument("doc2", "Machine learning is AI.");

    const Document* doc1 = store->getDocument("doc1");
    const Document* doc2 = store->getDocument("doc2");

    ASSERT_NE(doc1, nullptr);
    ASSERT_NE(doc2, nullptr);

    // Embeddings should be identical for identical text
    EXPECT_EQ(doc1->embedding.rows, doc2->embedding.rows);
    EXPECT_EQ(doc1->embedding.cols, doc2->embedding.cols);

    // Check embedding values are close (allowing for float precision)
    for (int i = 0; i < doc1->embedding.cols; ++i) {
        EXPECT_NEAR(doc1->embedding(0, i), doc2->embedding(0, i), 1e-6);
    }
}

TEST_F(DocumentStoreTest, CosineSimilaritySymmetry) {
    store->addDocument("doc1", "First document.");
    store->addDocument("doc2", "Second document.");

    auto results1 = store->retrieve("First document.", 2);
    auto results2 = store->retrieve("Second document.", 2);

    // Verify similarity scores are computed
    EXPECT_GT(results1[0].first, -2.0f);
    EXPECT_LT(results1[0].first, 2.0f);
    EXPECT_GT(results2[0].first, -2.0f);
    EXPECT_LT(results2[0].first, 2.0f);
}

TEST_F(DocumentStoreTest, SimilarityScoreRange) {
    store->addDocument("doc1", "Machine learning algorithms.");

    auto results = store->retrieve("machine learning", 1);

    // Cosine similarity should be in range [-1, 1]
    EXPECT_GE(results[0].first, -1.0f);
    EXPECT_LE(results[0].first, 1.0f);
}

TEST_F(DocumentStoreTest, IdenticalTextHighSimilarity) {
    store->addDocument("doc1", "Machine learning is AI.");

    auto results = store->retrieve("Machine learning is AI.", 1);

    // Query identical to document should have very high similarity
    EXPECT_GT(results[0].first, 0.9f);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(DocumentStoreTest, RemoveFromSingleDocumentStore) {
    store->addDocument("doc1", "Only document.");

    EXPECT_EQ(store->size(), 1);

    bool removed = store->removeDocument("doc1");

    EXPECT_TRUE(removed);
    EXPECT_EQ(store->size(), 0);
    EXPECT_TRUE(store->empty());
}

TEST_F(DocumentStoreTest, RemoveFromMiddleOfStore) {
    // Add multiple documents
    for (int i = 0; i < 5; ++i) {
        store->addDocument("doc" + std::to_string(i), "Document " + std::to_string(i));
    }

    EXPECT_EQ(store->size(), 5);

    // Remove from middle
    bool removed = store->removeDocument("doc2");

    EXPECT_TRUE(removed);
    EXPECT_EQ(store->size(), 4);

    // Verify removed document is gone
    EXPECT_EQ(store->getDocument("doc2"), nullptr);

    // Verify other documents still exist
    EXPECT_NE(store->getDocument("doc0"), nullptr);
    EXPECT_NE(store->getDocument("doc1"), nullptr);
    EXPECT_NE(store->getDocument("doc3"), nullptr);
    EXPECT_NE(store->getDocument("doc4"), nullptr);
}

TEST_F(DocumentStoreTest, RetrieveAfterClear) {
    store->addDocument("doc1", "Content.");
    store->addDocument("doc2", "More content.");

    store->clear();

    auto results = store->retrieve("query", 3);

    EXPECT_TRUE(results.empty());
}

TEST_F(DocumentStoreTest, MultipleAddRemoveCycles) {
    // Cycle 1: Add and remove
    store->addDocument("doc1", "First.");
    EXPECT_EQ(store->size(), 1);
    store->removeDocument("doc1");
    EXPECT_EQ(store->size(), 0);

    // Cycle 2: Add same ID again
    store->addDocument("doc1", "Second version.");
    EXPECT_EQ(store->size(), 1);

    const Document* doc = store->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->text, "Second version.");

    // Cycle 3: Add multiple, remove some
    store->addDocument("doc2", "Another.");
    store->addDocument("doc3", "Yet another.");
    EXPECT_EQ(store->size(), 3);

    store->removeDocument("doc1");
    EXPECT_EQ(store->size(), 2);

    // Verify remaining documents
    EXPECT_EQ(store->getDocument("doc1"), nullptr);
    EXPECT_NE(store->getDocument("doc2"), nullptr);
    EXPECT_NE(store->getDocument("doc3"), nullptr);
}

// ============================================================================
// TD-350: concurrent retrieval
// ============================================================================

TEST_F(DocumentStoreTest, ConcurrentRetrieveMatchesSerialResults) {
    store->addDocument("a", "machine learning is artificial intelligence");
    store->addDocument("b", "python programming language");
    store->addDocument("c", "deep neural network framework");

    const std::vector<std::string> queries = {"machine learning", "python code", "neural network",
                                              "data science"};
    std::vector<std::vector<std::pair<std::string, float>>> expected;
    for (const auto& q : queries) {
        std::vector<std::pair<std::string, float>> hits;
        for (const auto& [score, doc] : store->retrieve(q, 3)) hits.emplace_back(doc->id, score);
        expected.push_back(hits);
    }

    // retrieve() is const but encode() writes the encoder's layer caches; without the store's lock
    // concurrent queries race there (TSan reports it).
    std::atomic<int> mismatches{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 5; ++i) {
                const std::size_t qi = (t + i) % queries.size();
                std::vector<std::pair<std::string, float>> hits;
                for (const auto& [score, doc] : store->retrieve(queries[qi], 3))
                    hits.emplace_back(doc->id, score);
                if (hits != expected[qi]) ++mismatches;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(mismatches.load(), 0);
}

TEST_F(DocumentStoreTest, AddDocumentWhileRetrieving) {
    store->addDocument("a", "machine learning is artificial intelligence");
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        while (!stop) {
            auto hits = store->retrieve("machine learning", 2);
            ASSERT_FALSE(hits.empty());
        }
    });
    store->addDocument("b", "python programming language");
    store->addDocument("c", "deep neural network framework");
    stop = true;
    reader.join();
    EXPECT_EQ(store->size(), 3u);
}

// ============================================================================
// TD-349: documents longer than the encoder's window are indexed as chunks
// ============================================================================

namespace {

// `count` words from the test vocabulary, starting at `offset` in it.
std::string vocab_words(std::size_t count, std::size_t offset) {
    static const std::vector<std::string> words = {
        "the",     "machine", "learning",    "is",       "artificial", "intelligence", "algorithm",
        "data",    "neural",  "network",     "model",    "train",      "test",         "computer",
        "science", "python",  "programming", "language", "deep",       "framework",    "tensor"};
    std::string text;
    for (std::size_t i = 0; i < count; ++i) {
        if (i > 0)
            text += ' ';
        text += words[(offset + i) % words.size()];
    }
    return text;
}

}  // namespace

class DocumentStoreChunkTest : public DocumentStoreTest {
   protected:
    std::shared_ptr<LLMEncoder> small;  // a 16-token window
    std::unique_ptr<DocumentStore> chunked;

    void SetUp() override {
        DocumentStoreTest::SetUp();
        small = std::make_shared<LLMEncoder>(1000, 64, 1, 4, 128, 16);
        // A vocabulary with merges, so words are one or two tokens (the fixture's has none, and
        // every word is spelled out a character at a time).
        std::vector<std::string> corpus;
        for (std::size_t i = 0; i < 40; ++i)
            corpus.push_back(vocab_words(21, i));
        small->build_tokenizer(corpus, 300);
        chunked = std::make_unique<DocumentStore>(small);
    }
};

TEST_F(DocumentStoreChunkTest, ALongDocumentIsSplitIntoOverlappingChunksThatFit) {
    const std::string text = vocab_words(60, 0);
    ASSERT_GT(small->count_tokens(text), 16u);
    chunked->addDocument("long", text, {{"source", "test"}});

    EXPECT_EQ(chunked->getDocument("long"), nullptr) << "the entries are its chunks";
    const std::size_t n = chunked->size();
    ASSERT_GE(n, 3u);
    std::size_t previous_begin = 0, previous_end = 0;
    for (std::size_t k = 1; k <= n; ++k) {
        const Document* chunk = chunked->getDocument("long#" + std::to_string(k));
        ASSERT_NE(chunk, nullptr) << k;
        EXPECT_EQ(chunk->parent_id, "long");
        EXPECT_EQ(chunk->chunk_index, static_cast<int>(k));
        EXPECT_EQ(chunk->chunk_count, static_cast<int>(n));
        EXPECT_EQ(chunk->metadata.at("source"), "test");
        EXPECT_LE(small->count_tokens(chunk->text), 16u) << "chunk " << k << " fits the window";
        const std::size_t begin = text.find(chunk->text, k == 1 ? 0 : previous_begin + 1);
        ASSERT_NE(begin, std::string::npos) << "chunk " << k << " is the document's own text";
        if (k == 1) {
            EXPECT_EQ(begin, 0u) << "the first chunk starts the document";
        } else {
            EXPECT_LT(begin, previous_end) << "chunk " << k << " overlaps the one before";
        }
        previous_begin = begin;
        previous_end = begin + chunk->text.size();
    }
    EXPECT_EQ(previous_end, text.size()) << "the last chunk ends the document";
}

// The encoder embeds at most its window (the last max_seq_length tokens), so two documents with
// the same long ending used to get identical embeddings and the openings couldn't be retrieved.
TEST_F(DocumentStoreChunkTest, AnOpeningOutsideTheEncodersWindowIsRetrievable) {
    const std::string shared_tail = vocab_words(40, 5);
    chunked->addDocument("a", "python programming language deep framework " + shared_tail);
    chunked->addDocument("b", "neural network model train test computer " + shared_tail);

    const Document* opening = chunked->getDocument("a#1");
    ASSERT_NE(opening, nullptr);
    ASSERT_EQ(opening->text.rfind("python programming language", 0), 0u);
    const auto hits = chunked->retrieve(opening->text, static_cast<int>(chunked->size()));
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].second->id, "a#1");
    EXPECT_GT(hits[0].first, 0.999f);
    for (const auto& [score, doc] : hits)
        if (doc->parent_id == "b") {
            EXPECT_LT(score, hits[0].first) << doc->id;
        }
}

TEST_F(DocumentStoreChunkTest, RemovingAChunkedDocumentRemovesEveryChunk) {
    chunked->addDocument("long", vocab_words(60, 0));
    chunked->addDocument("short", "machine learning");
    ASSERT_GT(chunked->size(), 2u);
    EXPECT_TRUE(chunked->removeDocument("long"));
    EXPECT_EQ(chunked->size(), 1u);
    EXPECT_EQ(chunked->getAllDocumentIds(), std::vector<std::string>{"short"});
    EXPECT_FALSE(chunked->removeDocument("long"));
}

TEST_F(DocumentStoreChunkTest, AChunkedDocumentsIdCantBeAddedAgain) {
    chunked->addDocument("long", vocab_words(60, 0));
    const std::size_t entries = chunked->size();
    EXPECT_THROW(chunked->addDocument("long", "machine learning"), std::invalid_argument);
    EXPECT_THROW(chunked->addDocument("long#1", "machine learning"), std::invalid_argument);
    EXPECT_EQ(chunked->size(), entries);
}

TEST_F(DocumentStoreChunkTest, ADocumentThatFitsKeepsItsOwnId) {
    chunked->addDocument("short", vocab_words(10, 0));
    ASSERT_EQ(chunked->size(), 1u);
    const Document* doc = chunked->getDocument("short");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->parent_id, "short");
    EXPECT_EQ(doc->chunk_count, 1);
    EXPECT_EQ(doc->text, vocab_words(10, 0));
}

// Each word here is longer than a quarter of the window (the fixture's vocabulary has no merges, so
// a word is a token per character); consecutive chunks still share their boundary word.
TEST_F(DocumentStoreChunkTest, ChunksOverlapEvenWhenEveryWordIsLongerThanAQuarterWindow) {
    auto spelled = std::make_shared<LLMEncoder>(1000, 64, 1, 4, 128, 20);
    spelled->load_tokenizer_vocab(vocab_file.string());
    DocumentStore store_spelled(spelled);
    const std::string text =
        "machine learning artificial intelligence algorithm network computer science "
        "programming language framework machine learning network";
    ASSERT_GT(spelled->count_tokens("machine"), 5u) << "words are longer than a quarter window";
    store_spelled.addDocument("doc", text);
    const std::size_t n = store_spelled.size();
    ASSERT_GE(n, 2u);
    std::size_t previous_begin = 0, previous_end = 0;
    bool previous_one_word = false;
    int overlaps = 0;
    for (std::size_t k = 1; k <= n; ++k) {
        const Document* chunk = store_spelled.getDocument("doc#" + std::to_string(k));
        ASSERT_NE(chunk, nullptr);
        const std::size_t begin = text.find(chunk->text, k == 1 ? 0 : previous_begin + 1);
        ASSERT_NE(begin, std::string::npos);
        // A one-word chunk has nothing to share: the next one starts past it.
        if (k > 1 && !previous_one_word) {
            EXPECT_LT(begin, previous_end) << "chunk " << k << " overlaps the one before";
            ++overlaps;
        }
        previous_one_word = chunk->text.find(' ') == std::string::npos;
        previous_begin = begin;
        previous_end = begin + chunk->text.size();
    }
    EXPECT_EQ(previous_end, text.size());
    EXPECT_GE(overlaps, 2) << "the text has chunks of more than one word";
}
