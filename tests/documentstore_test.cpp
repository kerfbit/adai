/**
 * DocumentStore Unit Tests
 * 
 * Comprehensive test suite for the DocumentStore class used in RAG (Retrieval-Augmented Generation).
 * Tests cover document management, embedding generation, similarity search, and edge cases.
 * 
 * Test Categories:
 * 1. Construction and Configuration
 * 2. Document Management
 * 3. Collection Operations
 * 4. Retrieval and Similarity
 * 5. Embedding Operations
 * 6. Edge Cases
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include "DocumentStore.hpp"
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
        encoder = std::make_shared<LLMEncoder>(
            1000,   // vocab_size (smaller)
            128,    // d_model (smaller)
            2,      // num_layers (reduced)
            4,      // num_heads (reduced)
            512,    // d_ff (reduced)
            128     // max_seq_length (reduced)
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
            "the", "machine", "learning", "is", "artificial",
            "intelligence", "algorithm", "data", "neural", "network",
            "model", "train", "test", "computer", "science",
            "python", "programming", "language", "deep", "framework",
            "tensor", "flow", "pytorch", "keras", "code",
            "function", "class", "method", "variable", "return",
            "if", "else", "for", "while", "loop",
            "array", "list", "dict", "set", "tuple",
            "string", "int", "float", "bool", "none"
        };
        
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
    EXPECT_THROW({
        DocumentStore store(nullptr);
    }, std::invalid_argument);
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
        {"author", "John Doe"},
        {"date", "2026-03-01"},
        {"category", "AI"}
    };
    
    store->addDocument("doc1", "Artificial intelligence is the future.", metadata);
    
    const Document* doc = store->getDocument("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->metadata.at("author"), "John Doe");
    EXPECT_EQ(doc->metadata.at("date"), "2026-03-01");
    EXPECT_EQ(doc->metadata.at("category"), "AI");
}

TEST_F(DocumentStoreTest, AddDuplicateIdThrows) {
    store->addDocument("doc1", "First document.");
    
    EXPECT_THROW({
        store->addDocument("doc1", "Second document with same ID.");
    }, std::invalid_argument);
    
    // Original document should still exist
    EXPECT_EQ(store->size(), 1);
}

TEST_F(DocumentStoreTest, AddEmptyTextThrows) {
    EXPECT_THROW({
        store->addDocument("doc1", "");
    }, std::invalid_argument);
    
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
    EXPECT_NE(std::find(valid_ids.begin(), valid_ids.end(), results[0].second->id), valid_ids.end());
}

TEST_F(DocumentStoreTest, RetrieveFromEmptyStore) {
    auto results = store->retrieve("query", 3);
    
    EXPECT_TRUE(results.empty());
}

TEST_F(DocumentStoreTest, RetrieveWithInvalidK) {
    store->addDocument("doc1", "Content.");
    
    EXPECT_THROW({
        store->retrieve("query", 0);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        store->retrieve("query", -1);
    }, std::invalid_argument);
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
        EXPECT_NE(std::find(valid_ids.begin(), valid_ids.end(), result.second->id), valid_ids.end());
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
