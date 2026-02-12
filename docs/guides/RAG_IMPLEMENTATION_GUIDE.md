# RAG Implementation Guide

## Overview

This directory contains a complete implementation of **RAG (Retrieval-Augmented Generation)** for the ADAI chatbot system. RAG combines information retrieval with text generation to produce factually-grounded, context-aware responses.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a hybrid approach that:

1. **Retrieves** relevant documents from a knowledge base using semantic search
2. **Augments** the user's query with retrieved context
3. **Generates** a response conditioned on both query and evidence

### Benefits of RAG

✅ **Factual Grounding**: Responses based on retrieved evidence, not just learned parameters
✅ **Up-to-date Knowledge**: Add new information by simply adding documents (no retraining)
✅ **Reduced Hallucination**: Model has access to real information, not just memorized patterns
✅ **Explainability**: Can cite source documents for generated responses
✅ **Scalable Knowledge**: Knowledge base can grow without model retraining
✅ **Cost-Effective**: Avoids expensive retraining for knowledge updates

## RAG vs BERT Comparison

| Feature | RAG | BERT |
| --------- | ----- | ------ |
| **Primary Use** | Question answering, factual generation | Classification, embeddings, NER |
| **Text Generation** | ✅ Yes (decoder component) | ❌ No (encoder-only) |
| **Knowledge Updates** | ✅ Add documents instantly | ❌ Requires retraining |
| **Factual Accuracy** | ✅ High (grounded in evidence) | ⚠️ Limited to training data |
| **Hallucination** | ✅ Reduced | ⚠️ Can hallucinate |
| **Inference Speed** | ⚠️ Slower (retrieval overhead) | ✅ Fast (encoder-only) |
| **Storage** | ⚠️ Requires vector database | ✅ Model weights only |
| **Explainability** | ✅ Can cite sources | ⚠️ Black box |
| **Best For** | Q&A, chatbots, dynamic knowledge | Embeddings, classification, semantic search |

### When to Use RAG

- ✅ Building Q&A systems requiring factual accuracy
- ✅ Chatbots needing up-to-date information
- ✅ Applications where knowledge changes frequently
- ✅ Systems requiring source citation
- ✅ Reducing hallucination in generated text

### When to Use BERT

- ✅ Text classification tasks
- ✅ Named entity recognition (NER)
- ✅ Sentiment analysis
- ✅ Semantic similarity / embedding generation
- ✅ Fast inference requirements
- ✅ No text generation needed

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    RAG System Architecture                  │
└─────────────────────────────────────────────────────────────┘

User Query: "What is machine learning?"
     │
     ▼
┌─────────────────────┐
│  DocumentStore      │──── Contains knowledge base
│  (Vector Database)  │     with document embeddings
└─────────────────────┘
     │
     │ Semantic Search (cosine similarity)
     ▼
┌─────────────────────┐
│  Top-k Documents    │──── Retrieved: "ML is a subset of AI..."
│  (Retriever)        │              "Deep learning uses neural nets..."
└─────────────────────┘
     │
     │ Context Formatting
     ▼
Augmented Prompt:
"Context: [Retrieved documents...]
 Question: What is machine learning?"
     │
     ▼
┌─────────────────────┐
│ EncoderDecoderModel │──── Generates response conditioned
│  (Seq2Seq)          │     on query + retrieved context
└─────────────────────┘
     │
     ▼
Response: "Machine learning is a subset of artificial intelligence..."
```

## Components

### 1. DocumentStore (`DocumentStore.hpp/cpp`)

Manages the knowledge base with semantic search capabilities.

**Features:**

- Document indexing with encoder-based embeddings
- Cosine similarity search for top-k retrieval
- Document CRUD operations (add, remove, update)
- Metadata storage

**API:**

```cpp
auto doc_store = std::make_shared<DocumentStore>(encoder);

// Add documents
doc_store->addDocument("ml_def", "Machine learning is...");

// Retrieve top-k similar documents
auto results = doc_store->retrieve("What is ML?", k=3);
// Returns: vector<pair<float, const Document*>> with scores

// Manage documents
doc_store->removeDocument("ml_def");
auto doc = doc_store->getDocument("ml_def");
```

### 2. RAGInference (`RAGInference.hpp/cpp`)

Complete RAG pipeline combining retrieval and generation.

**Features:**

- Configurable retrieval (num docs, threshold)
- Context formatting and truncation
- Integration with EncoderDecoderModel
- Multiple generation strategies

**API:**

```cpp
RAGInference::RAGConfig config;
config.num_retrieved_docs = 3;
config.max_context_length = 256;

auto rag = std::make_unique<RAGInference>(model, doc_store, config);

// Generate with RAG
std::string response = rag->generate("What is Python?");

// Generate with retrieval details
std::vector<std::pair<float, const Document*>> docs;
std::string response = rag->generateWithRetrieval(query, docs);
```

### 3. Example Programs

#### RAGExample.cpp

Interactive demo showing RAG capabilities:

- Builds sample knowledge base
- Demonstrates Q&A with retrieval
- Interactive mode for testing
- Shows retrieval scores and sources

#### RAGBERTComparisonTest.cpp

Comprehensive test suite comparing RAG vs BERT:

- Retrieval quality tests
- Generation comparison
- Knowledge update demonstration
- Performance benchmarks
- Embedding quality analysis

## Building

The RAG system is integrated into the CMake build:

```bash
cd build
cmake .. -DBUILD_EXAMPLES=ON
make rag_example
make rag_bert_comparison
```

## Usage Examples

### Basic RAG Setup

```cpp
#include "DocumentStore.hpp"
#include "RAGInference.hpp"
#include "EncoderDecoderModel.hpp"
#include "encoder.hpp"

// 1. Create encoder and model
auto encoder = std::make_shared<LLMEncoder>(vocab_size, d_model);
auto model = std::make_shared<EncoderDecoderModel>(vocab_size, d_model);

// 2. Create document store
auto doc_store = std::make_shared<DocumentStore>(encoder);

// 3. Add knowledge
doc_store->addDocument("python_def",
    "Python is a high-level programming language...");
doc_store->addDocument("cpp_def",
    "C++ is a general-purpose programming language...");

// 4. Create RAG inference
RAGInference::RAGConfig config;
config.num_retrieved_docs = 2;
auto rag = std::make_unique<RAGInference>(model, doc_store, config);

// 5. Generate responses
std::string answer = rag->generate("What is Python?");
```

### Retrieval-Only Mode

```cpp
// Just retrieve documents without generation
auto results = rag->retrieveOnly("machine learning", k=5);

for (const auto& [score, doc] : results) {
    std::cout << "Score: " << score << "\n";
    std::cout << "Doc: " << doc->id << "\n";
    std::cout << "Text: " << doc->text << "\n\n";
}
```

### Dynamic Knowledge Updates

```cpp
// Add new knowledge without retraining
rag->addDocument("new_tech",
    "Quantum computing uses quantum mechanics...");

// Immediately available for retrieval and generation
auto response = rag->generate("Tell me about quantum computing");
// Will retrieve and use the newly added document
```

### Custom Configuration

```cpp
RAGInference::RAGConfig config;

// Retrieval settings
config.num_retrieved_docs = 5;          // Retrieve top-5 docs
config.retrieval_threshold = 0.3f;      // Min similarity score
config.include_scores = true;           // Show scores in context

// Context formatting
config.context_prefix = "Knowledge:\n";
config.query_prefix = "\n\nQ: ";
config.context_separator = "\n---\n";
config.max_context_length = 512;        // Max tokens

// Generation settings
config.gen_config.max_length = 150;
config.gen_config.temperature = 0.7f;
config.gen_config.top_p = 0.9f;

auto rag = std::make_unique<RAGInference>(model, doc_store, config);
```

## Running the Examples

### RAG Demo

```bash
./build/src/rag_example [model_path] [vocab_path]

# Example:
./build/src/rag_example chatbot_model.bin vocab.txt
```

**Features:**

- Preloaded knowledge base with ML/AI definitions
- Sample Q&A demonstrations
- Interactive mode for custom queries
- Document management commands

### RAG vs BERT Comparison

```bash
./build/src/rag_bert_comparison
```

**Tests:**

1. **Retrieval Quality**: Measures semantic search accuracy
2. **RAG vs Standard**: Compares generation with/without retrieval
3. **Knowledge Update**: Demonstrates dynamic knowledge addition
4. **Embedding Quality**: Tests BERT-style encoder embeddings
5. **Performance**: Benchmarks retrieval and encoding latency

**Expected Output:**

```text
═══════════════════════════════════════════════════════════════
          RAG vs BERT Comparison - Test Summary
═══════════════════════════════════════════════════════════════

Retrieval Quality        PASS (12.34 ms) 4/4 correct
RAG vs Standard          PASS (45.67 ms) Comparison completed
Knowledge Update         PASS (8.91 ms) New knowledge immediately accessible
Embedding Quality        PASS (23.45 ms) BERT-style encoder produces contextual embeddings
Performance             PASS (34.56 ms) Retrieval adds 5.2 ms overhead

Overall: 5/5 tests passed
```

## Performance Considerations

### Retrieval Latency

- Encoder forward pass: ~5-10ms per query
- Cosine similarity (100 docs): ~1-2ms
- **Total retrieval overhead: ~10-20ms**

### Optimizations

1. **Batch Retrieval**: Encode multiple queries in parallel
2. **Caching**: Cache frequently accessed document embeddings
3. **Approximate Search**: Use FAISS/Annoy for large document sets
4. **Index Sharding**: Distribute documents across multiple stores

### Scaling

- **< 1K documents**: In-memory brute-force search (current implementation)
- **1K-100K docs**: Use approximate nearest neighbor (ANN) libraries
- **> 100K docs**: Distributed vector database (Milvus, Weaviate, Pinecone)

## Hybrid RAG + BERT Approach

The current implementation already uses a BERT-like encoder for retrieval! This combines the best of both:

```text
┌──────────────────────────────────────────────────┐
│         Hybrid: RAG with BERT Encoder            │
└──────────────────────────────────────────────────┘

1. BERT Encoder → Generate high-quality embeddings
2. DocumentStore → Semantic search with cosine similarity
3. Decoder → Generate factually-grounded responses

✅ BERT's strength: Excellent semantic embeddings
✅ RAG's strength: Factual grounding and knowledge updates
```

## Future Enhancements

- [ ] **Advanced Retrieval**: BM25, dense-sparse hybrid, reranking
- [ ] **Vector Databases**: Integration with FAISS, Annoy, Hnswlib
- [ ] **Cross-Encoder Reranking**: Improve retrieval quality
- [ ] **Multi-hop Reasoning**: Iterative retrieval for complex queries
- [ ] **Citation Generation**: Automatically cite source documents
- [ ] **Negative Examples**: Filter out irrelevant retrieved docs
- [ ] **Query Expansion**: Improve retrieval with query reformulation
- [ ] **Caching Layer**: Cache embeddings and retrieval results

## Troubleshooting

### Low Retrieval Quality

- **Issue**: Retrieved documents not relevant
- **Solutions**:
  - Increase `num_retrieved_docs` to retrieve more candidates
  - Lower `retrieval_threshold` to allow lower similarity scores
  - Improve document quality and granularity
  - Fine-tune encoder on domain-specific data

### Generation Ignoring Context

- **Issue**: Model generates without using retrieved context
- **Solutions**:
  - Increase `max_context_length` to include more information
  - Improve context formatting (prefixes, separators)
  - Ensure encoder-decoder model is properly trained
  - Use temperature/top-p to control generation randomness

### Slow Performance

- **Issue**: RAG inference too slow
- **Solutions**:
  - Reduce number of documents in store
  - Implement approximate nearest neighbor search
  - Cache document embeddings
  - Use batched retrieval for multiple queries
  - Enable OpenMP for parallel matrix operations

## References

- **RAG Paper**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **BERT Paper**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **Transformers**: "Attention Is All You Need" (Vaswani et al., 2017)

## License

Part of the ADAI project. See main repository LICENSE for details.
