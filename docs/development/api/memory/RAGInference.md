# RAGInference Class - Technical Context Documentation

## Overview

The `RAGInference` class implements a retrieval-augmented generation (RAG) inference engine. It combines document retrieval with generative language models to produce factually grounded, context-aware responses. The engine retrieves relevant documents for a query, augments the query with retrieved context, and generates a response conditioned on both.

Files:

- `src/RAGInference.hpp` - Header file with class declaration and interface
- `src/RAGInference.cpp` - Implementation file with all method definitions

**Purpose:** Enable scalable, updatable, and explainable question answering and generation by integrating retrieval and generation in a unified API.

---

## Class Structure

### RAGConfig Structure

```cpp
struct RAGConfig {
    int num_retrieved_docs;          // Number of documents to retrieve
    float retrieval_threshold;       // Minimum similarity score (0 = no filter)
    bool include_scores;             // Include similarity scores in context
    std::string context_separator;   // Separator between documents
    std::string context_prefix;      // Prefix before context
    std::string query_prefix;        // Prefix before query
    int max_context_length;          // Maximum tokens for context
    TextGenerator::GenerationConfig gen_config;  // Generation parameters
};
```

### RAGInference Members

```cpp
std::shared_ptr<EncoderDecoderModel> model;   // Model for generation
std::shared_ptr<DocumentStore> doc_store;     // Document store for retrieval
RAGConfig config;                             // Inference configuration
```

---

## Core Operations

### Constructor

```cpp
RAGInference(std::shared_ptr<EncoderDecoderModel> model,
             std::shared_ptr<DocumentStore> doc_store,
             const RAGConfig& config = RAGConfig());
```

- **Initializes** the inference engine with a model, document store, and configuration.
- **Throws** if model or doc_store is null.

### Generate Response

```cpp
std::string generate(const std::string& query);
```

- **Retrieves** relevant documents and generates a response conditioned on both the query and retrieved context.

### Generate with Retrieval Info

```cpp
std::string generateWithRetrieval(
    const std::string& query,
    std::vector<std::pair<float, const Document*>>& retrieved_docs);
```

- **Outputs** both the generated response and the retrieved documents with similarity scores.

### Add/Remove/Get Document

```cpp
void addDocument(const std::string& id, const std::string& text,
                 const std::unordered_map<std::string, std::string>& metadata = {});
bool removeDocument(const std::string& id);
const Document* getDocument(const std::string& id) const;
```

- **Convenience methods** that forward to the underlying DocumentStore.

### Get Number of Documents

```cpp
size_t getNumDocuments() const;
```

- **Returns** the number of documents in the knowledge base.

### Update and Access Configuration

```cpp
void setConfig(const RAGConfig& new_config);
const RAGConfig& getConfig() const;
```

- **Update** or access the current RAG configuration.

### Retrieve Only

```cpp
std::vector<std::pair<float, const Document*>> retrieveOnly(const std::string& query, int k = -1) const;
```

- **Retrieves** documents for a query without generating a response (for inspection or debugging).

---

## Context and Prompt Construction

### Format Context

- **Method:**

  ```cpp
  std::string formatContext(const std::vector<std::pair<float, const Document*>>& retrieved_docs) const;
  ```

- **Description:** Formats retrieved documents into a context string, optionally including similarity scores and separators.

### Build Augmented Prompt

- **Method:**

  ```cpp
  std::string buildAugmentedPrompt(const std::string& query, const std::string& context) const;
  ```

- **Description:** Combines context and query into a single prompt for the model.

### Truncate Context

- **Method:**

  ```cpp
  std::string truncateContext(const std::string& context, int max_tokens) const;
  ```

- **Description:** Truncates the context string to fit within a token limit (approximate by character count).

---

## Example Usage

```cpp
#include "RAGInference.hpp"
#include <memory>

std::shared_ptr<EncoderDecoderModel> model = ...;
std::shared_ptr<DocumentStore> doc_store = ...;
RAGInference rag(model, doc_store);

rag.addDocument("doc1", "Python is a programming language.");
std::string response = rag.generate("What is Python?");

std::vector<std::pair<float, const Document*>> retrieved;
std::string answer = rag.generateWithRetrieval("What is Python?", retrieved);
for (const auto& [score, doc] : retrieved) {
    std::cout << doc->id << ": " << score << std::endl;
}
```

---

## Error Handling

- **Throws `std::invalid_argument`** if model or doc_store is null.
- **Propagates** errors from DocumentStore (e.g., duplicate IDs, empty text).
- **Handles** empty retrieval gracefully (empty context).

---

## Performance Characteristics

- **Retrieval** is O(N) for N documents (linear scan in DocumentStore).
- **Generation** time depends on model and prompt length.
- **Context truncation** prevents excessive prompt size.

---

## Integration Patterns

- **RAG pipelines:** Use as the main inference engine for retrieval-augmented generation systems.
- **Chatbots and QA:** Provide up-to-date, grounded answers with source citation.
- **Knowledge base updates:** Add or remove documents without retraining the model.
- **Explainability:** Expose retrieved documents and scores for transparency.

---

## Limitations and Constraints

- **No persistent storage:** DocumentStore is in-memory only.
- **No ANN retrieval:** Linear scan only; for large-scale, integrate with vector search libraries.
- **Not thread-safe:** No locking for concurrent access.
- **Token truncation is approximate:** Uses character count, not true tokenization.

---

## Future Enhancement Opportunities

- **Persistent knowledge base:** Add serialization for DocumentStore.
- **ANN integration:** Support for FAISS, HNSW, or similar libraries for fast retrieval.
- **Token-aware truncation:** Use tokenizer to truncate context by true token count.
- **Batch inference:** Support multiple queries at once.
- **Thread safety:** Add locking for concurrent use.

---

## Testing Recommendations

- **Unit tests:**
  - Add, remove, and retrieve documents
  - Generate responses for known queries
  - Edge cases: empty store, empty retrieval, long context
- **Integration tests:**
  - End-to-end RAG pipeline with retrieval and generation

---

## Debugging Tips

- **Print formatted context** to verify document selection and formatting.
- **Inspect retrieved document scores** for retrieval quality.
- **Check prompt length** after truncation.
- **Use retrieveOnly** to debug retrieval without generation.

---

## Summary

The `RAGInference` class provides a modular, extensible, and production-ready engine for retrieval-augmented generation. It enables factually grounded, updatable, and explainable responses by combining document retrieval and generative modeling in a unified API.
