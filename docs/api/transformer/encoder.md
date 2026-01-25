# LLM Encoder - Context Document

## Overview

The `LLMEncoder` class implements a transformer-based encoder architecture for natural language processing tasks, specifically designed for chatbot applications. It combines BPE tokenization with multi-layer transformer encoding to produce contextualized representations of text sequences.

## Purpose

The LLMEncoder is designed to:
- Convert text into rich, contextualized embeddings suitable for downstream NLP tasks
- Provide transformer-based feature extraction for chatbot applications
- Support variable-length sequence encoding with positional information
- Enable sentence-level and token-level representations
- Integrate with the existing BPETokenizer for subword handling
- Serve as a feature extractor for neural network classifiers

## Architecture

### Component Hierarchy

```
LLMEncoder
├── BPETokenizer (text → token IDs)
├── TokenEmbedding (token IDs → dense vectors)
├── PositionalEncoding (add position information)
├── EncoderBlocks (N layers)
│   ├── MultiHeadAttention
│   ├── LayerNorm
│   ├── FeedForward
│   └── LayerNorm
└── Final LayerNorm
```

### Information Flow

```
Input Text
    ↓
BPE Tokenization → [token_1, token_2, ..., token_n]
    ↓
Token Embedding → [emb_1, emb_2, ..., emb_n]  (shape: [seq_len, d_model])
    ↓
Positional Encoding → [emb_1 + pos_1, emb_2 + pos_2, ..., emb_n + pos_n]
    ↓
Encoder Block 1 (Attention + FFN)
    ↓
Encoder Block 2 (Attention + FFN)
    ↓
    ...
    ↓
Encoder Block N (Attention + FFN)
    ↓
Final Layer Normalization
    ↓
Contextualized Embeddings [seq_len, d_model]
    ↓ (optional pooling)
Sentence Embedding [d_model]
```

## Class Interfaces

### Matrix Class

```cpp
class Matrix {
public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
    // Constructors
    Matrix(int r, int c);
    Matrix(const std::vector<std::vector<float>>& d);
    
    // Operators
    float& operator()(int i, int j);
    const float& operator()(int i, int j) const;
    Matrix operator*(const Matrix& other) const;  // Matrix multiplication
    Matrix operator+(const Matrix& other) const;  // Element-wise addition
    
    // Operations
    Matrix transpose() const;
    void randomize(float scale = 0.1f);
    Matrix scale(float scalar) const;  // Scalar multiplication
};
```

**Issues & Improvements:**
- No move semantics (performance issue with large matrices)
- No SIMD optimizations for matrix operations
- Memory layout not optimized for cache coherency
- Missing batch operations

### Activation Class

```cpp
class Activation {
public:
    // Softmax activation (used in attention)
    static Matrix softmax(const Matrix& input);
    
    // GELU activation (used in feed-forward)
    static Matrix gelu(const Matrix& input);
};
```

**Formula:**
- **Softmax**: `softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))`
- **GELU**: `gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

### LayerNorm Class

```cpp
class LayerNorm {
private:
    Matrix gamma, beta;  // Learnable parameters
    float eps;           // Small constant for numerical stability
    
public:
    LayerNorm(int dim, float epsilon = 1e-5f);
    Matrix forward(const Matrix& input);
};
```

**Formula:**
```
mean = (1/d) * Σ x_i
var = (1/d) * Σ (x_i - mean)²
output = gamma * ((x - mean) / √(var + ε)) + beta
```

### PositionalEncoding Class

```cpp
class PositionalEncoding {
private:
    Matrix pos_encoding;  // Pre-computed positional encodings
    
public:
    PositionalEncoding(int max_len, int d_model);
    Matrix forward(const Matrix& input);
};
```

**Formula (Sinusoidal):**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### TokenEmbedding Class

```cpp
class TokenEmbedding {
private:
    Matrix embedding_matrix;  // [vocab_size, d_model]
    int vocab_size;
    int d_model;
    
public:
    TokenEmbedding(int vocab_size, int d_model);
    Matrix forward(const std::vector<int>& token_ids);
    void load_pretrained(const std::string& filename);
};
```

**Initialization**: Xavier/Glorot initialization with `scale = √(2 / (vocab_size + d_model))`

### MultiHeadAttention Class

```cpp
class MultiHeadAttention {
private:
    int d_model;      // Model dimension
    int num_heads;    // Number of attention heads
    int d_k;          // Dimension per head (d_model / num_heads)
    Matrix W_q, W_k, W_v, W_o;  // Weight matrices
    
public:
    MultiHeadAttention(int d_model, int num_heads);
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);
};
```

**Formula (Scaled Dot-Product Attention):**
```
Q = input * W_q
K = input * W_k
V = input * W_v

scores = (Q * K^T) / √d_k
attention = softmax(scores)
output = attention * V * W_o
```

**Current Limitation**: Implementation treats multi-head as single head (simplified version)

### FeedForward Class

```cpp
class FeedForward {
private:
    Matrix W1, W2;    // Weight matrices
    Matrix b1, b2;    // Bias vectors
    int d_model, d_ff;
    
public:
    FeedForward(int d_model, int d_ff);
    Matrix forward(const Matrix& input);
};
```

**Formula:**
```
hidden = GELU(input * W1 + b1)
output = hidden * W2 + b2
```

### EncoderBlock Class

```cpp
class EncoderBlock {
private:
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;
    float dropout_rate;
    
public:
    EncoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f);
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);
};
```

**Forward Pass:**
```
x1 = LayerNorm(x + MultiHeadAttention(x))
x2 = LayerNorm(x1 + FeedForward(x1))
```

### LLMEncoder Class (Main)

```cpp
class LLMEncoder {
private:
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<EncoderBlock>> encoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    
    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int d_ff;
    int max_seq_length;
    
public:
    // Constructor
    LLMEncoder(int vocab_size, int d_model = 512, int num_layers = 6,
               int num_heads = 8, int d_ff = 2048, int max_seq_length = 512);
    
    // Encoding methods
    Matrix encode(const std::string& text);
    Matrix encode_with_mask(const std::vector<int>& token_ids, 
                           const Matrix& attention_mask);
    std::vector<float> get_sentence_embedding(const std::string& text);
    
    // Tokenizer management
    void load_tokenizer_vocab(const std::string& vocab_file);
    void build_tokenizer(const std::vector<std::string>& corpus, 
                        int vocab_size = 10000);
    
    // Utilities
    void print_config() const;
    int get_embedding_dim() const;
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);
};
```

## Integration with NeuralNetwork Class

### Current Architecture Compatibility

The LLMEncoder produces embeddings that can be consumed by the NeuralNetwork class:

```cpp
#include "encoder.hpp"
#include "NeuralNetwork.hpp"

// Example: Text Classification Pipeline
class TextClassifier {
private:
    LLMEncoder encoder;
    NeuralNetwork classifier;
    
public:
    TextClassifier(int vocab_size, int num_classes)
        : encoder(vocab_size, 256, 4, 8, 1024, 128),
          classifier({256, 128, 64, num_classes},
                    {ActivationType::RELU, ActivationType::RELU, 
                     ActivationType::LINEAR},
                    LossType::CATEGORICAL_CROSS_ENTROPY, 0.001f) {
        classifier.initialize_he();
    }
    
    std::vector<float> classify(const std::string& text) {
        // Get sentence embedding from encoder
        auto features = encoder.get_sentence_embedding(text);
        
        // Classify using neural network
        return classifier.predict(features);
    }
    
    void train(const std::vector<std::string>& texts,
              const std::vector<std::vector<float>>& labels,
              int epochs = 50) {
        
        // Encode all texts to features
        std::vector<std::vector<float>> features;
        for (const auto& text : texts) {
            features.push_back(encoder.get_sentence_embedding(text));
        }
        
        // Train classifier
        classifier.fit(features, labels, epochs, 32);
    }
};
```

### Recommended Improvements for Better Integration

#### 1. **Add Batch Processing Support**

```cpp
// Recommended addition to LLMEncoder
class LLMEncoder {
public:
    // Process multiple texts in parallel
    std::vector<std::vector<float>> get_batch_sentence_embeddings(
        const std::vector<std::string>& texts) {
        
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(texts.size());
        
        for (const auto& text : texts) {
            embeddings.push_back(get_sentence_embedding(text));
        }
        
        return embeddings;
    }
    
    // Get token-level embeddings for sequence tasks
    std::vector<Matrix> encode_batch(const std::vector<std::string>& texts) {
        std::vector<Matrix> encoded_batch;
        encoded_batch.reserve(texts.size());
        
        for (const auto& text : texts) {
            encoded_batch.push_back(encode(text));
        }
        
        return encoded_batch;
    }
};
```

#### 2. **Matrix to Vector Conversion Utilities**

```cpp
// Add to LLMEncoder or create separate utility class
namespace EncoderUtils {
    // Flatten matrix to vector for NeuralNetwork input
    std::vector<float> matrix_to_vector(const Matrix& mat) {
        std::vector<float> vec;
        vec.reserve(mat.rows * mat.cols);
        
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                vec.push_back(mat(i, j));
            }
        }
        return vec;
    }
    
    // Max pooling over sequence dimension
    std::vector<float> max_pool(const Matrix& mat) {
        std::vector<float> pooled(mat.cols);
        
        for (int j = 0; j < mat.cols; j++) {
            float max_val = mat(0, j);
            for (int i = 1; i < mat.rows; i++) {
                max_val = std::max(max_val, mat(i, j));
            }
            pooled[j] = max_val;
        }
        return pooled;
    }
    
    // Attention pooling with learned weights
    std::vector<float> attention_pool(const Matrix& mat, 
                                     const std::vector<float>& attention_weights) {
        std::vector<float> pooled(mat.cols, 0.0f);
        
        for (int j = 0; j < mat.cols; j++) {
            for (int i = 0; i < mat.rows; i++) {
                pooled[j] += mat(i, j) * attention_weights[i];
            }
        }
        return pooled;
    }
}
```

#### 3. **Trainable Classifier Head Integration**

```cpp
// Recommended: Add classification head directly to encoder
class LLMEncoder {
private:
    std::unique_ptr<NeuralNetwork> classification_head;
    bool use_classification_head;
    
public:
    // Enable end-to-end training with classifier
    void add_classification_head(int num_classes, 
                                 const std::vector<int>& hidden_sizes = {128, 64}) {
        
        std::vector<int> architecture = {d_model};
        architecture.insert(architecture.end(), hidden_sizes.begin(), hidden_sizes.end());
        architecture.push_back(num_classes);
        
        std::vector<ActivationType> activations;
        for (size_t i = 0; i < hidden_sizes.size(); i++) {
            activations.push_back(ActivationType::RELU);
        }
        activations.push_back(ActivationType::LINEAR);
        
        classification_head = std::make_unique<NeuralNetwork>(
            architecture, activations, 
            LossType::CATEGORICAL_CROSS_ENTROPY, 0.001f
        );
        classification_head->initialize_he();
        use_classification_head = true;
    }
    
    // Classify directly
    std::vector<float> classify(const std::string& text) {
        if (!use_classification_head) {
            throw std::runtime_error("Classification head not initialized");
        }
        
        auto sentence_emb = get_sentence_embedding(text);
        return classification_head->predict(sentence_emb);
    }
    
    // Train end-to-end (encoder frozen, only classifier trained)
    void train_classifier(const std::vector<std::string>& texts,
                         const std::vector<std::vector<float>>& labels,
                         int epochs = 50, int batch_size = 32) {
        
        if (!use_classification_head) {
            throw std::runtime_error("Classification head not initialized");
        }
        
        // Encode all texts (encoder weights frozen)
        std::vector<std::vector<float>> features;
        for (const auto& text : texts) {
            features.push_back(get_sentence_embedding(text));
        }
        
        // Train only the classification head
        classification_head->fit(features, labels, epochs, batch_size);
    }
};
```

#### 4. **Feature Extraction Modes**

```cpp
// Add different pooling strategies
enum class PoolingStrategy {
    MEAN,        // Average all token embeddings
    MAX,         // Max pooling over sequence
    CLS_TOKEN,   // Use first token (if special [CLS] token added)
    LAST_TOKEN,  // Use last token
    ATTENTION    // Learned attention pooling
};

class LLMEncoder {
public:
    std::vector<float> get_sentence_embedding(const std::string& text,
                                              PoolingStrategy strategy = PoolingStrategy::MEAN) {
        Matrix encoded = encode(text);
        
        switch (strategy) {
            case PoolingStrategy::MEAN:
                return mean_pool(encoded);
            
            case PoolingStrategy::MAX:
                return EncoderUtils::max_pool(encoded);
            
            case PoolingStrategy::CLS_TOKEN:
                return get_row_as_vector(encoded, 0);
            
            case PoolingStrategy::LAST_TOKEN:
                return get_row_as_vector(encoded, encoded.rows - 1);
            
            case PoolingStrategy::ATTENTION:
                // Would require implementing attention pooling
                return mean_pool(encoded);  // Fallback
        }
    }
    
private:
    std::vector<float> mean_pool(const Matrix& mat) {
        std::vector<float> pooled(d_model, 0.0f);
        for (int j = 0; j < d_model; j++) {
            for (int i = 0; i < mat.rows; i++) {
                pooled[j] += mat(i, j);
            }
            pooled[j] /= mat.rows;
        }
        return pooled;
    }
    
    std::vector<float> get_row_as_vector(const Matrix& mat, int row) {
        std::vector<float> vec(mat.cols);
        for (int j = 0; j < mat.cols; j++) {
            vec[j] = mat(row, j);
        }
        return vec;
    }
};
```

#### 5. **Gradient Flow for End-to-End Training**

**Current Limitation**: The encoder doesn't support backpropagation, so it can only serve as a frozen feature extractor.

**Recommended Addition**:

```cpp
// Add gradient computation capability
class LLMEncoder {
private:
    bool requires_grad;
    
public:
    void set_requires_grad(bool grad) {
        requires_grad = grad;
    }
    
    // Backward pass through encoder (simplified)
    void backward(const std::vector<float>& grad_output) {
        if (!requires_grad) return;
        
        // Would need to implement:
        // 1. Store activations during forward pass
        // 2. Implement backward pass for each component
        // 3. Accumulate gradients
        // 4. Update weights
        
        // This is a significant extension requiring full autograd system
    }
};
```

## Usage Examples

### Example 1: Sentiment Analysis

```cpp
#include "encoder.hpp"
#include "NeuralNetwork.hpp"

int main() {
    // Initialize encoder
    LLMEncoder encoder(5000, 256, 4, 8, 1024, 128);
    
    // Build tokenizer from corpus
    std::vector<std::string> corpus = load_training_texts();
    encoder.build_tokenizer(corpus, 5000);
    
    // Create sentiment classifier (positive/negative)
    NeuralNetwork classifier(
        {256, 128, 2},  // Input: encoder output, Output: 2 classes
        {ActivationType::RELU, ActivationType::LINEAR},
        LossType::CATEGORICAL_CROSS_ENTROPY,
        0.001f
    );
    classifier.initialize_he();
    
    // Prepare training data
    std::vector<std::string> train_texts = {
        "This movie was absolutely fantastic!",
        "Terrible experience, waste of time.",
        "Great product, highly recommend!",
        // ... more examples
    };
    
    std::vector<std::vector<float>> train_labels = {
        {1.0f, 0.0f},  // Positive
        {0.0f, 1.0f},  // Negative
        {1.0f, 0.0f},  // Positive
        // ... more labels
    };
    
    // Extract features using encoder
    std::vector<std::vector<float>> features;
    for (const auto& text : train_texts) {
        features.push_back(encoder.get_sentence_embedding(text));
    }
    
    // Train classifier
    classifier.fit(features, train_labels, 100, 16);
    
    // Test on new text
    std::string test_text = "Amazing quality and fast shipping!";
    auto test_features = encoder.get_sentence_embedding(test_text);
    auto prediction = classifier.predict(test_features);
    
    std::cout << "Positive: " << prediction[0] << std::endl;
    std::cout << "Negative: " << prediction[1] << std::endl;
    
    return 0;
}
```

### Example 2: Intent Classification for Chatbot

```cpp
#include "encoder.hpp"
#include "NeuralNetwork.hpp"

enum Intent {
    GREETING = 0,
    QUESTION = 1,
    COMPLAINT = 2,
    THANKS = 3,
    GOODBYE = 4
};

class IntentClassifier {
private:
    LLMEncoder encoder;
    NeuralNetwork classifier;
    
public:
    IntentClassifier()
        : encoder(5000, 256, 4, 8, 1024, 128),
          classifier({256, 128, 64, 5},
                    {ActivationType::RELU, ActivationType::RELU, 
                     ActivationType::LINEAR},
                    LossType::CATEGORICAL_CROSS_ENTROPY, 0.001f) {
        classifier.initialize_he();
    }
    
    void load_tokenizer(const std::string& vocab_file) {
        encoder.load_tokenizer_vocab(vocab_file);
    }
    
    Intent classify_intent(const std::string& user_message) {
        auto features = encoder.get_sentence_embedding(user_message);
        auto probabilities = classifier.predict(features);
        
        // Find intent with highest probability
        int max_idx = 0;
        for (int i = 1; i < probabilities.size(); i++) {
            if (probabilities[i] > probabilities[max_idx]) {
                max_idx = i;
            }
        }
        
        return static_cast<Intent>(max_idx);
    }
    
    void train(const std::vector<std::string>& messages,
              const std::vector<Intent>& intents,
              int epochs = 100) {
        
        // Convert to training format
        std::vector<std::vector<float>> features;
        std::vector<std::vector<float>> labels;
        
        for (size_t i = 0; i < messages.size(); i++) {
            // Extract features
            features.push_back(encoder.get_sentence_embedding(messages[i]));
            
            // One-hot encode intent
            std::vector<float> label(5, 0.0f);
            label[static_cast<int>(intents[i])] = 1.0f;
            labels.push_back(label);
        }
        
        // Train
        classifier.fit(features, labels, epochs, 32);
    }
};
```

### Example 3: Semantic Similarity Search

```cpp
#include "encoder.hpp"
#include <algorithm>

class SemanticSearch {
private:
    LLMEncoder encoder;
    std::vector<std::string> documents;
    std::vector<std::vector<float>> doc_embeddings;
    
public:
    SemanticSearch(int vocab_size = 5000)
        : encoder(vocab_size, 256, 4, 8, 1024, 128) {}
    
    void build_index(const std::vector<std::string>& docs,
                    const std::string& vocab_file) {
        encoder.load_tokenizer_vocab(vocab_file);
        
        documents = docs;
        doc_embeddings.clear();
        
        for (const auto& doc : documents) {
            doc_embeddings.push_back(
                encoder.get_sentence_embedding(doc)
            );
        }
    }
    
    std::vector<int> search(const std::string& query, int top_k = 5) {
        auto query_emb = encoder.get_sentence_embedding(query);
        
        // Compute cosine similarity with all documents
        std::vector<std::pair<float, int>> similarities;
        
        for (size_t i = 0; i < doc_embeddings.size(); i++) {
            float sim = cosine_similarity(query_emb, doc_embeddings[i]);
            similarities.push_back({sim, i});
        }
        
        // Sort by similarity (descending)
        std::sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Return top-k document indices
        std::vector<int> results;
        for (int i = 0; i < std::min(top_k, (int)similarities.size()); i++) {
            results.push_back(similarities[i].second);
        }
        
        return results;
    }
    
private:
    float cosine_similarity(const std::vector<float>& a,
                           const std::vector<float>& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        
        for (size_t i = 0; i < a.size(); i++) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
    }
};
```

## Performance Considerations

### Current Limitations

1. **No SIMD Optimizations**: Matrix operations are naive loops
2. **Single-threaded**: No parallel processing of batches
3. **Memory Inefficient**: Creates many temporary Matrix objects
4. **No Gradient Support**: Cannot train encoder end-to-end
5. **Simplified Multi-Head Attention**: Treats all heads as single head
6. **No Caching**: Recomputes everything for each forward pass

### Recommended Optimizations

#### 1. **Use Eigen or Similar Library**

```cpp
// Replace custom Matrix with Eigen
#include <Eigen/Dense>

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;

// Benefits:
// - SIMD optimizations
// - Cache-friendly memory layout
// - Lazy evaluation
// - Well-tested implementations
```

#### 2. **Batch Processing**

```cpp
// Process multiple sequences simultaneously
class BatchEncoder {
public:
    std::vector<Matrix> encode_batch(
        const std::vector<std::vector<int>>& token_ids_batch,
        int max_len) {
        
        // Pad sequences to same length
        std::vector<std::vector<int>> padded_batch = pad_sequences(
            token_ids_batch, max_len
        );
        
        // Process in parallel
        std::vector<Matrix> results(padded_batch.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < padded_batch.size(); i++) {
            results[i] = encode_sequence(padded_batch[i]);
        }
        
        return results;
    }
};
```

#### 3. **KV Caching for Inference**

```cpp
// Cache key and value matrices for faster repeated inference
class CachedAttention {
private:
    std::unordered_map<std::string, std::pair<Matrix, Matrix>> kv_cache;
    
public:
    Matrix forward_with_cache(const Matrix& input, const std::string& cache_key) {
        if (kv_cache.find(cache_key) != kv_cache.end()) {
            auto [K, V] = kv_cache[cache_key];
            // Reuse cached K, V
        } else {
            // Compute and cache
            Matrix K = input * W_k;
            Matrix V = input * W_v;
            kv_cache[cache_key] = {K, V};
        }
    }
};
```

## Best Practices for Integration

### 1. Feature Extraction Pipeline

```cpp
class FeatureExtractor {
private:
    LLMEncoder encoder;
    
public:
    // Extract features for classification
    std::vector<std::vector<float>> extract_for_classification(
        const std::vector<std::string>& texts) {
        
        std::vector<std::vector<float>> features;
        features.reserve(texts.size());
        
        for (const auto& text : texts) {
            features.push_back(encoder.get_sentence_embedding(text));
        }
        
        return features;
    }
    
    // Extract features for sequence labeling
    std::vector<std::vector<std::vector<float>>> extract_for_sequence_labeling(
        const std::vector<std::string>& texts) {
        
        std::vector<std::vector<std::vector<float>>> features;
        
        for (const auto& text : texts) {
            Matrix encoded = encoder.encode(text);
            
            // Convert matrix to vector of vectors
            std::vector<std::vector<float>> seq_features;
            for (int i = 0; i < encoded.rows; i++) {
                std::vector<float> token_features(encoded.cols);
                for (int j = 0; j < encoded.cols; j++) {
                    token_features[j] = encoded(i, j);
                }
                seq_features.push_back(token_features);
            }
            features.push_back(seq_features);
        }
        
        return features;
    }
};
```

### 2. Model Saving and Loading

```cpp
// Extend save/load to include both encoder and classifier
class EncoderClassifierPipeline {
private:
    LLMEncoder encoder;
    NeuralNetwork classifier;
    
public:
    void save(const std::string& model_dir) {
        // Save encoder weights
        encoder.save_weights(model_dir + "/encoder_weights.bin");
        
        // Save classifier
        classifier.save(model_dir + "/classifier.nn");
        
        // Save metadata
        std::ofstream meta(model_dir + "/metadata.txt");
        meta << "encoder_dim: " << encoder.get_embedding_dim() << "\n";
        meta << "num_classes: " << classifier.get_layer_sizes().back() << "\n";
        meta.close();
    }
    
    void load(const std::string& model_dir) {
        // Load encoder weights
        encoder.load_weights(model_dir + "/encoder_weights.bin");
        
        // Load classifier
        classifier.load(model_dir + "/classifier.nn");
    }
};
```

### 3. Input Validation and Preprocessing

```cpp
class TextPreprocessor {
public:
    static std::string clean_text(const std::string& text) {
        std::string cleaned = text;
        
        // Remove extra whitespace
        cleaned = std::regex_replace(cleaned, std::regex("\\s+"), " ");
        
        // Trim
        cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
        cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
        
        return cleaned;
    }
    
    static std::vector<std::string> prepare_batch(
        const std::vector<std::string>& texts,
        int max_length = 512) {
        
        std::vector<std::string> prepared;
        
        for (const auto& text : texts) {
            std::string cleaned = clean_text(text);
            
            // Truncate if too long
            if (cleaned.length() > max_length) {
                cleaned = cleaned.substr(0, max_length);
            }
            
            prepared.push_back(cleaned);
        }
        
        return prepared;
    }
};
```

## Testing and Validation

### Unit Tests

```cpp
#include <cassert>
#include <cmath>

void test_encoder_dimensions() {
    LLMEncoder encoder(1000, 128, 2, 4, 512, 64);
    
    std::string test_text = "Hello world";
    auto embedding = encoder.get_sentence_embedding(test_text);
    
    assert(embedding.size() == 128);  // Should match d_model
    std::cout << "✓ Dimension test passed" << std::endl;
}

void test_encoder_consistency() {
    LLMEncoder encoder(1000, 128, 2, 4, 512, 64);
    
    std::string text = "Test consistency";
    auto emb1 = encoder.get_sentence_embedding(text);
    auto emb2 = encoder.get_sentence_embedding(text);
    
    // Same input should give same output
    for (size_t i = 0; i < emb1.size(); i++) {
        assert(std::abs(emb1[i] - emb2[i]) < 1e-6f);
    }
    
    std::cout << "✓ Consistency test passed" << std::endl;
}

void test_integration_with_neural_network() {
    LLMEncoder encoder(1000, 128, 2, 4, 512, 64);
    NeuralNetwork classifier({128, 64, 3},
                            {ActivationType::RELU, ActivationType::LINEAR},
                            LossType::CATEGORICAL_CROSS_ENTROPY);
    
    std::string text = "Integration test";
    auto features = encoder.get_sentence_embedding(text);
    auto output = classifier.predict(features);
    
    assert(output.size() == 3);  // Should match output layer size
    std::cout << "✓ Integration test passed" << std::endl;
}

void run_all_tests() {
    test_encoder_dimensions();
    test_encoder_consistency();
    test_integration_with_neural_network();
    std::cout << "\nAll tests passed!" << std::endl;
}
```

## Future Enhancements

### 1. **Add Decoder for Sequence-to-Sequence Tasks**

```cpp
class LLMEncoderDecoder {
private:
    LLMEncoder encoder;
    // Add decoder components
    
public:
    std::string generate_response(const std::string& input) {
        // Encode input
        // Decode to output sequence
        // Return generated text
    }
};
```

### 2. **Fine-tuning Support**

```cpp
class FineTuner {
public:
    void fine_tune_encoder(LLMEncoder& encoder,
                          const std::vector<std::string>& texts,
                          const std::vector<std::vector<float>>& labels,
                          int epochs = 10,
                          float learning_rate = 1e-5f) {
        // Implement fine-tuning with gradient descent
        // Update encoder weights based on task loss
    }
};
```

### 3. **Attention Visualization**

```cpp
class AttentionVisualizer {
public:
    Matrix get_attention_weights(const LLMEncoder& encoder,
                                const std::string& text) {
        // Extract attention weights from encoder
        // Return attention matrix for visualization
    }
};
```

### 4. **Quantization for Deployment**

```cpp
class QuantizedEncoder {
public:
    void quantize_weights(LLMEncoder& encoder, int bits = 8) {
        // Convert float32 weights to int8
        // Reduce model size and inference time
    }
};
```

## References

- **Attention Is All You Need**: Vaswani et al. (2017) - Original Transformer paper
- **BERT**: Devlin et al. (2018) - Bidirectional encoder representations
- **GPT**: Radford et al. (2018) - Generative pre-training
- **Layer Normalization**: Ba et al. (2016)
- **GELU Activation**: Hendrycks & Gimpel (2016)
- **Positional Encoding**: Vaswani et al. (2017)

## Related Components

- **BPETokenizer**: Subword tokenization for handling unknown words
- **NeuralNetwork**: Feed-forward classifier for downstream tasks
- **Neuron/NeuronLayer**: Basic building blocks for neural networks

## Summary of Recommendations

### High Priority Integration Improvements

1. **Add batch processing support** to encoder for efficient feature extraction
2. **Implement classification head** directly in encoder for end-to-end usage
3. **Add multiple pooling strategies** (mean, max, CLS, attention)
4. **Create utility functions** for Matrix ↔ vector conversions
5. **Implement proper save/load** for encoder weights (currently stubs)

### Medium Priority Enhancements

6. **Add gradient computation** for end-to-end training (requires autograd)
7. **Optimize matrix operations** with Eigen or similar library
8. **Implement true multi-head attention** (currently simplified)
9. **Add KV caching** for faster inference
10. **Support variable batch sizes** with padding

### Low Priority Future Work

11. Add decoder for generative tasks
12. Implement attention visualization
13. Add model quantization
14. Support distributed training
15. Add benchmarking utilities

The encoder provides a solid foundation for transformer-based NLP tasks and integrates well with the NeuralNetwork class for classification and regression tasks. The recommended improvements focus on making the integration more seamless and efficient while maintaining the clean separation of concerns between feature extraction and downstream task learning.


---

## Usage Guide

# LLM Chatbot Encoder

A comprehensive transformer-based encoder implementation for Large Language Model (LLM) chatbot applications, built in C++ with modern neural network architecture patterns.

## Architecture Overview

The encoder implements a full transformer encoder stack with the following components:

### Core Components

1. **BPE Tokenizer**
   - Byte-Pair Encoding for subword tokenization
   - Handles out-of-vocabulary words gracefully
   - Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`

2. **Token Embedding Layer**
   - Converts token IDs to dense vector representations
   - Xavier initialization for stable training
   - Dimension: configurable (default: 512)

3. **Positional Encoding**
   - Sinusoidal position embeddings
   - Enables the model to capture sequential information
   - Formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
   - Formula: PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

4. **Multi-Head Self-Attention**
   - Scaled dot-product attention mechanism
   - Multiple attention heads for diverse representations
   - Query, Key, Value projections
   - Attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V

5. **Feed-Forward Networks**
   - Position-wise fully connected layers
   - GELU activation function
   - Two linear transformations with expansion

6. **Layer Normalization**
   - Stabilizes training and improves convergence
   - Applied after residual connections
   - Learnable scale (gamma) and shift (beta) parameters

7. **Encoder Blocks**
   - Stacked layers combining attention and feed-forward
   - Residual connections around each sub-layer
   - Pre-normalization architecture

### Architecture Diagram

```
Input Text
    ↓
Tokenizer (BPE)
    ↓
Token IDs
    ↓
Token Embedding
    ↓
Positional Encoding
    ↓
┌─────────────────────────┐
│   Encoder Block 1       │
│  ┌──────────────────┐   │
│  │ Multi-Head Attn  │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │  Layer Norm      │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │  Feed-Forward    │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │  Layer Norm      │   │
│  └──────────────────┘   │
└─────────────────────────┘
    ↓
  (Repeat N layers)
    ↓
Final Layer Norm
    ↓
Contextualized Embeddings
```

## Features

- **Modular Design**: Clean separation of components for easy modification
- **Flexible Configuration**: Adjustable model size, layers, heads, etc.
- **Sentence Embeddings**: Mean pooling for sentence-level representations
- **Attention Masking**: Support for padding masks
- **Pre-trained Support**: Load pre-trained embeddings and weights
- **Memory Efficient**: Optimized matrix operations

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 5000 | Size of the vocabulary |
| `d_model` | 512 | Dimension of embeddings and hidden states |
| `num_layers` | 6 | Number of encoder layers |
| `num_heads` | 8 | Number of attention heads |
| `d_ff` | 2048 | Dimension of feed-forward layer |
| `max_seq_length` | 512 | Maximum sequence length |

## Usage

### Basic Example

```cpp
#include "LLMEncoder.hpp"

int main() {
    // Initialize encoder
    LLMEncoder encoder(
        5000,    // vocab_size
        256,     // d_model
        4,       // num_layers
        8,       // num_heads
        1024,    // d_ff
        128      // max_seq_length
    );
    
    // Build tokenizer from corpus
    std::vector<std::string> corpus = {
        "Hello, how can I help you?",
        "I am an AI assistant.",
        "What can I do for you today?"
    };
    encoder.build_tokenizer(corpus, 5000);
    
    // Encode text
    std::string input = "Hello, how can I help you?";
    Matrix embeddings = encoder.encode(input);
    
    // Get sentence embedding
    std::vector<float> sentence_emb = encoder.get_sentence_embedding(input);
    
    return 0;
}
```

### Advanced Usage - Batch Processing

```cpp
// Load pre-trained vocabulary
encoder.load_tokenizer_vocab("vocab.txt");

// Process multiple texts
std::vector<std::string> texts = {
    "First message",
    "Second message",
    "Third message"
};

for (const auto& text : texts) {
    auto embeddings = encoder.encode(text);
    // Process embeddings...
}
```

### Custom Attention Masks

```cpp
// Create attention mask (1 for valid tokens, 0 for padding)
std::vector<int> token_ids = {101, 2023, 2003, 0, 0};  // Last two are padding
Matrix mask(5, 5);
for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
        mask(i, j) = (j < 3) ? 1.0f : 0.0f;  // Mask out padding
    }
}

Matrix embeddings = encoder.encode_with_mask(token_ids, mask);
```

## Building

### Prerequisites
- CMake 3.10+
- C++17 compatible compiler
- Google Test (for testing)

### Build Instructions

```bash
# From project root
cd build
cmake ..
cmake --build . --target encoder

# Run the encoder demo
./src/encoder
```

## Applications

This encoder can be used for various NLP tasks:

1. **Chatbot Intent Classification**: Encode user messages for intent detection
2. **Semantic Similarity**: Compare sentence embeddings for similarity
3. **Question Answering**: Encode questions and context passages
4. **Text Classification**: Extract features for downstream classifiers
5. **Dialogue State Tracking**: Maintain conversation context
6. **Sentiment Analysis**: Encode text for emotion detection

## Performance Characteristics

### Time Complexity
- **Attention**: O(n²·d) where n is sequence length, d is model dimension
- **Feed-Forward**: O(n·d·d_ff)
- **Per Layer**: O(n²·d + n·d·d_ff)
- **Full Model**: O(L·(n²·d + n·d·d_ff)) where L is number of layers

### Space Complexity
- **Parameters**: O(V·d + L·(d² + d·d_ff)) where V is vocabulary size
- **Activations**: O(n·d)

### Typical Performance (on modern CPU)
- **Small Model** (d=256, L=4): ~10-50ms per sentence
- **Medium Model** (d=512, L=6): ~50-200ms per sentence
- **Large Model** (d=768, L=12): ~200-500ms per sentence

## Model Variants

### Small (Fast)
```cpp
LLMEncoder encoder(5000, 256, 4, 8, 1024, 128);
```

### Medium (Balanced)
```cpp
LLMEncoder encoder(10000, 512, 6, 8, 2048, 256);
```

### Large (High Quality)
```cpp
LLMEncoder encoder(30000, 768, 12, 12, 3072, 512);
```

## Future Enhancements

- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Quantization for inference speedup
- [ ] Knowledge distillation support
- [ ] Efficient attention variants (sparse, linear)
- [ ] Pre-training routines
- [ ] Fine-tuning capabilities
- [ ] Multi-language support
- [ ] Beam search decoding
- [ ] Dynamic batching
- [ ] Model compression

## Technical Details

### Initialization Strategies
- **Embeddings**: Xavier initialization - uniform(-√(6/(vocab_size + d_model)), √(6/(vocab_size + d_model)))
- **Linear Layers**: Normal distribution N(0, √(2/d_model))
- **Layer Norm**: gamma=1.0, beta=0.0

### Attention Mechanism
The scaled dot-product attention:
```
score = (Q · K^T) / √d_k
attention = softmax(score)
output = attention · V
```

### Residual Connections
Post-normalization pattern:
```
x = x + SubLayer(x)
x = LayerNorm(x)
```

## References

- Vaswani et al. "Attention Is All You Need" (2017)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

## License

This implementation is part of the ADAI project.

## Contributing

Contributions are welcome! Areas for improvement:
- Optimizations for specific hardware
- Additional activation functions
- Alternative attention mechanisms
- Visualization tools
- Comprehensive benchmarks
