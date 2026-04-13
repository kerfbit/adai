# Core Model Components: A Technical Reference

*ADAI Transformer Architecture — Internal Developer Guide*

---

## Preface

This reference documents the internal architecture of the ADAI language model system. It is intended for developers working on the C++ codebase who need to understand how individual components relate to one another, how data flows through the model, and where to locate the relevant implementation files.

Each chapter covers one layer of the build hierarchy, from primitive mathematical operations up through complete model assemblies, inference optimization, and retrieval augmentation. Mathematical notation follows standard transformer literature. Source file paths are given relative to the `src/` directory.

The reader is assumed to be familiar with basic linear algebra, C++17, and the high-level concept of neural network training.

---

## Table of Contents

1. Architecture Overview
2. Foundation Layer — `adai_core`
3. Layer Utilities — `adai_layers`
4. Attention Mechanisms — `adai_attention`
5. Feed-Forward Networks — `adai_feedforward`
6. Transformer Blocks — `adai_transformer`
7. Complete Model Assemblies — `adai_models`
8. Natural Language Processing Pipeline — `adai_nlp`
9. Performance Acceleration
10. API and Integration Layer
11. Inference Optimization — KV Cache
12. Retrieval-Augmented Generation
13. Glossary

---

## Chapter 1: Architecture Overview

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Describe the overall encoder-decoder transformer architecture used by ADAI.
> - Identify the static library targets and their dependency relationships.
> - Understand how the CMake build system assembles the model from layered components.

### 1.1 Introduction

ADAI is a transformer-based encoder-decoder language model implemented in C++17. The architecture follows the design introduced in *Attention is All You Need* (Vaswani et al., 2017), composed of stacked encoder and decoder blocks with multi-head attention, feed-forward networks, residual connections, and layer normalization.

### 1.2 Library Dependency Hierarchy

The build system composes the model from a set of layered static libraries defined in `src/CMakeLists.txt`. Each library depends only on libraries below it in the hierarchy, enforcing clean separation of concerns.

**Figure 1.1 — Static Library Dependency Graph**

```text
adai_core  ──────────────────────────────────────────────────┐
  └─► adai_layers    (LayerNorm, PositionalEncoding,         │
  │                   TokenEmbedding)                        │
  └─► adai_attention (MultiHeadAttention, CrossAttention)    │
  │         * KVCache — header-only, embedded here           │
  └─► adai_feedforward (FeedForward)                         │
        └─► adai_transformer (EncoderBlock, DecoderBlock)    │
                └─► adai_models (LLMEncoder, LLMDecoder,     │
                                 LanguageModelHead,          │
                                 EncoderDecoderModel)        │
                      └─► [DocumentStore + RAGInference]     │
                            (compiled per-executable)        │
adai_nlp  ◄──────────────────────────────────────────────────┘
  (BPETokenizer, TextGenerator, ConversationContext)
```

### 1.3 Build System Integration

The top-level `CMakeLists.txt` sets global compiler flags, fetches third-party dependencies via `FetchContent` (spdlog v1.12.0, Google Test v1.14.0), and delegates to `src/CMakeLists.txt` for library and executable targets. Optional features are controlled by CMake options:

| Option | Default | Effect |
| --- | --- | --- |
| `ENABLE_GPU` | OFF | Adds CUDA support to `adai_core` |
| `ENABLE_OPENMP` | auto-detected | Enables parallel matrix operations |
| `BUILD_API_SERVER` | ON | Builds the chatbot REST API (requires cpp-httplib) |
| `BUILD_METRICS_API_SERVER` | ON | Builds the training metrics REST API |
| `BUILD_TESTING` | ON | Fetches and builds Google Test suite |

> **Note:** Optional acceleration via OpenMP is detected automatically. No option flag is required; CMake reports whether it was found at configure time.

### 1.4 Chapter Map

The remaining chapters follow the dependency hierarchy bottom-up, starting with the mathematical primitives in Chapter 2 and progressing through to the retrieval-augmented generation layer in Chapter 12.

### Summary

ADAI is an encoder-decoder transformer built as a layered set of C++ static libraries. The CMake build system enforces a strict dependency hierarchy from mathematical primitives (`adai_core`) up through complete model assemblies (`adai_models`) and optional application layers (RAG, API servers). Optional GPU, OpenMP, and BLAS acceleration can be enabled at configure time.

### Review Questions

1. Which CMake option enables GPU acceleration, and which library target does it affect?
2. What source files are compiled into the `adai_api` static library, and what enables the RAG path at runtime?
3. What is the role of `spdlog` in the build dependency graph?

---

## Chapter 2: Foundation Layer — `adai_core`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Describe the primary data structure used to represent tensors in ADAI.
> - Identify all activation functions and their mathematical definitions.
> - Distinguish between the four supported optimization algorithms.
> - Explain the role of the logger and metrics service in the training loop.

The `adai_core` library is the lowest level of the dependency hierarchy. Every other library links against it. It provides the tensor type, activation functions, optimization algorithms, logging infrastructure, and training metrics collection. All learnable weight matrices, activations, and gradients in the system are represented as types defined in this layer.

### 2.1 Matrix Operations (`Matrix.cpp` / `Matrix.hpp`)

> **Definition:** A **Matrix** is the primary tensor type in ADAI, backed by a `std::vector<std::vector<float>>`. It is used to represent weight matrices, bias vectors, activations, gradients, and positional encodings throughout the model.

All weight matrices, activations, and gradients are represented as `Matrix` objects. The class provides both a dimension constructor `Matrix(int r, int c)` and a data constructor `Matrix(const std::vector<std::vector<float>>& d)` with row-length validation.

**Key operations:**

- Element access via `operator()(int i, int j)` with bounds checking
- Matrix multiplication (`operator*`) — parallelized with OpenMP when available (`#pragma omp parallel for collapse(2)`)
- Element-wise addition, subtraction, and scalar operations
- Transpose, Hadamard product, and gradient utilities
- Xavier/Glorot random initialization

> **Note:** The matrix multiplication inner loop uses `#pragma omp simd reduction(+:sum)` for SIMD vectorization. BLAS integration (OpenBLAS / Intel MKL) for matrices larger than 256×256 is planned but not yet implemented (see TD-007).

### 2.2 Activation Functions (`Activation.cpp` / `Activation.hpp`)

The `Activation` class is a static utility providing the activation functions and their derivatives needed for forward and backward passes throughout the model.

**Table 2.1 — Activation Functions**

| Function | Formula | Used By |
| --- | --- | --- |
| `softmax` | $\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$ | Attention, LM Head |
| `gelu` | $\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}(x + 0.044715x^3)\right)\right)$ | FeedForward |
| `gelu_derivative` | Derivative of the GELU tanh approximation | FeedForward backward |
| `softmax_derivative` | Jacobian of softmax | Attention backward |

The softmax implementation subtracts the row maximum before exponentiation for numerical stability, preventing floating-point overflow on large logits.

### 2.3 Optimization Algorithms (`Optimizer.cpp` / `Optimizer.hpp`)

> **Definition:** An **Optimizer** manages the update of a collection of `ParameterGroup` objects, each of which pairs a weight matrix with its accumulated gradient matrix and optimizer-specific state.

The `Optimizer` class provides centralized gradient management. All trainable components hold either an owned or pointer reference to an `Optimizer` instance. Training a model requires calling `zero_grad()` before each batch, accumulating gradients through the backward pass, optionally calling `clip_gradients()`, and then calling `step()` to apply updates.

**Table 2.2 — Optimizer Types (`OptimizerType` enum)**

| Type | Description |
| --- | --- |
| `SGD` | Stochastic Gradient Descent — $\theta \leftarrow \theta - \eta \nabla_\theta L$ |
| `SGD_MOMENTUM` | SGD with first-moment momentum to dampen oscillations |
| `ADAM` | Adaptive Moment Estimation — per-parameter adaptive learning rates |
| `ADAMW` | Adam with decoupled weight decay regularization |

The `ParameterGroup` struct holds a pointer to the weight `Matrix`, a pointer to its gradient `Matrix`, a momentum `Matrix`, a velocity `Matrix`, and a step counter `int`. Gradient clipping and L2 weight decay are applied globally before the per-parameter update rule.

### 2.4 Logger (`Logger.cpp` / `Logger.hpp`)

The logging infrastructure wraps **spdlog v1.12.0**, fetched automatically via `FetchContent` at configure time. It provides leveled, timestamped log records consumed by training loops, inference paths, and the metrics service. Log levels follow the standard spdlog hierarchy: `trace`, `debug`, `info`, `warn`, `error`, `critical`.

### 2.5 Training Metrics Service (`TrainingMetricsService.cpp` / `TrainingMetricsService.hpp`)

The `TrainingMetricsService` collects per-step and per-epoch training metrics including loss, accuracy, learning rate, and throughput. When built with cpp-httplib and the `BUILD_METRICS_API_SERVER` option enabled, it can push metrics to the REST API server at runtime (see Section 10.2).

### Summary

`adai_core` provides the four building blocks that all other libraries depend on: the `Matrix` tensor type, the `Activation` function library, the `Optimizer` gradient management system, and the logging and metrics infrastructure. No model component operates without objects defined in this layer.

### Key Terms

- **Matrix** — The fundamental 2D float tensor type.
- **GELU** — Gaussian Error Linear Unit; the activation function used in feed-forward layers.
- **AdamW** — Preferred optimizer combining Adam's adaptive rates with decoupled weight decay.
- **Parameter group** — A coupled pair of (weight matrix, gradient matrix) tracked by the optimizer.

### Review Questions

1. What data structure backs the `Matrix` type, and what validation does the data constructor perform?
2. Why is max-subtraction applied before exponentiation in the softmax implementation?
3. How does `ADAMW` differ from `ADAM` in its treatment of weight decay?
4. Which optimizer state matrices does a `ParameterGroup` maintain, and for which optimizer types are they used?

---

## Chapter 3: Layer Utilities — `adai_layers`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Derive the layer normalization formula and explain why it differs from batch normalization.
> - Reproduce the sinusoidal positional encoding construction.
> - Describe how token embeddings are initialized, updated, and why gradients are sparse.

The `adai_layers` library provides the three utility components placed immediately after the input and between every transformer sub-layer: layer normalization, positional encoding, and token embedding. All three depend on `adai_core` and are linked into `adai_transformer`.

### 3.1 Layer Normalization (`LayerNorm.cpp` / `LayerNorm.hpp`)

> **Definition:** **Layer Normalization** normalizes the activations within each individual sample across the feature dimension, independently of other samples in the batch. This contrasts with batch normalization, which normalizes across the batch dimension.

Layer normalization is applied after every attention and feed-forward sub-layer in both the encoder and decoder. For each sample row of input $x \in \mathbb{R}^d$:

$$\mu = \frac{1}{d}\sum_{i} x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i}(x_i - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad \text{output} = \gamma \odot \hat{x} + \beta$$

The small constant $\varepsilon$ (default $10^{-5}$) prevents division by zero when the variance is near zero.

**Learnable parameters:** affine scale $\gamma \in \mathbb{R}^{1 \times d}$ and shift $\beta \in \mathbb{R}^{1 \times d}$, each initialized to ones and zeros respectively. Both are updated by the attached `Optimizer` instance during training.

For the backward pass, the module caches the original input, the normalized values $\hat{x}$, and the per-sample means and variances to avoid redundant recomputation during gradient accumulation.

### 3.2 Positional Encoding (`PositionalEncoding.cpp` / `PositionalEncoding.hpp`)

> **Definition:** **Positional Encoding** adds a fixed vector to each token embedding to provide the model with information about the position of each token in the sequence. Without it, the self-attention mechanism is permutation-invariant.

Because the self-attention mechanism has no inherent notion of order, positional encodings inject sequence position information. The ADAI implementation uses sinusoidal encodings as described in the original transformer paper. For position $pos$ and embedding dimension index $i$:

$$PE(pos,\, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right), \qquad PE(pos,\, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

The full encoding matrix `pos_encoding [max_len, d_model]` is pre-computed once at construction. During the forward pass, the appropriate rows are element-wise added to the input embedding matrix. Because these encodings are deterministic and fixed, they have no learnable parameters and require no backward pass.

> **Note:** Sinusoidal encodings generalize gracefully to sequences longer than those seen during training, because relative position relationships are encoded in the ratio of frequencies. Learned positional embeddings do not share this property.

### 3.3 Token Embedding (`TokenEmbedding.cpp` / `TokenEmbedding.hpp`)

> **Definition:** A **Token Embedding** is a trainable lookup table that maps each discrete token ID to a dense continuous vector of dimension $d_\text{model}$. It is the first trainable layer the input passes through.

The embedding matrix `embedding_matrix [vocab_size, d_model]` is initialized with Xavier scaling:

$$\text{scale} = \sqrt{\frac{1}{d_\text{model}}}$$

**Forward pass:** For a sequence of token IDs $[t_1, t_2, \ldots, t_n]$, the output is the stacked matrix of corresponding rows:

$$\text{output}[i, :] = \text{embedding\_matrix}[t_i, :]$$

This produces an output of shape `[seq_len, d_model]`.

**Backward pass:** Gradients are sparse — they accumulate only at the rows corresponding to tokens present in the input sequence. The gradient matrix `embedding_grad [vocab_size, d_model]` is zeroed at the start of each batch and accumulated across positions referencing the same token ID.

### Summary

`adai_layers` provides the three conditioning components applied at the boundary between raw token IDs and the transformer stack: layer normalization (stabilizing intermediate activations), positional encoding (injecting sequence order), and token embedding (mapping discrete tokens to continuous space). None of these components are computationally expensive relative to attention; their correctness is, however, critical to training convergence.

### Key Terms

- **Layer Normalization** — Per-sample normalization across features; more stable than batch normalization for sequence models.
- **Sinusoidal Encoding** — A fixed, non-learned positional encoding using sine and cosine functions of varying frequency.
- **Token Embedding** — A trainable $\mathbb{R}^{\text{vocab\_size} \times d_\text{model}}$ lookup table.
- **Xavier Initialization** — A weight initialization strategy scaling variance by $1/d_\text{model}$ to maintain gradient magnitude at initialization.

### Review Questions

1. What is the value of $\varepsilon$ in layer normalization and why is it necessary?
2. Why do sinusoidal positional encodings generalize better to longer sequences than learned embeddings?
3. Why are token embedding gradients described as "sparse," and what implication does this have for batch efficiency?
4. At what point in the encoder pipeline does positional encoding occur relative to token embedding?

---

## Chapter 4: Attention Mechanisms — `adai_attention`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Derive the scaled dot-product attention formula and explain the role of the scaling factor $\sqrt{d_k}$.
> - Distinguish between self-attention and cross-attention and identify where each is used.
> - Explain how `KVCache` integrates with the attention forward pass during inference.

The `adai_attention` library implements the two attention variants used in the transformer: multi-head self-attention and cross-attention. It also embeds the `KVCache` header for inference optimization (covered in depth in Chapter 11).

### 4.1 Multi-Head Self-Attention (`MultiHeadAttention.cpp` / `MultiHeadAttention.hpp`)

> **Definition:** **Multi-Head Self-Attention** is a mechanism that allows every position in a sequence to attend to every other position simultaneously, using multiple independent attention "heads" that learn complementary relationship patterns.

This is the primary attention mechanism, used in both encoder blocks (unrestricted) and decoder blocks (causally masked). For input $X \in \mathbb{R}^{L \times d_\text{model}}$, the forward pass computes:

$$Q = XW_q, \qquad K = XW_k, \qquad V = XW_v$$

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$$\text{Output} = \text{Attention} \cdot W_o$$

The scaling factor $1/\sqrt{d_k}$ prevents the dot products from growing large in magnitude, which would push the softmax into regions of vanishingly small gradient.

**Learnable parameters:** projection matrices $W_q,\, W_k,\, W_v,\, W_o$, each of shape `[d_model, d_model]`, with corresponding gradient matrices.

**KV Cache integration:** `forward()` accepts an optional `KVCache*` argument. When non-null, new keys and values for the current step are appended to the cache, and attention is computed over the full accumulated sequence. See Chapter 11 for the complete caching mechanism.

For the backward pass, the module caches the input $X$, projected queries $Q$, keys $K$, values $V$, the pre-softmax scores, the attention weight matrix, and the attention output.

### 4.2 Cross-Attention (`CrossAttention.cpp` / `CrossAttention.hpp`)

> **Definition:** **Cross-Attention** is a variant of multi-head attention where the queries originate from the decoder and the keys and values originate from the encoder output. This allows the decoder to selectively focus on relevant parts of the encoded input at each generation step.

The fundamental difference from self-attention is the source of the key and value inputs:

**Table 4.1 — Self-Attention vs. Cross-Attention Input Sources**

| Tensor | Self-Attention source | Cross-Attention source |
| --- | --- | --- |
| Query ($Q$) | Decoder input | Decoder input |
| Key ($K$) | Decoder input | Encoder output |
| Value ($V$) | Decoder input | Encoder output |

For decoder input $X_\text{dec} \in \mathbb{R}^{L_\text{tgt} \times d}$ and encoder output $X_\text{enc} \in \mathbb{R}^{L_\text{src} \times d}$:

$$Q = X_\text{dec} \cdot W_q, \qquad K = X_\text{enc} \cdot W_k, \qquad V = X_\text{enc} \cdot W_v$$

$$\text{Scores} \in \mathbb{R}^{L_\text{tgt} \times L_\text{src}}, \qquad \text{Output} = \text{softmax}(\text{Scores}) \cdot V \cdot W_o$$

`CrossAttention` also accepts an optional `KVCache*`. Because the encoder output is fixed once computed, the cross-attention K/V pairs are computed a single time and reused across all generation steps — a significant inference saving for long encoder sequences.

### Summary

`adai_attention` provides scaled dot-product multi-head self-attention and cross-attention. Self-attention enables every position to attend to every other within the same sequence; cross-attention bridges encoder and decoder by routing decoder queries to encoder keys and values. Both integrate with `KVCache` for efficient autoregressive inference.

### Key Terms

- **Scaled Dot-Product Attention** — Attention score computation $\text{softmax}(QK^\top/\sqrt{d_k})V$.
- **Multi-Head Attention** — Running multiple parallel attention operations with independently learned projections.
- **Cross-Attention** — Attention where queries and key/values come from different sequences.
- **Causal Mask** — A mask applied during decoder self-attention that zeroes out future positions.

### Review Questions

1. Why is the dot product scaled by $1/\sqrt{d_k}$ before the softmax?
2. In the decoder, which attention sub-layer uses a causal mask, and why?
3. Why can cross-attention K/V pairs be cached once and reused, while self-attention K/V must be updated at every generation step?
4. What tensors does `MultiHeadAttention` cache for use in the backward pass?

---

## Chapter 5: Feed-Forward Networks — `adai_feedforward`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - State the mathematical operations of the position-wise feed-forward network.
> - Explain the role of $d_\text{ff}$ relative to $d_\text{model}$ and the reasoning for the 4× convention.
> - Identify the initialization strategy used for weights and biases.

### 5.1 Position-wise Feed-Forward Layer (`FeedForward.cpp` / `FeedForward.hpp`)

> **Definition:** A **Position-wise Feed-Forward Network** is a two-layer MLP applied identically and independently to each position in the sequence. It introduces non-linearity and expands the model's representational capacity between attention layers.

The feed-forward sub-layer is the second operation in every encoder and decoder block, applied after attention and residual normalization. The forward computation is:

$$\text{hidden} = \text{GELU}\!\left(X \cdot W_1 + b_1\right), \qquad \text{output} = \text{hidden} \cdot W_2 + b_2$$

The hidden dimension $d_\text{ff}$ is conventionally set to $4 \times d_\text{model}$. This expansion allows the network to compute richer intermediate representations before projecting back to the residual stream dimension.

**Table 5.1 — Feed-Forward Parameter Shapes**

| Parameter | Shape | Initialization |
| --- | --- | --- |
| $W_1$ | $[d_\text{model} \times d_\text{ff}]$ | Xavier/He |
| $b_1$ | $[1 \times d_\text{ff}]$ | Zero |
| $W_2$ | $[d_\text{ff} \times d_\text{model}]$ | Xavier/He |
| $b_2$ | $[1 \times d_\text{model}]$ | Zero |

The GELU activation (see Table 2.1) is used in preference to ReLU for its smoother gradient properties. For the backward pass, the module caches the pre-activation hidden state alongside the GELU-activated hidden state, allowing efficient gradient computation through both weight matrices.

The module supports gradient clipping and weight persistence via `save()` and `load()` methods.

### Summary

The feed-forward sub-layer expands each position's representation into a higher-dimensional space with GELU non-linearity, then projects back to $d_\text{model}$. It contains the majority of the parameter count in a transformer relative to the attention sub-layer, and its position-wise independence makes it trivially parallelizable.

### Review Questions

1. What is the conventional relationship between $d_\text{ff}$ and $d_\text{model}$, and why?
2. Why is GELU preferred over ReLU in transformer feed-forward layers?
3. Which cached tensors from the forward pass are required to compute gradients for $W_1$ and $W_2$?

---

## Chapter 6: Transformer Blocks — `adai_transformer`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Trace the complete data flow through an encoder block, including residual connections.
> - Explain the purpose of causal masking in the decoder self-attention sub-layer.
> - Compare the three-sub-layer structure of the decoder block to the two-sub-layer encoder block.

The `adai_transformer` library composes the components from Chapters 3–5 into complete, stackable transformer blocks. These blocks are the repeating units assembled into the full encoder and decoder models in Chapter 7.

### 6.1 Encoder Block (`EncoderBlock.cpp` / `EncoderBlock.hpp`)

> **Definition:** An **Encoder Block** is a single transformer encoder layer that applies multi-head self-attention followed by a feed-forward network, with residual connections and layer normalization around each sub-layer.

A single encoder block is stacked $N$ times to form the full encoder. Its internal data flow is:

**Figure 6.1 — Encoder Block Architecture**

```text
Input (x)
  └─► MultiHeadAttention(x, x, x, mask)
        └─► Add residual:  x + attn_output
              └─► LayerNorm₁  →  r₁
                    └─► FeedForward(r₁)
                          └─► Add residual:  r₁ + ff_output
                                └─► LayerNorm₂  →  Output
```

The full sequence of mathematical operations is:

$$\text{attn} = \text{MultiHeadAttention}(x,\, x,\, x,\, \text{mask})$$

$$r_1 = \text{LayerNorm}_1(x + \text{attn})$$

$$\text{ff} = \text{FeedForward}(r_1)$$

$$\text{output} = \text{LayerNorm}_2(r_1 + \text{ff})$$

The two residual connections ensure gradient flow through deep stacks without vanishing. The two `LayerNorm` instances are independent with their own learnable $\gamma$ and $\beta$ parameters.

**Subcomponents** (owned via `unique_ptr`): `MultiHeadAttention`, `FeedForward`, `LayerNorm` ×2.

**Constructor signature:** `EncoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f)`

### 6.2 Decoder Block (`DecoderBlock.cpp` / `DecoderBlock.hpp`)

> **Definition:** A **Decoder Block** extends the encoder block with a second attention sub-layer: a causally masked self-attention over the decoder's own output sequence, and a cross-attention sub-layer that attends to the encoder output. Three residual connections and three layer normalizations are used.

The decoder block has three sub-layers, requiring an additional residual connection and layer normalization compared to the encoder block.

**Figure 6.2 — Decoder Block Architecture**

```text
Input (x)
  └─► MaskedSelfAttention(x, x, x, causal_mask)
        └─► Add residual + LayerNorm₁  →  r₁
              └─► CrossAttention(r₁, encoder_output, encoder_output)
                    └─► Add residual + LayerNorm₂  →  r₂
                          └─► FeedForward(r₂)
                                └─► Add residual + LayerNorm₃  →  Output
```

$$r_1 = \text{LayerNorm}_1\!\left(x + \text{MaskedSelfAttn}(x,\, x,\, x,\, \text{causal\_mask})\right)$$

$$r_2 = \text{LayerNorm}_2\!\left(r_1 + \text{CrossAttn}(r_1,\, \text{enc},\, \text{enc})\right)$$

$$\text{output} = \text{LayerNorm}_3\!\left(r_2 + \text{FeedForward}(r_2)\right)$$

The causal mask in the self-attention sub-layer is a lower-triangular binary matrix that prevents position $t$ from attending to positions $t+1, t+2, \ldots$, enforcing the autoregressive property.

**Subcomponents** (owned via `unique_ptr`): `MultiHeadAttention` (self), `CrossAttention`, `FeedForward`, `LayerNorm` ×3.

### Summary

Transformer blocks wrap attention and feed-forward sub-layers with residual connections and layer normalization. The encoder block has two sub-layers; the decoder block has three, adding cross-attention to the encoder output. These blocks are the repeating units of the full model and the primary site of gradient flow during backpropagation.

### Review Questions

1. How many `LayerNorm` instances does a decoder block contain, and what does each normalize?
2. What is the shape of the causal mask for a sequence of length $L$, and how does it enforce the autoregressive property?
3. Why are residual connections essential in deep transformer stacks?
4. In the decoder block's forward pass, what is the source of the encoder output argument, and when is it computed?

---

## Chapter 7: Complete Model Assemblies — `adai_models`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Trace the end-to-end pipeline from raw text to output embeddings through the LLM encoder.
> - Describe the default hyperparameter configuration and identify where it is set.
> - Explain the role of the language model head and how it maps decoder outputs to vocabulary logits.
> - Describe the full autoregressive generation loop within `EncoderDecoderModel`.

The `adai_models` library assembles the sub-libraries from previous chapters into complete, deployable model classes. It is the top of the library dependency hierarchy before the application-layer components.

### 7.1 LLM Encoder (`LLMEncoder.cpp` / `encoder.hpp`)

> **Definition:** The **LLM Encoder** is the full transformer encoder stack. It accepts raw text as input, tokenizes it, embeds it, applies $N$ stacked encoder blocks, and returns a matrix of contextualized representations for each token position.

**Figure 7.1 — LLM Encoder Processing Pipeline**

```text
Input text (string)
  └─► BPETokenizer  →  token IDs  [seq_len]
        └─► TokenEmbedding  →  [seq_len, d_model]
              └─► PositionalEncoding  →  [seq_len, d_model]
                    └─► EncoderBlock × N
                          └─► LayerNorm (final)
                                └─► Output embeddings  [seq_len, d_model]
```

**Table 7.1 — Default LLM Encoder Configuration**

| Parameter | Default Value | Description |
| --- | --- | --- |
| `d_model` | 512 | Embedding and hidden dimension |
| `num_layers` | 6 | Number of stacked encoder blocks |
| `num_heads` | 8 | Attention heads per block |
| `d_ff` | 2048 | Feed-forward hidden dimension ($4 \times d_\text{model}$) |
| `max_seq_length` | 512 | Maximum supported sequence length |

**Key methods:** `encode(text)` — tokenizes and encodes a string, returning `[seq_len, d_model]`; `backward(grad)` — propagates gradients through the full stack; `update_weights()` and `zero_gradients()` — training utilities.

> **Note:** `LLMEncoder` also serves as a BERT-equivalent embedding model in the RAG system. When used by `DocumentStore`, it encodes documents to `[d_model]`-dimensional vectors for semantic similarity search. See Chapter 12.

### 7.2 LLM Decoder (`Decoder.cpp` / `Decoder.hpp`)

> **Definition:** The **LLM Decoder** is the full transformer decoder stack. It accepts token IDs and optional encoder output, applies $N$ stacked decoder blocks with causal self-attention and cross-attention, and returns contextual representations to be projected by the language model head.

**Pipeline:** Input tokens → `TokenEmbedding` → `PositionalEncoding` → `DecoderBlock` ×N → `LayerNorm` (final) → Output `[seq_len, d_model]`

The internal method `create_causal_mask(seq_length)` constructs a lower-triangular matrix of shape `[seq_len, seq_len]`, ensuring that position $t$ can only attend to positions $0, 1, \ldots, t$ during both training and inference. When encoder output is provided, each decoder block applies cross-attention to it; otherwise, the model operates in decoder-only mode.

For efficient inference, `forward_with_cache(token_ids, DecoderKVCache&)` processes one new token at a time, reusing cached K/V pairs from previous steps (see Chapter 11).

### 7.3 Language Model Head (`LanguageModelHead.cpp` / `LanguageModelHead.hpp`)

> **Definition:** The **Language Model Head** is the final linear projection layer that maps the decoder's $d_\text{model}$-dimensional output to raw unnormalized scores (logits) over the full vocabulary for next-token prediction.

$$\text{logits} = X \cdot W_\text{output} + \text{bias}$$

**Parameter shapes:** input `[seq_len, d_model]`; $W_\text{output}\ [d_\text{model} \times \text{vocab\_size}]$; bias `[1, vocab_size]`; output `[seq_len, vocab_size]`. Weights are Xavier-initialized; bias is zero-initialized. The module supports optimizer attachment and model persistence via `save()` and `load()`.

The output logits are passed to a softmax (during training for loss computation) or to the `TextGenerator` (during inference for token selection).

### 7.4 Encoder-Decoder Model (`EncoderDecoderModel.cpp` / `EncoderDecoderModel.hpp`)

> **Definition:** The **Encoder-Decoder Model** is the top-level sequence-to-sequence transformer, composing the encoder, decoder, language model head, tokenizer, and text generator into a single class with a unified training and generation interface.

**Composed subcomponents** (each owned via `unique_ptr`): `BPETokenizer`, `LLMEncoder`, `LLMDecoder`, `LanguageModelHead`, `TextGenerator`.

**Figure 7.2 — Autoregressive Generation Flow**

```text
Input text → BPETokenizer → LLMEncoder → encoder context vector
                                               ↓
              <bos> token → LLMDecoder (cross-attention to encoder)
                               └─► LanguageModelHead → logits → token₁
                            token₁ → LLMDecoder → LM Head → token₂
                                          ↓
                                       <eos> → stop
```

Supported generation strategies are delegated to `TextGenerator`: greedy, beam search, temperature sampling, top-k sampling, and nucleus (top-p) sampling.

### Summary

`adai_models` is the apex of the library hierarchy. It assembles the encoder, decoder, language model head, tokenizer, and generator into the four concrete model classes. `LLMEncoder` is reused both as the encoder half of the full model and as a BERT-style embedding backbone for the RAG retrieval system.

### Key Terms

- **LLM Encoder** — Full transformer encoder; outputs contextualized embeddings per token position.
- **LLM Decoder** — Full transformer decoder with causal masking and optional cross-attention.
- **Language Model Head** — Linear projection from $d_\text{model}$ to vocabulary size.
- **Encoder-Decoder Model** — Top-level seq2seq class composing all submodules.
- **Causal Mask** — Lower-triangular mask preventing attention to future positions.

### Review Questions

1. What is the shape of the output of `LLMEncoder.encode(text)`, and what does each dimension represent?
2. At inference time, `LLMDecoder` can operate in two modes depending on whether encoder output is provided. What changes between the two modes?
3. Why is the language model head's weight matrix Xavier-initialized rather than zero-initialized?
4. How many `unique_ptr` subcomponents does `EncoderDecoderModel` own, and what are they?

---

## Chapter 8: Natural Language Processing Pipeline — `adai_nlp`

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Describe the Byte-Pair Encoding algorithm and identify its role at the model boundary.
> - Compare the five supported text generation strategies and their determinism properties.
> - Explain the conversation context management model, including how limits are enforced.

The `adai_nlp` library provides the text-facing components that sit at the boundary between raw strings and numerical tensors. It links against `adai_core` and is consumed by `adai_models` for training and inference pipelines.

### 8.1 BPE Tokenizer (`BPETokenizer.cpp` / `BPETokenizer.hpp`)

> **Definition:** **Byte-Pair Encoding (BPE)** is a subword tokenization algorithm that iteratively merges the most frequent adjacent byte or character pairs in the training corpus, producing a vocabulary of subword units. It allows the model to handle out-of-vocabulary words by decomposing them into known subwords.

The `BPETokenizer` converts raw text strings to integer token ID sequences and back, forming the interface between the NLP pipeline and all model components.

**Vocabulary:** The vocabulary is loaded from a file and pre-populated with four special tokens defined in `SpecialTokens.hpp`:

| Token | Symbol | Role |
| --- | --- | --- |
| `PAD` | `<pad>` | Sequence padding to uniform length |
| `UNK` | `<unk>` | Unknown / out-of-vocabulary token |
| `BOS` | `<bos>` | Beginning-of-sequence marker |
| `EOS` | `<eos>` | End-of-sequence marker |

**Pre-tokenization** uses a regex pattern that splits on English contractions (`'s`, `'t`, `'re`, etc.), words, digit sequences, punctuation, and whitespace before BPE merge rules are applied.

**Error handling** uses typed exception classes for precise failure diagnosis:

- `TokenizerInputError` — empty or invalid input
- `TokenizerEncodingError` — malformed UTF-8
- `VocabularyFileError` — malformed vocabulary file
- `TokenIDError` — token ID out of range

**Key methods:** `encode(text)` → `vector<int>`; `decode(ids)` → `string`; `load_vocab(path)`; `save_vocab(path)`.

### 8.2 Text Generator (`TextGenerator.cpp` / `TextGenerator.hpp`)

> **Definition:** A **Text Generator** implements decoding strategies that select one token at a time from the model's output logit distribution, appending each selected token to the sequence until a stopping criterion is met.

The `TextGenerator` class wraps any model that emits logits and provides five generation strategies, parameterized by the `GenerationConfig` struct.

**Table 8.1 — Generation Strategy Comparison**

| Strategy | Determinism | Mechanism |
| --- | --- | --- |
| Greedy | Deterministic | Select $\arg\max$ of logits at each step |
| Beam Search | Semi-deterministic | Maintain $k$ best partial sequences (`BeamHypothesis`); select highest-scoring complete sequence |
| Temperature sampling | Stochastic | Scale logits by $1/T$; sample from resulting distribution |
| Top-$k$ sampling | Stochastic | Restrict sampling to the $k$ highest-probability tokens |
| Nucleus (Top-$p$) | Stochastic | Restrict sampling to the smallest token set whose cumulative probability $\geq p$ |

The `GenerationConfig` struct holds all strategy parameters: `max_length`, `temperature`, `top_k`, `top_p`, `beam_width`, `repetition_penalty`, and `length_normalization` (used for beam search score normalization).

### 8.3 Conversation Context (`ConversationContext.cpp` / `ConversationContext.hpp`)

> **Definition:** The **Conversation Context** manages a bounded history of role-labeled messages for multi-turn dialogue, automatically enforcing message count and token budget limits by evicting the oldest non-system messages.

The `ConversationContext` maintains a `deque<Message>` with role-based tracking across three roles: `user`, `assistant`, and `system`. Each `Message` carries a `role` string, a `content` string, and an estimated `token_count`.

**Context management rules:**

- Configurable `max_messages` and `max_tokens` limits
- Sliding-window truncation — oldest messages are dropped when either limit is exceeded
- System message pinning — the system prompt is always kept when `keep_system_message` is `true`
- Conversation state can be persisted and restored via `save()` and `load()`
- The full history is formatted into a single model-input string with appropriate special tokens before each inference call

### Summary

`adai_nlp` is the human-facing layer of the pipeline. The `BPETokenizer` converts text to token IDs; the `TextGenerator` converts logit matrices back to text using configurable decoding strategies; and the `ConversationContext` maintains bounded multi-turn dialogue state. Together they form the complete round-trip text interface for the model.

### Key Terms

- **BPE (Byte-Pair Encoding)** — A subword tokenization algorithm; produces compact vocabularies that generalize to unseen words.
- **Greedy Decoding** — Selecting the highest-probability token at each step; fast but can produce suboptimal sequences.
- **Beam Search** — Maintaining multiple candidate sequences simultaneously; trades compute for output quality.
- **Nucleus Sampling** — Sampling from the minimal token set covering cumulative probability $\geq p$; balances diversity and coherence.
- **System Message** — A privileged prompt providing persistent behavioral context, always retained under truncation.

### Review Questions

1. Why does the BPE tokenizer pre-tokenize with a regex before applying merge rules?
2. A temperature $T > 1$ makes the output distribution flatter. What effect does this have on generation?
3. Why is the system message retained under context truncation while user and assistant messages are evicted?
4. What is the `BeamHypothesis` data structure, and what information does it track?

---

## Chapter 9: Performance Acceleration

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Configure the build for GPU, OpenMP, SIMD, or BLAS acceleration.
> - Identify which OpenMP pragmas and SIMD intrinsic paths are used and the performance gains they provide.
> - Explain the BLAS SGEMM integration and why it targets matrices larger than 256×256.

Performance acceleration in ADAI is entirely optional and does not alter the mathematical correctness of any computation. Each feature is detected or enabled at CMake configure time and guarded by preprocessor macros at compile time.

### 9.1 GPU Acceleration (`gpu/MatrixGPU.cu`)

GPU support is compiled into `adai_core` when the project is configured with `-DENABLE_GPU=ON`. It provides CUDA-accelerated matrix operations through the `CUDA::cudart` and `CUDA::cublas` toolkit libraries.

GPU-specific code is isolated behind the preprocessor guard `ADAI_ENABLE_GPU` and declared in `gpu/GPUUtils.hpp` and `gpu/MatrixGPU.hpp`. The CUDA source file `gpu/MatrixGPU.cu` is compiled as a separate `adai_gpu` static library and linked into `adai_core`.

**Table 9.1 — Targeted CUDA Compute Capabilities**

| Architecture | Compute Capability | Examples |
| --- | --- | --- |
| Pascal | 6.0, 6.1 | GTX 1080, Tesla P100 |
| Volta | 7.0 | Tesla V100 |
| Turing | 7.5 | RTX 2080 |
| Ampere | 8.0, 8.6 | A100, RTX 3090 |

The default target list `60 61 70 75 80 86` can be overridden at configure time with `-DCMAKE_CUDA_ARCHITECTURES`.

### 9.2 OpenMP and SIMD Parallel Processing

OpenMP support is detected automatically by CMake via `find_package(OpenMP)`. When found, `adai_core` is linked against `OpenMP::OpenMP_CXX` and the compile definition `ADAI_ENABLE_OPENMP` is set. No explicit option flag is required.

SIMD intrinsic support is opt-in via `-DENABLE_SIMD=ON`. Runtime CPU feature detection (`has_avx2()` / `has_fma()`) selects the appropriate code path at startup. Capability macros (`ADAI_SIMD_AVX2`, `ADAI_SIMD_FMA`, `ADAI_SIMD_NEON`) and horizontal-reduction helpers (`hsum256()` / `hsum128()`) are defined in `src/MatrixSIMD.hpp`.

The two acceleration layers compose: all SIMD inner loops are wrapped in an `#ifdef ADAI_ENABLE_OPENMP` outer `#pragma omp parallel for`, so thread-level and data-level parallelism are active simultaneously.

**OpenMP parallelism** applies to matrix multiplication:

1. **Outer loop parallelism** — `#pragma omp parallel for collapse(2) schedule(dynamic, 32)` distributes row-column pairs across CPU cores. Active for matrices larger than 64×64.
2. **Inner loop vectorization** — `#pragma omp simd reduction(+:sum)` instructs the compiler to emit SIMD instructions for the dot product accumulation.

**Explicit SIMD intrinsics** accelerate all six performance-critical `Matrix` operations:

| Operation | AVX2/FMA path | ARM NEON path |
| --- | --- | --- |
| `operator*` | `ikj`-order loop, 8 floats per FMA instruction | `vfmaq_f32` |
| `operator+` | `_mm256_add_ps` row-wise loop | `vaddq_f32` |
| `operator-` | `_mm256_sub_ps` row-wise loop | `vsubq_f32` |
| `scale()` | `_mm256_mul_ps` row-wise loop | `vmulq_f32` |
| `hadamard()` | `_mm256_mul_ps` element-wise loop | `vmulq_f32` |
| `apply_gradients()` | `_mm256_fmadd_ps(−lr, g, w)` fused multiply-subtract | `vfmsq_n_f32` |
| `sum()` | horizontal reduction via `hsum256()` | `vaddvq_f32` |

All SIMD paths include scalar remainder loops to handle column widths that are not multiples of 8 (AVX2) or 4 (NEON). The combined effect of OpenMP and SIMD yields a reported **5–8× speedup** over the sequential fallback on multi-core CPUs.

### 9.3 BLAS Integration

BLAS SGEMM integration is opt-in via `-DENABLE_BLAS=ON`. CMake runs `find_package(BLAS)` and `find_path(CBLAS_INCLUDE_DIR cblas.h)` to detect any compatible installation (OpenBLAS, Intel MKL, Apple Accelerate). When both are found, `adai_core` is linked against `${BLAS_LIBRARIES}` and the compile definition `ADAI_ENABLE_BLAS` is set.

For matrices where all three dimensions (rows of A, columns of B, and the shared inner dimension) are at least 256, `Matrix::operator*` routes through `cblas_sgemm` instead of the hand-written loop. The implementation packs the row-major `Matrix` data, calls `cblas_sgemm`, and unpacks the result — no changes to callers are required. For smaller matrices the AVX2/FMA or NEON paths remain active.

The key integration points are:

- `src/CMakeLists.txt` — BLAS detection block sets `ADAI_ENABLE_BLAS` and links `${BLAS_LIBRARIES}`.
- `src/Matrix.cpp` — `#ifdef ADAI_ENABLE_BLAS` guard around the `cblas_sgemm` path in `operator*`.
- `-DENABLE_BLAS=ON` CMake option at the top-level `CMakeLists.txt`.

### Summary

Four acceleration layers are available: CUDA GPU operations (opt-in, `-DENABLE_GPU=ON`), OpenMP multi-core parallelism (auto-detected), explicit SIMD intrinsics for AVX2/FMA and ARM NEON (opt-in, `-DENABLE_SIMD=ON`), and BLAS SGEMM integration (opt-in, `-DENABLE_BLAS=ON`). All fall back gracefully to the sequential C++ implementation when not available. OpenMP and SIMD compose — SIMD inner loops are wrapped in OpenMP parallel regions when both are active.

### Key Terms

- **SIMD (Single Instruction, Multiple Data)** — A CPU parallelism model that applies one instruction to multiple data elements simultaneously; AVX2 processes 8 floats per cycle, NEON processes 4.
- **FMA (Fused Multiply-Add)** — A single instruction computing $a \times b + c$ with one rounding step, improving both throughput and numerical accuracy.
- **BLAS SGEMM** — The single-precision General Matrix Multiply routine from the BLAS standard; highly optimized by vendors for large matrix workloads.
- **`MatrixSIMD.hpp`** — Header defining compile-time capability macros, runtime CPUID detection, and horizontal-reduction helpers for SIMD code paths.

### Review Questions

1. Which preprocessor macro enables GPU-specific code paths, and in which header is it consumed?
2. Why is the OpenMP parallel-for pragma conditional on matrix dimensions greater than 64×64?
3. Why is BLAS targeted specifically at matrices where all dimensions are at least 256 rather than all matrices?
4. What is the role of `MatrixSIMD.hpp`, and what two runtime detection functions does it provide?
5. How do OpenMP and SIMD intrinsics compose in the matrix operations implementation?

---

## Chapter 10: API and Integration Layer

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Identify the build conditions required for the chatbot and metrics API servers.
> - Describe the dependency relationship between the API layer and the core model libraries.

The API layer provides external HTTP interfaces for model inference and training observability. Both servers are optional and compile only when cpp-httplib is detected and the corresponding CMake option is enabled.

### 10.1 Chatbot API (`ChatbotAPI.cpp` / `ChatbotAPI.hpp`)

The chatbot API server exposes a REST interface for inference against `adai_models` and `adai_nlp`. It is built when both `-DBUILD_API_SERVER=ON` (default) and a cpp-httplib installation are detected. The library is auto-sought in `/usr/include`, `/usr/local/include`, and `external/cpp-httplib`.

The `adai_api` static library now incorporates `DocumentStore.cpp` and `RAGInference.cpp` alongside `ChatbotAPI.cpp`. When the server starts with `RAG_ENABLED=true` in the configuration, `ChatbotAPIServer` instantiates a `DocumentStore`, indexes the document directory specified by `RAG_DOCS_PATH`, constructs a `RAGInference` engine, and calls `ChatbotAPI::enableRAG()`. All subsequent `/chat` and `/chat/session` requests are then routed through `RAGInference::generate()` instead of calling `EncoderDecoderModel` directly.

The server is consumed by the `chatbot` CLI executable, which acts as a lightweight client connecting to the API rather than loading the model directly.

### 10.2 Training Metrics API (`TrainingMetricsAPI.cpp` / `TrainingMetricsAPI.hpp`)

The metrics API server exposes training metrics collected by `TrainingMetricsService` (Section 2.5) over HTTP. It runs on its own `pthread` and is built when `-DBUILD_METRICS_API_SERVER=ON` (default) and cpp-httplib are available. A separate `adai_metrics_api` static library is produced and optionally linked into the `incremental_trainer` executable.

### Summary

Both API servers are thin HTTP wrappers over existing library functionality. They compile independently of the core model libraries and impose no link-time penalty when disabled.

---

## Chapter 11: Inference Optimization — KV Cache

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Explain the computational problem that KV caching solves during autoregressive generation.
> - Describe the structure of `KVCache` and `DecoderKVCache` and their relationship.
> - Identify every integration point between the cache and the model components.
> - Estimate the memory cost of maintaining a full `DecoderKVCache`.

### 11.1 Motivation

During autoregressive generation, the model produces one token per forward pass. Without caching, each pass recomputes the key and value matrices for all preceding tokens. For a sequence of $t$ tokens this costs $O(t)$ unnecessary work at every step, making generation cost $O(t^2)$ overall. The KV cache eliminates this by storing computed K/V pairs and appending incrementally.

> **Definition:** A **KV Cache** (Key-Value Cache) is a data structure that stores the key and value projection matrices computed by an attention layer for previously processed token positions, so they can be reused without recomputation in subsequent generation steps.

### 11.2 `KVCache` Struct (`KVCache.hpp`)

`KVCache` is **header-only** — it contains no `.cpp` file and produces no library target. It is included directly wherever attention is computed.

**Table 11.1 — `KVCache` Fields**

| Field | Type | Description |
| --- | --- | --- |
| `keys` | `Matrix [seq_len, d_model]` | Accumulated key projections |
| `values` | `Matrix [seq_len, d_model]` | Accumulated value projections |
| `current_length` | `int` | Number of cached token positions |

**Usage pattern during generation:**

1. **Step 0 (first token):** Cache is empty. K/V are computed for the single input token and stored via `append()`.
2. **Step $t$ (subsequent tokens):** K/V are computed for the single new token only. They are appended to the cache. Attention is computed over the full `[0, t]` range using cached + new K/V.
3. **Sequence complete:** `clear()` resets the cache for the next request.

**Performance:** Eliminates $O(t)$ matrix operations per step. Reported generation speedup is **2–3× for long sequences**. Memory cost per cache is:

$$\text{Memory} = 2 \times L \times d_\text{model} \times \text{sizeof(float)} \text{ bytes}$$

where $L$ is the current sequence length.

### 11.3 `DecoderKVCache` Struct

`DecoderKVCache` aggregates one `KVCache` per decoder layer for both self-attention and cross-attention.

**Figure 11.1 — `DecoderKVCache` Structure**

```text
DecoderKVCache (num_layers = N)
  ├─ self_attention_caches[0]    ← grows by 1 position each generation step
  ├─ self_attention_caches[1]
  │   ...
  ├─ self_attention_caches[N-1]
  ├─ cross_attention_caches[0]  ← computed once from encoder output; reused every step
  ├─ cross_attention_caches[1]
  │   ...
  └─ cross_attention_caches[N-1]
```

Self-attention caches grow monotonically throughout a generation sequence. Cross-attention caches are populated once when the encoder output is first processed and then held constant.

**Key methods:** `get_self_attention_cache(layer_idx)`, `get_cross_attention_cache(layer_idx)`, `clear()`, `clear_self_attention()`, `current_length()`, `is_empty()`.

### 11.4 Integration Points

The `KVCache` is threaded through the model stack via optional pointer arguments, allowing the same code paths to serve both cached inference and uncached training:

| Component | Method signature (excerpt) | Cache argument |
| --- | --- | --- |
| `MultiHeadAttention` | `forward(..., KVCache* kv_cache = nullptr)` | Self-attention cache |
| `CrossAttention` | `forward(..., KVCache* kv_cache = nullptr)` | Cross-attention cache |
| `DecoderBlock` | `forward(..., KVCache* self_attn_cache, KVCache* cross_attn_cache)` | Both caches |
| `LLMDecoder` | `forward_with_cache(token_ids, DecoderKVCache&)` | Full multi-layer cache |
| `EncoderDecoderModel` | Owns and manages `DecoderKVCache` lifecycle | — |

### Summary

The KV cache converts autoregressive generation from $O(t^2)$ to $O(t)$ attention computation by storing and reusing previously computed key and value projections. `KVCache` handles a single layer; `DecoderKVCache` manages the full decoder stack. The cache integrates via optional pointer arguments that default to `nullptr`, preserving uncached training code paths without change.

### Key Terms

- **KV Cache** — Cached key and value matrices for a single attention layer.
- **DecoderKVCache** — Multi-layer container holding per-layer `KVCache` instances for both self- and cross-attention.
- **Autoregressive Generation** — Sequential token-by-token generation where each token is conditioned on all prior tokens.
- **Cache Append** — The operation of extending cached K/V matrices with new-token K/V values.

### Review Questions

1. Without KV caching, what is the total computational complexity of generating a sequence of $T$ tokens? Why?
2. Why are cross-attention caches populated once while self-attention caches grow at each step?
3. How does the optional `KVCache*` argument allow the same attention `forward()` function to serve both training and inference?
4. Given $N=6$ decoder layers, $d_\text{model}=512$, and a sequence length of 256, compute the total memory (in MB) consumed by a fully populated `DecoderKVCache`.

---

## Chapter 12: Retrieval-Augmented Generation

> **Learning Objectives**
>
> After reading this chapter, you will be able to:
>
> - Explain the retrieve-then-generate paradigm and its advantages over pure generation.
> - Describe how `LLMEncoder` functions as a BERT-style embedding model in this context.
> - Trace the full data flow from query to response through the RAG pipeline.
> - Configure a `RAGInference` instance using the `RAGConfig` struct.

### 12.1 Overview and Motivation

Standard language model generation is bounded by the knowledge encoded in its weights at training time. Retrieval-Augmented Generation (RAG) extends generation with a dynamic knowledge source: a document store that is queried at inference time and whose contents are prepended to the model input as context.

> **Definition:** **Retrieval-Augmented Generation (RAG)** is an inference architecture that first retrieves relevant documents from an external store using semantic similarity search, then conditions the generative model on both the retrieved documents and the original query.

RAG is an **optional inference layer** that sits above `adai_models`. `DocumentStore.cpp` and `RAGInference.cpp` are compiled into the `adai_api` static library and are therefore available to every executable that links `adai_api` — including the production `chatbot_api_server`. RAG is activated at runtime via configuration (see Section 12.5); no separate binary is required.

**Figure 12.1 — RAG Architecture**

```text
LLMEncoder  ──► DocumentStore    (embed documents; cosine-similarity retrieval)
                      ↓
              RAGInference  ──► EncoderDecoderModel  ──► Response
```

**Benefits over plain generation:**

- **Factual grounding:** Responses are anchored to retrieved evidence rather than parametric memory.
- **Reduced hallucination:** The model has access to actual document text for the topic at hand.
- **Updatable knowledge:** New documents can be added without retraining.
- **Source attribution:** Document IDs can be included in the response for citation.

### 12.2 `LLMEncoder` as a BERT-equivalent

`LLMEncoder` (Section 7.1) serves a dual role in this codebase. It functions as the encoder half of the full seq2seq model during training and inference, and it also serves as the document embedding backbone for the RAG retrieval system. There is no separate BERT class. The encoder's ability to produce contextualized representations per token is exploited to generate a single document-level embedding via mean pooling across the sequence dimension.

### 12.3 Document Store (`DocumentStore.cpp` / `DocumentStore.hpp`)

> **Definition:** A **Document Store** is an indexed collection of documents where each document is paired with a dense embedding vector produced by the encoder. Given a query, the store returns the top-$k$ most semantically similar documents using cosine similarity.

**`Document` struct fields:**

| Field | Type | Description |
| --- | --- | --- |
| `id` | `string` | Unique document identifier |
| `text` | `string` | Raw document text |
| `embedding` | `Matrix [d_model]` | Dense vector produced by `LLMEncoder` |
| `metadata` | `unordered_map<string, string>` | Optional key-value annotation |

**Indexing:** On `addDocument(id, text)`, the text is passed through `LLMEncoder` to produce a `[d_model]` embedding vector, which is stored alongside the document in an internally maintained `vector<Document>`.

**Retrieval:** `retrieve(query, k)` encodes the query with `LLMEncoder`, then computes cosine similarity against all stored embeddings:

$$\text{cosine\_sim}(a,\, b) = \frac{a \cdot b}{\|a\|\,\|b\|}$$

The top-$k$ scoring `(float score, const Document*)` pairs are returned for use by `RAGInference`.

**Key methods:** `addDocument(id, text)`, `retrieve(query, k)`, `removeDocument(id)`, `size()`.

> **Note:** Retrieval is currently a linear scan over all stored embeddings. For large document collections this becomes a bottleneck; an approximate nearest-neighbour index (e.g. HNSW or FAISS) would reduce retrieval cost from $O(N \cdot d)$ to $O(\log N \cdot d)$.

### 12.4 RAG Inference Engine (`RAGInference.cpp` / `RAGInference.hpp`)

The `RAGInference` class composes `DocumentStore` and `EncoderDecoderModel` into the full retrieve-then-generate pipeline.

**Table 12.1 — `RAGConfig` Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `num_retrieved_docs` | 3 | Number of top-$k$ documents to retrieve |
| `retrieval_threshold` | 0.0 | Minimum cosine similarity; lower scores are filtered |
| `max_context_length` | 512 | Token budget for the retrieved context block |
| `include_scores` | `false` | Prepend similarity scores to each document in context |
| `context_separator` | `"\n\n"` | Delimiter inserted between retrieved documents |
| `context_prefix` | `"Context:\n"` | Text prepended to the context block |
| `query_prefix` | `"\n\nQuestion: "` | Text prepended to the user query |
| `gen_config` | — | `TextGenerator::GenerationConfig` for response generation |

### 12.5 Runtime Configuration

RAG is controlled through the service configuration system (`ServiceConfig` / `config.conf`) with no code changes required.

**Table 12.2 — RAG `config.conf` Keys**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `RAG_ENABLED` | bool | `false` | Master switch; routes all inference through `RAGInference` when `true` |
| `RAG_DOCS_PATH` | string | *(empty)* | Directory of `.txt` files to index at startup; each file becomes one document |
| `RAG_NUM_DOCS` | int | `3` | Maps to `RAGConfig::num_retrieved_docs` |
| `RAG_THRESHOLD` | float | `0.0` | Maps to `RAGConfig::retrieval_threshold`; `0.0` disables filtering |
| `RAG_MAX_CONTEXT_LENGTH` | int | `512` | Maps to `RAGConfig::max_context_length` (tokens) |

All five keys are also readable from environment variables of the same name. They are loaded by `ConfigLoader::load_from_file()` and `ConfigLoader::load_from_env()` using the same priority rules as all other service settings (environment overrides file, file overrides defaults).

**Startup sequence when `RAG_ENABLED=true`:**

```text
ChatbotAPIServer main()
  ├─ [1/4] Load tokenizer
  ├─ [2/4] Initialize EncoderDecoderModel (shared_ptr)
  ├─ [3/4] Initialize ChatbotAPI
  ├─ [+]   Create LLMEncoder (same architecture as model encoder)
  │          └─ load_tokenizer_vocab(config.vocab_path)
  │          └─ Create DocumentStore(rag_encoder)
  │          └─ Load *.txt files from RAG_DOCS_PATH → DocumentStore::addDocument()
  │          └─ Build RAGConfig from service config
  │          └─ Create RAGInference(model, doc_store, rag_config)
  │          └─ ChatbotAPI::enableRAG(rag_engine)
  └─ [4/4] Start HTTP server
```

If `RAG_DOCS_PATH` is not set or the directory is empty, RAG is still active but retrieval returns no documents; the model receives an unaug­mented query. If RAG initialization throws, the server logs a warning and falls back to direct generation.

**Figure 12.2 — RAG Inference Data Flow**

```text
Query (string)
  └─► DocumentStore.retrieve(query, k)
        └─► top-k (score, Document*) pairs
              └─► formatContext()
                    └─► context string (truncated to max_context_length tokens)
                          └─► buildAugmentedPrompt()
                                └─► "Context:\n<docs>\n\nQuestion: <query>"
                                      └─► EncoderDecoderModel.generate()
                                            └─► Response (string)
```

The `truncateContext()` method enforces the `max_context_length` token budget by truncating the concatenated document string. This prevents the augmented prompt from exceeding the model's `max_seq_length`.

### Summary

RAG augments the base `EncoderDecoderModel` with a dynamically queried document store, using `LLMEncoder` as both the document indexer and the query encoder. The `DocumentStore` maps text to dense embeddings and retrieves by cosine similarity; `RAGInference` orchestrates the retrieval, context formatting, and generation steps. Both classes compile into the `adai_api` static library and are active in the production `chatbot_api_server`. The RAG path is enabled at runtime with `RAG_ENABLED=true` in `config.conf` — no recompilation is needed.

### Key Terms

- **RAG (Retrieval-Augmented Generation)** — Inference pattern combining semantic document retrieval with generative language modeling.
- **Document Store** — Indexed collection of (text, embedding) pairs supporting cosine-similarity retrieval.
- **Cosine Similarity** — Normalized dot product measuring the angular distance between two embedding vectors.
- **Augmented Prompt** — The concatenation of retrieved context and the original user query passed to the generative model.

### Review Questions

1. What static library now contains `DocumentStore.cpp` and `RAGInference.cpp`, and what `config.conf` key activates the RAG inference path at runtime?
2. How does `LLMEncoder` serve as a BERT-equivalent in the RAG system?
3. What is the purpose of `retrieval_threshold` in `RAGConfig`, and what value disables it?
4. What algorithmic improvement could address the linear scan cost of `DocumentStore.retrieve()`?
5. A user adds 10,000 documents to the store. Describe the tradeoffs between retrieval accuracy and retrieval speed as $k$ increases.
6. What happens at startup if `RAG_ENABLED=true` but `RAG_DOCS_PATH` is empty or missing?

---

## Chapter 13: Glossary

| Term | Definition |
| --- | --- |
| **AdamW** | Adam optimizer with decoupled weight decay; preferred optimizer for transformer training. |
| **Attention mask** | A binary matrix applied before softmax to prevent certain positions from being attended to. |
| **Augmented prompt** | The concatenation of retrieved context and user query passed to the generative model in RAG. |
| **Autoregressive generation** | Token-by-token generation where each output token is conditioned on all previous tokens. |
| **Beam search** | A generation strategy that maintains multiple candidate sequences and returns the highest-scoring completed sequence. |
| **BPE (Byte-Pair Encoding)** | A subword tokenization algorithm that iteratively merges the most frequent adjacent character pairs. |
| **Causal mask** | A lower-triangular mask preventing decoder self-attention from attending to future positions. |
| **Cosine similarity** | A normalized dot product $\cos\theta = (a \cdot b) / (\|a\|\|b\|)$ measuring angular similarity between two vectors. |
| **Cross-attention** | Attention where queries originate from the decoder and keys/values originate from the encoder output. |
| **DecoderKVCache** | Multi-layer container holding per-layer KVCache instances for both self- and cross-attention in the decoder. |
| **Document store** | An indexed collection of (text, embedding) pairs supporting semantic similarity retrieval. |
| **Encoder block** | A transformer layer applying multi-head self-attention and feed-forward, each followed by residual addition and layer normalization. |
| **Decoder block** | A transformer layer with masked self-attention, cross-attention, and feed-forward sub-layers. |
| **GELU** | Gaussian Error Linear Unit; smooth approximation to ReLU used as the transformer feed-forward activation. |
| **KV Cache** | Stored key and value projection matrices for an attention layer, reused during autoregressive inference. |
| **Language model head** | The final linear projection from $d_\text{model}$ to vocabulary size, producing per-token logits. |
| **Layer normalization** | Per-sample normalization across the feature dimension; stabilizes transformer training. |
| **Matrix** | The primary 2D float tensor type in ADAI; backs all weights, activations, and gradients. |
| **Multi-head attention** | Running multiple parallel scaled dot-product attention operations with independently learned projections. |
| **Nucleus sampling (Top-$p$)** | Sampling from the minimal set of tokens whose cumulative probability is at least $p$. |
| **Parameter group** | A coupled pair of (weight `Matrix`, gradient `Matrix`) tracked by the optimizer. |
| **Positional encoding** | Fixed sinusoidal vectors added to token embeddings to supply sequence position information. |
| **RAG** | Retrieval-Augmented Generation; inference pattern combining semantic document retrieval with generative modeling. |
| **Residual connection** | Adding the input of a sub-layer directly to its output to facilitate gradient flow in deep networks. |
| **Scaled dot-product attention** | $\text{softmax}(QK^\top / \sqrt{d_k})V$; the core attention computation. |
| **Sinusoidal encoding** | Positional encoding using sine and cosine functions of varying frequency; not learned. |
| **System message** | A privileged conversation turn providing persistent behavioral context, retained under context truncation. |
| **Token embedding** | A trainable $\mathbb{R}^{\text{vocab\_size} \times d_\text{model}}$ lookup table mapping token IDs to dense vectors. |
| **Xavier initialization** | Weight initialization scaling variance by $1/d_\text{model}$ to maintain gradient magnitude at initialization. |
