# Decoder Architecture Diagrams

This document provides visual representations of the decoder architecture and its integration with the existing encoder.

---

## 1. Complete Encoder-Decoder Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENCODER-DECODER MODEL                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT TEXT: "What is the weather like?"
    │
    ├─────────────────────────────────────────────────────────────────────┐
    │                          ENCODER PATH                                │
    └─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────┐
│  BPE Tokenizer   │  → [45, 67, 12, 89, 23, 56]
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Token Embedding  │  → [6 × 512] matrix
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Positional Enc.  │  → [6 × 512] matrix (with position info)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Encoder Block 1  │  → Self-Attention + FFN
├──────────────────┤
│ Encoder Block 2  │
├──────────────────┤
│      ...         │
├──────────────────┤
│ Encoder Block N  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Final LayerNorm │
└──────────────────┘
    │
    ▼
 ENCODER OUTPUT: [6 × 512] matrix ─────────────┐
                                                │
    ┌───────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────────────────┐
    │                          DECODER PATH                                │
    └─────────────────────────────────────────────────────────────────────┘
    │
    │  START TOKEN: <BOS> → [101]
    │      │
    │      ▼
    │  ┌──────────────────┐
    │  │ Token Embedding  │  → [1 × 512] matrix
    │  └──────────────────┘
    │      │
    │      ▼
    │  ┌──────────────────┐
    │  │ Positional Enc.  │  → [1 × 512] matrix
    │  └──────────────────┘
    │      │
    │      ▼
    │  ┌──────────────────────────────────────┐
    │  │ Decoder Block 1                       │
    │  │  ┌────────────────────────────────┐  │
    │  │  │ Masked Self-Attention (causal) │  │ ← Decoder input
    │  │  └────────────────────────────────┘  │
    │  │           │                           │
    │  │  ┌────────────────────────────────┐  │
    ├──┼─→│ Cross-Attention to Encoder     │  │ ← Encoder output
    │  │  └────────────────────────────────┘  │
    │  │           │                           │
    │  │  ┌────────────────────────────────┐  │
    │  │  │ Feed-Forward Network           │  │
    │  │  └────────────────────────────────┘  │
    │  └──────────────────────────────────────┘
    │      │
    │  ┌──────────────────────────────────────┐
    │  │ Decoder Block 2                       │
    ├──┼─→│ (same structure)                   │
    │  └──────────────────────────────────────┘
    │      │
    │      ... (more decoder blocks)
    │      │
    │      ▼
    │  ┌──────────────────┐
    │  │  Final LayerNorm │
    │  └──────────────────┘
    │      │
    │      ▼
    │  ┌──────────────────────────────────────┐
    │  │ Language Model Head                   │
    │  │ Linear: [512 → vocab_size (10000)]   │
    │  └──────────────────────────────────────┘
    │      │
    │      ▼
    │  ┌──────────────────┐
    │  │    Softmax       │  → Probability distribution [10000]
    │  └──────────────────┘
    │      │
    │      ▼
    │  ┌──────────────────┐
    │  │  Sample Token    │  → Token ID: 234 ("It")
    │  └──────────────────┘
    │      │
    │      ▼
    └─────(REPEAT: append token, generate next)
           │
           ▼
    GENERATED SEQUENCE: [101, 234, 456, 789, ..., 102]
           │
           ▼
    ┌──────────────────┐
    │  BPE Decode      │
    └──────────────────┘
           │
           ▼
    OUTPUT TEXT: "It is sunny and warm today!"
```

---

## 2. DecoderBlock Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DECODER BLOCK                                     │
│                                                                               │
│  Input: [seq_len × d_model]                                                 │
│  Encoder Output: [enc_seq_len × d_model]                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Decoder Input
         │
         │ (cached for backward)
         ▼
    ┌─────────────────────────────────────┐
    │  MASKED SELF-ATTENTION              │
    │                                      │
    │  Q = K = V = input                  │
    │  Mask: Causal (lower triangular)   │
    │                                      │
    │  MultiHeadAttention.forward(        │
    │      input, input, input, mask)     │
    └─────────────────────────────────────┘
         │
         │ attn_output [seq_len × d_model]
         │
    ┌────┴────┐
    │    +    │  ← RESIDUAL CONNECTION 1
    └────┬────┘
         │ (input + attn_output)
         ▼
    ┌─────────────────────────────────────┐
    │  LAYER NORM 1                        │
    │  norm1.forward(residual1)            │
    └─────────────────────────────────────┘
         │
         │ normed1 [seq_len × d_model]
         │
         ├──────────────────────────────────┐
         │                                   │
         │                                   │
    ┌────┴────────────────────────────┐     │
    │  CROSS-ATTENTION                │     │
    │                                  │     │
    │  Q = normed1                    │     │
    │  K = V = encoder_output         │     │
    │  Mask: Padding (optional)       │     │
    │                                  │     │
    │  MultiHeadAttention.forward(    │     │
    │      normed1, encoder_output,   │     │
    │      encoder_output, mask)      │     │
    └──────────────────────────────────┘     │
         │                                   │
         │ cross_attn_output                 │
         │                                   │
    ┌────┴────┐                              │
    │    +    │  ← RESIDUAL CONNECTION 2 ────┘
    └────┬────┘
         │ (normed1 + cross_attn_output)
         ▼
    ┌─────────────────────────────────────┐
    │  LAYER NORM 2                        │
    │  norm2.forward(residual2)            │
    └─────────────────────────────────────┘
         │
         │ normed2 [seq_len × d_model]
         │
         ├──────────────────────────────────┐
         │                                   │
    ┌────┴────────────────────────────┐     │
    │  FEED-FORWARD NETWORK           │     │
    │                                  │     │
    │  FFN(x) = GELU(xW₁ + b₁)W₂ + b₂│     │
    │                                  │     │
    │  feed_forward.forward(normed2)  │     │
    └──────────────────────────────────┘     │
         │                                   │
         │ ff_output [seq_len × d_model]    │
         │                                   │
    ┌────┴────┐                              │
    │    +    │  ← RESIDUAL CONNECTION 3 ────┘
    └────┬────┘
         │ (normed2 + ff_output)
         ▼
    ┌─────────────────────────────────────┐
    │  LAYER NORM 3                        │
    │  norm3.forward(residual3)            │
    └─────────────────────────────────────┘
         │
         ▼
    Output: [seq_len × d_model]
```

---

## 3. Attention Masking Visualization

### Self-Attention Mask (Causal)

```
Sequence: ["<BOS>", "It", "is", "sunny"]
Positions:    0      1     2      3

ATTENTION MATRIX (before mask):
                To Position →
From Pos ↓    0      1      2      3
    0       [0.25   0.25   0.25   0.25]  ← BOS can see all (before mask)
    1       [0.25   0.25   0.25   0.25]  ← "It" can see all (before mask)
    2       [0.25   0.25   0.25   0.25]  ← "is" can see all (before mask)
    3       [0.25   0.25   0.25   0.25]  ← "sunny" can see all (before mask)

CAUSAL MASK:
                To Position →
From Pos ↓    0      1      2      3
    0       [ 0     -∞     -∞     -∞ ]
    1       [ 0      0     -∞     -∞ ]
    2       [ 0      0      0     -∞ ]
    3       [ 0      0      0      0 ]

ATTENTION MATRIX (after mask + softmax):
                To Position →
From Pos ↓    0      1      2      3
    0       [1.0    0      0      0  ]  ← BOS only sees itself
    1       [0.5   0.5     0      0  ]  ← "It" sees BOS, It
    2       [0.33  0.33   0.33    0  ]  ← "is" sees BOS, It, is
    3       [0.25  0.25   0.25   0.25]  ← "sunny" sees all
```

### Cross-Attention Mask (Padding)

```
Encoder Tokens: ["What", "is", "the", "weather", "<PAD>", "<PAD>"]
Decoder Token:  "It"

CROSS-ATTENTION MASK:
Decoder Pos → Encoder Positions
    "It"  →  [What] [is] [the] [weather] [PAD] [PAD]
             [  0     0    0      0       -∞    -∞  ]
                                          ↑     ↑
                                     Blocked  Blocked

ATTENTION WEIGHTS (after mask + softmax):
    "It"  →  [0.3   0.25  0.2   0.25    0     0  ]
             ↑     ↑     ↑     ↑      ↑     ↑
           What   is    the  weather PAD   PAD
           (attended)                (ignored)
```

---

## 4. Autoregressive Generation Flow

```
STEP-BY-STEP TOKEN GENERATION

Encoder Output (computed once): [6 × 512]
    │
    └─────────────────────────────────────┐
                                          │
Step 1: Generate Token 1                  │
────────────────────────                  │
Input: [<BOS>]                            │
    │                                     │
    ▼                                     │
Decoder([101]) ──────────────────────────┤
    │                                     │
    ▼                                     │
Logits: [10000]                           │
    │                                     │
    ▼                                     │
Softmax → Probs: [0.001, 0.002, ..., 0.05]│
    │                                     │
    ▼                                     │
Sample → Token: 234 ("It")                │
    │                                     │
Generated: [101, 234]                     │
                                          │
Step 2: Generate Token 2                  │
────────────────────────                  │
Input: [<BOS>, "It"]                      │
    │                                     │
    ▼                                     │
Decoder([101, 234]) ──────────────────────┤
    │                                     │
    ▼                                     │
Logits: [10000] (only last position used) │
    │                                     │
    ▼                                     │
Softmax → Sample → Token: 456 ("is")      │
    │                                     │
Generated: [101, 234, 456]                │
                                          │
Step 3: Generate Token 3                  │
────────────────────────                  │
Input: [<BOS>, "It", "is"]                │
    │                                     │
    ▼                                     │
Decoder([101, 234, 456]) ─────────────────┤
    │                                     │
    ▼                                     │
Sample → Token: 789 ("sunny")             │
    │                                     │
Generated: [101, 234, 456, 789]           │
                                          │
Continue until <EOS> or max_length...     │
                                          │
Final: [101, 234, 456, 789, 890, 102]     │
        └────────────────────────────┘    │
                  │                       │
                  ▼                       │
        "It is sunny today <EOS>"         │
```

---

## 5. Component Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                      COMPONENT DEPENDENCIES                      │
└─────────────────────────────────────────────────────────────────┘

                        ┌─────────────┐
                        │   Matrix    │ (foundation)
                        └──────┬──────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
            ┌───────▼──────┐   │   ┌──────▼─────┐
            │  Activation  │   │   │  LayerNorm │
            └──────────────┘   │   └────────────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
            ┌───────▼──────┐   │   ┌──────▼─────────┐
            │ FeedForward  │   │   │ MultiHeadAttn  │
            └──────────────┘   │   └────────────────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
        ┌───────────▼─────┐    │    ┌─────▼──────────┐
        │ TokenEmbedding  │    │    │ PositionalEnc  │
        └─────────────────┘    │    └────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    ┌───────▼────────┐  ┌──────▼────────┐  ┌─────▼─────────┐
    │ EncoderBlock   │  │ DecoderBlock  │  │ LMHead (NEW)  │
    └───────┬────────┘  └──────┬────────┘  └───────────────┘
            │                  │
    ┌───────▼────────┐  ┌──────▼────────┐
    │  LLMEncoder    │  │  LLMDecoder   │  (NEW)
    └───────┬────────┘  └──────┬────────┘
            │                  │
            └──────────┬───────┘
                       │
            ┌──────────▼────────────┐
            │ EncoderDecoderModel   │  (NEW)
            └──────────┬────────────┘
                       │
            ┌──────────▼────────────┐
            │   TextGenerator       │  (NEW)
            └───────────────────────┘

Legend:
  Existing components: No marker
  New components: (NEW)
  Dependencies: ▼ arrows
```

---

## 6. Training vs Inference Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Teacher Forcing)                    │
└─────────────────────────────────────────────────────────────────┘

Input:  "What is the weather?"
Target: "It is sunny today"

                        Encoder
                           │
                           ▼
                    Encoder Output
                           │
    ┌──────────────────────┼──────────────────────┐
    │                                              │
Decoder Input: [<BOS>, "It", "is", "sunny"]       │
    │ (target shifted right)                       │
    ▼                                              │
Decoder Forward Pass (parallel) ───────────────────┤
    │                                              │
    ▼                                              │
Logits: [4 × vocab_size]                           │
    │                                              │
    ▼                                              │
Loss = CrossEntropy(logits, ["It", "is", "sunny", "today"])
    │
    ▼
Backward Pass → Update Weights

Characteristics:
✓ Parallel processing (all tokens at once)
✓ Faster training
✓ Uses ground truth tokens
✗ Different from inference


┌─────────────────────────────────────────────────────────────────┐
│                  INFERENCE (Autoregressive)                      │
└─────────────────────────────────────────────────────────────────┘

Input:  "What is the weather?"

                        Encoder
                           │
                           ▼
                    Encoder Output (cached)
                           │
    ┌──────────────────────┼──────────────────────┐
    │                                              │
Step 1:                                            │
Decoder([<BOS>]) ──────────────────────────────────┤
    │                                              │
    ▼                                              │
Sample → "It"                                      │
                                                   │
Step 2:                                            │
Decoder([<BOS>, "It"]) ────────────────────────────┤
    │                                              │
    ▼                                              │
Sample → "is"                                      │
                                                   │
Step 3:                                            │
Decoder([<BOS>, "It", "is"]) ──────────────────────┤
    │                                              │
    ▼                                              │
Sample → "sunny"                                   │
                                                   │
... continue until <EOS>

Characteristics:
✗ Sequential processing (one token at a time)
✗ Slower generation
✓ Uses model's own predictions
✓ Matches real-world usage
```

---

## 7. Memory Layout During Generation

```
GENERATION STATE EVOLUTION

Initial State:
┌──────────────────────────────────────────┐
│ Encoder Output: [6 × 512] (cached)       │
│ Generated Tokens: [<BOS>]                │
│ Decoder State: Ready                     │
└──────────────────────────────────────────┘

After Token 1:
┌──────────────────────────────────────────┐
│ Encoder Output: [6 × 512] (cached)       │
│ Generated Tokens: [<BOS>, 234]           │
│ Decoder Cache: [1 × 512] embeddings      │
└──────────────────────────────────────────┘

After Token 2:
┌──────────────────────────────────────────┐
│ Encoder Output: [6 × 512] (cached)       │
│ Generated Tokens: [<BOS>, 234, 456]      │
│ Decoder Cache: [2 × 512] embeddings      │
└──────────────────────────────────────────┘

After Token N:
┌──────────────────────────────────────────┐
│ Encoder Output: [6 × 512] (cached)       │
│ Generated Tokens: [<BOS>, ..., <EOS>]    │
│ Decoder Cache: [N × 512] embeddings      │
└──────────────────────────────────────────┘

Memory Growth: O(N × d_model) per sequence
```

---

## 8. Class Hierarchy and Relationships

```
┌────────────────────────────────────────────────────────────────┐
│                        CLASS DIAGRAM                            │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│   EncoderDecoderModel       │
│─────────────────────────────│
│ - encoder: LLMEncoder*      │
│ - decoder: LLMDecoder*      │◄─────────┐
│ - generator: TextGenerator* │          │
│─────────────────────────────│          │
│ + generate_response()       │          │
│ + train_step()              │          │
│ + train_batch()             │          │
└─────────────────────────────┘          │
                                         │
┌─────────────────────────────┐          │
│      TextGenerator          │          │
│─────────────────────────────│          │
│ - encoder: LLMEncoder*      │──────────┤
│ - decoder: LLMDecoder*      │──────────┤
│─────────────────────────────│          │
│ + generate_greedy()         │          │
│ + generate_sampling()       │          │
│ + generate_beam_search()    │          │
└─────────────────────────────┘          │
                                         │
┌─────────────────────────────┐          │
│       LLMDecoder            │◄─────────┘
│─────────────────────────────│
│ - tokenizer                 │
│ - token_embedding           │
│ - positional_encoding       │
│ - decoder_blocks[]          │──────────┐
│ - final_norm                │          │
│ - lm_head                   │──────┐   │
│─────────────────────────────│      │   │
│ + forward()                 │      │   │
│ + backward()                │      │   │
│ + generate_next_token()     │      │   │
└─────────────────────────────┘      │   │
                                     │   │
┌─────────────────────────────┐      │   │
│    LanguageModelHead        │◄─────┘   │
│─────────────────────────────│          │
│ - W_output: Matrix          │          │
│ - bias: Matrix              │          │
│─────────────────────────────│          │
│ + forward()                 │          │
│ + backward()                │          │
│ + get_probabilities()       │          │
└─────────────────────────────┘          │
                                         │
┌─────────────────────────────┐          │
│      DecoderBlock           │◄─────────┘
│─────────────────────────────│
│ - self_attention            │──────┐
│ - cross_attention           │──────┤
│ - feed_forward              │──────┤
│ - norm1, norm2, norm3       │──────┤
│─────────────────────────────│      │
│ + forward()                 │      │
│ + backward()                │      │
└─────────────────────────────┘      │
                                     │
    ┌────────────────────────────────┴───────────────┐
    │                                                 │
    ▼                                                 ▼
┌───────────────────┐                    ┌──────────────────┐
│ MultiHeadAttn     │                    │  FeedForward     │
│ (existing)        │                    │  (existing)      │
└───────────────────┘                    └──────────────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** January 18, 2026  
**Related Documents:** DECODER_DESIGN.md, DECODER_DESIGN_SUMMARY.md
