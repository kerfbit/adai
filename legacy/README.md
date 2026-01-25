# Legacy Components - Dense Neural Networks

**Last Updated:** January 24, 2026

## Purpose

This directory contains archived implementations of traditional dense (fully-connected) neural networks that were part of the ADAI project's early development but are no longer part of the active codebase.

## Directory Structure

```
legacy/
└── dense_networks/
    ├── README.md           # Detailed documentation of archived components
    ├── src/                # Original source files
    │   ├── Neuron.hpp
    │   ├── Neuron.cpp
    │   ├── NeuralNetwork.hpp
    │   └── NeuralNetwork.cpp
    ├── tests/              # Original test files
    │   ├── neuron_test.cpp
    │   ├── neuronlayer_test.cpp
    │   └── neuralnetwork_test.cpp
    └── examples/           # Original example files
        ├── NeuronLayerExample.cpp
        └── NeuralNetworkExample.cpp
```

## Why Archived?

The ADAI project evolved to focus exclusively on transformer-based architectures for natural language processing. The dense neural network components:

1. **Were architecturally isolated** - No integration with transformer components
2. **Represented a different paradigm** - Per-neuron abstraction vs matrix operations
3. **Would require significant maintenance** - To align with the Optimizer refactoring pattern
4. **Did not align with project direction** - Transformers are the core focus

See `dense_networks/README.md` for complete analysis and reasoning.

## Status

- ✅ **Working Code** - All archived files were functional at archive time
- ✅ **Full Test Coverage** - 82 tests, all passing
- ✅ **Documented** - Examples and usage patterns preserved
- ⚠️ **Not Built** - Removed from CMakeLists.txt
- ⚠️ **Not Maintained** - No updates planned

## Accessing Archived Code

These files are preserved for:
- **Reference** - Understanding traditional neural network implementations
- **Education** - Learning basic neural network concepts
- **Comparison** - Contrasting dense networks with transformers
- **Recovery** - Can be restored if needed (though not recommended)

To view the code, browse the subdirectories. To use it, you would need to manually copy files back to `src/` and update build configuration.

## Active Development

For current ADAI development, use the transformer components in `src/`:
- MultiHeadAttention
- CrossAttention
- FeedForward
- LayerNorm
- TokenEmbedding
- LanguageModelHead
- Encoder/Decoder architectures

All active components use the unified Optimizer class and follow modern transformer architecture patterns.

## Git History

Full development history of these components is preserved in git. Use:
```bash
git log --follow legacy/dense_networks/src/Neuron.hpp
```

---

**Note:** This is an archive directory. Do not add new code here. Use for reference only.
