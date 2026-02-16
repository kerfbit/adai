# Development Documentation

This folder contains all **coding and development-related documentation** for the ADAI project.

## Contents

### 📁 [api/](api/)

API documentation for all core components, including:

- Core classes (Matrix, Neuron, NeuralNetwork, Optimizer)
- Transformer components (Encoder, Decoder, Attention mechanisms)
- NLP components (Tokenizer, TextGenerator)
- Data processing utilities
- Advanced features (Quantization, Speculative Decoding, PPO)

### 📁 [architecture/](architecture/)

System architecture and design documentation:

- Decoder architecture and design
- Encoder-decoder comparison
- Neural network architecture
- Transformer design patterns

### 📁 [testing/](testing/)

Testing documentation and test specifications:

- BPE tokenizer tests
- Chatbot CLI tests
- Chatbot trainer tests
- Encoder-decoder tests
- Neural network tests
- Neuron layer tests
- Optimizer integration tests
- Test coverage reports

### 📁 [reference/](reference/)

Technical reference documentation:

- Gradient operations reference
- Batch processor internals
- KV cache implementation
- Performance profiler details
- Vocabulary training analysis
- Component completeness tracking

### 📁 [guides/](guides/)

Developer guides and implementation documentation:

- **Building & CI/CD**: building.md, ci-cd.md, windows-cross-compilation.md
- **Git Workflow**: git-workflow.md, branch-protection.md, contributing.md
- **Implementation Guides**: implementation-guide.md, implementation-checklist.md
- **Internal Architecture**: chatbot-cli-internals.md, training-internals.md
- **Feature Implementation**:
  - Augmentation: AUGMENTATION_IMPLEMENTATION.md
  - Batch Processing: BATCH_PROCESSING_INTEGRATION.md
  - OpenMP: OPENMP_IMPLEMENTATION.md
  - Neuron Layer: NEURON_LAYER_IMPLEMENTATION.md
  - RAG: RAG_IMPLEMENTATION_GUIDE.md
- **Data Pipeline**: data-pipeline-enhancement.md, dataset-enhanced-features.md, enhanced-training-pipeline.md
- **Technical Debt**: technical-debt-management.md, TECHNICAL_DEBT.md
- **Advanced Features**: phase5-advanced-features.md
- **Model Management**: save-load.md
- **Process Improvement**: PROCESS_IMPROVEMENT_PLAN.md
- **Trainer Development**: chatbot-trainer-improvements-2026.md, chatbot-trainer-metrics-logging.md

### 📁 [archive/](archive/)

Historical documentation and completed phase summaries for reference.

## Target Audience

This documentation is intended for:

- **Developers** contributing to the codebase
- **Architects** designing new features
- **Code reviewers** understanding implementation details
- **CI/CD engineers** managing build and test infrastructure

## Related Documentation

For user-facing and operational documentation, see [../operations/](../operations/)
