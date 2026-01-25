# ADAI Reference Documentation

This directory contains reference materials, technical specifications, and implementation details for the ADAI project.

## 📖 Contents

### Core References

- **[KVCache API Reference](kvcache.md)** - Complete API documentation for the Key-Value cache system
  - Single-layer and multi-layer caching
  - Usage patterns and best practices
  - Performance characteristics
  - Integration examples
  - Troubleshooting guide

- **[BatchProcessor API Reference](batchprocessor.md)** - Complete API documentation for batch processing utilities
  - TokenBatch and BatchStats structures
  - Dynamic batching by sequence length
  - Padding and masking utilities
  - Performance optimization strategies
  - Real-world usage patterns
  - Troubleshooting guide

- **[PerformanceProfiler API Reference](performanceprofiler.md)** - Complete API documentation for profiling tools
  - Timer, ScopedTimer, and Profiler classes
  - Statistical analysis (mean, median, percentiles)
  - Automated benchmarking
  - Before/after optimization comparison
  - Production monitoring patterns
  - Troubleshooting guide

### Project Planning

- **[Chatbot Completeness](chatbot-completeness.md)** - Feature completeness tracking and roadmap
  - Implementation status
  - Phase-by-phase development plan
  - Feature requirements

### Technical Documentation

- **[Gradient Operations Without Optimizer](GRADIENT_OPERATIONS_WITHOUT_OPTIMIZER.md)** - Manual gradient computation reference
  - Low-level gradient calculations
  - Implementation details

## 🔗 Related Documentation

### For API Usage
- See [API Reference](../api/README.md) for component APIs
- See [Guides](../guides/README.md) for usage tutorials

### For Performance Optimization
- [Inference Optimization Guide](../guides/inference-optimization.md) - Complete optimization guide
- [Quick Start](../guides/inference-optimization-quickstart.md) - 5-minute tutorial
- [KVCache API](kvcache.md) - Detailed cache API reference

### For Development
- [Architecture Documentation](../architecture/) - System design and patterns
- [Testing Documentation](../testing/) - Test suites and validation

## 📝 Adding New Reference Docs

When adding new reference documentation:

1. **Create the document** in this directory
2. **Update this README** with a link and brief description
3. **Update the main docs README** (`docs/README.md`) if it's a major reference
4. **Cross-link** from related guides and API docs

### Document Types

This directory is for:
- ✅ API reference documentation
- ✅ Technical specifications
- ✅ Implementation details
- ✅ Performance characteristics
- ✅ Project planning documents

This directory is NOT for:
- ❌ User guides (use `guides/` instead)
- ❌ Architecture overviews (use `architecture/` instead)
- ❌ Test documentation (use `testing/` instead)
- ❌ API component docs (use `api/` instead)

---

**Last Updated:** January 25, 2026
