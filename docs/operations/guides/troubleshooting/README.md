# ADAI Troubleshooting Guide

This directory contains solutions and fixes for common issues encountered during ADAI development and deployment.

## 🔧 Common Issues & Fixes

### Build & Compilation Issues

- **[CPP Wrapper Solution](CPP_WRAPPER_SOLUTION.md)** - Resolving C++ wrapper compilation issues
  - Template instantiation problems
  - Linking errors
  - Header inclusion fixes

### Model Issues

- **[Model Loading Fix](MODEL_LOADING_FIX.md)** - Troubleshooting model loading problems
  - Binary format compatibility
  - File corruption detection
  - Path resolution issues

- **[Input Length Fix](INPUT_LENGTH_FIX.md)** - Fixing input sequence length issues
  - Max length validation
  - Padding strategies
  - Memory allocation fixes

- **[Fixing UNK Generation](FIXING_UNK_GENERATION.md)** - Resolving `<unk>` token generation after vocabulary repair
  - Vocabulary/model mismatch diagnosis
  - Recompile and retrain steps
  - Verification checklist

- **[Special Token Issues](SPECIAL_TOKEN_ISSUES.md)** - Diagnosing and fixing special token ID mismatches
  - Incorrect BOS/EOS token IDs
  - Tokenizer/model synchronization
  - Training vs. inference inconsistencies

### Runtime Issues

- **[Thread Error Fix](THREAD_ERROR_FIX.md)** - Resolving threading and concurrency issues
  - OpenMP thread management
  - Race condition fixes
  - Deadlock prevention

### Training Issues

- **[Training Fix Strategy](TRAINING_FIX_STRATEGY.md)** - Comprehensive training troubleshooting
  - Loss divergence solutions
  - Gradient issues
  - Learning rate tuning
  - Checkpoint recovery

### GUI Issues

- **[Chatbot GUI Troubleshooting](CHATBOT_GUI_TROUBLESHOOTING.md)**

  - GUI-specific problems
  - Display issues
  - Event handling bugs
  - Performance optimization
  - Cross-platform compatibility

## 🚨 Quick Diagnostic Steps

### Step 1: Check Build Configuration

```bash
# Verify build was successful
cmake --build build --config Release
./build/tests/run_all_tests
```

### Step 2: Validate Model Files

```bash
# Check model file integrity
ls -lh chatbot_model.bin*
# Verify vocab exists
cat vocab.txt | wc -l
```

### Step 3: Test with Minimal Configuration

```bash
# Run with verbose output
./build/chatbot_cli --verbose
```

### Step 4: Check System Resources

```bash
# Memory usage
free -h
# Thread limits
ulimit -u
# OpenMP settings
echo $OMP_NUM_THREADS
```

## 📋 Reporting Issues

If you encounter an issue not covered here:

1. **Check existing documentation**:
   - [Building Guide](../../../development/guides/building/building.md)
   - [Training Guide](../training-guide.md)
   - [API Documentation](../../../development/api/README.md)

2. **Gather diagnostic information**:
   - OS and compiler version
   - Build configuration (Debug/Release)
   - Error messages and stack traces
   - Steps to reproduce

3. **Search closed issues** on GitHub

4. **Create a new issue** with diagnostic info

## 🔍 Related Documentation

- **[Operations Documentation](../../README.md)** - Operations docs home
- **[Building ADAI](../../../development/guides/building/building.md)** - Build instructions
- **[Contributing Guide](../../../development/guides/workflow/contributing.md)** - Development guidelines
- **[Technical Debt](../../../development/guides/TECHNICAL_DEBT.md)** - Known limitations

## 💡 Prevention Tips

### Before Building

- Ensure all dependencies are installed
- Use recommended compiler versions
- Check CMake version compatibility

### Before Training

- Validate training data format
- Set appropriate batch sizes
- Monitor memory usage

### Before Deployment

- Test thoroughly in staging
- Review performance profiles
- Document configuration changes

## 🆘 Getting Help

- **Documentation**: Start with [Operations Documentation](../../README.md)
- **GitHub Issues**: Search/create issues
- **Contributing**: See [Contributing Guide](../../../development/guides/workflow/contributing.md)
