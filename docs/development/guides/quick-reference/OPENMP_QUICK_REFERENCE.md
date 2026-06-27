# OpenMP Quick Reference

## ⚡ Quick Start

```bash
# Install OpenMP
sudo apt-get install libomp-dev

# Build
cd build && cmake .. && make -j$(nproc)

# Test
./src/openmp_benchmark 512
```

## 🎯 Usage Examples

### Automatic (Recommended)

```cpp
// Just use Matrix operations normally
Matrix A(1024, 1024), B(1024, 1024);
Matrix C = A * B;  // Automatically parallelized!
```

### Control Threads

```bash
# Environment variable (best)
export OMP_NUM_THREADS=8

# At runtime
#ifdef ADAI_ENABLE_OPENMP
omp_set_num_threads(8);
#endif
```

## 📊 Expected Performance

|Matrix Size|Sequential|Parallel (8 cores)|Speedup|
|-------------|------------|-------------------|---------|
|256×256|159 ms|41 ms|3.9x|
|512×512|1391 ms|331 ms|4.2x|
|1024×1024|~11000 ms|~2600 ms|4.2x|

## ⚙️ Configuration

### Optimal Settings

```bash
# Use physical cores (not hyperthreads)
export OMP_NUM_THREADS=$(nproc --all)

# Pin threads to cores
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

### For Shared Systems

```bash
# Reduce thread count to share resources
export OMP_NUM_THREADS=4
```

## ✅ Verify Installation

```bash
# Check OpenMP in build
cmake .. 2>&1 | grep "OpenMP found"

# Run benchmark
./src/openmp_benchmark 256

# Should show: ✓ OpenMP ENABLED
```

## 🐛 Common Issues

### "OpenMP not found"

```bash
sudo apt-get install libomp-dev
cmake .. && make
```

### "Benchmark shows NOT ENABLED"

```bash
rm -rf build && mkdir build && cd build
cmake .. && make
```

### "No speedup"

- Use larger matrices (>256)
- Check: `htop` - verify threads running
- Set: `export OMP_NUM_THREADS=8`

## 📈 Parallelized Operations

|Operation|Threshold|Expected Speedup|
|-----------|-----------|------------------|
|Matrix Multiplication|>64×64|4-8x|
|Addition/Subtraction|>10k elements|4-6x|
|Transpose|>10k elements|3-5x|
|Hadamard Product|>10k elements|4-6x|
|Scalar Operations|>10k elements|4-6x|
|Gradient Update|>10k elements|4-7x|

## 🔍 Performance Tips

1. **Larger matrices = better speedup** (less overhead)
2. **Use physical cores** (not hyperthreads)
3. **Close background apps** during benchmarking
4. **Batch operations** when possible

## 📝 Status Check

```bash
# Check if OpenMP is working
./src/openmp_benchmark 512 | head -10

# Expected output:
# ✓ OpenMP ENABLED
# Max Threads Available: 8
```

## 📚 Documentation

- Full Guide: [OPENMP_IMPLEMENTATION.md](OPENMP_IMPLEMENTATION.md)
- Summary: [OPENMP_PRIORITY1_SUMMARY.md](OPENMP_PRIORITY1_SUMMARY.md)
- Analysis: [PARALLEL_PROCESSING_ANALYSIS_REPORT.md](PARALLEL_PROCESSING_ANALYSIS_REPORT.md)

---

**Version:** 1.0
**Status:** Production Ready ✅
