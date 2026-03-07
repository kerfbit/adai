# Chatbot GUI - Quick Start with Parallel Processing

---

## REBUILD GUI WITH PARALLEL PROCESSING

Quick Rebuild:

```bash
./scripts/build_and_vocab.sh build-gui
```

Verify Parallel Support:

```bash
./scripts/build_and_vocab.sh verify-gui
```

Manual Rebuild:

```bash
cd build && cmake .. && make chatbot_gui_binary chatbot_gui -j$(nproc)
```

---

## RUN GUI

Standard Run:

```bash
./build/src/chatbot_gui --vocab vocab.txt --model chatbot_model.bin
```

With Convenience Script:

```bash
./scripts/run_chatbot_gui.sh
```

Maximum Performance (Set OpenMP Threads):

```bash
export OMP_NUM_THREADS=8
./build/src/chatbot_gui --vocab vocab.txt --model chatbot_model.bin
```

With Specific Model Epoch:

```bash
./build/src/chatbot_gui --vocab vocab.txt \
                         --model chatbot_model.bin.epoch10
```

---

## PARALLEL PROCESSING STATUS

### Enabled Features

- ✅ Priority 1: OpenMP matrix operations (adai_core)
- ✅ Priority 4: Parallel attention heads (adai_attention)
- ✅ Multi-core CPU utilization (8 cores available)
- ✅ Vectorized operations with SIMD
- ✅ Optimized memory access patterns

### Build Configuration

- **Build Type:** Release (-O3 -march=native)
- **OpenMP Flags:** -fopenmp
- **Binary Size:** 881K
- **Linked OpenMP:** ✅ libgomp.so.1

---

## PERFORMANCE MONITORING

Watch CPU Usage:

```bash
# Run GUI, then in another terminal:
htop
# Look for 60-100% usage across multiple cores during generation
```

Detailed Thread View:

```bash
top -H -p $(pgrep chatbot_gui)
```

Check OpenMP Threads:

```bash
# See how many threads are configured
echo $OMP_NUM_THREADS
# If empty, it uses all available cores by default
```

---

## GENERATION SETTINGS IN GUI

Best Settings for Quality:

- Strategy: Nucleus (top-p sampling)
- Temperature: 0.7 - 0.8
- Max Length: 100 - 150 tokens

For Faster Response:

- Strategy: Greedy
- Temperature: 0.3 - 0.5
- Max Length: 50 - 100 tokens

For Creative Output:

- Strategy: Sampling
- Temperature: 1.0 - 1.2
- Max Length: 150 - 200 tokens

---

## TROUBLESHOOTING

### GUI Won't Start?

- ✓ Check if vocab and model files exist
- ✓ Try: `./scripts/run_chatbot_gui.sh` (uses wrapper to fix library paths)
- ✓ Verify Qt5 is installed: `apt list --installed | grep qt5`

### Slow Generation?

- ✓ Verify OpenMP is linked: `ldd build/src/chatbot_gui_binary | grep gomp`
- ✓ Set threads: `export OMP_NUM_THREADS=8`
- ✓ Use greedy strategy for faster responses
- ✓ Reduce max length

### Model Generates Gibberish?

- ✓ Train for more epochs (20-50)
- ✓ Lower temperature (0.3-0.7)
- ✓ Use greedy strategy initially
- ✓ Verify training loss decreased

### GUI Freezes During Generation?

- ✓ This is expected - generation is compute-intensive
- ✓ Parallel processing helps but UI still blocks
- ✓ Check CPU usage - should be high during generation

---

## EXPECTED PERFORMANCE

- **Matrix Operations:** Up to 8x faster with 8 cores
- **Attention Computation:** Parallelized across attention heads
- **Overall Inference:** 2-4x faster than sequential version
- **CPU Utilization:** 60-100% across all cores during generation

---

## COMPARISON WITH CLI

|Feature|CLI Chatbot|GUI Chatbot|
|-------|-----------|-----------|
|OpenMP Support|✅|✅|
|Parallel Attention|✅|✅|
|Multi-core Usage|✅|✅|
|Optimization Level|Release|Release|
|Build Flags|-O3 -march|-O3 -march|

Both now have IDENTICAL parallel processing capabilities!

---

## FILES AND DOCUMENTATION

### Executables

- `build/src/chatbot_gui_binary` - Main GUI executable
- `build/src/chatbot_gui` - Wrapper (fixes library paths)

### Scripts

- `scripts/run_chatbot_gui.sh` - Convenience launcher
- `scripts/build_and_vocab.sh` - Build automation
- `scripts/verify_gui_parallel.sh` - Verification tool

### Documentation

- [CHATBOT_GUI_PARALLEL_REBUILD.md](../../../development/archive/CHATBOT_GUI_PARALLEL_REBUILD.md) - Technical details
- [CHATBOT_GUI_PARALLEL_COMPLETE_SUMMARY.md](../../../development/archive/CHATBOT_GUI_PARALLEL_COMPLETE_SUMMARY.md) - Complete summary
- [GUI_QUICK_START.md](GUI_QUICK_START.md) - This file

---

> **🚀 Your GUI chatbot now runs with full parallel processing!**
