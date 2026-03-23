# Matrix Class - Technical Context Documentation

## Overview

The `Matrix` class is a fundamental component for tensor operations in the neural network framework. It provides essential matrix operations needed for forward propagation, backpropagation, and gradient-based optimization in deep learning applications.

Files:

- `src/Matrix.hpp` - Header file with class declaration and interface
- `src/Matrix.cpp` - Implementation file with all method definitions

**Purpose:** Serve as the core mathematical abstraction for representing and manipulating multi-dimensional numerical data in neural networks, including weight matrices, activations, gradients, and embeddings.

---

## Class Structure

### Public Members

```cpp
std::vector<std::vector<float>> data;  // 2D storage for matrix elements
int rows;                               // Number of rows
int cols;                               // Number of columns
```

### Memory Layout

- **Row-major order**: Elements stored as `data[row][col]`
- **Contiguous rows**: Each row is a contiguous `std::vector<float>`
- **Dynamic allocation**: Automatically managed by `std::vector`

---

## Constructors

### 1. Default Constructor

```cpp
Matrix()
```

Creates an empty matrix with `rows = 0` and `cols = 0`.

**Use case:** Default initialization, container storage

### 2. Dimensional Constructor

```cpp
Matrix(int r, int c)
```

Creates a zero-initialized matrix with specified dimensions.

Parameters:

- `r` - Number of rows
- `c` - Number of columns

**Initialization:** All elements set to `0.0f`

**Use case:** Creating weight matrices, gradient buffers, activation tensors

Example:

```cpp
Matrix weights(256, 512);  // 256x512 zero matrix for layer weights
Matrix gradients(10, 10);  // 10x10 gradient accumulator
```

### 3. Data Constructor

```cpp
Matrix(const std::vector<std::vector<float>>& d)
```

Creates a matrix from existing 2D vector data.

**Validation:** Ensures all rows have the same number of columns

**Exception:** `std::invalid_argument` if rows have inconsistent sizes

**Use case:** Loading pre-initialized data, testing, constant matrices

Example:

```cpp
Matrix identity({{1, 0, 0},
                 {0, 1, 0},
                 {0, 0, 1}});
```

---

## Core Operations

### Element Access

```cpp
float& operator()(int i, int j)              // Modifiable access
const float& operator()(int i, int j) const  // Read-only access
```

**Bounds Checking:** Throws `std::out_of_range` for invalid indices

**Complexity:** O(1)

Example:

```cpp
Matrix A(3, 3);
A(0, 0) = 1.5;           // Set element
float val = A(1, 2);     // Get element
```

### Matrix Multiplication

```cpp
Matrix operator*(const Matrix& other) const
```

**Mathematical Operation:** Standard matrix multiplication (dot product)

**Dimensions:** `[m × n] * [n × p] = [m × p]`

**Validation:** Throws `std::invalid_argument` if `this->cols != other.rows`

**Algorithm:** Triple nested loop with O(m × n × p) complexity

Example:

```cpp
Matrix W(512, 256);      // Weights
Matrix X(100, 512);      // Batch of 100 inputs
Matrix Y = X * W;        // Result: 100x256
```

Formula:

```text
Y[i][j] = Σ(k=0 to n-1) A[i][k] * B[k][j]
```

### Matrix Addition

```cpp
Matrix operator+(const Matrix& other) const
```

**Mathematical Operation:** Element-wise addition

**Validation:** Requires matching dimensions

**Complexity:** O(rows × cols)

**Use case:** Residual connections, bias addition

Example:

```cpp
Matrix A(5, 5);
Matrix B(5, 5);
Matrix C = A + B;  // Element-wise sum
```

### Matrix Subtraction

```cpp
Matrix operator-(const Matrix& other) const
```

**Mathematical Operation:** Element-wise subtraction

**Validation:** Requires matching dimensions

**Complexity:** O(rows × cols)

**Use case:** Computing gradients, error terms

Example:

```cpp
Matrix predicted(10, 1);
Matrix target(10, 1);
Matrix error = predicted - target;
```

### Transpose

```cpp
Matrix transpose() const
```

**Mathematical Operation:** Swaps rows and columns

**Dimensions:** `[m × n]` → `[n × m]`

**Complexity:** O(rows × cols)

**Use case:** Weight matrix transpose for backprop, attention computation

Example:

```cpp
Matrix W(256, 512);
Matrix W_T = W.transpose();  // 512x256
```

Formula:

```text
B[j][i] = A[i][j]  for all i, j
```

---

## Specialized Operations

---

## GPU-Accelerated Operations

When built with GPU support (`ENABLE_GPU=ON`), the `Matrix` class provides GPU-accelerated versions of core operations. These leverage CUDA and cuBLAS for significant speedups on large matrices.

### Enabling GPU Support

- **Build:**

  ```bash
  cmake -DENABLE_GPU=ON ..
  make
  ```

- **Requirements:** CUDA Toolkit 11.0+, compatible NVIDIA GPU, drivers

### Initialization and Device Management

```cpp
// Initialize GPU subsystem (call once at startup)
Matrix::gpu_initialize();

// Check if GPU is available
if (Matrix::gpu_available()) {
    std::cout << Matrix::gpu_info() << std::endl;
}

// Cleanup before exit
Matrix::gpu_cleanup();
```

### GPU Methods (API)

All GPU methods throw if the GPU is not initialized or available.

```cpp
Matrix multiply_gpu(const Matrix& other) const;   // Matrix multiplication (cuBLAS)
Matrix add_gpu(const Matrix& other) const;        // Element-wise addition
Matrix hadamard_gpu(const Matrix& other) const;   // Element-wise multiplication
Matrix transpose_gpu() const;                     // Matrix transpose
Matrix scale_gpu(float scalar) const;             // Scalar multiplication
```

#### Example Usage

```cpp
#include "Matrix.hpp"

int main() {
    Matrix::gpu_initialize();
    if (Matrix::gpu_available()) {
        Matrix A(1000, 1000);
        Matrix B(1000, 1000);
        A.randomize();
        B.randomize();

        Matrix C = A.multiply_gpu(B);      // Fast matrix multiplication
        Matrix D = A.add_gpu(B);           // Fast element-wise addition
        Matrix E = A.transpose_gpu();      // Fast transpose
        Matrix F = A.scale_gpu(2.5f);      // Fast scalar multiplication
        Matrix G = A.hadamard_gpu(B);      // Fast element-wise multiply
    }
    Matrix::gpu_cleanup();
    return 0;
}
```

#### Performance Notes

- GPU operations are most beneficial for large matrices (e.g., 500x500 or larger).
- Data is transferred between CPU and GPU memory for each operation; batching operations can improve throughput.
- Underlying implementation uses CUDA kernels and cuBLAS for optimal performance.

#### Supported Operations

| Operation                | Method                | Backend      |
|--------------------------|-----------------------|--------------|
| Matrix multiplication    | multiply_gpu()        | cuBLAS       |
| Element-wise addition    | add_gpu()             | CUDA kernel  |
| Element-wise multiply    | hadamard_gpu()        | CUDA kernel  |
| Scalar multiplication    | scale_gpu()           | CUDA kernel  |
| Transpose                | transpose_gpu()       | CUDA kernel  |

#### Error Handling (GPU operations)

- Throws `std::runtime_error` if GPU is not initialized or available.
- Throws `std::invalid_argument` for dimension mismatches.

#### Integration

- GPU operations are drop-in replacements for their CPU counterparts.
- See `examples/GPUExample.cpp` for benchmarks and usage patterns.

#### Implementation Files

- `src/gpu/MatrixGPU.hpp`, `src/gpu/MatrixGPU.cu` — CUDA kernels and API
- `src/gpu/GPUUtils.hpp` — Device/memory management
- `src/Matrix.hpp`, `src/Matrix.cpp` — Matrix class integration

---

### Hadamard Product (Element-wise Multiplication)

```cpp
Matrix hadamard(const Matrix& other) const
```

**Mathematical Operation:** Element-wise multiplication (⊙)

**Validation:** Requires matching dimensions

**Complexity:** O(rows × cols)

**Use case:** Activation derivatives, masking, gating mechanisms

Example:

```cpp
Matrix activations(100, 256);
Matrix derivatives(100, 256);
Matrix grad = activations.hadamard(derivatives);
```

Formula:

```text
C[i][j] = A[i][j] * B[i][j]  for all i, j
```

### Scalar Multiplication

```cpp
Matrix scale(float scalar) const
```

**Mathematical Operation:** Multiply all elements by a scalar

**Complexity:** O(rows × cols)

**Use case:** Learning rate scaling, normalization constants

Example:

```cpp
Matrix scores(10, 10);
float scale_factor = 1.0 / sqrt(64);  // Attention scaling
Matrix scaled = scores.scale(scale_factor);
```

Formula:

```text
B[i][j] = scalar * A[i][j]  for all i, j
```

---

## Training Operations

### Apply Gradients

```cpp
void apply_gradients(const Matrix& gradients, float learning_rate)
```

**Mathematical Operation:** Gradient descent weight update

**In-place:** Modifies the matrix directly

**Validation:** Requires matching dimensions

**Complexity:** O(rows × cols)

**Use case:** Primary mechanism for updating neural network weights

Example:

```cpp
Matrix weights(512, 256);
Matrix weight_gradients(512, 256);
float lr = 0.001f;

weights.apply_gradients(weight_gradients, lr);
```

Formula:

```text
W[i][j] = W[i][j] - learning_rate * grad[i][j]  for all i, j
```

Gradient Descent Variants:

- **SGD:** Direct application as shown
- **Momentum:** Requires velocity buffer (external)
- **Adam:** Requires momentum and velocity buffers (external)

### Randomization

```cpp
void randomize(float scale = 0.1f)
```

**Mathematical Operation:** Xavier/He initialization with Gaussian distribution

**Distribution:** N(0, scale²)

**In-place:** Modifies the matrix directly

**Seed:** Uses `std::random_device` for non-deterministic seed

**Use case:** Weight initialization before training

Example:

```cpp
Matrix W(256, 512);
float scale = sqrt(2.0f / 256);  // He initialization
W.randomize(scale);
```

Initialization Strategies:

- **Xavier:** `scale = sqrt(2.0 / (fan_in + fan_out))`
- **He:** `scale = sqrt(2.0 / fan_in)`
- **Uniform:** Custom implementation needed

---

## Utility Operations

### Fill

```cpp
void fill(float value)
```

**In-place:** Sets all elements to specified value

**Complexity:** O(rows × cols)

**Use case:** Zeroing gradients, initializing bias, creating constant matrices

Example:

```cpp
Matrix gradients(256, 512);
gradients.fill(0.0f);  // Zero all gradients
```

### Sum

```cpp
float sum() const
```

**Returns:** Sum of all matrix elements

**Complexity:** O(rows × cols)

**Use case:** Computing total loss, gradient magnitude

Example:

```cpp
Matrix loss(10, 1);
float total_loss = loss.sum();
```

### Mean

```cpp
float mean() const
```

**Returns:** Average of all matrix elements

**Complexity:** O(rows × cols)

**Use case:** Statistics, normalization

Example:

```cpp
Matrix activations(100, 256);
float avg_activation = activations.mean();
```

Formula:

```text
mean = sum(all elements) / (rows * cols)
```

### Validation

```cpp
bool is_valid() const
```

Checks:

1. Dimensions are positive
2. Data vector size matches `rows`
3. All row vectors have size `cols`

**Returns:** `true` if matrix is structurally valid

**Use case:** Debugging, assertion checks

---

## Advanced Utilities

### Reshape

```cpp
Matrix reshape(int new_rows, int new_cols) const
```

**Constraint:** `rows * cols == new_rows * new_cols`

**Order:** Row-major flattening and reformation

**Complexity:** O(rows × cols)

**Use case:** Changing tensor dimensions while preserving data

Example:

```cpp
Matrix A(2, 6);     // 2x6 matrix
Matrix B = A.reshape(3, 4);  // Reshape to 3x4
```

Process:

1. Flatten to 1D: `[a₀₀, a₀₁, ..., a₁₅]`
2. Reform to new shape: 12 elements → 3 rows × 4 cols

### Row Operations

```cpp
std::vector<float> get_row(int row_idx) const
void set_row(int row_idx, const std::vector<float>& values)
```

**Bounds Checking:** Throws `std::out_of_range` for invalid index

**Use case:** Row-wise operations, extracting embeddings

Example:

```cpp
Matrix embeddings(1000, 256);
std::vector<float> word_vec = embeddings.get_row(42);

Matrix batch(10, 256);
batch.set_row(0, word_vec);
```

### Column Operations

```cpp
std::vector<float> get_col(int col_idx) const
void set_col(int col_idx, const std::vector<float>& values)
```

**Bounds Checking:** Throws `std::out_of_range` for invalid index

**Use case:** Feature extraction, dimension-wise statistics

Example:

```cpp
Matrix features(100, 10);
std::vector<float> feature_0 = features.get_col(0);
```

### Print (Debug)

```cpp
void print(const std::string& name = "",
           int max_rows = 10,
           int max_cols = 10) const
```

**Format:** Formatted output with 4 decimal precision

**Truncation:** Shows first `max_rows` × `max_cols` with ellipsis for overflow

**Use case:** Debugging, visualization

Example:

```cpp
Matrix W(512, 256);
W.print("Layer1 Weights", 5, 5);
```

Output:

```text
Layer1 Weights (512x256):
[  0.0234  -0.0156   0.0089  -0.0234   0.0145 ...]
[ -0.0123   0.0267  -0.0089   0.0156  -0.0234 ...]
...
```

---

## Integration Patterns

### 1. Neural Network Layers

```cpp
// Feed-forward layer
class LinearLayer {
    Matrix W;  // Weights [in_features, out_features]
    Matrix b;  // Bias [1, out_features]

    Matrix forward(const Matrix& input) {
        // input: [batch_size, in_features]
        Matrix output = input * W;  // [batch_size, out_features]

        // Broadcast bias addition
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                output(i, j) += b(0, j);
            }
        }
        return output;
    }

    void backward(const Matrix& grad_output, const Matrix& input) {
        // Gradient w.r.t. weights
        Matrix grad_W = input.transpose() * grad_output;

        // Gradient w.r.t. bias
        Matrix grad_b(1, b.cols);
        for (int j = 0; j < grad_output.cols; j++) {
            float sum = 0.0f;
            for (int i = 0; i < grad_output.rows; i++) {
                sum += grad_output(i, j);
            }
            grad_b(0, j) = sum;
        }

        // Update weights
        W.apply_gradients(grad_W, learning_rate);
        b.apply_gradients(grad_b, learning_rate);
    }
};
```

### 2. Attention Mechanism

```cpp
// Scaled dot-product attention
Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V, float d_k) {
    // Q: [seq_len, d_k], K: [seq_len, d_k], V: [seq_len, d_v]

    // Compute attention scores
    Matrix scores = Q * K.transpose();  // [seq_len, seq_len]

    // Scale
    float scale = 1.0f / sqrt(d_k);
    scores = scores.scale(scale);

    // Softmax (implemented separately)
    Matrix attn_weights = softmax(scores);

    // Apply attention to values
    Matrix output = attn_weights * V;  // [seq_len, d_v]

    return output;
}
```

### 3. Batch Processing

```cpp
// Process mini-batch
Matrix batch_input(32, 512);    // 32 samples, 512 features
Matrix weights(512, 256);        // Layer weights
Matrix batch_output = batch_input * weights;  // 32x256 output

// Each row is one sample's output
for (int i = 0; i < 32; i++) {
    std::vector<float> sample_output = batch_output.get_row(i);
    // Process individual sample
}
```

### 4. Gradient Accumulation

```cpp
// Training loop with gradient accumulation
Matrix weight_gradients(256, 512);
weight_gradients.fill(0.0f);

for (int step = 0; step < accumulation_steps; step++) {
    // Forward and backward pass
    Matrix batch_grad = compute_gradients();

    // Accumulate gradients
    weight_gradients = weight_gradients + batch_grad;
}

// Average and apply
float scale = 1.0f / accumulation_steps;
weight_gradients = weight_gradients.scale(scale);
weights.apply_gradients(weight_gradients, learning_rate);
```

---

## Error Handling

### Exception Types

1. **`std::out_of_range`**
   - Element access with invalid indices
   - Row/column access beyond bounds

2. **`std::invalid_argument`**
   - Dimension mismatch in operations
   - Inconsistent row sizes in data constructor
   - Invalid reshape dimensions

### Dimension Checking

All binary operations validate dimensions:

```cpp
Matrix A(10, 20);
Matrix B(15, 20);

try {
    Matrix C = A + B;  // Throws: dimension mismatch
} catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### Safe Access Pattern

```cpp
Matrix M(100, 100);

if (M.is_valid() && i < M.rows && j < M.cols) {
    float value = M(i, j);
} else {
    // Handle invalid access
}
```

---

## Performance Characteristics

### Time Complexity

|Operation|Complexity|Notes|
|-----------|-----------|-------|
|Element access|O(1)|Direct indexing|
|Matrix multiplication|O(m×n×p)|For [m×n] * [n×p]|
|Addition/Subtraction|O(m×n)|Element-wise|
|Transpose|O(m×n)|Copy all elements|
|Hadamard product|O(m×n)|Element-wise|
|Scale|O(m×n)|Element-wise|
|Fill|O(m×n)|Element-wise|
|Sum/Mean|O(m×n)|Scan all elements|
|Reshape|O(m×n)|Copy all elements|
|Get/Set row|O(n)|Single row operation|
|Get/Set column|O(m)|Single column operation|

### Space Complexity

- **Storage:** O(rows × cols) for the data
- **Overhead:** O(1) for dimension variables
- **Operations:** Most create new matrix (copy), except in-place operations

### In-Place vs Copy

**In-place operations** (modify existing matrix):

- `apply_gradients()`
- `randomize()`
- `fill()`
- `set_row()`
- `set_col()`

**Copy operations** (create new matrix):

- All arithmetic operators (+, -, *)
- `transpose()`
- `scale()`
- `hadamard()`
- `reshape()`

---

## Optimization Opportunities

### 1. Cache-Friendly Iteration

```cpp
// Good: Row-major order (cache-friendly)
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        process(M(i, j));
    }
}

// Bad: Column-major order (cache-unfriendly)
for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
        process(M(i, j));
    }
}
```

### 2. SIMD Vectorization

**SIMD (Single Instruction, Multiple Data) vectorization** is a hardware-level optimization where a single CPU instruction operates on multiple data elements in parallel. Modern CPUs provide SIMD instruction sets (e.g., SSE, AVX on x86, NEON on ARM) that can accelerate common matrix operations.

#### Relevance to Matrix Class

- **Element-wise operations** (addition, subtraction, Hadamard product, scaling) can be vectorized so that multiple elements are processed per instruction.
- **Matrix multiplication** inner loops can be vectorized to compute several products and sums simultaneously.
- **Reduction operations** (sum, mean) can use SIMD to accumulate multiple values in parallel.

#### Example (Conceptual)

```cpp
// Pseudocode for SIMD vectorized addition
for (int i = 0; i < size; i += SIMD_WIDTH) {
    simd_vec a = load_simd(&A[i]);
    simd_vec b = load_simd(&B[i]);
    simd_vec c = a + b;
    store_simd(&C[i], c);
}
```

#### Benefits

- 2x–8x speedup for large matrices, depending on hardware and operation
- Lower CPU utilization and improved cache efficiency

#### Implementation Notes

- The current Matrix implementation uses standard C++ loops, but can be extended with compiler intrinsics or libraries (e.g., Eigen, OpenBLAS) for SIMD.
- Compilers like GCC and Clang can auto-vectorize simple loops if written in a SIMD-friendly way (contiguous memory, no aliasing).
- SIMD is most effective for large, contiguous data blocks (row-major layout helps).

#### Limitations

- SIMD width varies by CPU (e.g., 128/256/512 bits)
- Alignment and memory layout must be considered for best results
- Not all operations or hardware support SIMD equally

#### Further Reading

- [SIMD on Wikipedia](https://en.wikipedia.org/wiki/SIMD)
- [Auto-vectorization in GCC](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

### 3. Parallel Processing

Large matrices can benefit from:

- OpenMP parallelization of loops
- Thread-based batch processing
- GPU acceleration (requires CUDA/OpenCL)

### 4. Memory Pooling

```cpp
// Reuse matrix storage to avoid allocations
Matrix temp(100, 100);
for (int epoch = 0; epoch < 1000; epoch++) {
    temp = compute_something();
    // Reuses same memory
}
```

---

## Common Patterns and Best Practices

### 1. Weight Initialization

```cpp
// He initialization for ReLU networks
Matrix weights(fan_in, fan_out);
float scale = sqrt(2.0f / fan_in);
weights.randomize(scale);

// Xavier initialization for tanh/sigmoid networks
float scale_xavier = sqrt(2.0f / (fan_in + fan_out));
weights.randomize(scale_xavier);
```

### 2. Gradient Zeroing

```cpp
// Clear gradients before backward pass
gradient_matrix.fill(0.0f);

// Or create new zero matrix
Matrix gradients(rows, cols);  // Automatically zero-initialized
```

### 3. Dimension Validation

```cpp
void validate_dimensions(const Matrix& A, const Matrix& B,
                         const std::string& operation) {
    if (A.cols != B.rows) {
        throw std::invalid_argument(
            operation + " requires A.cols == B.rows, got " +
            std::to_string(A.cols) + " vs " + std::to_string(B.rows)
        );
    }
}
```

### 4. Batch Processing Template

```cpp
template<typename Func>
Matrix batch_process(const Matrix& batch_input, Func operation) {
    Matrix batch_output(batch_input.rows, expected_output_cols);

    for (int i = 0; i < batch_input.rows; i++) {
        std::vector<float> sample = batch_input.get_row(i);
        std::vector<float> result = operation(sample);
        batch_output.set_row(i, result);
    }

    return batch_output;
}
```

---

## Limitations and Constraints

### Current Limitations

1. **No BLAS Integration:** Manual matrix operations (slower than optimized libraries)
2. **Single Precision:** Only `float` type, no `double` or custom precision
3. **No Sparse Matrix Support:** All elements stored explicitly
4. **No Broadcasting:** Manual implementation required for broadcasting operations
5. **No Inplace Arithmetic:** Operations like `+=`, `-=` not implemented

### Design Constraints

1. **Memory Overhead:** Each matrix stores dimensions separately
2. **Copy Semantics:** Most operations create new matrices
3. **Thread Safety:** Not thread-safe without external synchronization
4. **Exception Safety:** Operations may leave matrices in inconsistent state on exception

---

## Future Enhancement Opportunities

### 1. BLAS Integration

```cpp
// Potential Eigen library integration
#ifdef USE_EIGEN
    Matrix operator*(const Matrix& other) const {
        Eigen::MatrixXf A = to_eigen();
        Eigen::MatrixXf B = other.to_eigen();
        return from_eigen(A * B);
    }
#endif
```

### 2. Broadcasting Support

```cpp
// Numpy-style broadcasting
Matrix broadcast_add(const Matrix& A, const Matrix& bias) {
    // A: [batch, features], bias: [1, features]
    // Automatically broadcast bias to all rows
}
```

### 3. Inplace Operations

```cpp
Matrix& operator+=(const Matrix& other);
Matrix& operator*=(float scalar);
```

### 4. Template Support

```cpp
template<typename T>
class MatrixT {
    std::vector<std::vector<T>> data;
    // Support double, int, custom types
};
```

### 5. Memory Views (Avoid Copies)

```cpp
class MatrixView {
    Matrix& parent;
    int row_offset, col_offset;
    int view_rows, view_cols;
    // Zero-copy sub-matrix access
};
```

---

## Testing Recommendations

### Unit Tests

```cpp
// Test dimensions
Matrix A(3, 4);
assert(A.rows == 3 && A.cols == 4);

// Test multiplication dimensions
Matrix B(4, 5);
Matrix C = A * B;
assert(C.rows == 3 && C.cols == 5);

// Test transpose
Matrix A_T = A.transpose();
assert(A_T.rows == 4 && A_T.cols == 3);

// Test element access
A(1, 2) = 3.14f;
assert(abs(A(1, 2) - 3.14f) < 1e-6);

// Test gradient application
Matrix W(2, 2);
W.fill(1.0f);
Matrix grad(2, 2);
grad.fill(0.1f);
W.apply_gradients(grad, 1.0f);
assert(abs(W(0, 0) - 0.9f) < 1e-6);
```

### Numerical Gradient Checking

```cpp
float numerical_gradient(Matrix& W, int i, int j,
                         std::function<float()> loss_fn) {
    float epsilon = 1e-5f;

    float orig = W(i, j);

    W(i, j) = orig + epsilon;
    float loss_plus = loss_fn();

    W(i, j) = orig - epsilon;
    float loss_minus = loss_fn();

    W(i, j) = orig;

    return (loss_plus - loss_minus) / (2 * epsilon);
}
```

---

## Debugging Tips

### 1. Dimension Tracking

```cpp
#define DEBUG_DIMS
#ifdef DEBUG_DIMS
    std::cout << "A: " << A.rows << "x" << A.cols
              << ", B: " << B.rows << "x" << B.cols << std::endl;
#endif
```

### 2. Value Inspection

```cpp
// Check for NaN/Inf
bool has_nan_or_inf(const Matrix& M) {
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            if (std::isnan(M(i, j)) || std::isinf(M(i, j))) {
                return true;
            }
        }
    }
    return false;
}
```

### 3. Gradient Magnitude Monitoring

```cpp
float gradient_norm(const Matrix& grad) {
    float sum_sq = 0.0f;
    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            sum_sq += grad(i, j) * grad(i, j);
        }
    }
    return sqrt(sum_sq);
}
```

---

## Mathematical Foundations

### Matrix Multiplication (Detailed)

For matrices A[m×n] and B[n×p]:

```text
C[i][j] = Σ(k=0 to n-1) A[i][k] × B[k][j]
```

Geometric Interpretation:

- Each element is the dot product of a row from A and a column from B
- Result dimensions: number of rows from A, number of columns from B

Properties:

- Associative: (AB)C = A(BC)
- Distributive: A(B+C) = AB + AC
- Not commutative: AB ≠ BA (generally)

### Transpose Properties

```text
(A^T)^T = A
(AB)^T = B^T A^T
(A + B)^T = A^T + B^T
```

### Gradient Descent Update Rule

```text
W(t+1) = W(t) - η × ∇L/∂W
```

Where:

- W(t): weights at time step t
- η: learning rate
- ∇L/∂W: gradient of loss with respect to weights

---

## Summary

The `Matrix` class provides a complete, self-contained implementation for neural network tensor operations with:

Strengths:

- Clean, intuitive interface
- Comprehensive operation coverage
- Strong error checking
- Good documentation
- Training-ready with gradient support

Use Cases:

- Neural network weight matrices
- Activation tensors
- Gradient storage and propagation
- Embedding tables
- Attention mechanisms

Integration:

- Used by LayerNorm, TokenEmbedding, MultiHeadAttention, FeedForward
- Core component of LLMEncoder architecture
- Standalone utility for matrix computations

The class successfully balances simplicity with functionality, making it suitable for educational purposes and small-to-medium scale neural network implementations.
