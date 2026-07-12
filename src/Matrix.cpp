#include "Matrix.hpp"
#include <iomanip>
#include <sstream>
#include "Logger.hpp"

#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ADAI_ENABLE_BLAS
#include <cblas.h>
#endif

// Default constructor
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(Matrix&& other) noexcept
    : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        rows = other.rows;
        cols = other.cols;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// Constructor with dimensions
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(rows, std::vector<float>(cols, 0.0f));
}

// Constructor from existing data
Matrix::Matrix(const std::vector<std::vector<float>>& d)
    : data(d), cols((rows > 0) ? d[0].size() : 0), rows(d.size()) {
    // Validate that all rows have the same number of columns
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
}

// Element access (non-const)
float& Matrix::operator()(int i, int j) {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[i][j];
}

// Element access (const)
const float& Matrix::operator()(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[i][j];
}

// Matrix multiplication
// Acceleration priority:
//   1. cuBLAS SGEMM (ADAI_ENABLE_GPU, matrices where any dim ≥ 32)
//   2. BLAS SGEMM   (ADAI_ENABLE_BLAS, matrices ≥ 256 in every dimension)
//   3. AVX2/FMA     (ADAI_SIMD_AVX2)  — ikj loop order, 8-wide FMA per iteration
//   4. ARM NEON     (ADAI_SIMD_NEON)  — ikj loop order, 4-wide FMA per iteration
//   5. OpenMP       (ADAI_ENABLE_OPENMP) — original ijk with #pragma omp simd
//   6. Scalar fallback
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication: [" +
                                    std::to_string(rows) + "x" + std::to_string(cols) + "] * [" +
                                    std::to_string(other.rows) + "x" + std::to_string(other.cols) +
                                    "]");
    }

    Matrix result(rows, other.cols);  // zero-initialised

#ifdef ADAI_ENABLE_GPU
    // cuBLAS path: dispatch to GPU when the inner dimensions (cols / other.cols)
    // are large enough for cuBLAS SGEMM to outperform AVX2.  We do NOT gate on
    // 'rows' so that short-sequence inputs (seq_len < 32) still benefit from
    // fast weight-matrix multiplications (d_model × d_model, d_model × d_ff).
    if (adai::gpu::GPUManager::is_available() && cols >= 32 && other.cols >= 32) {
        return multiply_gpu(other);
    }
#endif  // ADAI_ENABLE_GPU

#ifdef ADAI_ENABLE_BLAS
    // BLAS path: pack to flat row-major arrays, call cblas_sgemm, unpack result.
    // Only worthwhile for large matrices where packing overhead is negligible.
    if (rows >= 256 && cols >= 256 && other.cols >= 256) {
        std::vector<float> a_flat(rows * cols);
        std::vector<float> b_flat(other.rows * other.cols);
        for (int i = 0; i < rows; ++i)
            std::copy(data[i].begin(), data[i].end(), a_flat.data() + i * cols);
        for (int i = 0; i < other.rows; ++i)
            std::copy(other.data[i].begin(), other.data[i].end(), b_flat.data() + i * other.cols);

        std::vector<float> c_flat(rows * other.cols, 0.0f);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, other.cols, cols, 1.0f,
                    a_flat.data(), cols, b_flat.data(), other.cols, 0.0f, c_flat.data(),
                    other.cols);

        for (int i = 0; i < rows; ++i)
            std::copy(c_flat.data() + i * other.cols, c_flat.data() + (i + 1) * other.cols,
                      result.data[i].data());
        return result;
    }
#endif  // ADAI_ENABLE_BLAS

#if defined(ADAI_SIMD_AVX2)
    // ikj loop order with AVX2 FMA: broadcasts A[i][k], processes 8 floats of
    // row k of B simultaneously into row i of C.  B rows are contiguous so
    // cache-friendly; each inner loop touches only one row of C (good locality).
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) if (rows > 64)
#endif
    for (int i = 0; i < rows; ++i) {
        float* c_row = result.data[i].data();
        const float* a_row = data[i].data();
        for (int k = 0; k < cols; ++k) {
            const float* b_row = other.data[k].data();
#ifdef ADAI_SIMD_FMA
            __m256 va = _mm256_set1_ps(a_row[k]);
            int j = 0;
            for (; j <= other.cols - 8; j += 8) {
                __m256 vc = _mm256_loadu_ps(c_row + j);
                vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j), vc);
                _mm256_storeu_ps(c_row + j, vc);
            }
#else  // AVX2 without FMA
            __m256 va = _mm256_set1_ps(a_row[k]);
            int j = 0;
            for (; j <= other.cols - 8; j += 8) {
                __m256 vc = _mm256_loadu_ps(c_row + j);
                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, _mm256_loadu_ps(b_row + j)));
                _mm256_storeu_ps(c_row + j, vc);
            }
#endif
            // scalar remainder for cols not divisible by 8
            for (; j < other.cols; ++j) {
                c_row[j] += a_row[k] * b_row[j];
            }
        }
    }

#elif defined(ADAI_SIMD_NEON)
    // ikj loop order with ARM NEON 4-wide FMA
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) if (rows > 64)
#endif
    for (int i = 0; i < rows; ++i) {
        float* c_row = result.data[i].data();
        const float* a_row = data[i].data();
        for (int k = 0; k < cols; ++k) {
            const float* b_row = other.data[k].data();
            float32x4_t va_ik = vdupq_n_f32(a_row[k]);
            int j = 0;
            for (; j <= other.cols - 4; j += 4) {
                float32x4_t vc = vld1q_f32(c_row + j);
                vc = vfmaq_f32(vc, va_ik, vld1q_f32(b_row + j));
                vst1q_f32(c_row + j, vc);
            }
            for (; j < other.cols; ++j)
                c_row[j] += a_row[k] * b_row[j];
        }
    }

#elif defined(ADAI_ENABLE_OPENMP)
// Parallel version with OpenMP — 5-8x speedup on multi-core CPUs
#pragma omp parallel for collapse(2) schedule(dynamic, 32) if (rows > 64 && other.cols > 64)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }

#else
    // Scalar fallback
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
#endif

    return result;
}

// Matrix addition — element-wise, SIMD row-by-row
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows, cols);

#if defined(ADAI_SIMD_AVX2)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        const float* b = other.data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            _mm256_storeu_ps(r + j, _mm256_add_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)));
        }
        for (; j < cols; ++j) {
            r[j] = a[j] + b[j];
        }
    }

#elif defined(ADAI_SIMD_NEON)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        const float* b = other.data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 4; j += 4)
            vst1q_f32(r + j, vaddq_f32(vld1q_f32(a + j), vld1q_f32(b + j)));
        for (; j < cols; ++j)
            r[j] = a[j] + b[j];
    }

#elif defined(ADAI_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
#endif

    return result;
}

// Matrix subtraction — element-wise, SIMD row-by-row
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows, cols);

#if defined(ADAI_SIMD_AVX2)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        const float* b = other.data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            _mm256_storeu_ps(r + j, _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)));
        }
        for (; j < cols; ++j) {
            r[j] = a[j] - b[j];
        }
    }

#elif defined(ADAI_SIMD_NEON)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        const float* b = other.data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 4; j += 4)
            vst1q_f32(r + j, vsubq_f32(vld1q_f32(a + j), vld1q_f32(b + j)));
        for (; j < cols; ++j)
            r[j] = a[j] - b[j];
    }

#elif defined(ADAI_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
#endif

    return result;
}

// Matrix transpose
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

#ifdef ADAI_ENABLE_OPENMP
// Parallel version with OpenMP
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
#else
    // Sequential fallback
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
#endif

    return result;
}

// Randomize matrix with Xavier/He initialization
void Matrix::randomize(float scale) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

// Scalar multiplication
Matrix Matrix::scale(float scalar) const {
    Matrix result(rows, cols);

#if defined(ADAI_SIMD_AVX2)
    __m256 vs = _mm256_set1_ps(scalar);
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            _mm256_storeu_ps(r + j, _mm256_mul_ps(vs, _mm256_loadu_ps(a + j)));
        }
        for (; j < cols; ++j) {
            r[j] = a[j] * scalar;
        }
    }

#elif defined(ADAI_SIMD_NEON)
    float32x4_t vs = vdupq_n_f32(scalar);
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 4; j += 4)
            vst1q_f32(r + j, vmulq_f32(vs, vld1q_f32(a + j)));
        for (; j < cols; ++j)
            r[j] = a[j] * scalar;
    }

#elif defined(ADAI_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
#endif

    return result;
}

// Hadamard product (element-wise multiplication) — SIMD row-by-row
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }

    Matrix result(rows, cols);

#if defined(ADAI_SIMD_AVX2)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        const float* b = other.data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            _mm256_storeu_ps(r + j, _mm256_mul_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)));
        }
        for (; j < cols; ++j) {
            r[j] = a[j] * b[j];
        }
    }

#elif defined(ADAI_SIMD_NEON)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* a = data[i].data();
        const float* b = other.data[i].data();
        float* r = result.data[i].data();
        int j = 0;
        for (; j <= cols - 4; j += 4)
            vst1q_f32(r + j, vmulq_f32(vld1q_f32(a + j), vld1q_f32(b + j)));
        for (; j < cols; ++j)
            r[j] = a[j] * b[j];
    }

#elif defined(ADAI_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
#endif

    return result;
}

// Apply gradients (W = W − lr × grad) — fused multiply-add, SIMD row-by-row
void Matrix::apply_gradients(const Matrix& gradients, float learning_rate) {
    if (rows != gradients.rows || cols != gradients.cols) {
        throw std::invalid_argument("Gradient matrix dimensions must match");
    }

#if defined(ADAI_SIMD_AVX2)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        float* d = data[i].data();
        const float* g = gradients.data[i].data();
        // d[j] = d[j] + (−lr) * g[j]  →  fmadd(neg_lr, g, d)
#ifdef ADAI_SIMD_FMA
        __m256 neg_lr = _mm256_set1_ps(-learning_rate);
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 vd = _mm256_loadu_ps(d + j);
            vd = _mm256_fmadd_ps(neg_lr, _mm256_loadu_ps(g + j), vd);
            _mm256_storeu_ps(d + j, vd);
        }
#else  // AVX2 without FMA
        __m256 neg_lr = _mm256_set1_ps(-learning_rate);
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 vd = _mm256_loadu_ps(d + j);
            vd = _mm256_sub_ps(
                vd, _mm256_mul_ps(_mm256_set1_ps(learning_rate), _mm256_loadu_ps(g + j)));
            _mm256_storeu_ps(d + j, vd);
        }
#endif
        for (; j < cols; ++j) {
            d[j] -= learning_rate * g[j];
        }
    }

#elif defined(ADAI_SIMD_NEON)
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for if (rows * cols > 10000)
#endif
    for (int i = 0; i < rows; ++i) {
        float* d = data[i].data();
        const float* g = gradients.data[i].data();
        int j = 0;
        for (; j <= cols - 4; j += 4) {
            float32x4_t vd = vld1q_f32(d + j);
            // vd = vd - lr * g  →  vfmsq_f32(vd, lr_vec, g)
            vd = vfmsq_n_f32(vd, vld1q_f32(g + j), learning_rate);
            vst1q_f32(d + j, vd);
        }
        for (; j < cols; ++j)
            d[j] -= learning_rate * g[j];
    }

#elif defined(ADAI_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= learning_rate * gradients.data[i][j];
        }
    }
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= learning_rate * gradients.data[i][j];
        }
    }
#endif
}

// Fill matrix with constant value
void Matrix::fill(float value) {
#ifdef ADAI_ENABLE_OPENMP
// Parallel version with OpenMP
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = value;
        }
    }
#else
    // Sequential fallback
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = value;
        }
    }
#endif
}

// Sum of all elements — SIMD horizontal reduction
float Matrix::sum() const {
    float total = 0.0f;

#if defined(ADAI_SIMD_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < rows; ++i) {
        const float* row = data[i].data();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(row + j));
        }
        for (; j < cols; ++j) {
            total += row[j];
        }
    }
    total += adai::simd::hsum256(acc);

#elif defined(ADAI_SIMD_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < rows; ++i) {
        const float* row = data[i].data();
        int j = 0;
        for (; j <= cols - 4; j += 4)
            acc = vaddq_f32(acc, vld1q_f32(row + j));
        for (; j < cols; ++j)
            total += row[j];
    }
    total += adai::simd::hsum128(acc);

#elif defined(ADAI_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) reduction(+ : total) if (rows * cols > 10000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += data[i][j];
        }
    }
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += data[i][j];
        }
    }
#endif

    return total;
}

// Mean of all elements
float Matrix::mean() const {
    if (rows == 0 || cols == 0) {
        return 0.0f;
    }
    return sum() / static_cast<float>(rows * cols);
}

// Print matrix
void Matrix::print(const std::string& name, int max_rows, int max_cols) const {
    if (!name.empty()) {
        std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    }

    int print_rows = std::min(rows, max_rows);
    int print_cols = std::min(cols, max_cols);

    for (int i = 0; i < print_rows; i++) {
        std::cout << "[";
        for (int j = 0; j < print_cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data[i][j];
            if (j < print_cols - 1) {
                std::cout << " ";
            }
        }
        if (cols > max_cols) {
            std::cout << " ...";
        }
        std::cout << "]" << std::endl;
    }
    if (rows > max_rows) {
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

// Check if matrix is valid
bool Matrix::is_valid() const {
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    if (data.size() != rows) {
        return false;
    }
    for (const auto& row : data) {
        if (row.size() != cols)
            return false;
    }
    return true;
}

// Reshape matrix
Matrix Matrix::reshape(int new_rows, int new_cols) const {
    if (rows * cols != new_rows * new_cols) {
        throw std::invalid_argument("Total number of elements must remain the same when reshaping");
    }

    Matrix result(new_rows, new_cols);
    int idx = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int new_i = idx / new_cols;
            int new_j = idx % new_cols;
            result.data[new_i][new_j] = data[i][j];
            idx++;
        }
    }

    return result;
}

// Get row
std::vector<float> Matrix::get_row(int row_idx) const {
    if (row_idx < 0 || row_idx >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    return data[row_idx];
}

// Get column
std::vector<float> Matrix::get_col(int col_idx) const {
    if (col_idx < 0 || col_idx >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }

    std::vector<float> column(rows);
    for (int i = 0; i < rows; i++) {
        column[i] = data[i][col_idx];
    }
    return column;
}

// Set row
void Matrix::set_row(int row_idx, const std::vector<float>& values) {
    if (row_idx < 0 || row_idx >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    if (values.size() != cols) {
        throw std::invalid_argument("Row values size must match number of columns");
    }
    data[row_idx] = values;
}

// Set column
void Matrix::set_col(int col_idx, const std::vector<float>& values) {
    if (col_idx < 0 || col_idx >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }
    if (values.size() != rows) {
        throw std::invalid_argument("Column values size must match number of rows");
    }

    for (int i = 0; i < rows; i++) {
        data[i][col_idx] = values[i];
    }
}

// ============================================================================
// GPU Management — always defined (stubs in non-GPU builds)
// ============================================================================

#ifdef ADAI_ENABLE_GPU

bool Matrix::gpu_available() {
    return adai::gpu::GPUManager::is_available();
}

bool Matrix::gpu_initialize(int device_id, float memory_fraction, bool use_low_priority) {
    return adai::gpu::GPUManager::initialize(device_id, memory_fraction, use_low_priority);
}

bool Matrix::gpu_try_initialize(int device_id, float memory_fraction, bool use_low_priority) {
    if (!adai::gpu::GPUManager::probe()) {
        return false;
    }
    try {
        return adai::gpu::GPUManager::initialize(device_id, memory_fraction, use_low_priority);
    } catch (const std::exception& e) {
        adai::Logger::warn("[GPU] Initialisation failed, falling back to CPU: {}", e.what());
        return false;
    }
}

void Matrix::gpu_cleanup() {
    adai::gpu::GPUManager::cleanup();
}

std::string Matrix::gpu_info(int device) {
    return adai::gpu::GPUManager::get_device_info(device);
}

#else  // !ADAI_ENABLE_GPU — CPU-only stubs

bool Matrix::gpu_available() {
    return false;
}

bool Matrix::gpu_initialize(int, float, bool) {
    return false;
}

bool Matrix::gpu_try_initialize(int, float, bool) {
    return false;
}

void Matrix::gpu_cleanup() {}

std::string Matrix::gpu_info(int) {
    return "GPU support not compiled (rebuild with -DENABLE_GPU=ON for CUDA or -DENABLE_SYCL=ON for Intel Arc)";
}

#endif  // ADAI_ENABLE_GPU

#ifdef ADAI_ENABLE_GPU
// ============================================================================
// GPU-Accelerated Matrix Operations
// ============================================================================

// Helper function to flatten matrix to 1D array
static std::vector<float> flatten_matrix(const Matrix& mat) {
    std::vector<float> flat(mat.rows * mat.cols);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            flat[i * mat.cols + j] = mat.data[i][j];
        }
    }
    return flat;
}

// Helper function to unflatten 1D array to matrix
static Matrix unflatten_matrix(const std::vector<float>& flat, int rows, int cols) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = flat[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::multiply_gpu(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    if (!gpu_available()) {
        // CPU fallback
        return (*this) * other;
    }

    // Flatten matrices
    auto a_flat = flatten_matrix(*this);
    auto b_flat = flatten_matrix(other);

    // Allocate GPU memory
    adai::gpu::GPUMemory<float> d_a(rows * cols);
    adai::gpu::GPUMemory<float> d_b(other.rows * other.cols);
    adai::gpu::GPUMemory<float> d_c(rows * other.cols);

    // Copy to GPU
    d_a.copy_from_host(a_flat.data(), rows * cols);
    d_b.copy_from_host(b_flat.data(), other.rows * other.cols);

    // Perform multiplication on GPU
    adai::gpu::matrix_multiply_gpu(d_a.get(), d_b.get(), d_c.get(), rows, cols, other.cols);

    // Copy result back
    std::vector<float> c_flat(rows * other.cols);
    d_c.copy_to_host(c_flat.data(), rows * other.cols);

    return unflatten_matrix(c_flat, rows, other.cols);
}

Matrix Matrix::add_gpu(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    if (!gpu_available()) {
        // CPU fallback
        return (*this) + other;
    }

    int size = rows * cols;
    auto a_flat = flatten_matrix(*this);
    auto b_flat = flatten_matrix(other);

    adai::gpu::GPUMemory<float> d_a(size);
    adai::gpu::GPUMemory<float> d_b(size);
    adai::gpu::GPUMemory<float> d_c(size);

    d_a.copy_from_host(a_flat.data(), size);
    d_b.copy_from_host(b_flat.data(), size);

    adai::gpu::matrix_add_gpu(d_a.get(), d_b.get(), d_c.get(), size);

    std::vector<float> c_flat(size);
    d_c.copy_to_host(c_flat.data(), size);

    return unflatten_matrix(c_flat, rows, cols);
}

Matrix Matrix::transpose_gpu() const {
    if (!gpu_available()) {
        // CPU fallback
        return this->transpose();
    }

    auto a_flat = flatten_matrix(*this);

    adai::gpu::GPUMemory<float> d_input(rows * cols);
    adai::gpu::GPUMemory<float> d_output(rows * cols);

    d_input.copy_from_host(a_flat.data(), rows * cols);

    adai::gpu::matrix_transpose_gpu(d_input.get(), d_output.get(), rows, cols);

    std::vector<float> output_flat(rows * cols);
    d_output.copy_to_host(output_flat.data(), rows * cols);

    return unflatten_matrix(output_flat, cols, rows);
}

Matrix Matrix::scale_gpu(float scalar) const {
    if (!gpu_available()) {
        // CPU fallback
        return this->scale(scalar);
    }

    int size = rows * cols;
    auto a_flat = flatten_matrix(*this);

    adai::gpu::GPUMemory<float> d_a(size);
    adai::gpu::GPUMemory<float> d_c(size);

    d_a.copy_from_host(a_flat.data(), size);

    adai::gpu::matrix_multiply_scalar_gpu(d_a.get(), scalar, d_c.get(), size);

    std::vector<float> c_flat(size);
    d_c.copy_to_host(c_flat.data(), size);

    return unflatten_matrix(c_flat, rows, cols);
}

Matrix Matrix::hadamard_gpu(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }

    if (!gpu_available()) {
        // CPU fallback
        return this->hadamard(other);
    }

    int size = rows * cols;
    auto a_flat = flatten_matrix(*this);
    auto b_flat = flatten_matrix(other);

    adai::gpu::GPUMemory<float> d_a(size);
    adai::gpu::GPUMemory<float> d_b(size);
    adai::gpu::GPUMemory<float> d_c(size);

    d_a.copy_from_host(a_flat.data(), size);
    d_b.copy_from_host(b_flat.data(), size);

    adai::gpu::matrix_multiply_elementwise_gpu(d_a.get(), d_b.get(), d_c.get(), size);

    std::vector<float> c_flat(size);
    d_c.copy_to_host(c_flat.data(), size);

    return unflatten_matrix(c_flat, rows, cols);
}

// ============================================================================
// Persistent GPU-resident matrix operations (TD-003)
// ============================================================================

adai::gpu::GPUMatrix Matrix::to_gpu() const {
    if (!gpu_available()) {
        throw std::runtime_error(
            "GPU not available: call Matrix::gpu_initialize() before to_gpu()");
    }
    adai::gpu::GPUMatrix gm(rows, cols);
    auto flat = flatten_matrix(*this);
    gm.upload(flat.data(), rows * cols);
    return gm;
}

/*static*/
Matrix Matrix::from_gpu(const adai::gpu::GPUMatrix& gm) {
    std::vector<float> flat(static_cast<size_t>(gm.rows * gm.cols));
    gm.download(flat.data(), gm.rows * gm.cols);
    return unflatten_matrix(flat, gm.rows, gm.cols);
}

#endif  // ADAI_ENABLE_GPU
