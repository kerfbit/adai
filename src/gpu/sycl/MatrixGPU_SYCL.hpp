#ifndef MATRIX_GPU_SYCL_HPP
#define MATRIX_GPU_SYCL_HPP

// @adai-status: beta        (capped by TD-030 — see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07


#include "GPUUtils_SYCL.hpp"

namespace adai {
namespace gpu {

enum class ActivationType { RELU = 0, SIGMOID = 1, TANH = 2, GELU = 3 };

void matrix_add_gpu(const float* a, const float* b, float* c, int size);
void matrix_add_scalar_gpu(const float* a, float scalar, float* c, int size);
void matrix_multiply_elementwise_gpu(const float* a, const float* b, float* c, int size);
void matrix_multiply_scalar_gpu(const float* a, float scalar, float* c, int size);
void matrix_transpose_gpu(const float* input, float* output, int rows, int cols);
void matrix_multiply_gpu(const float* a, const float* b, float* c, int m, int k, int n);
void matrix_apply_activation_gpu(float* data, int size, ActivationType type);
float matrix_sum_gpu(const float* data, int size);

void matrix_batch_add_gpu(const float** a_batch, const float** b_batch, float** c_batch,
                          int batch_size, int size);
void matrix_batch_multiply_gpu(const float** a_batch, const float** b_batch, float** c_batch,
                               int batch_size, int m, int k, int n);

// TD-003 — persistent GPU-resident training kernels
void matrix_add_inplace_gpu(const float* src, float* dst, int size);
void matrix_softmax_rows_gpu(float* data, int rows, int cols);
void matrix_softmax_backward_gpu(const float* s, const float* dout, float* din, int rows, int cols);
void matrix_gelu_backward_gpu(const float* pre_act, const float* dout, float* din, int size);
void matrix_layer_norm_fwd_gpu(const float* input, float* output, float* out_normed,
                               const float* gamma, const float* beta, float* out_mean,
                               float* out_rstd, int rows, int cols, float eps);
void matrix_layer_norm_bwd_gpu(const float* dout, const float* input_norm, const float* gamma,
                               const float* mean, const float* rstd, float* dx, float* dgamma,
                               float* dbeta, int rows, int cols);
void matrix_add_row_bias_gpu(const float* mat, const float* bias, float* out, int rows, int cols);
void matrix_sum_rows_gpu(const float* mat, float* out, int rows, int cols);
void matrix_masked_fill_gpu(float* data, const float* mask, float fill_val, int size);
float matrix_cross_entropy_loss_gpu(const float* logits, const int* targets, int seq_len,
                                    int vocab_size);
void matrix_cross_entropy_grad_gpu(const float* logits, const int* targets, float* grad,
                                   int seq_len, int vocab_size);

// Training-diagnostics reductions (GPU-native activation saturation / attention
// entropy — see ChatbotTrainer's gpu_activation_stats_hook_/gpu_attention_stats_hook_).
float matrix_count_below_threshold_gpu(const float* data, int size, float threshold);
void matrix_row_entropy_gpu(const float* data, float* out_row_entropy, int rows, int cols);

// Small backend-agnostic transfer helpers (replace direct GPUManager::get_queue()
// calls that only compiled under SYCL).
void matrix_copy_device_to_device_gpu(const float* src, float* dst, int count);
void matrix_download_gpu(const float* device_ptr, float* host_ptr, int count);

class GPUMatrix {
   public:
    int rows = 0;
    int cols = 0;

   private:
    GPUMemory<float> data_;

   public:
    GPUMatrix(int r, int c) : rows(r), cols(c), data_(r * c) {}

    GPUMatrix(const GPUMatrix&) = delete;
    GPUMatrix& operator=(const GPUMatrix&) = delete;
    GPUMatrix(GPUMatrix&&) = default;
    GPUMatrix& operator=(GPUMatrix&&) = default;

    float* device_ptr() {
        return data_.get();
    }
    const float* device_ptr() const {
        return data_.get();
    }
    int size() const {
        return rows * cols;
    }

    void upload(const float* host_ptr, int count) {
        data_.copy_from_host(host_ptr, static_cast<size_t>(count));
    }

    void download(float* host_ptr, int count) const {
        data_.copy_to_host(host_ptr, static_cast<size_t>(count));
    }

    GPUMatrix copy() const {
        GPUMatrix result(rows, cols);
        auto& q = GPUManager::get_queue();
        q.memcpy(result.data_.get(), data_.get(), static_cast<size_t>(rows * cols) * sizeof(float))
            .wait();
        return result;
    }

    GPUMatrix operator*(const GPUMatrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("GPUMatrix dimensions incompatible for multiply");
        GPUMatrix result(rows, other.cols);
        matrix_multiply_gpu(data_.get(), other.data_.get(), result.data_.get(), rows, cols,
                            other.cols);
        return result;
    }

    GPUMatrix operator+(const GPUMatrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("GPUMatrix dimensions must match for add");
        GPUMatrix result(rows, cols);
        matrix_add_gpu(data_.get(), other.data_.get(), result.data_.get(), size());
        return result;
    }

    GPUMatrix operator-(const GPUMatrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("GPUMatrix dimensions must match for subtract");
        GPUMatrix neg_b(rows, cols);
        matrix_multiply_scalar_gpu(other.data_.get(), -1.0f, neg_b.data_.get(), size());
        GPUMatrix result(rows, cols);
        matrix_add_gpu(data_.get(), neg_b.data_.get(), result.data_.get(), size());
        return result;
    }

    GPUMatrix scale(float scalar) const {
        GPUMatrix result(rows, cols);
        matrix_multiply_scalar_gpu(data_.get(), scalar, result.data_.get(), size());
        return result;
    }

    GPUMatrix hadamard(const GPUMatrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("GPUMatrix dimensions must match for hadamard");
        GPUMatrix result(rows, cols);
        matrix_multiply_elementwise_gpu(data_.get(), other.data_.get(), result.data_.get(), size());
        return result;
    }

    GPUMatrix transpose() const {
        GPUMatrix result(cols, rows);
        matrix_transpose_gpu(data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    void apply_activation_inplace(ActivationType type) {
        matrix_apply_activation_gpu(data_.get(), size(), type);
    }

    float sum() const {
        return matrix_sum_gpu(data_.get(), size());
    }

    // ---- TD-003 persistent-training operations ----------------------------

    void zero() {
        auto& q = GPUManager::get_queue();
        q.memset(data_.get(), 0, static_cast<size_t>(rows * cols) * sizeof(float));
    }

    void add_inplace(const GPUMatrix& other) {
        matrix_add_inplace_gpu(other.data_.get(), data_.get(), size());
    }

    void softmax_rows_inplace() {
        matrix_softmax_rows_gpu(data_.get(), rows, cols);
    }

    GPUMatrix softmax_backward(const GPUMatrix& dout) const {
        GPUMatrix result(rows, cols);
        matrix_softmax_backward_gpu(data_.get(), dout.data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    GPUMatrix gelu_backward(const GPUMatrix& dout) const {
        GPUMatrix result(rows, cols);
        matrix_gelu_backward_gpu(data_.get(), dout.data_.get(), result.data_.get(), size());
        return result;
    }

    GPUMatrix add_row_bias(const GPUMatrix& bias) const {
        GPUMatrix result(rows, cols);
        matrix_add_row_bias_gpu(data_.get(), bias.data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    GPUMatrix sum_rows() const {
        GPUMatrix result(1, cols);
        matrix_sum_rows_gpu(data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    void masked_fill_inplace(const GPUMatrix& mask, float fill_val) {
        matrix_masked_fill_gpu(data_.get(), mask.data_.get(), fill_val, size());
    }

    GPUMatrix layer_norm(const GPUMatrix& gamma, const GPUMatrix& beta, float eps,
                         GPUMatrix& out_normed, GPUMatrix& out_mean, GPUMatrix& out_rstd) const {
        GPUMatrix result(rows, cols);
        matrix_layer_norm_fwd_gpu(data_.get(), result.data_.get(), out_normed.data_.get(),
                                  gamma.data_.get(), beta.data_.get(), out_mean.data_.get(),
                                  out_rstd.data_.get(), rows, cols, eps);
        return result;
    }

    GPUMatrix layer_norm_backward(const GPUMatrix& input_norm, const GPUMatrix& gamma,
                                  const GPUMatrix& mean, const GPUMatrix& rstd, GPUMatrix& dgamma,
                                  GPUMatrix& dbeta) const {
        GPUMatrix dx(rows, cols);
        matrix_layer_norm_bwd_gpu(data_.get(), input_norm.data_.get(), gamma.data_.get(),
                                  mean.data_.get(), rstd.data_.get(), dx.data_.get(),
                                  dgamma.data_.get(), dbeta.data_.get(), rows, cols);
        return dx;
    }

    // ---- Training-diagnostics reductions (activation saturation / attention entropy) --

    /** @brief Fraction of elements with |x| < threshold (e.g. post-GELU saturation). */
    float count_below_threshold(float threshold) const {
        return matrix_count_below_threshold_gpu(data_.get(), size(), threshold) /
               static_cast<float>(size());
    }

    /** @brief Average per-row Shannon entropy of an already-normalized (e.g. post-softmax) matrix.
     */
    float row_entropy_avg() const {
        GPUMemory<float> row_ent(rows);
        matrix_row_entropy_gpu(data_.get(), row_ent.get(), rows, cols);
        return matrix_sum_gpu(row_ent.get(), rows) / static_cast<float>(rows);
    }
};

}  // namespace gpu
}  // namespace adai

#endif  // MATRIX_GPU_SYCL_HPP
