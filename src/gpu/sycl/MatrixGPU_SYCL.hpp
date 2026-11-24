#ifndef MATRIX_GPU_SYCL_HPP
#define MATRIX_GPU_SYCL_HPP

// @adai-status: beta        (capped by TD-389 — see TECHNICAL_DEBT.md)
// @adai-version: 0.21.0
// @adai-reviewed: 2026-11-23

#include "../GPUOptimizerKernels.hpp"
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
// TD-388: row-major strided GEMM; see MatrixGPU.hpp's declaration.
void matrix_gemm_strided_gpu(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
                             const float* a, int lda, const float* b, int ldb, float* c, int ldc);
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
// TD-276: how many of the seq_len rows have their argmax (lowest index on ties) at the row's
// target id; a target outside [0, vocab_size) never matches. Downloads one number.
int matrix_token_correct_gpu(const float* logits, const int* targets, int seq_len, int vocab_size);
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
// TD-395: sets `count` floats to zero, queued like every kernel (no wait).
void matrix_zero_gpu(float* data, int count);

// TD-468: device-resident optimizer kernels (see GPUOptimizerKernels.hpp and GPUOptimizer.hpp).
// Adds sum(data[i]^2) into *out_device (a device float) without waiting; zero it first.
void matrix_sum_squares_accumulate_gpu(const float* data, int size, float* out_device);
// One optimizer update of `size` parameters in place. m/v may be null when the rule doesn't use
// them (SGD: neither; SGD_MOMENTUM: m only).
void optimizer_update_gpu(float* weights, const float* grads, float* m, float* v, int size,
                          const GPUOptimizerUpdate& update);

// TD-547: token embedding on the device. out[i,:] = table[ids[i],:] + pe[pos_offset+i,:] (no
// positional row when pos_offset+i >= pe_rows or pe is null, as PositionalEncoding::forward()).
void embedding_gather_add_pe_gpu(const float* table, const int* ids, const float* pe, int pe_rows,
                                 int pos_offset, float* out, int n, int d);
// grad_table[ids[i],:] += dout[i,:] (atomic: a token can repeat within a sequence).
void embedding_scatter_add_gpu(const float* dout, const int* ids, float* grad_table, int n, int d);

// TD-342: the gated world-model/hippocampal residual of DecoderBlock, with the gate read from
// device memory (a 1x1 parameter), so no gate value crosses to the host.
// residual += tanh(gate[0]) * attn
void gated_residual_add_gpu(float* residual, const float* attn, const float* gate, int size);
// d_attn = tanh(gate[0]) * d_res, and gate_grad[0] += (1 - tanh(gate[0])^2) * sum(d_res * attn)
void gated_residual_backward_gpu(const float* d_res, const float* attn, const float* gate,
                                 float* d_attn, float* gate_grad, int size);

// TD-344: inverted dropout on the device. mask[i] = adai::dropout_mask_value(seed, i, keep_prob)
// (1 / keep_prob or 0, the same values the CPU draws for that seed) and data[i] *= mask[i]. The
// backward pass multiplies the gradient by the kept mask (matrix_multiply_elementwise_gpu()).
void dropout_forward_gpu(float* data, float* mask, int size, float keep_prob,
                         unsigned long long seed);

// TD-586: SIGReg on the device, with SIGRegMath.hpp's formulas (the host SIGReg uses the same ones).
// Fills directions [d_model, num_sketches] for (seed, draw) and normalizes every column to unit length.
void sigreg_directions_gpu(float* directions, int d_model, int num_sketches, unsigned long long seed,
                           unsigned long long draw);
// Per direction k of proj [batch, num_sketches]: loss_terms[k], the unbiased characteristic-function gap
// summed over the frequencies (TD-503; 0 when batch < 2), and var_terms[k], the projections' population
// variance. With grad_proj non-null, also d(loss)/d(proj) [batch, num_sketches], scaled by norm_factor.
void sigreg_terms_gpu(const float* proj, int batch, int num_sketches, float norm_factor, float* loss_terms,
                      float* var_terms, float* grad_proj);
// out[0] = norm_factor * sum(loss_terms), out[1] = mean(var_terms), out[2] = stddev(var_terms) (population).
void sigreg_finish_gpu(const float* loss_terms, const float* var_terms, int num_sketches, float norm_factor,
                       float* out);

// TD-587: LeJEPA's view, loss and pooling arithmetic (LeJEPAEncoder::train_tokens()/train_batch()).
// context_ids = target_ids with [span_start, span_start + span_len) replaced by unk_id.
void lejepa_context_ids_gpu(const int* target_ids, int* context_ids, int seq_len, int span_start, int span_len,
                            int unk_id);
// Over the masked rows [span_start, span_start + span_len) of predicted/target [seq_len, d_model]:
// *loss_out = mean squared error, *cos_out = mean row cosine similarity (0 for a row with a near-zero norm),
// grad_predicted = 2 (predicted - target) / (span_len * d_model) and grad_target = its negative on those rows,
// 0 elsewhere. loss_out and cos_out are device floats (nothing is downloaded).
void lejepa_masked_mse_gpu(const float* predicted, const float* target, int seq_len, int d_model, int span_start,
                           int span_len, float* loss_out, float* cos_out, float* grad_predicted, float* grad_target);
// out_row[j] = mean over the rows of x[:, j] (x is [rows, d_model]): one text's pooled embedding.
void lejepa_mean_pool_gpu(const float* x, int rows, int d_model, float* out_row);
// grad[i, j] += scale * row[j] for every row i of grad [rows, d_model].
void lejepa_add_row_broadcast_gpu(float* grad, int rows, int d_model, const float* row, float scale);

// TD-589: device-accumulating diagnostics, for training that downloads its diagnostics once per epoch.
// acc[0] += (number of |data[i]| < threshold) / size, acc[1] += 1 (an activation-saturation sample).
void matrix_count_below_threshold_accumulate_gpu(const float* data, int size, float threshold, float* acc);
// acc[0] += weight * mean(data[0..n)), acc[1] += count_increment.
void matrix_mean_accumulate_gpu(const float* data, int n, float weight, float count_increment, float* acc);

}  // namespace gpu
}  // namespace adai

#endif  // MATRIX_GPU_SYCL_HPP
