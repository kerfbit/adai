#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Forward declarations to avoid pulling in heavy headers
class EncoderDecoderModel;
class Matrix;

/**
 * ModelSerializer — HuggingFace SafeTensors import/export
 *
 * Provides bi-directional weight interoperability between ADAI's native binary
 * format and the HuggingFace SafeTensors ecosystem.
 *
 * Tensor naming follows the T5ForConditionalGeneration convention.
 * Weight matrices are transposed on both export and import because ADAI stores
 * them as right-multiplied operands (output = input @ W) while HuggingFace /
 * PyTorch uses the transposed convention (output = F.linear(input, W) = input @ W.T).
 *
 * File layout written/expected:
 *   <dir>/model.safetensors   — tensors in SafeTensors binary format
 *   <dir>/config.json         — architecture hyperparameters (T5-style schema)
 *
 * Reference: https://huggingface.co/docs/safetensors
 */
class ModelSerializer {
   public:
    /**
     * Export a fully-loaded EncoderDecoderModel to SafeTensors format.
     *
     * Writes:
     *   <output_dir>/model.safetensors
     *   <output_dir>/config.json
     *
     * @param model        Loaded ADAI model (weights must already be in memory)
     * @param output_dir   Directory to write output files; created if absent
     * @throws std::runtime_error on I/O failure or if output_dir cannot be made
     */
    static void export_safetensors(const EncoderDecoderModel& model, const std::string& output_dir);

    /**
     * Import weights from a HuggingFace SafeTensors directory into a live model.
     *
     * Reads:
     *   <input_dir>/model.safetensors
     *   <input_dir>/config.json
     *
     * The model's architecture (d_model, num_heads, d_ff, layer counts) must
     * already match the config.json values; mismatches throw std::runtime_error.
     * Unknown tensor keys in the file are silently ignored (e.g. T5 position
     * biases that ADAI does not use).
     *
     * @param model     Live ADAI model whose weights will be overwritten
     * @param input_dir Directory containing model.safetensors and config.json
     * @throws std::runtime_error on I/O failure, architecture mismatch, or
     *         missing required tensor
     */
    static void import_safetensors(EncoderDecoderModel& model, const std::string& input_dir);

    // ── Low-level SafeTensors protocol (exposed for testing) ─────────────────

    struct TensorDescriptor {
        std::string name;
        std::string dtype;  // e.g. "F32"
        std::vector<int64_t> shape;
        std::vector<float> data;  // always float32
    };

    /**
     * Write a collection of named tensors to a SafeTensors binary file.
     *
     * @param path      Output file path
     * @param tensors   Ordered list of tensors to write
     * @param metadata  Key-value pairs for the __metadata__ field
     */
    static void write_safetensors(const std::string& path,
                                  const std::vector<TensorDescriptor>& tensors,
                                  const std::map<std::string, std::string>& metadata);

    /**
     * Read all tensors from a SafeTensors binary file.
     *
     * @param path  Input file path
     * @return      Map from tensor name to TensorDescriptor (data fully loaded)
     */
    static std::map<std::string, TensorDescriptor> read_safetensors(const std::string& path);
};
