#include "ModelSerializer.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CrossAttention.hpp"
#include "Decoder.hpp"
#include "DecoderBlock.hpp"
#include "EncoderBlock.hpp"
#include "EncoderDecoderModel.hpp"
#include "FeedForward.hpp"
#include "LanguageModelHead.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"
#include "TokenEmbedding.hpp"
#include "encoder.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// SafeTensors is always little-endian.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "ModelSerializer: SafeTensors I/O assumes a little-endian host. "
              "Add byte-swap logic before enabling on big-endian platforms.");

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Transpose a Matrix (ADAI row-major: output = input @ W → HF: output = W.T @ input.T)
Matrix transpose_matrix(const Matrix& m) {
    return m.transpose();
}

// Flatten a 1-row Matrix to a vector (for biases / LayerNorm params).
std::vector<float> flatten(const Matrix& m) {
    std::vector<float> out;
    out.reserve(static_cast<size_t>(m.rows * m.cols));
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out.push_back(m(i, j));
    return out;
}

// Reshape a flat float vector back into a Matrix.
Matrix from_flat(const std::vector<float>& data, int rows, int cols) {
    if (static_cast<int>(data.size()) != rows * cols)
        throw std::runtime_error("ModelSerializer: size mismatch in from_flat");
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = data[static_cast<size_t>(i * cols + j)];
    return m;
}

// Build a TensorDescriptor from a Matrix, optionally transposing it.
ModelSerializer::TensorDescriptor make_td(const std::string& name,
                                          const Matrix& m,
                                          bool do_transpose = false) {
    const Matrix& src = do_transpose ? transpose_matrix(m) : m;
    ModelSerializer::TensorDescriptor td;
    td.name  = name;
    td.dtype = "F32";
    td.shape = {static_cast<int64_t>(src.rows), static_cast<int64_t>(src.cols)};
    td.data  = flatten(src);
    return td;
}

// Build a 1-D TensorDescriptor from a 1-row bias / norm matrix.
ModelSerializer::TensorDescriptor make_td_1d(const std::string& name,
                                             const Matrix& m) {
    ModelSerializer::TensorDescriptor td;
    td.name  = name;
    td.dtype = "F32";
    td.shape = {static_cast<int64_t>(m.rows * m.cols)};
    td.data  = flatten(m);
    return td;
}

// Emit a JSON string where all special characters are escaped.
std::string json_escape(const std::string& s) {
    // nlohmann handles this during serialization; this is only for metadata values.
    return s;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// write_safetensors
// ─────────────────────────────────────────────────────────────────────────────

void ModelSerializer::write_safetensors(const std::string& path,
                                        const std::vector<TensorDescriptor>& tensors,
                                        const std::map<std::string, std::string>& metadata) {
    // Build the JSON header.
    json header;

    // __metadata__
    json meta_obj;
    for (const auto& kv : metadata)
        meta_obj[kv.first] = kv.second;
    header["__metadata__"] = meta_obj;

    // Compute per-tensor byte offsets (relative to start of data region).
    uint64_t offset = 0;
    for (const auto& td : tensors) {
        uint64_t n_bytes = static_cast<uint64_t>(td.data.size()) * sizeof(float);
        json td_entry;
        td_entry["dtype"]        = td.dtype;
        json shape_arr = json::array();
        for (auto s : td.shape) shape_arr.push_back(s);
        td_entry["shape"]        = shape_arr;
        td_entry["data_offsets"] = json::array({offset, offset + n_bytes});
        header[td.name]          = td_entry;
        offset += n_bytes;
    }

    // Serialise header to UTF-8 string and pad to an 8-byte boundary.
    std::string header_str = header.dump();
    while (header_str.size() % 8 != 0) header_str += ' ';
    uint64_t header_size = static_cast<uint64_t>(header_str.size());

    // Write file.
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("ModelSerializer: cannot open for writing: " + path);

    // 8-byte little-endian header length.
    f.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    // JSON header.
    f.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    // Tensor data.
    for (const auto& td : tensors)
        f.write(reinterpret_cast<const char*>(td.data.data()),
                static_cast<std::streamsize>(td.data.size() * sizeof(float)));

    if (!f.good())
        throw std::runtime_error("ModelSerializer: write error: " + path);
}

// ─────────────────────────────────────────────────────────────────────────────
// read_safetensors
// ─────────────────────────────────────────────────────────────────────────────

std::map<std::string, ModelSerializer::TensorDescriptor>
ModelSerializer::read_safetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("ModelSerializer: cannot open for reading: " + path);

    // Read 8-byte header length.
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!f.good() || header_size == 0)
        throw std::runtime_error("ModelSerializer: invalid SafeTensors header size in " + path);

    // Read JSON header.
    std::string header_str(header_size, '\0');
    f.read(header_str.data(), static_cast<std::streamsize>(header_size));
    if (!f.good())
        throw std::runtime_error("ModelSerializer: failed to read header from " + path);

    json header = json::parse(header_str);

    // The data region starts immediately after the header.
    std::streampos data_start = f.tellg();

    std::map<std::string, TensorDescriptor> result;

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string& key = it.key();
        if (key == "__metadata__") continue;

        const json& entry = it.value();

        TensorDescriptor td;
        td.name  = key;
        td.dtype = entry.value("dtype", "F32");

        for (auto& s : entry.at("shape"))
            td.shape.push_back(s.get<int64_t>());

        auto offsets = entry.at("data_offsets");
        uint64_t start_off = offsets[0].get<uint64_t>();
        uint64_t end_off   = offsets[1].get<uint64_t>();
        uint64_t n_bytes   = end_off - start_off;
        uint64_t n_floats  = n_bytes / sizeof(float);

        td.data.resize(n_floats);
        f.seekg(data_start + static_cast<std::streamoff>(start_off));
        f.read(reinterpret_cast<char*>(td.data.data()),
               static_cast<std::streamsize>(n_bytes));
        if (!f.good())
            throw std::runtime_error("ModelSerializer: failed to read tensor '" +
                                     key + "' from " + path);

        result[key] = std::move(td);
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// export_safetensors
// ─────────────────────────────────────────────────────────────────────────────

void ModelSerializer::export_safetensors(const EncoderDecoderModel& model,
                                         const std::string& output_dir) {
    fs::create_directories(output_dir);

    const int enc_layers = model.get_encoder_layers();
    const int dec_layers = model.get_decoder_layers();
    const int d_model    = model.get_d_model();
    const int vocab_size = model.get_vocab_size();
    const int num_heads  = model.get_num_heads();
    const int d_ff       = model.get_d_ff();
    const int max_seq    = model.get_max_seq_length();

    // Non-const access needed to reach sub-components through non-const getters.
    // Cast away top-level const only to traverse the component tree.
    EncoderDecoderModel& mutable_model = const_cast<EncoderDecoderModel&>(model);

    LLMEncoder*        enc      = mutable_model.get_encoder();
    LLMDecoder*        dec      = mutable_model.get_decoder();
    LanguageModelHead* lm_head  = mutable_model.get_lm_head();

    std::vector<TensorDescriptor> tensors;

    // ── Shared embedding ──────────────────────────────────────────────────────
    // encoder and decoder share the same token embedding; use encoder's.
    // Shape: [vocab_size, d_model] — no transpose needed.
    {
        TokenEmbedding* emb = enc->get_token_embedding();
        tensors.push_back(make_td("model.shared.weight", emb->get_embeddings(), false));
    }

    // ── Encoder layers ────────────────────────────────────────────────────────
    for (int i = 0; i < enc_layers; ++i) {
        EncoderBlock* blk  = enc->get_encoder_block(i);
        MultiHeadAttention* sa = blk->get_self_attention();
        FeedForward*        ff = blk->get_feed_forward();
        LayerNorm*         ln1 = blk->get_norm1();
        LayerNorm*         ln2 = blk->get_norm2();

        const std::string b = "model.encoder.block." + std::to_string(i);

        // Self-attention weights — transposed
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.q.weight", sa->get_Wq(), true));
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.k.weight", sa->get_Wk(), true));
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.v.weight", sa->get_Wv(), true));
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.o.weight", sa->get_Wo(), true));
        // Self-attention layer norm
        tensors.push_back(make_td_1d(b + ".layer.0.layer_norm.weight", ln1->get_gamma()));
        tensors.push_back(make_td_1d(b + ".layer.0.layer_norm.bias",   ln1->get_beta()));

        // Feed-forward — W1: [d_model,d_ff] → HF [d_ff,d_model]; W2: [d_ff,d_model] → HF [d_model,d_ff]
        tensors.push_back(make_td(b + ".layer.1.DenseReluDense.wi.weight", ff->get_W1(), true));
        tensors.push_back(make_td_1d(b + ".layer.1.DenseReluDense.wi.bias", ff->get_b1()));
        tensors.push_back(make_td(b + ".layer.1.DenseReluDense.wo.weight", ff->get_W2(), true));
        tensors.push_back(make_td_1d(b + ".layer.1.DenseReluDense.wo.bias", ff->get_b2()));
        // Feed-forward layer norm
        tensors.push_back(make_td_1d(b + ".layer.1.layer_norm.weight", ln2->get_gamma()));
        tensors.push_back(make_td_1d(b + ".layer.1.layer_norm.bias",   ln2->get_beta()));
    }

    // Encoder final norm
    if (LayerNorm* fn = enc->get_final_norm()) {
        tensors.push_back(make_td_1d("model.encoder.final_layer_norm.weight", fn->get_gamma()));
        tensors.push_back(make_td_1d("model.encoder.final_layer_norm.bias",   fn->get_beta()));
    }

    // ── Decoder layers ────────────────────────────────────────────────────────
    for (int j = 0; j < dec_layers; ++j) {
        DecoderBlock*       blk  = dec->get_decoder_block(j);
        MultiHeadAttention* sa   = blk->get_self_attention();
        CrossAttention*     ca   = blk->get_cross_attention();
        FeedForward*        ff   = blk->get_feed_forward();
        LayerNorm*          ln1  = blk->get_norm1();
        LayerNorm*          ln2  = blk->get_norm2();
        LayerNorm*          ln3  = blk->get_norm3();

        const std::string b = "model.decoder.block." + std::to_string(j);

        // Self-attention
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.q.weight", sa->get_Wq(), true));
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.k.weight", sa->get_Wk(), true));
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.v.weight", sa->get_Wv(), true));
        tensors.push_back(make_td(b + ".layer.0.SelfAttention.o.weight", sa->get_Wo(), true));
        tensors.push_back(make_td_1d(b + ".layer.0.layer_norm.weight", ln1->get_gamma()));
        tensors.push_back(make_td_1d(b + ".layer.0.layer_norm.bias",   ln1->get_beta()));

        // Cross-attention
        tensors.push_back(make_td(b + ".layer.1.EncDecAttention.q.weight", ca->get_Wq(), true));
        tensors.push_back(make_td(b + ".layer.1.EncDecAttention.k.weight", ca->get_Wk(), true));
        tensors.push_back(make_td(b + ".layer.1.EncDecAttention.v.weight", ca->get_Wv(), true));
        tensors.push_back(make_td(b + ".layer.1.EncDecAttention.o.weight", ca->get_Wo(), true));
        tensors.push_back(make_td_1d(b + ".layer.1.layer_norm.weight", ln2->get_gamma()));
        tensors.push_back(make_td_1d(b + ".layer.1.layer_norm.bias",   ln2->get_beta()));

        // Feed-forward
        tensors.push_back(make_td(b + ".layer.2.DenseReluDense.wi.weight", ff->get_W1(), true));
        tensors.push_back(make_td_1d(b + ".layer.2.DenseReluDense.wi.bias", ff->get_b1()));
        tensors.push_back(make_td(b + ".layer.2.DenseReluDense.wo.weight", ff->get_W2(), true));
        tensors.push_back(make_td_1d(b + ".layer.2.DenseReluDense.wo.bias", ff->get_b2()));
        tensors.push_back(make_td_1d(b + ".layer.2.layer_norm.weight", ln3->get_gamma()));
        tensors.push_back(make_td_1d(b + ".layer.2.layer_norm.bias",   ln3->get_beta()));
    }

    // Decoder final norm
    if (LayerNorm* fn = dec->get_final_norm()) {
        tensors.push_back(make_td_1d("model.decoder.final_layer_norm.weight", fn->get_gamma()));
        tensors.push_back(make_td_1d("model.decoder.final_layer_norm.bias",   fn->get_beta()));
    }

    // ── LM Head ───────────────────────────────────────────────────────────────
    // W_output: [d_model, vocab_size] → HF [vocab_size, d_model]
    tensors.push_back(make_td("lm_head.weight", lm_head->get_W_output(), true));
    tensors.push_back(make_td_1d("lm_head.bias", lm_head->get_bias()));

    // ── Write SafeTensors file ────────────────────────────────────────────────
    std::map<std::string, std::string> metadata = {
        {"format",       "pt"},
        {"adai_version", "1.0"},
        {"model_type",   "t5"}
    };
    write_safetensors(output_dir + "/model.safetensors", tensors, metadata);

    // ── Write config.json ─────────────────────────────────────────────────────
    json cfg;
    cfg["architectures"]           = json::array({"T5ForConditionalGeneration"});
    cfg["model_type"]              = "t5";
    cfg["adai_native"]             = true;
    cfg["vocab_size"]              = vocab_size;
    cfg["d_model"]                 = d_model;
    cfg["d_ff"]                    = d_ff;
    cfg["d_kv"]                    = d_model / num_heads;
    cfg["num_heads"]               = num_heads;
    cfg["num_layers"]              = enc_layers;
    cfg["num_decoder_layers"]      = dec_layers;
    cfg["max_length"]              = max_seq;
    cfg["decoder_start_token_id"]  = model.get_bos_token_id();
    cfg["eos_token_id"]            = model.get_eos_token_id();
    cfg["pad_token_id"]            = model.get_pad_token_id();
    cfg["feed_forward_proj"]       = "relu";
    cfg["torch_dtype"]             = "float32";

    std::ofstream cfg_f(output_dir + "/config.json");
    if (!cfg_f.is_open())
        throw std::runtime_error("ModelSerializer: cannot write config.json to " + output_dir);
    cfg_f << cfg.dump(2) << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// import_safetensors
// ─────────────────────────────────────────────────────────────────────────────

void ModelSerializer::import_safetensors(EncoderDecoderModel& model,
                                         const std::string& input_dir) {
    // ── Read and validate config.json ─────────────────────────────────────────
    std::ifstream cfg_f(input_dir + "/config.json");
    if (!cfg_f.is_open())
        throw std::runtime_error("ModelSerializer: cannot open " + input_dir + "/config.json");

    json cfg;
    cfg_f >> cfg;

    const int file_d_model    = cfg.value("d_model",           -1);
    const int file_enc_layers = cfg.value("num_layers",        -1);
    const int file_dec_layers = cfg.value("num_decoder_layers",-1);
    const int file_num_heads  = cfg.value("num_heads",         -1);
    const int file_d_ff       = cfg.value("d_ff",              -1);

    if (file_d_model    != model.get_d_model()         ||
        file_enc_layers != model.get_encoder_layers()  ||
        file_dec_layers != model.get_decoder_layers()  ||
        file_num_heads  != model.get_num_heads()       ||
        file_d_ff       != model.get_d_ff()) {
        throw std::runtime_error(
            "ModelSerializer: architecture mismatch between config.json and live model. "
            "file=(d_model=" + std::to_string(file_d_model) +
            " enc=" + std::to_string(file_enc_layers) +
            " dec=" + std::to_string(file_dec_layers) +
            " heads=" + std::to_string(file_num_heads) +
            " d_ff=" + std::to_string(file_d_ff) + ") "
            "model=(d_model=" + std::to_string(model.get_d_model()) +
            " enc=" + std::to_string(model.get_encoder_layers()) +
            " dec=" + std::to_string(model.get_decoder_layers()) +
            " heads=" + std::to_string(model.get_num_heads()) +
            " d_ff=" + std::to_string(model.get_d_ff()) + ")");
    }

    const int d_model    = model.get_d_model();
    const int enc_layers = model.get_encoder_layers();
    const int dec_layers = model.get_decoder_layers();
    const int vocab_size = model.get_vocab_size();
    const int d_ff       = model.get_d_ff();

    // ── Read SafeTensors file ─────────────────────────────────────────────────
    auto tensors = read_safetensors(input_dir + "/model.safetensors");

    // Helper: look up a required tensor and verify its total element count.
    auto require = [&](const std::string& name, int expected_elems) -> const TensorDescriptor& {
        auto it = tensors.find(name);
        if (it == tensors.end())
            throw std::runtime_error("ModelSerializer: required tensor missing: " + name);
        const TensorDescriptor& td = it->second;
        int actual = 1;
        for (auto s : td.shape) actual *= static_cast<int>(s);
        if (actual != expected_elems)
            throw std::runtime_error("ModelSerializer: tensor '" + name + "' has " +
                                     std::to_string(actual) + " elements, expected " +
                                     std::to_string(expected_elems));
        return td;
    };

    // Helper: load a 2-D tensor and transpose it back to ADAI convention.
    auto load2d_transposed = [&](const std::string& name, int hf_rows, int hf_cols) -> Matrix {
        const auto& td = require(name, hf_rows * hf_cols);
        Matrix hf = from_flat(td.data, hf_rows, hf_cols);
        return transpose_matrix(hf);
    };

    // Helper: load a 1-D tensor into a 1-row Matrix.
    auto load1d = [&](const std::string& name, int n) -> Matrix {
        const auto& td = require(name, n);
        return from_flat(td.data, 1, n);
    };

    LLMEncoder*        enc     = model.get_encoder();
    LLMDecoder*        dec_ptr = model.get_decoder();
    LanguageModelHead* lm_head = model.get_lm_head();

    // ── Shared embedding ──────────────────────────────────────────────────────
    {
        Matrix emb = from_flat(require("model.shared.weight", vocab_size * d_model).data,
                               vocab_size, d_model);
        enc->get_token_embedding()->set_embeddings(emb);
        // Decoder uses its own embedding; set it to the same values.
        if (dec_ptr->get_token_embedding())
            dec_ptr->get_token_embedding()->set_embeddings(emb);
    }

    // ── Encoder layers ────────────────────────────────────────────────────────
    for (int i = 0; i < enc_layers; ++i) {
        EncoderBlock*       blk = enc->get_encoder_block(i);
        MultiHeadAttention* sa  = blk->get_self_attention();
        FeedForward*        ff  = blk->get_feed_forward();
        LayerNorm*          ln1 = blk->get_norm1();
        LayerNorm*          ln2 = blk->get_norm2();

        const std::string b = "model.encoder.block." + std::to_string(i);

        sa->set_Wq(load2d_transposed(b + ".layer.0.SelfAttention.q.weight", d_model, d_model));
        sa->set_Wk(load2d_transposed(b + ".layer.0.SelfAttention.k.weight", d_model, d_model));
        sa->set_Wv(load2d_transposed(b + ".layer.0.SelfAttention.v.weight", d_model, d_model));
        sa->set_Wo(load2d_transposed(b + ".layer.0.SelfAttention.o.weight", d_model, d_model));
        ln1->set_gamma(load1d(b + ".layer.0.layer_norm.weight", d_model));
        ln1->set_beta (load1d(b + ".layer.0.layer_norm.bias",   d_model));

        // W1: HF [d_ff, d_model] → ADAI [d_model, d_ff] (transpose)
        ff->set_W1(load2d_transposed(b + ".layer.1.DenseReluDense.wi.weight", d_ff, d_model));
        ff->set_b1(load1d(b + ".layer.1.DenseReluDense.wi.bias", d_ff));
        // W2: HF [d_model, d_ff] → ADAI [d_ff, d_model] (transpose)
        ff->set_W2(load2d_transposed(b + ".layer.1.DenseReluDense.wo.weight", d_model, d_ff));
        ff->set_b2(load1d(b + ".layer.1.DenseReluDense.wo.bias", d_model));
        ln2->set_gamma(load1d(b + ".layer.1.layer_norm.weight", d_model));
        ln2->set_beta (load1d(b + ".layer.1.layer_norm.bias",   d_model));
    }

    // Encoder final norm (optional — tolerate absence for 3rd-party checkpoints)
    if (LayerNorm* fn = enc->get_final_norm()) {
        auto it_w = tensors.find("model.encoder.final_layer_norm.weight");
        auto it_b = tensors.find("model.encoder.final_layer_norm.bias");
        if (it_w != tensors.end() && it_b != tensors.end()) {
            fn->set_gamma(load1d("model.encoder.final_layer_norm.weight", d_model));
            fn->set_beta (load1d("model.encoder.final_layer_norm.bias",   d_model));
        }
    }

    // ── Decoder layers ────────────────────────────────────────────────────────
    for (int j = 0; j < dec_layers; ++j) {
        DecoderBlock*       blk = dec_ptr->get_decoder_block(j);
        MultiHeadAttention* sa  = blk->get_self_attention();
        CrossAttention*     ca  = blk->get_cross_attention();
        FeedForward*        ff  = blk->get_feed_forward();
        LayerNorm*          ln1 = blk->get_norm1();
        LayerNorm*          ln2 = blk->get_norm2();
        LayerNorm*          ln3 = blk->get_norm3();

        const std::string b = "model.decoder.block." + std::to_string(j);

        // Self-attention
        sa->set_Wq(load2d_transposed(b + ".layer.0.SelfAttention.q.weight", d_model, d_model));
        sa->set_Wk(load2d_transposed(b + ".layer.0.SelfAttention.k.weight", d_model, d_model));
        sa->set_Wv(load2d_transposed(b + ".layer.0.SelfAttention.v.weight", d_model, d_model));
        sa->set_Wo(load2d_transposed(b + ".layer.0.SelfAttention.o.weight", d_model, d_model));
        ln1->set_gamma(load1d(b + ".layer.0.layer_norm.weight", d_model));
        ln1->set_beta (load1d(b + ".layer.0.layer_norm.bias",   d_model));

        // Cross-attention
        ca->set_Wq(load2d_transposed(b + ".layer.1.EncDecAttention.q.weight", d_model, d_model));
        ca->set_Wk(load2d_transposed(b + ".layer.1.EncDecAttention.k.weight", d_model, d_model));
        ca->set_Wv(load2d_transposed(b + ".layer.1.EncDecAttention.v.weight", d_model, d_model));
        ca->set_Wo(load2d_transposed(b + ".layer.1.EncDecAttention.o.weight", d_model, d_model));
        ln2->set_gamma(load1d(b + ".layer.1.layer_norm.weight", d_model));
        ln2->set_beta (load1d(b + ".layer.1.layer_norm.bias",   d_model));

        // Feed-forward
        ff->set_W1(load2d_transposed(b + ".layer.2.DenseReluDense.wi.weight", d_ff, d_model));
        ff->set_b1(load1d(b + ".layer.2.DenseReluDense.wi.bias", d_ff));
        ff->set_W2(load2d_transposed(b + ".layer.2.DenseReluDense.wo.weight", d_model, d_ff));
        ff->set_b2(load1d(b + ".layer.2.DenseReluDense.wo.bias", d_model));
        ln3->set_gamma(load1d(b + ".layer.2.layer_norm.weight", d_model));
        ln3->set_beta (load1d(b + ".layer.2.layer_norm.bias",   d_model));
    }

    // Decoder final norm
    if (LayerNorm* fn = dec_ptr->get_final_norm()) {
        auto it_w = tensors.find("model.decoder.final_layer_norm.weight");
        auto it_b = tensors.find("model.decoder.final_layer_norm.bias");
        if (it_w != tensors.end() && it_b != tensors.end()) {
            fn->set_gamma(load1d("model.decoder.final_layer_norm.weight", d_model));
            fn->set_beta (load1d("model.decoder.final_layer_norm.bias",   d_model));
        }
    }

    // ── LM Head ───────────────────────────────────────────────────────────────
    // HF lm_head.weight: [vocab_size, d_model] → ADAI [d_model, vocab_size] (transpose)
    lm_head->set_W_output(load2d_transposed("lm_head.weight", vocab_size, d_model));
    lm_head->set_bias(load1d("lm_head.bias", vocab_size));
}
