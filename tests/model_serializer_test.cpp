#include <../gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../src/EncoderDecoderModel.hpp"
#include "../src/Matrix.hpp"
#include "../src/ModelSerializer.hpp"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool matrices_approx_equal(const Matrix& a, const Matrix& b,
                                   float eps = 1e-6f) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            if (std::abs(a(i, j) - b(i, j)) > eps) return false;
    return true;
}

// A temporary directory that cleans itself up.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& prefix) {
        path = fs::temp_directory_path() / (prefix + std::to_string(
            std::hash<std::string>{}(std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()))));
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
};

// Small model parameters used throughout — kept tiny for speed.
static constexpr int VOCAB   = 64;
static constexpr int D_MODEL = 32;
static constexpr int HEADS   = 2;
static constexpr int D_FF    = 64;
static constexpr int ENC_L   = 2;
static constexpr int DEC_L   = 2;
static constexpr int MAX_SEQ = 32;

// ─────────────────────────────────────────────────────────────────────────────
// 1. SafeTensors protocol: write then read a known tensor set
// ─────────────────────────────────────────────────────────────────────────────

TEST(ModelSerializerTest, SafeTensorsRoundTripProtocol) {
    TempDir tmp("st_proto_");
    std::string path = (tmp.path / "test.safetensors").string();

    // Build two small tensors.
    ModelSerializer::TensorDescriptor t1, t2;
    t1.name  = "foo.weight";
    t1.dtype = "F32";
    t1.shape = {2, 3};
    t1.data  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

    t2.name  = "bar.bias";
    t2.dtype = "F32";
    t2.shape = {4};
    t2.data  = {0.1f, 0.2f, 0.3f, 0.4f};

    ModelSerializer::write_safetensors(path, {t1, t2},
                                       {{"format", "pt"}, {"test", "yes"}});

    auto result = ModelSerializer::read_safetensors(path);
    ASSERT_EQ(result.count("foo.weight"), 1u);
    ASSERT_EQ(result.count("bar.bias"),   1u);

    // Verify shapes
    EXPECT_EQ(result["foo.weight"].shape, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(result["bar.bias"].shape,   (std::vector<int64_t>{4}));

    // Verify data values
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(result["foo.weight"].data[i], t1.data[i]);
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(result["bar.bias"].data[i], t2.data[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Full model round-trip: export then import, all weights match
// ─────────────────────────────────────────────────────────────────────────────

TEST(ModelSerializerTest, FullModelRoundTrip) {
    TempDir tmp("st_roundtrip_");
    std::string dir = tmp.path.string();

    // Build source model with random weights.
    EncoderDecoderModel src(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);

    // Snapshot some weights before export.
    Matrix orig_enc0_Wq =
        src.get_encoder()->get_encoder_block(0)->get_self_attention()->get_Wq();
    Matrix orig_dec0_Wq =
        src.get_decoder()->get_decoder_block(0)->get_self_attention()->get_Wq();
    Matrix orig_lm_W = src.get_lm_head()->get_W_output();

    // Export.
    ASSERT_NO_THROW(ModelSerializer::export_safetensors(src, dir));
    EXPECT_TRUE(fs::exists(dir + "/model.safetensors"));
    EXPECT_TRUE(fs::exists(dir + "/config.json"));

    // Build a second (freshly-initialised) model with identical architecture.
    EncoderDecoderModel dst(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);

    // Import into destination.
    ASSERT_NO_THROW(ModelSerializer::import_safetensors(dst, dir));

    // Verify encoder layer 0 self-attention Wq matches original.
    Matrix dst_enc0_Wq =
        dst.get_encoder()->get_encoder_block(0)->get_self_attention()->get_Wq();
    EXPECT_TRUE(matrices_approx_equal(orig_enc0_Wq, dst_enc0_Wq))
        << "Encoder block 0 Wq mismatch after round-trip";

    // Verify decoder layer 0 self-attention Wq.
    Matrix dst_dec0_Wq =
        dst.get_decoder()->get_decoder_block(0)->get_self_attention()->get_Wq();
    EXPECT_TRUE(matrices_approx_equal(orig_dec0_Wq, dst_dec0_Wq))
        << "Decoder block 0 Wq mismatch after round-trip";

    // Verify LM head weight.
    Matrix dst_lm_W = dst.get_lm_head()->get_W_output();
    EXPECT_TRUE(matrices_approx_equal(orig_lm_W, dst_lm_W))
        << "LM head W_output mismatch after round-trip";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Transpose correctness: (W.T).T == W
// ─────────────────────────────────────────────────────────────────────────────

TEST(ModelSerializerTest, TransposeIsItsOwnInverse) {
    EncoderDecoderModel model(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);

    // Test with attention weight (square) and FF weight (rectangular).
    const Matrix& Wq = model.get_encoder()
                           ->get_encoder_block(0)
                           ->get_self_attention()
                           ->get_Wq();
    EXPECT_TRUE(matrices_approx_equal(Wq, Wq.transpose().transpose()))
        << "transpose(transpose(Wq)) != Wq";

    const Matrix& W1 = model.get_encoder()
                           ->get_encoder_block(0)
                           ->get_feed_forward()
                           ->get_W1();
    EXPECT_TRUE(matrices_approx_equal(W1, W1.transpose().transpose()))
        << "transpose(transpose(W1)) != W1";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Config mismatch throws on import
// ─────────────────────────────────────────────────────────────────────────────

TEST(ModelSerializerTest, ConfigMismatchThrows) {
    TempDir tmp("st_mismatch_");
    std::string dir = tmp.path.string();

    // Export a small model.
    EncoderDecoderModel src(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);
    ModelSerializer::export_safetensors(src, dir);

    // Try to import into a model with a different d_model.
    EncoderDecoderModel wrong(VOCAB, D_MODEL * 2, ENC_L, DEC_L, HEADS,
                               D_FF * 2, MAX_SEQ);
    EXPECT_THROW(ModelSerializer::import_safetensors(wrong, dir),
                 std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Unknown tensor keys are silently ignored
// ─────────────────────────────────────────────────────────────────────────────

TEST(ModelSerializerTest, UnknownTensorKeysIgnored) {
    TempDir tmp("st_unknown_");
    std::string dir = tmp.path.string();

    // Export a valid model.
    EncoderDecoderModel src(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);
    ModelSerializer::export_safetensors(src, dir);

    // Manually inject an extra tensor into the file by rebuilding it.
    auto tensors = ModelSerializer::read_safetensors(dir + "/model.safetensors");

    ModelSerializer::TensorDescriptor extra;
    extra.name  = "relative_attention_bias.weight";  // T5-style key ADAI doesn't use
    extra.dtype = "F32";
    extra.shape = {8, 32};
    extra.data.assign(8 * 32, 0.0f);

    std::vector<ModelSerializer::TensorDescriptor> all_tensors;
    for (auto& [name, td] : tensors) all_tensors.push_back(td);
    all_tensors.push_back(extra);

    ModelSerializer::write_safetensors(dir + "/model.safetensors", all_tensors,
                                       {{"format", "pt"}});

    // Import should succeed without throwing.
    EncoderDecoderModel dst(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);
    EXPECT_NO_THROW(ModelSerializer::import_safetensors(dst, dir));
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Missing required tensor throws
// ─────────────────────────────────────────────────────────────────────────────

TEST(ModelSerializerTest, MissingRequiredTensorThrows) {
    TempDir tmp("st_missing_");
    std::string dir = tmp.path.string();

    // Export a valid model.
    EncoderDecoderModel src(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);
    ModelSerializer::export_safetensors(src, dir);

    // Remove a required tensor (lm_head.weight) by rewriting the file without it.
    auto tensors = ModelSerializer::read_safetensors(dir + "/model.safetensors");
    std::vector<ModelSerializer::TensorDescriptor> pruned;
    for (auto& [name, td] : tensors)
        if (name != "lm_head.weight") pruned.push_back(td);

    ModelSerializer::write_safetensors(dir + "/model.safetensors", pruned,
                                       {{"format", "pt"}});

    EncoderDecoderModel dst(VOCAB, D_MODEL, ENC_L, DEC_L, HEADS, D_FF, MAX_SEQ);
    EXPECT_THROW(ModelSerializer::import_safetensors(dst, dir), std::runtime_error);
}
