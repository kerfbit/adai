/**
 * @file lejepa_training_pass_test.cpp
 * @brief The LeJEPA world-model pass (LeJEPATrainingPass.cpp) against a local registry, with a tiny
 * world model. TD-566: a lost GPU stops the pass without a checkpoint and gives its files back, a
 * failed batch on a working GPU is skipped, and an epoch in which nothing trained fails the pass.
 *
 * A GPU device loss can't be produced on demand, so the tests make batches fail through
 * LeJEPAPassHooks::before_batch (as a GPU error surfaces, an exception from train_batch()) and
 * decide what the device check answers through LeJEPAPassHooks::device_usable.
 */

#include "LeJEPATrainingPass.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "BPETokenizer.hpp"
#include "DatasetRegistry.hpp"
#include "IncrementalTrainerArgs.hpp"
#include "LeJEPAEncoder.hpp"

namespace fs = std::filesystem;

namespace {

class LeJEPATrainingPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() /
               ("adai_lejepa_pass_" + std::to_string(rd()) + std::to_string(rd()));
        fs::create_directories(dir_);

        // 6 pairs → 12 texts → 6 batches of 2 per epoch.
        const std::vector<std::pair<std::string, std::string>> pairs = {
            {"the cat sat on the warm mat today", "the dog ran to the old red barn"},
            {"a bird sang in the tall green tree", "the sun rose over the quiet hills"},
            {"we walked along the river after lunch", "the boat drifted slowly past the dock"},
            {"she read a book by the open window", "he cooked dinner for the whole family"},
            {"rain fell softly on the tin roof", "the children laughed in the small yard"},
            {"an owl watched from the dark branch", "the wind moved through the long grass"}};
        data_ = (dir_ / "pairs.jsonl").string();
        std::ofstream out(data_);
        std::vector<std::string> texts;
        for (const auto& [in, resp] : pairs) {
            out << "{\"input\":\"" << in << "\",\"response\":\"" << resp << "\"}\n";
            texts.push_back(in);
            texts.push_back(resp);
        }
        out.close();

        vocab_ = (dir_ / "vocab").string();
        BPETokenizer tok;
        tok.build_vocab(texts, 120);
        tok.save_vocab(vocab_);

        svc_.session_dir = (dir_ / "sessions").string();
        svc_.registry_server_url.clear();
        svc_.name_service_url.clear();
        svc_.metrics_server_url.clear();
        svc_.run_id.clear();
        svc_.dataset_kind = "world_model";
        svc_.world_model_d_model = 64;
        svc_.world_model_num_layers = 1;
        svc_.world_model_num_heads = 2;
        svc_.world_model_d_ff = 128;
        svc_.world_model_sigreg_num_sketches = 8;
        svc_.world_model_sigreg_batch_size = 2;
        svc_.max_seq_length = 16;
        svc_.auto_save_enabled = false;
        world_model_dir_ = (dir_ / "world_model_checkpoint").string();

        DatasetRegistry reg(DatasetRegistry::make_config(svc_));
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data_));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    int run(const adai::LeJEPAPassHooks& hooks, int epochs = 1) {
        return adai::run_lejepa_training_pass(svc_, vocab_, world_model_dir_, epochs, nullptr,
                                              hooks);
    }

    bool trained() {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_));
        reg.load_registry();
        const auto files = reg.trained_files();
        return std::find(files.begin(), files.end(), data_) != files.end();
    }

    // The file is back in the pool: pending, and claimable again by any run.
    bool pending_and_unclaimed() {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_));
        reg.load_pending_list();
        for (const auto& e : reg.pending_entries()) {
            if (e.path == data_)
                return e.run_id.empty();
        }
        return false;
    }

    fs::path dir_;
    std::string data_, vocab_, world_model_dir_;
    adai::ServiceConfig svc_;
};

}  // namespace

TEST_F(LeJEPATrainingPassTest, APassTrainsChecksAndRecordsItsFiles) {
    EXPECT_EQ(run({}), 0);
    EXPECT_TRUE(LeJEPAEncoder::has_checkpoint(world_model_dir_));
    EXPECT_TRUE(trained());
}

// A batch that fails on a working GPU is skipped, as before; the device is checked once for it.
TEST_F(LeJEPATrainingPassTest, AFailedBatchOnAWorkingGpuIsSkipped) {
    int checks = 0;
    adai::LeJEPAPassHooks hooks;
    hooks.device_usable = [&] {
        ++checks;
        return true;
    };
    hooks.before_batch = [](std::size_t i) {
        if (i == 1)
            throw std::runtime_error("level_zero backend failed with error: 39");
    };
    EXPECT_EQ(run(hooks), 0);
    EXPECT_EQ(checks, 1);
    EXPECT_TRUE(trained());
    EXPECT_TRUE(LeJEPAEncoder::has_checkpoint(world_model_dir_));
}

// TD-566: a failed batch on a lost GPU stops the pass at once, writes no checkpoint and gives the
// files back. It used to skip every remaining batch and then "complete", marking the files trained.
TEST_F(LeJEPATrainingPassTest, ALostGpuStopsThePassAndReleasesItsFiles) {
    int checks = 0;
    std::size_t batches_started = 0;
    adai::LeJEPAPassHooks hooks;
    hooks.device_usable = [&] {
        ++checks;
        return false;
    };
    hooks.before_batch = [&](std::size_t i) {
        ++batches_started;
        if (i >= 2)
            throw std::runtime_error(
                "level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)");
    };
    EXPECT_EQ(run(hooks, 2), adai::kResumeExitFailed);
    EXPECT_EQ(checks, 1) << "one check, at the first failure";
    EXPECT_EQ(batches_started, 3u) << "no batch after the failure";
    EXPECT_FALSE(LeJEPAEncoder::has_checkpoint(world_model_dir_))
        << "no checkpoint from a lost device";
    EXPECT_FALSE(trained());
    EXPECT_TRUE(pending_and_unclaimed());
}

// TD-566: an epoch in which no batch trained fails the pass (as for the chatbot). Here epoch 1
// trains and every batch of epoch 2 fails on a working GPU; the pass used to finish and record the
// files as trained.
TEST_F(LeJEPATrainingPassTest, AnEpochThatTrainsNothingFailsThePass) {
    std::size_t batches = 0;
    adai::LeJEPAPassHooks hooks;
    hooks.device_usable = [] { return true; };
    hooks.before_batch = [&](std::size_t) {
        if (++batches > 6)
            throw std::runtime_error("every batch of epoch 2 fails");  // 6 batches per epoch
    };
    EXPECT_EQ(run(hooks, 2), adai::kResumeExitFailed);
    EXPECT_FALSE(LeJEPAEncoder::has_checkpoint(world_model_dir_));
    EXPECT_FALSE(trained());
    EXPECT_TRUE(pending_and_unclaimed());
}
