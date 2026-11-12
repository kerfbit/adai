#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-11-12

// The LeJEPA world-model training pass (`incremental_trainer --objective=lejepa train|resume`),
// moved out of IncrementalTrainingTool.cpp (TD-566) so it can be tested. See LeJEPATrainingPass.md
// for the pass.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "Config.hpp"
#include "TrainerControlState.hpp"

namespace adai {

/// Ways to observe or steer a LeJEPA pass. Both are empty in production.
struct LeJEPAPassHooks {
    /// TD-566: after a batch fails, whether the GPU still runs work. Empty means the real check
    /// (GPUManager::device_usable() when a GPU is in use; always true in a CPU build).
    std::function<bool()> device_usable;
    /// Called before each batch with its index within the epoch; a test throws from it to make that
    /// batch fail as a GPU error would.
    std::function<void(std::size_t)> before_batch;
};

/// One LeJEPA pass: claims pending files, trains the world model in `world_model_dir` for `epochs`
/// epochs, checkpoints, and marks the files trained. Returns 0 (trained), kResumeExitNothingPending
/// (3) or 1 (failed: registry, download, checkpoint, a lost GPU, or an epoch that trained nothing;
/// the files are released). `control`, when given (a trainer_service pass or a SIGTERM drain), is
/// the per-pass admin state: progress is reported there, /admin/checkpoint writes a checkpoint, and
/// a pause stops the pass at the next batch, checkpoints, releases the files and returns 1.
int run_lejepa_training_pass(const ServiceConfig& svc_config, const std::string& default_vocab,
                             const std::string& world_model_dir, int epochs,
                             const std::shared_ptr<TrainerControlState>& control = nullptr,
                             const LeJEPAPassHooks& hooks = {});

}  // namespace adai
