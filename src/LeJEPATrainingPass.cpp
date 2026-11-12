// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-12

// TD-566: the LeJEPA world-model pass, moved out of IncrementalTrainingTool.cpp so it can be tested
// (lejepaTrainingPassTests), and given the chatbot pass's GPU-loss handling.

#include "LeJEPATrainingPass.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <thread>

#include "BPETokenizer.hpp"
#include "DatasetRegistry.hpp"
#include "IncrementalTrainerArgs.hpp"
#include "LeJEPAEncoder.hpp"
#include "Logger.hpp"
#include "MetricsPushClient.hpp"
#include "Optimizer.hpp"
#include "PassHeartbeat.hpp"
#include "StartupSweep.hpp"
#include "TrainingPassFiles.hpp"
#include "gpu/GPUUtils.hpp"

namespace adai {

namespace fs = std::filesystem;

namespace {

// ── LeJEPA world-model pretraining pass (TD-183) ────────────────────────────────
//
// Runs one `--objective=lejepa train` pass: acquires pending files via the SAME dataset
// registry/FTP-download machinery the chatbot objective uses (the "train" command's own
// pre-fork pending-count check, shared by both objectives — see the call site in main()), but
// trains a standalone LeJEPAEncoder world model via LeJEPAEncoder::train_step() instead of
// building a ChatbotTrainer/IncrementalTrainer at all. Resumes from `world_model_dir`'s own
// checkpoint if one already exists (TD-184) — genuinely incremental across separate invocations,
// the same way IncrementalTrainer's own constructor resumes the chatbot model. No MNS
// registration or IncrementalTrainer::begin_run() here — the world model is its own,
// independently MNS-registered artifact (via a manual `mns_cli register`, TD-184), not something
// this pass registers itself.
//
// Data format: reuses DatasetRegistry::load_conversation_pairs() — the only file-content parser
// this codebase has — even though LeJEPA's own objective needs no (input, target) pairing at all
// (see LeJEPAEncoder's class doc). Both a ConversationPair's `input` and `response` fields are
// fed to train_step() as independent, unpaired plain-text samples: the JSONL half of that parser
// only requires a non-empty "input" field (see parse_jsonl_sample in TrainingSampleMeta.hpp), so
// an operator queuing genuinely unpaired text for this objective can write bare `{"input":
// "..."}` lines with no "response" field at all — and any already-queued ordinary (input,
// response) chatbot data gets both of its sides reused for pretraining for free, with zero new
// file-format code.
// Compact JSON config snapshot for the LeJEPA metrics session — mirrors
// IncrementalTrainer.cpp's own build_config_snapshot(), but reads the
// WORLD_MODEL_* fields off ServiceConfig directly since this path has no
// IncrementalConfig/TrainingConfig of its own.
std::string build_lejepa_config_snapshot(const adai::ServiceConfig& svc_config) {
    std::ostringstream json;
    json << std::fixed;
    json << "{" << "\"d_model\":" << svc_config.world_model_d_model
         << ",\"heads\":" << svc_config.world_model_num_heads
         << ",\"d_ff\":" << svc_config.world_model_d_ff
         << ",\"layers\":" << svc_config.world_model_num_layers
         << ",\"sigreg_lambda\":" << svc_config.world_model_sigreg_lambda
         << ",\"sigreg_sketches\":" << svc_config.world_model_sigreg_num_sketches
         << ",\"lr\":" << svc_config.learning_rate << "}";
    return json.str();
}

}  // namespace

// TD-403: `control`, when given (a trainer_service pass), is the per-pass admin API's state: the
// pass reports its phase and progress there, writes a checkpoint on /admin/checkpoint, takes the
// auto-save settings live from it, and on /admin/pause stops at the next batch, checkpoints,
// releases its files and returns kResumeExitFailed (as an aborted chatbot pass does).
int run_lejepa_training_pass(const adai::ServiceConfig& svc_config,
                             const std::string& default_vocab, const std::string& world_model_dir,
                             int epochs, const std::shared_ptr<adai::TrainerControlState>& control,
                             const LeJEPAPassHooks& hooks) {
    // The admin API reports idle again whichever way the pass ends.
    struct IdleOnExit {
        const std::shared_ptr<adai::TrainerControlState>& control;
        ~IdleOnExit() {
            if (control)
                control->set_phase(adai::TrainerPhase::Idle);
        }
    } idle_on_exit{control};
    if (control) {
        control->set_phase(adai::TrainerPhase::LoadingData);
        control->log(adai::TrainerLogLevel::Info, "LeJEPA world-model pass starting");
    }

    // Determine vocab_size the same way IncrementalTrainer.cpp's own model-construction path
    // does: load the vocab file into a throwaway tokenizer first, then size the encoder's fixed
    // TokenEmbedding to match. LeJEPAEncoder::load_tokenizer_vocab() itself does not resize
    // anything (a pre-existing gap in that method, not introduced here — flagged separately),
    // so the constructor's own vocab_size must already be correct before that call.
    // TD-416: TOKENIZER_MODE, as for the chatbot (it used to be ASCII always); a vocabulary that
    // records its own mode overrides it on load.
    const TokenizerMode tokenizer_mode =
        svc_config.unicode_tokenizer ? TokenizerMode::UNICODE : TokenizerMode::ASCII;
    BPETokenizer vocab_probe(tokenizer_mode);
    vocab_probe.load_vocab(default_vocab);
    const int vocab_size = vocab_probe.get_vocab_size();

    LeJEPAEncoder world_model(vocab_size, static_cast<int>(svc_config.world_model_d_model),
                              static_cast<int>(svc_config.world_model_num_layers),
                              static_cast<int>(svc_config.world_model_num_heads),
                              static_cast<int>(svc_config.world_model_d_ff),
                              static_cast<int>(svc_config.max_seq_length),
                              static_cast<int>(svc_config.world_model_sigreg_num_sketches));
    world_model.set_tokenizer_mode(tokenizer_mode);
    world_model.load_tokenizer_vocab(default_vocab);

    // TD-184: genuinely incremental across separate `train` invocations — resume from a prior
    // pass's checkpoint if one exists, the same way IncrementalTrainer's own constructor resumes
    // the chatbot model from its best saved checkpoint before training on newly-acquired data
    // (see IncrementalTrainer.cpp's constructor). Without this, every pass would silently discard
    // all prior progress and retrain a fresh, randomly-initialized encoder — defeating the whole
    // point of feeding it new pending-data batches over time. A mismatched architecture (the
    // WORLD_MODEL_* config changed since the checkpoint was saved) is reported clearly and this
    // pass aborts rather than either silently discarding the old checkpoint or crashing uncaught.
    // TD-548: saves go to world_model_dir; the checkpoint to resume may still be in the
    // pre-TD-548 <session_dir>/world_model (resolve_checkpoint_dir() finds it).
    const std::string resume_dir = LeJEPAEncoder::resolve_checkpoint_dir(
        svc_config.session_dir, svc_config.world_model_artifact_path);
    if (LeJEPAEncoder::has_checkpoint(resume_dir)) {
        try {
            world_model.load(resume_dir);
            adai::Logger::info("[lejepa] Resumed world-model weights from '{}'", resume_dir);
            if (resume_dir != world_model_dir) {
                adai::Logger::info("[lejepa] Saving to '{}' from now on (TD-548)", world_model_dir);
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ [lejepa] Failed to load existing world-model checkpoint at '"
                      << resume_dir << "': " << e.what() << "\n"
                      << "   (WORLD_MODEL_* config must match the architecture the checkpoint "
                         "was saved with — or move/remove the old checkpoint directory to start "
                         "fresh)\n";
            return 1;
        }
    }

    world_model.set_sigreg_lambda(svc_config.world_model_sigreg_lambda);
    world_model.set_learning_rate(svc_config.learning_rate);

    Optimizer optimizer(OptimizerType::ADAM, svc_config.learning_rate);
    world_model.register_parameters_with_optimizer(optimizer);

    // TD-415: a run id kept with the world-model checkpoint, so a pass restarted after a crash,
    // kill or SIGTERM reclaims the files the last one claimed (registries hand a claim back to the
    // same run id). It used to be PID-based, so those files stayed claimed by an id no later
    // process had.
    std::string run_id = adai::persistent_run_id(svc_config.run_id, world_model_dir);
    if (control)
        control->set_run_id(run_id);

    DatasetConfig dcfg = DatasetRegistry::make_config(svc_config);
    DatasetRegistry reg(dcfg);
    reg.load_registry();
    startup_sweep(reg, run_id, dcfg.download_dir);  // loads the pending list itself (TD-512)

    auto resp = reg.acquire_pending(run_id);
    // TD-403: the same exit codes as the chatbot resume (TD-523), now that trainer_service runs
    // this pass too: an unreachable registry is a failure, an empty pool means idle.
    if (resp.failed) {
        adai::Logger::error("[lejepa] Could not ask the dataset registry for pending files");
        return adai::kResumeExitFailed;
    }
    if (resp.files.empty()) {
        adai::Logger::warn("[lejepa] No pending data files to train on");
        return adai::kResumeExitNothingPending;
    }

    // TD-411: the same download step as train/retrain/resume. This pass used to download only
    // when DOWNLOAD_DIR was set, so against an FTP registry without it, it read registry-host
    // paths that don't exist on the trainer host.
    std::vector<PendingFileRange> local_ranges;
    std::vector<fs::path> downloaded_paths;
    if (!resolve_pass_files(resp, dcfg, local_ranges, downloaded_paths)) {
        reg.release_pending(run_id, resp.file_ranges());  // TD-333: exact units
        return 1;
    }
    std::vector<std::string> texts;
    std::vector<int> pair_counts;  // TD-334: per file, 1:1 with local_ranges / resp.files
    for (const auto& range : local_ranges) {
        std::vector<ConversationPair> pairs;
        // TD-415: only the claimed segment's rows (TD-205), as the chatbot pass loads them; every
        // claimed segment of a file used to load the whole file.
        DatasetRegistry::load_conversation_pairs(range.path, pairs, range.segment_start,
                                                 range.segment_count);
        pair_counts.push_back(static_cast<int>(pairs.size()));
        for (const auto& pair : pairs) {
            if (!pair.input.empty())
                texts.push_back(pair.input);
            if (!pair.response.empty())
                texts.push_back(pair.response);
        }
    }

    // TD-333: exact units (whole files and segments)
    const auto reg_paths = resp.file_ranges();
    if (texts.empty()) {
        adai::Logger::warn("[lejepa] Acquired files contained no usable text samples");
        reg.release_pending(run_id, reg_paths);
        cleanup_downloads(downloaded_paths);  // empty unless files were downloaded
        return 1;
    }
    if (control) {
        control->total_epochs = epochs;
        control->current_epoch = 0;
        control->samples_trained_this_pass = 0;
        control->set_phase(adai::TrainerPhase::Training);
        control->log(adai::TrainerLogLevel::Info,
                     "Training the world model on " + std::to_string(texts.size()) + " text(s)");
    }

    // Metrics-dashboard reporting (previously entirely absent from this path — the only
    // signal during a potentially many-hour pass was two log lines total: one per epoch,
    // one at the very end). Mirrors IncrementalTrainer::run_training()'s MetricsPushClient
    // wiring (IncrementalTrainer.cpp), but with a deliberately STABLE session key —
    // "lejepa-<model_name>", not a per-invocation session_id+host key — so the dashboard
    // shows one continuous training-history curve across every separate `train` invocation
    // for this world model's lifetime, matching how the checkpoint itself is resumed above.
    // A 409 here means a prior pass crashed without ever reaching end_session() (this loop
    // has no periodic checkpoint, so a crash mid-pass — e.g. the real GPU engine-reset seen
    // on ai-machine — leaves the session marked "training" forever); MetricsSessionRegistry
    // only 409s a genuinely still-"training" session, never an already-ended one, so a short
    // retry on the SAME key (changing it would fragment the continuous curve) gives the
    // staleness sweep a chance to reclaim it.
    std::unique_ptr<MetricsPushClient> metrics_client;
    if (!svc_config.metrics_server_url.empty()) {
        const std::string session_key = sanitize_session_key(
            "lejepa-" +
            (svc_config.model_name.empty() ? std::string("world-model") : svc_config.model_name));
        const std::string push_url =
            build_metrics_session_push_base(svc_config.metrics_server_url, session_key);
        const std::string label = svc_config.metrics_session_label.empty()
                                      // "world_model": the same label as before TD-548 moved
                                      // the default directory
                                      ? derive_metrics_session_label(1, "world_model")
                                      : svc_config.metrics_session_label;
        const std::string config_snapshot = build_lejepa_config_snapshot(svc_config);

        auto pc = std::make_unique<MetricsPushClient>(
            push_url, svc_config.metrics_push_timeout_ms, 1024,
            svc_config.metrics_heartbeat_interval_ms,
            parse_metrics_server_headers(svc_config.metrics_server_headers),
            svc_config.metrics_server_ca_file);
        int rc = 0;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (attempt > 0) {
                adai::Logger::warn("[lejepa] Metrics session busy (409), retrying '{}'",
                                   session_key);
                std::this_thread::sleep_for(std::chrono::milliseconds(500 * attempt));
            }
            rc = pc->start_session(1, epochs, static_cast<int>(texts.size()), label,
                                   config_snapshot, /*reset_best=*/false);
            if (rc != 409)
                break;
        }
        if (rc >= 200 && rc < 300) {
            metrics_client = std::move(pc);
        } else {
            adai::Logger::error(
                "[lejepa] Metrics session/start failed for key '{}' (HTTP {}) — training will "
                "proceed WITHOUT metrics reporting for this pass",
                session_key, rc);
        }
    }

    // Mid-pass checkpointing: this loop previously only saved once, after every acquired
    // file had been fully processed — for a many-hour-to-many-day pass over the whole
    // pending pool, a crash (process kill, the real GPU engine-reset seen on ai-machine,
    // etc.) lost 100% of progress, not just the time since the last save. Reuses the same
    // AUTO_SAVE_ENABLED/AUTO_SAVE_EVERY_SAMPLES/AUTO_SAVE_EVERY_MINUTES config keys the
    // chatbot path already uses (same cross-objective concern, no new config surface) —
    // resuming from wherever this leaves off is already handled by the TD-184 load-on-
    // startup logic above, so a periodic world_model.save() here is the whole fix; nothing
    // else needs to change. A failed periodic save is logged and skipped rather than
    // aborting the pass — the resilience feature itself must not become a new crash source.
    int samples_since_last_save = 0;
    auto last_save_time = std::chrono::steady_clock::now();
    // One checkpoint write, reported to the admin API when there is one (TD-403).
    const auto save_checkpoint = [&](const std::string& why) {
        if (control)
            control->set_phase(adai::TrainerPhase::Checkpointing);
        bool saved = false;
        try {
            world_model.save(world_model_dir);
            saved = true;
            if (control) {
                control->checkpoints_written.fetch_add(1);
                control->last_checkpoint_time_unix = static_cast<long long>(std::time(nullptr));
                control->set_last_checkpoint_path(world_model_dir);
                control->log(adai::TrainerLogLevel::Info,
                             "Checkpoint saved (" + why + "): " + world_model_dir);
            } else {
                adai::Logger::info("[lejepa] Checkpoint saved to '{}' ({})", world_model_dir, why);
            }
        } catch (const std::exception& e) {
            adai::Logger::warn("[lejepa] Checkpoint save failed, continuing: {}", e.what());
        }
        if (control)
            control->set_phase(adai::TrainerPhase::Training);
        samples_since_last_save = 0;
        last_save_time = std::chrono::steady_clock::now();
        return saved;
    };
    bool aborted = false;
    // TD-566: after a failed batch, whether the GPU still runs work. A lost device (a driver engine
    // reset) used to fail every remaining batch, each skipped, until the pass "completed" on what
    // came before; now the pass stops at once without a checkpoint and gives its files back, so
    // trainer_service starts a fresh process. An epoch in which no batch trained fails the pass
    // too, as it does for the chatbot.
    const std::function<bool()> device_usable = hooks.device_usable ? hooks.device_usable : [] {
#ifdef ADAI_ENABLE_GPU
        return !adai::gpu::GPUManager::is_available() || adai::gpu::GPUManager::device_usable();
#else
        return true;
#endif
    };
    bool device_lost = false;
    int empty_epoch = 0;  // the 1-based epoch that trained nothing, if any

    // Advanced diagnostics (activation saturation / attention entropy): mirrors
    // ChatbotTrainer.cpp's own accumulate-via-hook/reset-per-epoch/report-then-clear pattern,
    // adapted to this loop's flat per-sample structure (there's no gradient-accumulation window
    // here, so the epoch is the natural reset boundary). Registered once, outside the epoch
    // loop, since world_model's layer structure doesn't change across epochs; accumulators are
    // reset at the start of each epoch below.
    //
    // Caveat: LeJEPAEncoder::train_step() calls encode_tokens() 3x per sample internally
    // (context view, target view, re-encoded context view for its second internal update), so
    // these hooks fire 3x per sample, blending the partially-masked context view and the
    // unmasked target view indiscriminately — a legitimate first-cut blended signal, not
    // view-separated.
    float sat_sum = 0.0f;
    int sat_count = 0;
    float ent_sum = 0.0f;
    int ent_count = 0;
    auto saturation_hook = [&sat_sum, &sat_count](const Matrix& activated) {
        const int total = activated.rows * activated.cols;
        if (total <= 0) {
            return;
        }
        int sat = 0;
        for (int r = 0; r < activated.rows; ++r) {
            for (int c = 0; c < activated.cols; ++c) {
                if (std::abs(activated(r, c)) < 0.01f) {
                    ++sat;
                }
            }
        }
        sat_sum += static_cast<float>(sat) / static_cast<float>(total);
        ++sat_count;
    };
    auto entropy_hook = [&ent_sum, &ent_count](const Matrix& attn_weights) {
        const int seq_len = attn_weights.rows;
        if (seq_len <= 0 || attn_weights.cols <= 0) {
            return;
        }
        float layer_entropy = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float row_entropy = 0.0f;
            for (int j = 0; j < attn_weights.cols; ++j) {
                float a = attn_weights(i, j);
                if (a > 0.0f) {
                    row_entropy -= a * std::log(a + 1e-10f);
                }
            }
            layer_entropy += row_entropy;
        }
        ent_sum += layer_entropy / static_cast<float>(seq_len);
        ++ent_count;
    };
    const int world_model_layers = world_model.get_num_layers();
    for (int l = 0; l < world_model_layers; ++l) {
        world_model.get_encoder_block(l)->get_feed_forward()->set_activation_hook(saturation_hook);
        world_model.get_encoder_block(l)->get_self_attention()->set_attention_hook(entropy_hook);
    }

    double predictor_loss_sum = 0.0;
    double sigreg_loss_sum = 0.0;
    size_t step_count = 0;
    for (int epoch = 0; epoch < epochs && !aborted; ++epoch) {
        if (metrics_client)
            metrics_client->start_epoch(epoch + 1, static_cast<int>(texts.size()));

        double epoch_predictor_sum = 0.0;
        double epoch_sigreg_sum = 0.0;
        size_t epoch_step_count = 0;
        sat_sum = 0.0f;
        sat_count = 0;
        ent_sum = 0.0f;
        ent_count = 0;
        double epoch_masking_ratio_sum = 0.0;
        double epoch_cosine_sim_sum = 0.0;
        double epoch_sigreg_variance_mean_sum = 0.0;
        double epoch_sigreg_variance_stddev_sum = 0.0;
        // compute_time_ratio here means "fraction of epoch wall time actually inside
        // train_step() calls" (surfaces checkpoint-save/metrics-push overhead between
        // samples) — train_step() is an indivisible forward+backward+both-optimizer-steps
        // black box, unlike the chatbot path's own forward+backward-vs-data-loading version
        // of this same metric.
        long long epoch_train_step_ns = 0;
        const auto epoch_wall_start = std::chrono::steady_clock::now();
        int sample_idx = 0;
        // SIGReg is applied across each batch's mean-pooled embeddings (LeJEPAEncoder::
        // train_batch(), as the LeJEPA paper does) rather than one text's tokens.
        const size_t batch_size = std::max<size_t>(2, svc_config.world_model_sigreg_batch_size);
        for (size_t first = 0; first < texts.size(); first += batch_size) {
            adai::note_pass_progress();        // TD-569: trainer_service's stall check
            if (control && control->paused) {  // TD-403: /admin/pause drains at a batch boundary
                aborted = true;
                break;
            }
            if (const long long ticket =
                    control ? control->take_checkpoint_request() : 0) {  // POST /admin/checkpoint
                if (save_checkpoint("requested"))
                    control->mark_checkpoint_served(ticket);  // TD-521
            }
            const size_t last = std::min(texts.size(), first + batch_size);
            const std::vector<std::string> batch(texts.begin() + static_cast<std::ptrdiff_t>(first),
                                                 texts.begin() + static_cast<std::ptrdiff_t>(last));
            const auto step_start = std::chrono::steady_clock::now();
            LeJEPAEncoder::BatchStepResult result;
            try {
                if (hooks.before_batch)
                    hooks.before_batch(first / batch_size);
                result = world_model.train_batch(batch);
            } catch (const std::exception& e) {
                if (!device_usable()) {  // TD-566
                    adai::Logger::error(
                        "[lejepa] The GPU stopped running work ({}); stopping the pass", e.what());
                    device_lost = true;
                    break;
                }
                adai::Logger::warn("[lejepa] Skipping batch of {} ({})", batch.size(), e.what());
                sample_idx += static_cast<int>(batch.size());
                continue;
            }
            epoch_train_step_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - step_start)
                                       .count();
            if (result.trained.size() < batch.size()) {
                adai::Logger::warn(
                    "[lejepa] Skipped {} of {} samples in a batch (too short or "
                    "untokenizable)",
                    batch.size() - result.trained.size(), batch.size());
            }
            sample_idx += static_cast<int>(batch.size());
            for (size_t k = 0; k < result.trained.size(); ++k) {
                const float predictor_loss = result.predictor_losses[k];
                // The batch's SIGReg loss and diagnostics count once per trained text, so the
                // per-step averages below keep their meaning.
                const float sigreg_loss = result.sigreg_loss;
                predictor_loss_sum += predictor_loss;
                sigreg_loss_sum += sigreg_loss;
                epoch_predictor_sum += predictor_loss;
                epoch_sigreg_sum += sigreg_loss;
                epoch_masking_ratio_sum += world_model.get_last_masking_ratio();
                epoch_cosine_sim_sum += world_model.get_last_predictor_target_cosine_sim();
                epoch_sigreg_variance_mean_sum += world_model.get_last_sigreg_variance_mean();
                epoch_sigreg_variance_stddev_sum += world_model.get_last_sigreg_variance_stddev();
                ++step_count;
                ++epoch_step_count;
                if (metrics_client) {
                    // Live per-sample curve on the existing loss/perplexity chart —
                    // predictor_loss alone (the actual predictive objective, most
                    // interpretable). sigreg_loss gets its own dedicated dashboard
                    // channel via update_lejepa_metrics() below (TD-178), not folded
                    // in here.
                    metrics_client->update_sample_metrics(
                        static_cast<int>(first + result.trained[k] + 1),
                        static_cast<float>(predictor_loss), world_model.get_last_gradient_norm(),
                        static_cast<float>(svc_config.learning_rate));
                }

                if (control) {
                    control->samples_trained_this_pass = static_cast<long long>(step_count);
                    control->last_loss = static_cast<double>(predictor_loss);
                }

                ++samples_since_last_save;
                // TD-403: live from the admin API when there is one (PUT /admin/config).
                const bool auto_save =
                    control ? control->auto_save_enabled.load() : svc_config.auto_save_enabled;
                const int every_samples = control ? control->auto_save_every_samples.load()
                                                  : svc_config.auto_save_every_samples;
                const int every_minutes = control ? control->auto_save_every_minutes.load()
                                                  : svc_config.auto_save_every_minutes;
                if (auto_save) {
                    const auto elapsed_minutes =
                        std::chrono::duration_cast<std::chrono::minutes>(
                            std::chrono::steady_clock::now() - last_save_time)
                            .count();
                    const bool due_by_samples =
                        every_samples > 0 && samples_since_last_save >= every_samples;
                    const bool due_by_time = every_minutes > 0 && elapsed_minutes >= every_minutes;
                    if (due_by_samples || due_by_time) {
                        save_checkpoint(std::to_string(step_count) + " steps this pass so far");
                    }
                }
            }
        }
        if (aborted || device_lost)
            break;
        if (epoch_step_count == 0) {  // TD-566: nothing to record as this epoch's loss
            empty_epoch = epoch + 1;
            break;
        }
        adai::Logger::info("[lejepa] Epoch {}/{} complete ({} samples)", epoch + 1, epochs,
                           texts.size());
        if (control) {
            control->current_epoch = epoch + 1;
            if (epoch_step_count > 0) {
                const double avg = epoch_predictor_sum / static_cast<double>(epoch_step_count);
                // TD-520: best_loss starts NaN; only a finite average can become it.
                const double best = control->best_loss.load();
                if (std::isfinite(avg) && (!std::isfinite(best) || avg < best))
                    control->best_loss = avg;
            }
        }
        if (metrics_client && epoch_step_count > 0) {
            const float avg_predictor_loss =
                static_cast<float>(epoch_predictor_sum / static_cast<double>(epoch_step_count));
            const float avg_sigreg_loss =
                static_cast<float>(epoch_sigreg_sum / static_cast<double>(epoch_step_count));
            // These buffered-field setters must all run BEFORE end_epoch() below —
            // MetricsPushClient::end_epoch() reads its buffers synchronously while building the
            // /epoch/end JSON body, so a setter called after it fires would have its value land
            // in the FOLLOWING epoch's POST instead of this one (still at this epoch's
            // start_epoch()-reset sentinel when end_epoch() runs) — the same ordering lesson
            // TD-208 already learned once server-side in TrainingMetricsAPI::
            // handle_post_epoch_end(), just on the client side this time. Mirrors
            // ChatbotTrainer.cpp's own call order (update_advanced_epoch_metrics/
            // update_activation_saturation/update_attention_entropy all precede its end_epoch()
            // call).
            //
            // TD-178: predictor_loss/sigreg_loss get their own dedicated dashboard channel —
            // two independent series, not a train/validation pair.
            metrics_client->update_lejepa_metrics(avg_predictor_loss, avg_sigreg_loss);

            const auto epoch_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - epoch_wall_start)
                                           .count();
            const float compute_time_ratio =
                (epoch_wall_ns > 0) ? static_cast<float>(static_cast<double>(epoch_train_step_ns) /
                                                         static_cast<double>(epoch_wall_ns))
                                    : 0.0f;
            // gradient_variance/weight_update_ratio have no LeJEPA-side equivalent (blocked on
            // how train_step()'s two sequential internal optimizer updates, TD-189, should
            // report gradients separately vs. combined — a deeper design question than this
            // change covers) — passed as 0.0f. Both already default to 0.0f in the snapshot, so
            // this isn't a new ambiguity; update_advanced_epoch_metrics() bundles all three
            // fields into one call, so sending compute_time_ratio at all means these two also
            // get persisted as literal 0.0 rather than the -1 "not applicable" sentinel used
            // elsewhere (a pre-existing API-shape wrinkle, not introduced here).
            metrics_client->update_advanced_epoch_metrics(0.0f, compute_time_ratio, 0.0f);

            const float avg_saturation =
                (sat_count > 0) ? (sat_sum / static_cast<float>(sat_count)) : -1.0f;
            metrics_client->update_activation_saturation(avg_saturation);
            const float avg_entropy =
                (ent_count > 0) ? (ent_sum / static_cast<float>(ent_count)) : -1.0f;
            metrics_client->update_attention_entropy(avg_entropy);

            const float avg_masking_ratio =
                static_cast<float>(epoch_masking_ratio_sum / static_cast<double>(epoch_step_count));
            const float avg_cosine_sim =
                static_cast<float>(epoch_cosine_sim_sum / static_cast<double>(epoch_step_count));
            const float avg_sigreg_variance_mean = static_cast<float>(
                epoch_sigreg_variance_mean_sum / static_cast<double>(epoch_step_count));
            const float avg_sigreg_variance_stddev = static_cast<float>(
                epoch_sigreg_variance_stddev_sum / static_cast<double>(epoch_step_count));
            metrics_client->update_lejepa_advanced_metrics(avg_masking_ratio, avg_cosine_sim,
                                                           avg_sigreg_variance_mean,
                                                           avg_sigreg_variance_stddev);

            // No real validation split for this objective — validation_loss stays 0.0f,
            // matching ChatbotTrainer's own convention of simply not reporting it when
            // there's no validation data (rather than an N/A sentinel).
            metrics_client->end_epoch(epoch + 1, avg_predictor_loss, /*validation_loss=*/0.0f,
                                      static_cast<float>(svc_config.learning_rate),
                                      /*perplexity=*/0.0f, world_model.get_last_gradient_norm());
        }
    }

    for (int l = 0; l < world_model_layers; ++l) {
        world_model.get_encoder_block(l)->get_feed_forward()->clear_activation_hook();
        world_model.get_encoder_block(l)->get_self_attention()->clear_attention_hook();
    }

    if (aborted) {
        // TD-403: like a paused chatbot pass, keep the progress but don't record the files as
        // trained; they go back to the pool for the next pass.
        if (control) {
            control->log(
                adai::TrainerLogLevel::Warn,
                "LeJEPA pass paused via admin API after " + std::to_string(step_count) +
                    " step(s) — checkpointing and releasing claimed files back to pending");
        }
        if (step_count > 0)
            save_checkpoint("paused");
        reg.release_pending(run_id, reg_paths);
        if (metrics_client)
            metrics_client->end_session();
        cleanup_downloads(downloaded_paths);
        return adai::kResumeExitFailed;
    }

    // TD-566: a lost GPU or an epoch that trained nothing fails the pass without a checkpoint
    // (after a lost device the weights can't be trusted) and gives the files back.
    if (device_lost || empty_epoch > 0) {
        if (device_lost) {
            adai::Logger::error(
                "[lejepa] Not saving the world model: the GPU was lost; releasing the pass's "
                "files");
        } else {
            adai::Logger::error(
                "[lejepa] Epoch {} trained nothing (every batch failed, or every text was too "
                "short); "
                "releasing the pass's files",
                empty_epoch);
        }
        if (control)
            control->log(adai::TrainerLogLevel::Error, "LeJEPA pass failed; files released");
        reg.release_pending(run_id, reg_paths);
        if (metrics_client)
            metrics_client->end_session();
        cleanup_downloads(downloaded_paths);
        return adai::kResumeExitFailed;
    }

    bool ok = step_count > 0;
    if (ok) {
        adai::Logger::info(
            "[lejepa] Pass complete — {} steps, avg predictor_loss={:.4f}, avg sigreg_loss={:.4f}",
            step_count, predictor_loss_sum / static_cast<double>(step_count),
            sigreg_loss_sum / static_cast<double>(step_count));
        // TD-571: the checkpoint first, and the files recorded as trained only once it is written.
        // They used to be marked first, so a failed save (it throws) lost the pass's training while
        // the registry said the data was done; the files now go back to the pool.
        try {
            world_model.save(world_model_dir);
            adai::Logger::info("[lejepa] World-model checkpoint saved to '{}'", world_model_dir);
        } catch (const std::exception& e) {
            adai::Logger::error(
                "[lejepa] Could not save the world-model checkpoint to '{}': {}; releasing the "
                "pass's files",
                world_model_dir, e.what());
            ok = false;
        }
    }
    if (ok) {
        reg.mark_trained(run_id, reg_paths, pair_counts);  // TD-334: pairs loaded per file
    } else if (step_count > 0) {
        reg.release_pending(run_id, reg_paths);
    } else {
        reg.release_pending(run_id, reg_paths);
        adai::Logger::warn("[lejepa] No samples were long enough to train on — all skipped");
    }

    if (metrics_client)
        metrics_client->end_session();
    cleanup_downloads(downloaded_paths);  // empty unless files were downloaded
    return ok ? 0 : 1;
}

}  // namespace adai
