// @adai-status: stable
// @adai-version: 1.3.0
// @adai-reviewed: 2026-10-19

/**
 * StartupSweep — stale download cleanup before the first acquire.
 *
 * Extracted from IncrementalTrainingTool.cpp as an inline function so it can
 * be unit-tested without compiling the full training binary.
 *
 * Conditions (see dataset-transport-proposal.md "Stale Download Cleanup"):
 *   D  — trained in registry → delete local copy
 *   G  — filename not found anywhere in the pending list (any model/run_id,
 *        not just ours — pending_files() is unfiltered) → orphaned, delete
 *   A  — zero-byte partial + assigned → delete + release back to pool
 *   B/C — non-zero + assigned → keep; DataTransport handles re-use/resume
 *
 * TD-337: downloads are named by download_file_name(registry_path), so they are matched to
 * pending/trained entries by that name, not by basename. A `<download>.checksum` sidecar
 * (TD-338) is kept with its download and removed with it, or on its own once its download is gone.
 */

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "DatasetRegistry.hpp"
#include "Logger.hpp"

inline void startup_sweep(DatasetRegistry& reg, const std::string& run_id,
                          const std::string& download_dir) {
    namespace fs = std::filesystem;

    if (download_dir.empty() || !fs::exists(download_dir))
        return;

    // TD-512: load the state the sweep decides on here, rather than trusting whatever the caller
    // loaded. Every caller used to load only the trained registry, so pending_files() was empty
    // and every download (complete, partial, or anything else in download_dir) was deleted as
    // orphaned on every pass. If the pending list can't be loaded (registry unreachable, no
    // pending list yet), skip the sweep: without it every file would look orphaned. If the
    // trained registry can't be loaded, carry on with an empty trained set, so nothing is
    // deleted as "already trained" on missing information.
    if (!reg.load_pending_list()) {
        adai::Logger::warn("[CLEANUP] Could not load the pending list; skipping the download_dir "
                           "sweep so no download is deleted on missing information");
        return;
    }
    reg.load_registry();  // false (none yet / unreachable) leaves trained_files() empty

    // Build O(1) lookup structures from registry state.
    // download name → registry_path for files currently in the pending list.
    std::unordered_map<std::string, std::string> assigned_by_filename;
    for (const auto& p : reg.pending_files()) {
        assigned_by_filename[download_file_name(p)] = p;
    }

    // Download names of files already marked trained in the registry.
    std::unordered_set<std::string> trained_filenames;
    for (const auto& p : reg.trained_files()) {
        trained_filenames.insert(download_file_name(p));
    }
    const std::string sidecar_suffix = kDownloadChecksumSuffix;
    const auto remove_with_sidecar = [&](const fs::path& p) {
        std::error_code ec;
        fs::remove(p, ec);
        fs::remove(p.string() + sidecar_suffix, ec);
    };

    for (const auto& entry : fs::directory_iterator(download_dir)) {
        if (!entry.is_regular_file())
            continue;
        const fs::path local_path = entry.path();
        const std::string filename = local_path.filename().string();

        // TD-338: a checksum sidecar goes with its download; drop it once the download is gone.
        if (filename.size() > sidecar_suffix.size() &&
            filename.compare(filename.size() - sidecar_suffix.size(), sidecar_suffix.size(),
                             sidecar_suffix) == 0) {
            const std::string download =
                local_path.string().substr(0, local_path.string().size() - sidecar_suffix.size());
            if (!fs::exists(download)) {
                std::error_code ec;
                fs::remove(local_path, ec);
            }
            continue;
        }

        // TODO: See TD-513 in TECHNICAL_DEBT.md - throwing fs calls; condition A's release is whole-file only.
        // (Its basename matching is fixed by TD-337's download names.)
        // Condition D: already trained — stale local copy. TD-333: a file can be trained in part
        // (some TD-205 segments) while other segments are still pending; keep its download then.
        if (trained_filenames.count(filename) && !assigned_by_filename.count(filename)) {
            adai::Logger::info(
                "[CLEANUP-D] Found locally retained file '{}' already marked "
                "trained in registry. Deleting local copy.",
                filename);
            remove_with_sidecar(local_path);
            continue;
        }

        auto it = assigned_by_filename.find(filename);
        if (it == assigned_by_filename.end()) {
            // Condition G: filename not found anywhere in the pending list
            // (pending_files() is unfiltered — this checks all models/runs,
            // not just ours).
            adai::Logger::info(
                "[CLEANUP-G] Found orphaned file '{}' in download_dir with no "
                "matching pending-list entry (checked while processing run_id "
                "'{}'). Deleting.",
                filename, run_id);
            remove_with_sidecar(local_path);
            continue;
        }

        // File IS in the pending list.
        const std::string& registry_path = it->second;
        const std::size_t file_size = fs::file_size(local_path);

        if (file_size == 0) {
            // Condition A (token expired): zero-byte partial download from a
            // previous crash.  The token from that session no longer exists, so
            // we cannot resume — release the assignment and delete the file so
            // another trainer (or ourselves on the next acquire) can start fresh.
            adai::Logger::info(
                "[CLEANUP-A] Found empty partial download '{}' in download_dir. "
                "Token from previous session expired — releasing assignment for "
                "run_id '{}' and deleting file.",
                filename, run_id);
            remove_with_sidecar(local_path);
            reg.release_pending(run_id, {registry_path});
            continue;
        }

        // Non-zero file in pending list (Conditions B or C).
        // Leave it in place: DataTransport::fetch() will detect Condition B
        // (size_bytes match → skip re-download) or resume from the current
        // offset (partial file with a valid token).
        adai::Logger::info(
            "[CLEANUP-B/C] Found local file '{}' ({} bytes) assigned to "
            "run_id '{}'. Keeping — DataTransport will verify size and "
            "skip or resume download.",
            filename, file_size, run_id);
    }
}
