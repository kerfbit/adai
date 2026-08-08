/**
 * StartupSweep — stale download cleanup before the first acquire.
 *
 * Extracted from IncrementalTrainingTool.cpp as an inline function so it can
 * be unit-tested without compiling the full training binary.
 *
 * Conditions (see dataset-transport-proposal.md "Stale Download Cleanup"):
 *   D  — trained in registry → delete local copy
 *   G  — no assignment for our run_id → delete
 *   A  — zero-byte partial + assigned → delete + release back to pool
 *   B/C — non-zero + assigned → keep; DataTransport handles re-use/resume
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

    // Build O(1) lookup structures from registry state.
    // filename → registry_path for files currently in the pending list.
    std::unordered_map<std::string, std::string> assigned_by_filename;
    for (const auto& p : reg.pending_files()) {
        assigned_by_filename[fs::path(p).filename().string()] = p;
    }

    // Set of filenames already marked trained in the registry.
    std::unordered_set<std::string> trained_filenames;
    for (const auto& p : reg.trained_files()) {
        trained_filenames.insert(fs::path(p).filename().string());
    }

    for (const auto& entry : fs::directory_iterator(download_dir)) {
        if (!entry.is_regular_file())
            continue;
        const fs::path local_path = entry.path();
        const std::string filename = local_path.filename().string();

        // Condition D: already trained — stale local copy
        if (trained_filenames.count(filename)) {
            adai::Logger::info(
                "[CLEANUP-D] Found locally retained file '{}' already marked "
                "trained in registry. Deleting local copy.",
                filename);
            fs::remove(local_path);
            continue;
        }

        auto it = assigned_by_filename.find(filename);
        if (it == assigned_by_filename.end()) {
            // Condition G: not referenced by any assignment for our run_id
            adai::Logger::info(
                "[CLEANUP-G] Found orphaned file '{}' in download_dir with "
                "no active registry assignment for run_id '{}'. Deleting.",
                filename, run_id);
            fs::remove(local_path);
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
            fs::remove(local_path);
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
