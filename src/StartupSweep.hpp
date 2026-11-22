// @adai-status: stable
// @adai-version: 1.4.0
// @adai-reviewed: 2026-11-21

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
 *   A  — zero-byte partial → delete, and release this run's claims on the file (whole file or each
 *        TD-205 segment) back to the pool; kept when another run holds a claim on it (TD-513)
 *   B/C — non-zero + assigned → keep; DataTransport handles re-use/resume
 *
 * TD-337: downloads are named by download_file_name(registry_path), so they are matched to
 * pending/trained entries by that name, not by basename. A `<download>.checksum` sidecar
 * (TD-338) is kept with its download and removed with it, or on its own once its download is gone.
 *
 * TD-513: every filesystem call uses its std::error_code overload, so a permission error or a file
 * removed meanwhile is logged and skipped instead of throwing out of the pass.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "DatasetRegistry.hpp"
#include "Logger.hpp"

inline void startup_sweep(DatasetRegistry& reg, const std::string& run_id,
                          const std::string& download_dir) {
    namespace fs = std::filesystem;

    if (download_dir.empty())
        return;
    std::error_code dir_ec;
    if (!fs::exists(download_dir, dir_ec)) {
        if (dir_ec)
            adai::Logger::warn("[CLEANUP] Can't check download_dir '{}': {}; skipping the sweep",
                               download_dir, dir_ec.message());
        return;
    }

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
    // download name → the pending entries of that file (TD-205: one per segment, sharing the path).
    std::unordered_map<std::string, std::vector<PendingEntry>> assigned_by_filename;
    for (const auto& e : reg.pending_entries()) {
        assigned_by_filename[download_file_name(e.path)].push_back(e);
    }

    // Download names of files already marked trained in the registry.
    std::unordered_set<std::string> trained_filenames;
    for (const auto& p : reg.trained_files()) {
        trained_filenames.insert(download_file_name(p));
    }
    const std::string sidecar_suffix = kDownloadChecksumSuffix;
    const auto remove_with_sidecar = [&](const fs::path& p) {
        for (const fs::path& f : {p, fs::path(p.string() + sidecar_suffix)}) {
            std::error_code ec;
            fs::remove(f, ec);  // false without an error when it doesn't exist
            if (ec)
                adai::Logger::warn("[CLEANUP] Couldn't delete '{}': {}", f.string(), ec.message());
        }
    };

    std::error_code it_ec;
    fs::directory_iterator it(download_dir, it_ec);
    if (it_ec) {
        adai::Logger::warn("[CLEANUP] Can't read download_dir '{}': {}; skipping the sweep",
                           download_dir, it_ec.message());
        return;
    }
    for (; it != fs::directory_iterator(); it.increment(it_ec)) {
        if (it_ec)
            break;
        const fs::directory_entry& entry = *it;
        std::error_code type_ec;
        if (!entry.is_regular_file(type_ec))
            continue;  // not a regular file, or gone meanwhile
        const fs::path local_path = entry.path();
        const std::string filename = local_path.filename().string();

        // TD-338: a checksum sidecar goes with its download; drop it once the download is gone.
        if (filename.size() > sidecar_suffix.size() &&
            filename.compare(filename.size() - sidecar_suffix.size(), sidecar_suffix.size(),
                             sidecar_suffix) == 0) {
            const std::string download =
                local_path.string().substr(0, local_path.string().size() - sidecar_suffix.size());
            std::error_code exists_ec;
            if (!fs::exists(download, exists_ec) && !exists_ec) {
                std::error_code ec;
                fs::remove(local_path, ec);
                if (ec)
                    adai::Logger::warn("[CLEANUP] Couldn't delete '{}': {}", local_path.string(),
                                       ec.message());
            }
            continue;
        }

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

        const auto found = assigned_by_filename.find(filename);
        if (found == assigned_by_filename.end()) {
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
        const std::vector<PendingEntry>& entries = found->second;
        std::error_code size_ec;
        const std::uintmax_t file_size = fs::file_size(local_path, size_ec);
        if (size_ec) {
            adai::Logger::warn("[CLEANUP] Can't read the size of '{}': {}; leaving it", filename,
                               size_ec.message());
            continue;
        }

        if (file_size == 0) {
            // Condition A (token expired): zero-byte partial download from a previous crash. The
            // token from that session no longer exists, so it can't resume: delete the file and
            // release this run's claims on it so another trainer (or ours, on the next acquire)
            // starts fresh. TD-513: every claimed unit of the file, whole file or TD-205 segment,
            // is released (a path-only release skipped segments); and when another run holds a
            // claim, its download may be starting right now, so the file is left alone.
            const std::string other_run = [&] {
                for (const auto& e : entries)
                    if (!e.run_id.empty() && e.run_id != run_id)
                        return e.run_id;
                return std::string();
            }();
            if (!other_run.empty()) {
                adai::Logger::info(
                    "[CLEANUP-A] Empty download '{}' is claimed by run '{}', not '{}'; leaving it.",
                    filename, other_run, run_id);
                continue;
            }
            std::vector<PendingFileRange> ours;
            for (const auto& e : entries)
                if (!run_id.empty() && e.run_id == run_id)
                    ours.push_back({e.path, e.segment_start, e.segment_count});
            adai::Logger::info(
                "[CLEANUP-A] Found empty partial download '{}' in download_dir. "
                "Token from previous session expired — releasing {} claim(s) of run_id '{}' and "
                "deleting file.",
                filename, ours.size(), run_id);
            remove_with_sidecar(local_path);
            if (!ours.empty())
                reg.release_pending(run_id, ours);
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
    if (it_ec)
        adai::Logger::warn("[CLEANUP] Stopped reading download_dir '{}': {}", download_dir,
                           it_ec.message());
}
