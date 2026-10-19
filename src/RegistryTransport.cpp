// @adai-status: stable
// @adai-version: 1.3.0
// @adai-reviewed: 2026-10-19

#include "RegistryTransport.hpp"
#include <fcntl.h>  // open(), O_RDWR
// TD-160: <sys/file.h>'s flock()/LOCK_EX/LOCK_UN are BSD/Linux-only — MinGW (Windows
// cross-compilation, scripts/build_windows.sh) has no equivalent, and previously failed to
// compile this file at all. <io.h>'s _locking() (with <sys/locking.h>'s _LK_LOCK/_LK_UNLCK) is
// MinGW/MSVC's own CRT-level file-locking primitive, operating on the same open()-returned fd —
// confirmed compiling, linking, and running correctly under Wine. Kept behind #ifdef _WIN32
// rather than replacing flock() outright so the non-Windows path (open()'s advisory whole-file
// lock via LOCK_EX, matching this function's existing semantics) is unchanged.
#ifdef _WIN32
#include <io.h>           // open(), close(), _locking()
#include <sys/locking.h>  // _LK_LOCK, _LK_UNLCK
#else
#include <sys/file.h>  // flock()
#include <unistd.h>    // close()
#endif
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include "Logger.hpp"

#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

using adai::Logger;
namespace fs = std::filesystem;

// ============================================================================
// LocalTransport
// ============================================================================

LocalTransport::LocalTransport(std::string registry_path, std::string pending_path)
    : registry_path_(std::move(registry_path)), pending_path_(std::move(pending_path)) {}

bool LocalTransport::load_registry(std::vector<DataVersion>& out) {
    if (!fs::exists(registry_path_)) {
        return false;
    }

    std::ifstream file(registry_path_);
    if (!file.is_open()) {
        return false;
    }

    out.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        DataVersion dv;
        int trained_int = 0;
        // TODO: See TD-496 in TECHNICAL_DEBT.md - '/data/my file.jsonl' loads as data_file '/data/my', trained=false.
        iss >> dv.data_file >> dv.checksum >> dv.num_samples >> trained_int;
        dv.trained = (trained_int == 1);
        if (dv.checksum == "-")
            dv.checksum.clear();  // written for an empty checksum so later columns don't shift

        // model_id (Phase 2), added_utc/source (Phase 15) are optional trailing
        // columns, absent in older files. "-" is a written placeholder for an
        // empty value: with whitespace-delimited >>, an empty middle column
        // can't otherwise be distinguished from "no more columns" and would
        // silently shift every column after it.
        std::string tok;
        if (iss >> tok)
            dv.model_id = (tok == "-") ? "" : tok;
        if (iss >> tok)
            dv.added_utc = (tok == "-") ? "" : tok;
        if (iss >> tok)
            dv.source = (tok == "-") ? "" : tok;
        // TD-333: optional segment columns (absent in older files = whole file).
        int seg_start = -1, seg_count = -1;
        if (iss >> seg_start >> seg_count) {
            dv.segment_start = seg_start;
            dv.segment_count = seg_count;
        }

        out.push_back(std::move(dv));
    }

    return true;
}

// TODO: See TD-495 in TECHNICAL_DEBT.md - truncate-and-rewrite in place; a failed write leaves a partial file.
// TODO: See TD-496 in TECHNICAL_DEBT.md - space-separated columns: a path with a space corrupts the record.
bool LocalTransport::save_registry(const std::vector<DataVersion>& entries) {
    fs::create_directories(fs::path(registry_path_).parent_path());

    std::ofstream file(registry_path_);
    if (!file.is_open()) {
        return false;
    }

    auto col = [](const std::string& s) { return s.empty() ? std::string("-") : s; };

    file << "# Data Registry: data_file checksum num_samples trained model_id added_utc source "
            "segment_start segment_count\n";
    for (const auto& dv : entries) {
        file << dv.data_file << " " << col(dv.checksum) << " " << dv.num_samples << " "
             << (dv.trained ? 1 : 0) << " " << col(dv.model_id) << " " << col(dv.added_utc) << " "
             << col(dv.source) << " " << dv.segment_start << " " << dv.segment_count << "\n";
    }

    return file.good();
}

bool LocalTransport::load_pending(std::vector<PendingEntry>& out) {
    if (!fs::exists(pending_path_)) {
        return false;
    }

    std::ifstream file(pending_path_);
    if (!file.is_open()) {
        return false;
    }

    out.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        // Format: path[\trun_id[\tmodel_name[\tsource[\tadded_utc[\tsize_bytes[\tnum_entries[\tchecksum[\tsegment_start[\tsegment_count]]]]]]]]]
        // Each trailing column was added in a later phase (run_id: Phase 9,
        // model_name: Phase 11-ish, source/added_utc/size_bytes/num_entries/
        // checksum: Phase 15, segment_start/segment_count: TD-205) — split on tab and assign
        // positionally so shorter lines from older files keep loading with defaults for the
        // columns they predate.
        std::vector<std::string> cols;
        std::size_t start = 0;
        while (true) {
            const auto tab = line.find('\t', start);
            if (tab == std::string::npos) {
                cols.push_back(line.substr(start));
                break;
            }
            cols.push_back(line.substr(start, tab - start));
            start = tab + 1;
        }

        PendingEntry e;
        e.path = cols[0];
        if (cols.size() > 1)
            e.run_id = cols[1];
        if (cols.size() > 2)
            e.model_name = cols[2];
        if (cols.size() > 3)
            e.source = cols[3];
        if (cols.size() > 4)
            e.added_utc = cols[4];
        if (cols.size() > 5 && !cols[5].empty()) {
            try {
                e.size_bytes = static_cast<std::size_t>(std::stoull(cols[5]));
            } catch (...) {
            }
        }
        if (cols.size() > 6 && !cols[6].empty()) {
            try {
                e.num_entries = std::stoi(cols[6]);
            } catch (...) {
            }
        }
        if (cols.size() > 7)
            e.checksum = cols[7];
        if (cols.size() > 8 && !cols[8].empty()) {
            try {
                e.segment_start = std::stoi(cols[8]);
            } catch (...) {
            }
        }
        if (cols.size() > 9 && !cols[9].empty()) {
            try {
                e.segment_count = std::stoi(cols[9]);
            } catch (...) {
            }
        }

        out.push_back(std::move(e));
    }

    return true;
}

// TODO: See TD-495 in TECHNICAL_DEBT.md - truncate-and-rewrite in place (simulated disk full kept 114 of 2,001
// entries); write a temp file and rename. TD-496: tab/newline in a path breaks the format.
bool LocalTransport::save_pending(const std::vector<PendingEntry>& entries) {
    fs::create_directories(fs::path(pending_path_).parent_path());

    std::ofstream file(pending_path_);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& e : entries) {
        file << e.path << '\t' << e.run_id << '\t' << e.model_name << '\t' << e.source << '\t'
             << e.added_utc << '\t' << e.size_bytes << '\t' << e.num_entries << '\t' << e.checksum
             << '\t' << e.segment_start << '\t' << e.segment_count << '\n';
    }

    return file.good();
}

// ── Lock helpers ──────────────────────────────────────────────────────────────

int LocalTransport::lock_pending() const {
    const std::string lock_path = pending_path_ + ".lock";
    fs::create_directories(fs::path(pending_path_).parent_path());
    const int fd = open(lock_path.c_str(), O_CREAT | O_WRONLY, 0600);
    if (fd < 0) {
        return -1;
    }
    // TD-160: _locking() locks a byte range rather than the whole file the way flock() does —
    // one byte (offset 0, where O_WRONLY without O_TRUNC positions a fresh write) is sufficient
    // for this file's only purpose (mutual exclusion via its mere existence/openness), and works
    // correctly even when the file is currently empty (confirmed under Wine).
#ifdef _WIN32
    if (_locking(fd, _LK_LOCK, 1) != 0) {
#else
    if (flock(fd, LOCK_EX) != 0) {
#endif
        close(fd);
        return -1;
    }
    return fd;
}

void LocalTransport::unlock_pending(int lock_fd) const {
    if (lock_fd >= 0) {
#ifdef _WIN32
        _locking(lock_fd, _LK_UNLCK, 1);
#else
        flock(lock_fd, LOCK_UN);
#endif
        close(lock_fd);
    }
}

// ── Phase 9: acquire / release / commit_trained ───────────────────────────────

AcquireResponse LocalTransport::acquire(const std::string& run_id, int max_files,
                                        const std::string& model_name) {
    AcquireResponse resp;
    resp.run_id = run_id;
    // ftp_server_host left empty → caller uses direct filesystem paths

    if (!fs::exists(pending_path_)) {
        return resp;
    }

    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::acquire — failed to acquire pending lock");
        return resp;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);

    const int limit = (max_files > 0) ? max_files : std::numeric_limits<int>::max();

    // Assignment-aware: an entry is claimable iff unassigned or assigned to
    // this exact model_name — never another model's assigned entry. A caller
    // with no model identity (model_name empty) can only claim unassigned
    // entries. See RegistryTransport::acquire's doc comment.
    const auto eligible = [&model_name](const PendingEntry& e) {
        return e.model_name.empty() || e.model_name == model_name;
    };

    for (auto& e : entries) {
        // Unclaimed, OR already claimed by this exact run_id — the latter
        // lets a crashed-and-restarted run (same run_id — see
        // IncrementalTrainer::begin_run()/MNS's new_run=false continuation)
        // reclaim its own in-flight files instead of finding them
        // permanently stuck. A *different* run_id still can never steal
        // another's claim.
        if ((e.run_id.empty() || e.run_id == run_id) && eligible(e) &&
            static_cast<int>(resp.files.size()) < limit) {
            e.run_id = run_id;
            FileToken tok;
            tok.registry_path = e.path;
            tok.segment_start = e.segment_start;
            tok.segment_count = e.segment_count;
            resp.files.push_back(std::move(tok));
        }
    }

    if (!resp.files.empty()) {
        save_pending(entries);
        Logger::info("Acquired {} pending files for run '{}'", resp.files.size(), run_id);
    }

    unlock_pending(lock_fd);
    return resp;
}

void LocalTransport::release(const std::string& run_id, const std::vector<std::string>& paths,
                             const std::vector<SegmentTarget>& segments) {
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::release — failed to acquire pending lock");
        return;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);

    // TD-205: whole-file paths match only whole-file entries (segment_start < 0); a segment
    // needs its own `segments` target, matched on its exact range so its siblings stay claimed.
    // TD-333: both lists are applied in one call. This used to be either/or (segments, else
    // paths), so a batch with both kinds released only half of them.
    const std::set<std::string> to_release(paths.begin(), paths.end());
    int released = 0;
    for (auto& e : entries) {
        if (e.run_id != run_id)
            continue;
        bool match = e.segment_start < 0 && to_release.count(e.path) > 0;
        for (std::size_t i = 0; !match && i < segments.size(); ++i)
            match = segment_matches(e, segments[i].path, segments[i].segment_start,
                                    segments[i].segment_count);
        if (match) {
            e.run_id.clear();
            ++released;
        }
    }
    save_pending(entries);
    Logger::info("Released {} pending entr{} back to the pool (run '{}')", released,
                 released == 1 ? "y" : "ies", run_id);

    unlock_pending(lock_fd);
}

// TODO: See TD-498 in TECHNICAL_DEBT.md - appends without dedupe or claim check (duplicate rows, verified); the
// registry rewrite runs outside the lock.
void LocalTransport::commit_trained(const std::string& run_id,
                                    const std::vector<DataVersion>& new_entries,
                                    const std::vector<std::string>& trained_paths,
                                    const std::vector<SegmentTarget>& trained_segments) {
    // ── Registry ──────────────────────────────────────────────────────────
    // Load existing registry, append new entries, save.
    std::vector<DataVersion> registry;
    load_registry(registry);
    for (const auto& e : new_entries) {
        registry.push_back(e);
    }
    save_registry(registry);

    // ── Pending ───────────────────────────────────────────────────────────
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::commit_trained — failed to acquire pending lock");
        return;
    }

    std::vector<PendingEntry> pending;
    load_pending(pending);

    const std::set<std::string> trained_set(trained_paths.begin(), trained_paths.end());
    const bool exact = !trained_segments.empty();  // TD-333
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](const PendingEntry& e) {
                                     if (!run_id.empty() && e.run_id != run_id)
                                         return false;
                                     if (!exact)
                                         return trained_set.count(e.path) > 0;
                                     if (e.segment_start < 0)
                                         return trained_set.count(e.path) > 0;
                                     return std::any_of(
                                         trained_segments.begin(), trained_segments.end(),
                                         [&](const SegmentTarget& t) {
                                             return segment_matches(e, t.path, t.segment_start,
                                                                    t.segment_count);
                                         });
                                 }),
                  pending.end());

    save_pending(pending);
    unlock_pending(lock_fd);
}

bool LocalTransport::add_pending(const std::string& path, int segment_start, int segment_count) {
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::add_pending — failed to acquire pending lock");
        return false;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);
    for (const auto& e : entries) {
        if (segment_matches(e, path, segment_start, segment_count)) {
            unlock_pending(lock_fd);
            // TODO: See TD-498 in TECHNICAL_DEBT.md - header (and RemoteTransport) say false for a duplicate.
            return true;  // already queued
        }
    }

    fs::create_directories(fs::path(pending_path_).parent_path());
    std::ofstream file(pending_path_, std::ios::app);
    bool ok;
    if (segment_start < 0) {
        // Whole-file: keep the original minimal single-token line, byte-for-byte identical
        // to every pre-TD-205 add_pending() caller/on-disk file.
        ok = file.is_open() && (file << path << '\n') && file.good();
    } else {
        // TD-205: a segment needs its range recorded, which requires the full tab-delimited
        // row shape load_pending() parses positionally — empty run_id/model_name/source/
        // added_utc/size_bytes/num_entries/checksum columns are fine, load_pending() already
        // defaults each of those for short/legacy lines.
        ok = file.is_open() &&
            (file << path << "\t\t\t\t\t\t\t\t" << segment_start << '\t' << segment_count
                 << '\n') &&
            file.good();
    }
    unlock_pending(lock_fd);
    return ok;
}

namespace {
// "N" -> "0N" for N < 10, otherwise unchanged — matches MNS's run_id padding
// (run-01 ... run-10, run-11, ...). Shared by both transports' next_session().
std::string zero_pad2(int n) {
    const std::string s = std::to_string(n);
    return s.size() < 2 ? "0" + s : s;
}
}  // namespace

std::string LocalTransport::next_session(const std::string& model_name,
                                         const std::string& run_id) {
    if (!session_store_) {
        try {
            const fs::path db_path = fs::path(pending_path_).parent_path() / "session_counters.db";
            fs::create_directories(db_path.parent_path());
            session_store_ = std::make_unique<adai::DaemonConfigStore>(db_path.string());
        } catch (const std::exception& e) {
            Logger::warn(
                "LocalTransport::next_session — session_counters.db unavailable ({}); session "
                "numbers won't persist across restarts",
                e.what());
            return "session-01";
        }
    }

    const std::string key = model_name + "|" + run_id;
    int next = 1;
    // TODO: See TD-304 in TECHNICAL_DEBT.md - non-atomic, unlocked read-modify-write; concurrent processes can get duplicate session numbers (TD-308: full-table scan)
    if (const auto all = session_store_->load_all(); all.count(key)) {
        try {
            next = std::stoi(all.at(key)) + 1;
        } catch (...) {
        }
    }
    session_store_->set(key, std::to_string(next));
    return "session-" + zero_pad2(next);
}

// ── Phase 16: assign-by-count / unassign / delete ──────────────────────────

AssignResult LocalTransport::assign(const std::string& model_name,
                                    const std::vector<std::string>& paths, int count,
                                    const std::vector<SegmentTarget>& segments) {
    AssignResult result;
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::assign — failed to acquire pending lock");
        return result;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);

    if (!segments.empty()) {
        // TD-205: explicit segment targets — highest priority, paths/count ignored.
        for (const auto& t : segments) {
            for (auto& e : entries) {
                if (segment_matches(e, t.path, t.segment_start, t.segment_count)) {
                    e.model_name = model_name;
                    result.segments.push_back(t);
                    break;
                }
            }
        }
    } else {
        const bool by_paths = !paths.empty();
        const bool by_count = !by_paths && count > 0;
        const bool assign_all = !by_paths && !by_count;
        const std::set<std::string> target_paths(paths.begin(), paths.end());
        for (auto& e : entries) {
            if (by_count && static_cast<int>(result.paths.size()) >= count) {
                break;
            }
            // TD-205: by_paths/by_count only ever match whole-file entries — a caller
            // targeting a specific segment must use `segments` above. assign_all (the bulk
            // "assign every pending entry" default) still applies to segments too.
            const bool matches = by_paths ? (e.segment_start < 0 && target_paths.count(e.path) > 0)
                                : by_count ? (e.segment_start < 0 && e.model_name.empty())
                                           : assign_all;
            if (matches) {
                e.model_name = model_name;
                result.paths.push_back(e.path);
            }
        }
    }
    result.assigned = static_cast<int>(result.paths.size() + result.segments.size());

    if (result.assigned > 0) {
        save_pending(entries);
    }
    unlock_pending(lock_fd);
    Logger::info("LocalTransport::assign — model='{}' assigned {} file(s)", model_name,
                result.assigned);
    return result;
}

UnassignResult LocalTransport::unassign(const std::string& model_name,
                                        const std::vector<std::string>& paths, bool force,
                                        const std::vector<SegmentTarget>& segments) {
    UnassignResult result;
    // Defense in depth: DatasetRegistry already rejects this combination before
    // it reaches the transport, but a caller could invoke LocalTransport
    // directly.
    if (paths.empty() && segments.empty() && model_name.empty()) {
        Logger::warn("LocalTransport::unassign — no-op: paths, segments, and model_name all empty");
        return result;
    }

    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::unassign — failed to acquire pending lock");
        return result;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);

    if (!segments.empty()) {
        for (const auto& t : segments) {
            for (auto& e : entries) {
                if (segment_matches(e, t.path, t.segment_start, t.segment_count) &&
                    (model_name.empty() || e.model_name == model_name)) {
                    if (!e.run_id.empty() && !force) {
                        ++result.skipped;
                    } else {
                        e.model_name.clear();
                        result.segments.push_back(t);
                    }
                    break;
                }
            }
        }
    } else {
        // bulk-by-model (paths empty) intentionally matches whole-file AND segment entries
        // alike — "clear every entry assigned to this model" shouldn't need to know which
        // entries happen to be segments. Explicit `paths` matches whole-file entries only.
        const bool bulk_by_model = paths.empty();
        const std::set<std::string> target_paths(paths.begin(), paths.end());
        for (auto& e : entries) {
            const bool matches = bulk_by_model
                                    ? e.model_name == model_name
                                    : (e.segment_start < 0 && target_paths.count(e.path) > 0 &&
                                       (model_name.empty() || e.model_name == model_name));
            if (!matches) {
                continue;
            }
            if (!e.run_id.empty() && !force) {
                ++result.skipped;
                continue;
            }
            e.model_name.clear();
            result.paths.push_back(e.path);
        }
    }
    result.unassigned = static_cast<int>(result.paths.size() + result.segments.size());

    if (result.unassigned > 0) {
        save_pending(entries);
    }
    unlock_pending(lock_fd);
    Logger::info("LocalTransport::unassign — model='{}' unassigned {} file(s), {} skipped",
                model_name, result.unassigned, result.skipped);
    return result;
}

// Note on delete_files containment: unlike RemoteTransport, local mode has no
// "server-owned data_dir" concept — the caller already has full filesystem
// access (Phase 11 fetch/upload are unsupported stubs here, see below), so
// any existing path is eligible for unlinking. The one guard kept is
// self-protection: never unlink this transport's own state files even if
// somehow passed as a delete target.
DeleteResult LocalTransport::delete_paths(const std::vector<std::string>& paths, bool force,
                                          bool delete_files,
                                          const std::vector<SegmentTarget>& segments) {
    DeleteResult result;
    if (paths.empty() && segments.empty()) {
        Logger::warn("LocalTransport::delete_paths — no-op: paths and segments both empty");
        return result;
    }

    // Registry half has no run_id/ownership concept and needs no lock,
    // mirroring commit_trained()'s split locking.
    std::vector<DataVersion> registry;
    load_registry(registry);

    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::delete_paths — failed to acquire pending lock");
        return result;
    }
    std::vector<PendingEntry> pending;
    load_pending(pending);

    bool pending_changed = false;
    bool registry_changed = false;

    // TD-205: one combined worklist — a whole-file path becomes a target with segment_start=-1,
    // so it and an explicit segment target share the exact same per-item logic below via
    // segment_matches().
    std::vector<SegmentTarget> targets;
    targets.reserve(paths.size() + segments.size());
    for (const auto& p : paths)
        targets.push_back({p, -1, -1});
    for (const auto& s : segments)
        targets.push_back(s);

    for (const auto& t : targets) {
        bool pending_blocked = false;
        bool found_anywhere = false;

        auto pit = std::find_if(pending.begin(), pending.end(), [&](const PendingEntry& e) {
            return segment_matches(e, t.path, t.segment_start, t.segment_count);
        });
        if (pit != pending.end()) {
            found_anywhere = true;
            if (!pit->run_id.empty() && !force) {
                pending_blocked = true;
            } else {
                pending.erase(pit);
                pending_changed = true;
            }
        }

        // TD-333: trained records carry their segment. A whole-file target removes every
        // record of the path; a segment target removes exactly that segment's record.
        {
            const auto before = registry.size();
            registry.erase(std::remove_if(registry.begin(), registry.end(),
                                          [&](const DataVersion& dv) {
                                              return t.segment_start < 0
                                                         ? dv.data_file == t.path
                                                         : trained_matches(dv, t.path,
                                                                           t.segment_start,
                                                                           t.segment_count);
                                          }),
                           registry.end());
            if (registry.size() != before) {
                found_anywhere = true;
                registry_changed = true;
            }
        }

        DeleteResult::Detail detail;
        detail.path = t.path;
        detail.segment_start = t.segment_start;
        detail.segment_count = t.segment_count;
        if (pending_blocked) {
            detail.status = "skipped_active_run";
            ++result.skipped;
        } else if (found_anywhere) {
            detail.status = "deleted";
            ++result.deleted;
            const bool is_own_state_file = fs::path(t.path) == fs::path(registry_path_) ||
                                           fs::path(t.path) == fs::path(pending_path_);
            // TD-205: never unlink the physical file for a segment target — other segments
            // (or a legacy whole-file consumer) may still reference the same physical path;
            // only a whole-file target's own bytes are ever eligible for deletion here.
            if (delete_files && t.segment_start < 0 && !is_own_state_file && fs::exists(t.path)) {
                std::error_code ec;
                fs::remove(t.path, ec);
                detail.file_deleted = !ec;
            } else if (delete_files && t.segment_start < 0 && is_own_state_file) {
                Logger::warn(
                    "LocalTransport::delete_paths — refusing to unlink own state file '{}'",
                    t.path);
            }
        } else {
            detail.status = "not_found";
            ++result.not_found;
        }
        result.details.push_back(std::move(detail));
    }

    if (pending_changed) {
        save_pending(pending);
    }
    unlock_pending(lock_fd);
    if (registry_changed) {
        save_registry(registry);
    }

    Logger::info("LocalTransport::delete_paths — {} deleted, {} skipped, {} not_found",
                result.deleted, result.skipped, result.not_found);
    return result;
}

// Phase 11: LocalTransport has no registry_server to delegate downloading to.
std::string LocalTransport::fetch_gutenberg(int /*book_id*/, int /*num_pairs*/,
                                            const std::string& /*model_name*/) {
    Logger::warn(
        "LocalTransport::fetch_gutenberg — not supported in local mode; use DataFetcher directly "
        "and DatasetRegistry::add_file()");
    return "";
}

std::string LocalTransport::fetch_huggingface(const std::string& /*dataset_id*/,
                                              int /*num_pairs*/, const std::string& /*split*/,
                                              const std::string& /*input_field*/,
                                              const std::string& /*output_field*/,
                                              const std::string& /*model_name*/) {
    Logger::warn(
        "LocalTransport::fetch_huggingface — not supported in local mode; use DataFetcher "
        "directly and DatasetRegistry::add_file()");
    return "";
}

std::string LocalTransport::upload_file(const std::string& /*local_path*/) {
    Logger::warn(
        "LocalTransport::upload_file — not supported in local mode; use "
        "DatasetRegistry::add_file() directly");
    return "";
}

RequeueResult LocalTransport::requeue_trained(const std::vector<std::string>& paths, bool all,
                                              const std::string& model_name) {
    RequeueResult result;
    if (paths.empty() && !all) {
        Logger::warn("LocalTransport::requeue_trained — no-op: no paths and all=false");
        return result;
    }

    std::vector<DataVersion> registry;
    load_registry(registry);

    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::requeue_trained — failed to acquire pending lock");
        result.ok = false;
        return result;
    }
    std::vector<PendingEntry> pending;
    load_pending(pending);

    std::vector<std::string> targets;
    for (const auto& p : all ? std::vector<std::string>{} : paths)
        if (std::find(targets.begin(), targets.end(), p) == targets.end())
            targets.push_back(p);
    if (all)
        for (const auto& dv : registry)
            if (std::find(targets.begin(), targets.end(), dv.data_file) == targets.end())
                targets.push_back(dv.data_file);

    bool changed = false;
    for (const auto& path : targets) {
        // TD-333: every trained record of the path -- the whole file, or each trained segment,
        // which comes back as that same segment.
        bool found = false;
        for (auto rit = registry.begin(); rit != registry.end();) {
            if (rit->data_file != path) {
                ++rit;
                continue;
            }
            found = true;
            RequeueResult::Detail detail;
            detail.path = path;
            const bool already_pending =
                std::any_of(pending.begin(), pending.end(), [&](const PendingEntry& e) {
                    return segment_matches(e, path, rit->segment_start, rit->segment_count);
                });
            if (already_pending) {
                detail.status = "already_pending";
                ++result.already_pending;
            } else {
                PendingEntry e;
                e.path = path;
                e.model_name = model_name;
                e.source = rit->source;
                e.added_utc = rit->added_utc;
                e.checksum = rit->checksum;
                e.segment_start = rit->segment_start;
                e.segment_count = rit->segment_count;
                pending.push_back(std::move(e));
                detail.status = "requeued";
                ++result.requeued;
            }
            rit = registry.erase(rit);
            changed = true;
            result.details.push_back(std::move(detail));
        }
        if (!found) {
            result.details.push_back({path, "not_found"});
            ++result.not_found;
        }
    }

    if (changed) {
        save_pending(pending);
    }
    unlock_pending(lock_fd);
    if (changed) {
        save_registry(registry);
    }
    Logger::info("LocalTransport::requeue_trained — {} requeued, {} already pending, {} not "
                 "found",
                 result.requeued, result.already_pending, result.not_found);
    return result;
}

// ============================================================================
// RemoteTransport
// ============================================================================

namespace {
// Minimal JSON helpers shared between RemoteTransport methods ----------------

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

// TODO: See TD-497 in TECHNICAL_DEBT.md - no unescaping (backslashes multiply per round trip) and objects are cut at
// the next '}' even inside strings (a path with '}' disappears); use nlohmann/json.
// Extract the value of a simple JSON string key: "key":"value"
// Returns empty string if not found.
std::string json_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return {};
    const auto start = pos + needle.size();
    const auto end = body.find('"', start);
    if (end == std::string::npos)
        return {};
    return body.substr(start, end - start);
}

// Extract the value of a bare JSON number key: "key":N
int json_int(const std::string& body, const std::string& key, int def = 0) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    try {
        return std::stoi(body.substr(pos + needle.size()));
    } catch (...) {
        return def;
    }
}

// Extract the value of a bare JSON boolean key (true/false, unquoted).
bool json_bool(const std::string& body, const std::string& key, bool def = false) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    const auto start = pos + needle.size();
    return body.compare(start, 4, "true") == 0;
}

// Parse a JSON array of strings from body["key"]: ["a","b",...]
std::vector<std::string> json_string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        // An empty array (or one already fully consumed) closes with ']'
        // before any further '"' — check that first, otherwise an empty
        // array immediately followed by another JSON field would wrongly
        // treat that field's key as an array element.
        const auto quote_pos = body.find('"', cur);
        const auto close_pos = body.find(']', cur);
        if (quote_pos == std::string::npos ||
            (close_pos != std::string::npos && close_pos < quote_pos)) {
            break;
        }
        cur = quote_pos + 1;
        const auto end = body.find('"', cur);
        if (end == std::string::npos)
            break;
        result.push_back(body.substr(cur, end - cur));
        cur = end + 1;
        // Stop at ]
        const auto next = body.find_first_of(",]", cur);
        if (next == std::string::npos || body[next] == ']')
            break;
        cur = next + 1;
    }
    return result;
}

// TD-205: parse a flat JSON array of numbers: [1,2,-1,...] — used for the legacy (non-FTP)
// acquire response's "segment_starts"/"segment_counts" parallel arrays.
std::vector<int> json_int_array(const std::string& body, const std::string& key) {
    std::vector<int> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return result;
    auto cur = pos + needle.size();
    const auto close_pos = body.find(']', cur);
    if (close_pos == std::string::npos)
        return result;
    std::string inner = body.substr(cur, close_pos - cur);
    std::istringstream iss(inner);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (token.empty())
            continue;
        try {
            result.push_back(std::stoi(token));
        } catch (...) {
        }
    }
    return result;
}

// Build a JSON array of quoted strings: ["a","b",...]
std::string json_array(const std::vector<std::string>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i)
            out += ',';
        out += '"';
        out += json_escape(v[i]);
        out += '"';
    }
    out += ']';
    return out;
}

// TD-205: build a JSON array of segment targets:
// [{"path":"...","segment_start":N,"segment_count":N},...]
std::string json_segments_array(const std::vector<SegmentTarget>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i)
            out += ',';
        out += "{\"path\":\"";
        out += json_escape(v[i].path);
        out += "\",\"segment_start\":";
        out += std::to_string(v[i].segment_start);
        out += ",\"segment_count\":";
        out += std::to_string(v[i].segment_count);
        out += "}";
    }
    out += ']';
    return out;
}

// TODO: See TD-498 in TECHNICAL_DEBT.md - https:// is stripped and plain HTTP is sent (default port 8082).
// Parse URL "http://host[:port]" → {host, port}
std::pair<std::string, int> parse_host_port(const std::string& url) {
    std::string s = url;
    for (const char* prefix : {"http://", "https://"}) {
        const std::string p = prefix;
        if (s.rfind(p, 0) == 0) {
            s = s.substr(p.size());
            break;
        }
    }
    // strip any trailing path
    const auto slash = s.find('/');
    if (slash != std::string::npos)
        s = s.substr(0, slash);

    const auto colon = s.rfind(':');
    if (colon != std::string::npos) {
        try {
            return {s.substr(0, colon), std::stoi(s.substr(colon + 1))};
        } catch (...) {
        }
    }
    return {s, 8082};
}

}  // anonymous namespace

RemoteTransport::RemoteTransport(std::string base_url, std::string run_group, int timeout_ms,
                                 std::string dataset_kind)
    : timeout_ms_(timeout_ms) {
    auto [host, port] = parse_host_port(base_url);
    host_ = std::move(host);
    port_ = port;
    group_prefix_ = "/registry/" + run_group + (dataset_kind.empty() ? "" : "/" + dataset_kind);
}

bool RemoteTransport::load_registry(std::vector<DataVersion>& out) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    const auto res = cli.Get((group_prefix_ + "/registry").c_str());
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::load_registry — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/registry");
        return false;
    }

    // Parse JSON array:
    // {"entries":[{"data_file":"...","checksum":"...","num_samples":N,"trained":T},...]}
    out.clear();
    const std::string& body = res->body;
    auto entries_start = body.find("\"entries\":[");
    if (entries_start == std::string::npos)
        return false;
    auto cur = entries_start + 11;  // len of "entries":[
    while (cur < body.size()) {
        auto obj_start = body.find('{', cur);
        if (obj_start == std::string::npos)
            break;
        auto obj_end = body.find('}', obj_start);
        if (obj_end == std::string::npos)
            break;
        const std::string obj = body.substr(obj_start, obj_end - obj_start + 1);
        DataVersion dv;
        dv.data_file = json_string(obj, "data_file");
        dv.checksum = json_string(obj, "checksum");
        dv.segment_start = json_int(obj, "segment_start", -1);  // TD-333; absent = whole file
        dv.segment_count = json_int(obj, "segment_count", -1);
        // num_samples is a JSON number, not a string
        const std::string ns_needle = "\"num_samples\":";
        const auto nsp = obj.find(ns_needle);
        if (nsp != std::string::npos) {
            try {
                dv.num_samples = std::stoi(obj.substr(nsp + ns_needle.size()));
            } catch (...) {
            }
        }
        // trained is a JSON boolean, not a string
        const std::string tr_needle = "\"trained\":";
        const auto trp = obj.find(tr_needle);
        if (trp != std::string::npos) {
            const std::string tval = obj.substr(trp + tr_needle.size(), 4);
            dv.trained = (tval == "true");
        }
        dv.added_utc = json_string(obj, "added_utc");
        dv.source = json_string(obj, "source");
        if (!dv.data_file.empty())
            out.push_back(std::move(dv));
        cur = obj_end + 1;
        if (body.find(']', cur) < body.find('{', cur))
            break;
    }
    return true;
#else
    Logger::error(
        "RemoteTransport::load_registry — not compiled (BUILD_METRICS_API_SERVER not set)");
    return false;
#endif
}

bool RemoteTransport::save_registry(const std::vector<DataVersion>& /*entries*/) {
    // In distributed mode the server owns the registry; clients use commit_trained().
    Logger::warn(
        "RemoteTransport::save_registry — no-op; use mark_trained(run_id,...) in distributed mode");
    return false;
}

bool RemoteTransport::load_pending(std::vector<PendingEntry>& out) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    const auto res = cli.Get((group_prefix_ + "/queue").c_str());
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::load_pending — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/queue");
        return false;
    }

    // Parse {"entries":[{"path":"...","run_id":"..."},...]}
    out.clear();
    const std::string& body = res->body;
    auto cur = body.find("\"entries\":[");
    if (cur == std::string::npos)
        return false;
    cur += 11;
    while (cur < body.size()) {
        auto obj_start = body.find('{', cur);
        if (obj_start == std::string::npos)
            break;
        auto obj_end = body.find('}', obj_start);
        if (obj_end == std::string::npos)
            break;
        const std::string obj = body.substr(obj_start, obj_end - obj_start + 1);
        const std::string path = json_string(obj, "path");
        if (!path.empty()) {
            PendingEntry e;
            e.path = path;
            e.run_id = json_string(obj, "run_id");
            e.model_name = json_string(obj, "model_name");
            e.source = json_string(obj, "source");
            e.added_utc = json_string(obj, "added_utc");
            e.checksum = json_string(obj, "checksum");
            const std::string sb_needle = "\"size_bytes\":";
            const auto sbp = obj.find(sb_needle);
            if (sbp != std::string::npos) {
                try {
                    e.size_bytes = static_cast<std::size_t>(std::stoull(obj.substr(sbp + sb_needle.size())));
                } catch (...) {
                }
            }
            const std::string ne_needle = "\"num_entries\":";
            const auto nep = obj.find(ne_needle);
            if (nep != std::string::npos) {
                try {
                    e.num_entries = std::stoi(obj.substr(nep + ne_needle.size()));
                } catch (...) {
                }
            }
            const std::string ss_needle = "\"segment_start\":";
            const auto ssp = obj.find(ss_needle);
            if (ssp != std::string::npos) {
                try {
                    e.segment_start = std::stoi(obj.substr(ssp + ss_needle.size()));
                } catch (...) {
                }
            }
            const std::string sc_needle = "\"segment_count\":";
            const auto scp = obj.find(sc_needle);
            if (scp != std::string::npos) {
                try {
                    e.segment_count = std::stoi(obj.substr(scp + sc_needle.size()));
                } catch (...) {
                }
            }
            out.push_back(std::move(e));
        }
        cur = obj_end + 1;
        if (body.find(']', cur) < body.find('{', cur))
            break;
    }
    return true;
#else
    Logger::error("RemoteTransport::load_pending — not compiled");
    return false;
#endif
}

bool RemoteTransport::save_pending(const std::vector<PendingEntry>& /*entries*/) {
    Logger::warn(
        "RemoteTransport::save_pending — no-op; use acquire/release/commit_trained in distributed "
        "mode");
    return false;
}

AcquireResponse RemoteTransport::acquire(const std::string& run_id, int max_files,
                                         const std::string& model_name) {
    AcquireResponse resp;
    resp.run_id = run_id;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape(run_id) << "\"," << "\"max_files\":" << max_files << ","
         << "\"model_name\":\"" << json_escape(model_name) << "\"}";

    const auto res = cli.Post((group_prefix_ + "/acquire").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::acquire — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/acquire");
        return resp;
    }

    const std::string& b = res->body;

    // Detect new structured response: has "files":[ array.
    // Old response format: {"acquired":["path",...]}
    const bool new_format = (b.find("\"files\":[") != std::string::npos);

    if (new_format) {
        // Parse top-level scalar fields
        resp.ftp_server_host = json_string(b, "ftp_server_host");
        // ftp_server_port is a JSON number, not a string
        const std::string port_needle = "\"ftp_server_port\":";
        const auto pp = b.find(port_needle);
        if (pp != std::string::npos) {
            try {
                resp.ftp_server_port = std::stoi(b.substr(pp + port_needle.size()));
            } catch (...) {
            }
        }
        // ftps_enabled is a JSON boolean (Phase 3)
        const std::string ftps_needle = "\"ftps_enabled\":";
        const auto fp = b.find(ftps_needle);
        if (fp != std::string::npos) {
            const std::string fval = b.substr(fp + ftps_needle.size(), 5);
            resp.ftps_enabled = (fval.compare(0, 4, "true") == 0);
        }

        // Parse each file object inside "files":[{...},{...}]
        auto files_pos = b.find("\"files\":[");
        if (files_pos != std::string::npos) {
            auto cur = files_pos + 9;  // len("\"files\":[")
            while (cur < b.size()) {
                auto obj_start = b.find('{', cur);
                if (obj_start == std::string::npos)
                    break;
                // Find matching closing brace (objects are flat, no nesting)
                auto obj_end = b.find('}', obj_start);
                if (obj_end == std::string::npos)
                    break;
                const std::string obj = b.substr(obj_start, obj_end - obj_start + 1);

                FileToken tok;
                tok.registry_path = json_string(obj, "registry_path");
                tok.ftp_path = json_string(obj, "ftp_path");
                tok.ftp_username = json_string(obj, "ftp_username");
                tok.ftp_password = json_string(obj, "ftp_password");
                tok.checksum = json_string(obj, "checksum");
                tok.token_expires_utc = json_string(obj, "token_expires_utc");
                // size_bytes is a JSON number
                const std::string sb_needle = "\"size_bytes\":";
                const auto sbp = obj.find(sb_needle);
                if (sbp != std::string::npos) {
                    try {
                        tok.size_bytes = static_cast<std::size_t>(
                            std::stoull(obj.substr(sbp + sb_needle.size())));
                    } catch (...) {
                    }
                }
                // TD-205: segment_start/segment_count are JSON numbers, same pattern as
                // size_bytes above.
                const std::string ss_needle = "\"segment_start\":";
                const auto ssp = obj.find(ss_needle);
                if (ssp != std::string::npos) {
                    try {
                        tok.segment_start = std::stoi(obj.substr(ssp + ss_needle.size()));
                    } catch (...) {
                    }
                }
                const std::string sc_needle = "\"segment_count\":";
                const auto scp = obj.find(sc_needle);
                if (scp != std::string::npos) {
                    try {
                        tok.segment_count = std::stoi(obj.substr(scp + sc_needle.size()));
                    } catch (...) {
                    }
                }
                if (!tok.registry_path.empty()) {
                    resp.files.push_back(std::move(tok));
                }
                cur = obj_end + 1;
                if (b.find(']', cur) < b.find('{', cur))
                    break;
            }
        }
    } else {
        // Legacy format: {"acquired":["path1","path2",...]}
        // Wrap each path in a minimal FileToken with ftp fields empty.
        // TD-205: "segment_starts"/"segment_counts" are new, purely-additive parallel arrays
        // (same order/length as "acquired") — an older server predating this field simply
        // omits them, and json_int_array() returns empty, so every entry defaults to -1
        // (whole file), matching pre-TD-205 behavior byte-for-byte.
        auto paths = json_string_array(b, "acquired");
        const auto segment_starts = json_int_array(b, "segment_starts");
        const auto segment_counts = json_int_array(b, "segment_counts");
        for (std::size_t i = 0; i < paths.size(); ++i) {
            FileToken tok;
            tok.registry_path = std::move(paths[i]);
            if (i < segment_starts.size())
                tok.segment_start = segment_starts[i];
            if (i < segment_counts.size())
                tok.segment_count = segment_counts[i];
            resp.files.push_back(std::move(tok));
        }
    }

    Logger::info("RemoteTransport: acquired {} files for run '{}'", resp.files.size(), run_id);
#else
    Logger::error("RemoteTransport::acquire — not compiled");
#endif
    return resp;
}

void RemoteTransport::release(const std::string& run_id, const std::vector<std::string>& paths,
                              const std::vector<SegmentTarget>& segments) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape(run_id) << "\"," << "\"files\":" << json_array(paths)
         << "," << "\"segments\":" << json_segments_array(segments) << "}";

    const auto res = cli.Post((group_prefix_ + "/release").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::release — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/release");
    }
#else
    Logger::error("RemoteTransport::release — not compiled");
#endif
}

void RemoteTransport::commit_trained(const std::string& run_id,
                                     const std::vector<DataVersion>& new_entries,
                                     const std::vector<std::string>& trained_paths,
                                     const std::vector<SegmentTarget>& trained_segments) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    // TODO: See TD-498 in TECHNICAL_DEBT.md - model_id is never sent (and never set), so history?model_id= never matches.
    // One aligned entry per trained unit: files[i], samples[i] and (TD-333, exact mode)
    // segment_starts[i]/segment_counts[i]. new_entries first (they carry the sample counts), then
    // any trained path/segment already recorded, so its pending entry is still removed.
    std::vector<std::string> files;
    std::vector<int> samples, seg_starts, seg_counts;
    auto has_unit = [&](const std::string& path, int start, int count) {
        for (std::size_t i = 0; i < files.size(); ++i)
            if (files[i] == path && seg_starts[i] == start && seg_counts[i] == count)
                return true;
        return false;
    };
    auto add_unit = [&](const std::string& path, int start, int count, int n) {
        if (!has_unit(path, start, count)) {
            files.push_back(path);
            samples.push_back(n);
            seg_starts.push_back(start);
            seg_counts.push_back(count);
        }
    };
    for (const auto& e : new_entries)
        add_unit(e.data_file, e.segment_start, e.segment_count, e.num_samples);
    for (const auto& p : trained_paths)
        add_unit(p, -1, -1, 0);
    for (const auto& t : trained_segments)
        add_unit(t.path, t.segment_start, t.segment_count, 0);

    auto int_array = [](const std::vector<int>& v) {
        std::string out = "[";
        for (std::size_t i = 0; i < v.size(); ++i)
            out += (i ? "," : "") + std::to_string(v[i]);
        return out + "]";
    };
    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape(run_id) << "\"," << "\"files\":" << json_array(files)
         << ",\"samples\":" << int_array(samples);
    if (!trained_segments.empty()) {
        // Exact mode: the server removes exactly these pending entries. Without the arrays it
        // keeps the pre-TD-333 behaviour (every entry sharing a path), for older clients.
        body << ",\"segment_starts\":" << int_array(seg_starts)
             << ",\"segment_counts\":" << int_array(seg_counts);
    }
    body << "}";

    const auto res = cli.Post((group_prefix_ + "/trained").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::commit_trained — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/trained");
    }
#else
    Logger::error("RemoteTransport::commit_trained — not compiled");
#endif
}

bool RemoteTransport::add_pending(const std::string& path, int segment_start,
                                  int segment_count) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"path\":\"" << json_escape(path) << "\"," << "\"segment_start\":" << segment_start
         << ",\"segment_count\":" << segment_count << "}";

    const auto res =
        cli.Post((group_prefix_ + "/pending/add").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::add_pending — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/pending/add");
        return false;
    }
    if (res->body.find("\"added\":false") != std::string::npos) {
        Logger::warn("RemoteTransport::add_pending — '{}' already in remote queue", path);
        return false;
    }
    return true;
#else
    Logger::error("RemoteTransport::add_pending — not compiled");
    return false;
#endif
}

std::string RemoteTransport::next_session(const std::string& model_name,
                                          const std::string& run_id) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(model_name) << "\"," << "\"run_id\":\""
         << json_escape(run_id) << "\"}";

    const auto res =
        cli.Post((group_prefix_ + "/session/next").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::next_session — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/session/next");
        return "session-01";
    }
    return json_string(res->body, "session_id");
#else
    Logger::error("RemoteTransport::next_session — not compiled");
    return "session-01";
#endif
}

// Phase 16: assign-by-count / unassign / delete ------------------------------

AssignResult RemoteTransport::assign(const std::string& model_name,
                                     const std::vector<std::string>& paths, int count,
                                     const std::vector<SegmentTarget>& segments) {
    AssignResult result;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(model_name) << "\","
         << "\"paths\":" << json_array(paths) << "," << "\"count\":" << count << ","
         << "\"segments\":" << json_segments_array(segments) << "}";

    const auto res = cli.Post((group_prefix_ + "/assign").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::assign — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/assign");
        return result;
    }
    // TODO: See TD-498 in TECHNICAL_DEBT.md - response "segments" not parsed; result.segments stays empty.
    result.assigned = json_int(res->body, "assigned");
    result.paths = json_string_array(res->body, "paths");
    Logger::info("RemoteTransport: assigned {} files to model '{}'", result.assigned, model_name);
#else
    Logger::error("RemoteTransport::assign — not compiled");
#endif
    return result;
}

UnassignResult RemoteTransport::unassign(const std::string& model_name,
                                         const std::vector<std::string>& paths, bool force,
                                         const std::vector<SegmentTarget>& segments) {
    UnassignResult result;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(model_name) << "\","
         << "\"paths\":" << json_array(paths) << "," << "\"force\":" << (force ? "true" : "false")
         << "," << "\"segments\":" << json_segments_array(segments) << "}";

    const auto res =
        cli.Post((group_prefix_ + "/unassign").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::unassign — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/unassign");
        return result;
    }
    result.unassigned = json_int(res->body, "unassigned");
    result.skipped = json_int(res->body, "skipped");
    result.paths = json_string_array(res->body, "paths");
    Logger::info("RemoteTransport: unassigned {} files from model '{}' ({} skipped)",
                result.unassigned, model_name, result.skipped);
#else
    Logger::error("RemoteTransport::unassign — not compiled");
#endif
    return result;
}

DeleteResult RemoteTransport::delete_paths(const std::vector<std::string>& paths, bool force,
                                           bool delete_files,
                                           const std::vector<SegmentTarget>& segments) {
    DeleteResult result;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"paths\":" << json_array(paths) << "," << "\"force\":" << (force ? "true" : "false")
         << "," << "\"delete_files\":" << (delete_files ? "true" : "false") << ","
         << "\"segments\":" << json_segments_array(segments) << "}";

    const auto res = cli.Post((group_prefix_ + "/delete").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::delete_paths — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/delete");
        return result;
    }
    const std::string& b = res->body;
    result.deleted = json_int(b, "deleted");
    result.skipped = json_int(b, "skipped");
    result.not_found = json_int(b, "not_found");

    // Parse "details":[{"path":"...","status":"...","file_deleted":bool},...]
    // — flat (non-nested) objects, same iteration idiom as load_registry/
    // load_pending/acquire above.
    auto cur = b.find("\"details\":[");
    if (cur != std::string::npos) {
        cur += 11;  // len of "details":[
        while (cur < b.size()) {
            auto obj_start = b.find('{', cur);
            if (obj_start == std::string::npos)
                break;
            auto obj_end = b.find('}', obj_start);
            if (obj_end == std::string::npos)
                break;
            const std::string obj = b.substr(obj_start, obj_end - obj_start + 1);
            DeleteResult::Detail detail;
            detail.path = json_string(obj, "path");
            detail.status = json_string(obj, "status");
            detail.file_deleted = json_bool(obj, "file_deleted");
            detail.segment_start = json_int(obj, "segment_start", -1);
            detail.segment_count = json_int(obj, "segment_count", -1);
            result.details.push_back(std::move(detail));
            cur = obj_end + 1;
            if (b.find(']', cur) < b.find('{', cur))
                break;
        }
    }
    Logger::info("RemoteTransport: delete — {} deleted, {} skipped, {} not_found", result.deleted,
                result.skipped, result.not_found);
#else
    Logger::error("RemoteTransport::delete_paths — not compiled");
#endif
    return result;
}

// Phase 11: server-side dataset fetch ---------------------------------------

std::string RemoteTransport::fetch_gutenberg(int book_id, int num_pairs,
                                             const std::string& model_name) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    // Fetching a book + converting it can take a while; give the server room.
    cli.set_read_timeout(std::max(timeout_ms_, 120000) / 1000, 0);

    std::ostringstream body;
    body << "{\"book_id\":" << book_id << ",\"num_pairs\":" << num_pairs << ",\"model_name\":\""
         << json_escape(model_name) << "\"}";

    const auto res =
        cli.Post((group_prefix_ + "/fetch/gutenberg").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::fetch_gutenberg — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/fetch/gutenberg");
        return "";
    }
    if (res->body.find("\"added\":true") == std::string::npos) {
        Logger::error("RemoteTransport::fetch_gutenberg — registry reported failure: {}",
                      res->body);
        return "";
    }
    return json_string(res->body, "path");
#else
    Logger::error("RemoteTransport::fetch_gutenberg — not compiled");
    return "";
#endif
}

std::string RemoteTransport::fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                               const std::string& split,
                                               const std::string& input_field,
                                               const std::string& output_field,
                                               const std::string& model_name) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    // HuggingFace downloads can be large; give the server room.
    cli.set_read_timeout(std::max(timeout_ms_, 120000) / 1000, 0);

    std::ostringstream body;
    body << "{\"dataset_id\":\"" << json_escape(dataset_id) << "\"," << "\"num_pairs\":"
         << num_pairs << "," << "\"split\":\"" << json_escape(split) << "\","
         << "\"input_field\":\"" << json_escape(input_field) << "\"," << "\"output_field\":\""
         << json_escape(output_field) << "\"," << "\"model_name\":\"" << json_escape(model_name)
         << "\"}";

    const auto res =
        cli.Post((group_prefix_ + "/fetch/huggingface").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::fetch_huggingface — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/fetch/huggingface");
        return "";
    }
    if (res->body.find("\"added\":true") == std::string::npos) {
        Logger::error("RemoteTransport::fetch_huggingface — registry reported failure: {}",
                      res->body);
        return "";
    }
    return json_string(res->body, "path");
#else
    Logger::error("RemoteTransport::fetch_huggingface — not compiled");
    return "";
#endif
}

std::string RemoteTransport::upload_file(const std::string& local_path) {
#ifdef BUILD_METRICS_API_SERVER
    std::ifstream in(local_path, std::ios::binary);
    if (!in.is_open()) {
        Logger::error("RemoteTransport::upload_file — cannot open local file: {}", local_path);
        return "";
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string contents = buf.str();

    const std::string filename = fs::path(local_path).filename().string();

    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    // Upload duration scales with file size; give the server room.
    cli.set_read_timeout(std::max(timeout_ms_, 120000) / 1000, 0);
    cli.set_write_timeout(std::max(timeout_ms_, 120000) / 1000, 0);

    const std::string path =
        group_prefix_ + "/upload?filename=" + httplib::detail::encode_query_param(filename);
    const auto res = cli.Post(path.c_str(), contents, "application/octet-stream");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::upload_file — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, path);
        return "";
    }
    if (res->body.find("\"added\":true") == std::string::npos) {
        Logger::error("RemoteTransport::upload_file — registry reported failure: {}", res->body);
        return "";
    }
    return json_string(res->body, "path");
#else
    Logger::error("RemoteTransport::upload_file — not compiled");
    return "";
#endif
}

RequeueResult RemoteTransport::requeue_trained(const std::vector<std::string>& paths, bool all,
                                               const std::string& model_name) {
    RequeueResult result;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"paths\":" << json_array(paths) << ",\"all\":" << (all ? "true" : "false")
         << ",\"model_name\":\"" << json_escape(model_name) << "\"}";
    const auto res =
        cli.Post((group_prefix_ + "/requeue").c_str(), body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::requeue_trained — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/requeue");
        result.ok = false;
        return result;
    }
    const std::string& b = res->body;
    result.requeued = json_int(b, "requeued");
    result.already_pending = json_int(b, "already_pending");
    result.not_found = json_int(b, "not_found");
    auto cur = b.find("\"details\":[");
    if (cur != std::string::npos) {
        cur += 11;
        while (cur < b.size()) {
            const auto obj_start = b.find('{', cur);
            if (obj_start == std::string::npos)
                break;
            const auto obj_end = b.find('}', obj_start);
            if (obj_end == std::string::npos)
                break;
            const std::string obj = b.substr(obj_start, obj_end - obj_start + 1);
            RequeueResult::Detail detail;
            detail.path = json_string(obj, "path");
            detail.status = json_string(obj, "status");
            result.details.push_back(std::move(detail));
            cur = obj_end + 1;
        }
    }
#else
    (void)paths;
    (void)all;
    (void)model_name;
    Logger::error("RemoteTransport::requeue_trained — built without HTTP support");
    result.ok = false;
#endif
    return result;
}
