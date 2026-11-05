// @adai-status: stable
// @adai-version: 1.10.0
// @adai-reviewed: 2026-11-05

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
#include "AtomicFile.hpp"
#include "Logger.hpp"
#include "nlohmann/json.hpp"

#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

using adai::Logger;
namespace fs = std::filesystem;

// ── TD-496: file formats ──────────────────────────────────────────────────────
//
// Both files are JSON Lines: a header line, then one JSON object per entry. They used to be a
// space-separated registry read with >> and a tab-separated queue, so '/data/my file.jsonl' came
// back as data_file '/data/my', checksum 'file.jsonl', trained=false, and a tab or newline in a
// path broke the queue the same way. A file without the header is read in the old format and is
// written in the new one by the next save.

namespace {

constexpr const char* kRegistryHeader = "# adai data registry, JSON Lines v2";
constexpr const char* kPendingHeader = "# adai pending queue, JSON Lines v2";

std::string dump_line(const nlohmann::json& j) {
    // replace: a path stored before TD-496 may hold invalid UTF-8; dump() would otherwise throw.
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string json_str(const nlohmann::json& j, const char* key) {
    const auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : std::string();
}

template <typename T>
T json_num(const nlohmann::json& j, const char* key, T fallback) {
    const auto it = j.find(key);
    return (it != j.end() && it->is_number_integer()) ? it->get<T>() : fallback;
}

// Reads @p path's lines; @p is_jsonl is set when the first line is @p header (it is not returned).
bool read_lines(const std::string& path, const char* header, std::vector<std::string>& lines,
                bool& is_jsonl) {
    if (!fs::exists(path)) {
        return false;
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    lines.clear();
    is_jsonl = false;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (first) {
            first = false;
            if (line == header) {
                is_jsonl = true;
                continue;
            }
        }
        lines.push_back(std::move(line));
    }
    return true;
}

// Parses one JSON Lines entry; logs and returns a discarded value when it isn't an object.
nlohmann::json parse_entry(const std::string& line, const std::string& file) {
    auto j = nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) {
        Logger::error("LocalTransport: skipping an unreadable line in '{}': {}", file, line);
        return nlohmann::json(nlohmann::json::value_t::discarded);
    }
    return j;
}

// The pre-TD-496 trained registry line: space-separated columns, "-" for an empty value.
DataVersion parse_legacy_registry_line(const std::string& line) {
    std::istringstream iss(line);
    DataVersion dv;
    int trained_int = 0;
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
    return dv;
}

// The pre-TD-496 queue line:
// path[\trun_id[\tmodel_name[\tsource[\tadded_utc[\tsize_bytes[\tnum_entries[\tchecksum[\tsegment_start[\tsegment_count[\tclaimed_unix]]]]]]]]]]
// Each trailing column was added in a later phase (run_id: Phase 9, model_name: Phase 11-ish,
// source/added_utc/size_bytes/num_entries/checksum: Phase 15, segment_start/segment_count:
// TD-205, claimed_unix: TD-494) — split on tab and assign positionally so shorter lines from
// older files keep loading with defaults for the columns they predate.
PendingEntry parse_legacy_pending_line(const std::string& line) {
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
    if (cols.size() > 10 && !cols[10].empty()) {  // TD-494
        try {
            e.claimed_unix = std::stoll(cols[10]);
        } catch (...) {
        }
    }
    return e;
}

}  // namespace

std::string registry_path_problem(const std::string& path) {
    if (path.empty()) {
        return "path is empty";
    }
    for (std::size_t i = 0; i < path.size();) {
        const auto c = static_cast<unsigned char>(path[i]);
        if (c < 0x20 || c == 0x7F) {
            return "path contains a control character (tab, newline, ...)";
        }
        // UTF-8: the sequence length from the lead byte, then that many continuation bytes,
        // rejecting overlong forms, surrogates and code points above U+10FFFF.
        std::size_t len = 1;
        unsigned int cp = c;
        if (c >= 0x80) {
            if (c >= 0xC2 && c <= 0xDF) {
                len = 2;
                cp = c & 0x1F;
            } else if (c >= 0xE0 && c <= 0xEF) {
                len = 3;
                cp = c & 0x0F;
            } else if (c >= 0xF0 && c <= 0xF4) {
                len = 4;
                cp = c & 0x07;
            } else {
                return "path is not valid UTF-8";
            }
            if (i + len > path.size()) {
                return "path is not valid UTF-8";
            }
            for (std::size_t k = 1; k < len; ++k) {
                const auto cc = static_cast<unsigned char>(path[i + k]);
                if ((cc & 0xC0) != 0x80) {
                    return "path is not valid UTF-8";
                }
                cp = (cp << 6) | (cc & 0x3F);
            }
            if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
                (cp >= 0xD800 && cp <= 0xDFFF)) {
                return "path is not valid UTF-8";
            }
            if (cp >= 0x80 && cp <= 0x9F) {
                return "path contains a control character (tab, newline, ...)";
            }
        }
        i += len;
    }
    return {};
}

// ============================================================================
// LocalTransport
// ============================================================================

LocalTransport::LocalTransport(std::string registry_path, std::string pending_path)
    : registry_path_(std::move(registry_path)), pending_path_(std::move(pending_path)) {}

bool LocalTransport::load_registry(std::vector<DataVersion>& out) {
    std::vector<std::string> lines;
    bool is_jsonl = false;
    if (!read_lines(registry_path_, kRegistryHeader, lines, is_jsonl)) {
        return false;
    }

    out.clear();
    for (const auto& line : lines) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!is_jsonl) {
            out.push_back(parse_legacy_registry_line(line));
            continue;
        }
        const auto j = parse_entry(line, registry_path_);
        if (j.is_discarded()) {
            continue;
        }
        DataVersion dv;
        dv.data_file = json_str(j, "data_file");
        dv.checksum = json_str(j, "checksum");
        dv.num_samples = json_num(j, "num_samples", 0);
        const auto trained = j.find("trained");
        dv.trained = trained != j.end() && trained->is_boolean() && trained->get<bool>();
        dv.model_id = json_str(j, "model_id");
        dv.added_utc = json_str(j, "added_utc");
        dv.source = json_str(j, "source");
        dv.segment_start = json_num(j, "segment_start", -1);
        dv.segment_count = json_num(j, "segment_count", -1);
        out.push_back(std::move(dv));
    }

    return true;
}

// TD-495: written to a temporary file and renamed over the registry, so a failed write (disk full)
// leaves the old file whole; it used to truncate and rewrite in place.
bool LocalTransport::save_registry(const std::vector<DataVersion>& entries) {
    std::ostringstream file;
    file << kRegistryHeader << '\n';
    for (const auto& dv : entries) {
        file << dump_line({{"data_file", dv.data_file},
                           {"checksum", dv.checksum},
                           {"num_samples", dv.num_samples},
                           {"trained", dv.trained},
                           {"model_id", dv.model_id},
                           {"added_utc", dv.added_utc},
                           {"source", dv.source},
                           {"segment_start", dv.segment_start},
                           {"segment_count", dv.segment_count}})
             << '\n';
    }

    const std::string error = adai::write_file_atomically(registry_path_, file.str());
    if (!error.empty()) {
        Logger::error("LocalTransport: could not save the trained registry: {}", error);
        return false;
    }
    return true;
}

bool LocalTransport::load_pending(std::vector<PendingEntry>& out) {
    std::vector<std::string> lines;
    bool is_jsonl = false;
    if (!read_lines(pending_path_, kPendingHeader, lines, is_jsonl)) {
        return false;
    }

    out.clear();
    for (const auto& line : lines) {
        if (line.empty())
            continue;
        if (!is_jsonl) {
            out.push_back(parse_legacy_pending_line(line));
            continue;
        }
        if (line[0] == '#')
            continue;
        const auto j = parse_entry(line, pending_path_);
        if (j.is_discarded()) {
            continue;
        }
        PendingEntry e;
        e.path = json_str(j, "path");
        e.run_id = json_str(j, "run_id");
        e.model_name = json_str(j, "model_name");
        e.source = json_str(j, "source");
        e.added_utc = json_str(j, "added_utc");
        e.size_bytes = json_num<std::size_t>(j, "size_bytes", 0);
        e.num_entries = json_num(j, "num_entries", -1);
        e.checksum = json_str(j, "checksum");
        e.segment_start = json_num(j, "segment_start", -1);
        e.segment_count = json_num(j, "segment_count", -1);
        e.claimed_unix = json_num<long long>(j, "claimed_unix", 0);
        out.push_back(std::move(e));
    }

    return true;
}

// TD-495: written to a temporary file and renamed over the queue (a simulated disk-full rewrite in
// place used to keep 114 of 2,001 entries).
bool LocalTransport::save_pending(const std::vector<PendingEntry>& entries) {
    std::ostringstream file;
    file << kPendingHeader << '\n';
    for (const auto& e : entries) {
        file << dump_line({{"path", e.path},
                           {"run_id", e.run_id},
                           {"model_name", e.model_name},
                           {"source", e.source},
                           {"added_utc", e.added_utc},
                           {"size_bytes", e.size_bytes},
                           {"num_entries", e.num_entries},
                           {"checksum", e.checksum},
                           {"segment_start", e.segment_start},
                           {"segment_count", e.segment_count},
                           {"claimed_unix", e.claimed_unix}})
             << '\n';
    }

    const std::string error = adai::write_file_atomically(pending_path_, file.str());
    if (!error.empty()) {
        Logger::error("LocalTransport: could not save the pending queue: {}", error);
        return false;
    }
    return true;
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
        resp.failed = true;
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
            // TD-494: recorded for display; standalone claims have no lease (claim_lease_seconds 0).
            e.claimed_unix = static_cast<long long>(std::time(nullptr));
            FileToken tok;
            tok.registry_path = e.path;
            tok.segment_start = e.segment_start;
            tok.segment_count = e.segment_count;
            resp.files.push_back(std::move(tok));
        }
    }

    if (!resp.files.empty()) {
        if (!save_pending(entries)) {  // TD-495: nothing was claimed
            resp.files.clear();
            resp.failed = true;
        } else {
            Logger::info("Acquired {} pending files for run '{}'", resp.files.size(), run_id);
        }
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
    if (!save_pending(entries)) {  // TD-495
        Logger::error("LocalTransport::release — the release wasn't saved");
    }
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
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::commit_trained — failed to acquire pending lock");
        return;
    }

    std::vector<PendingEntry> pending;
    load_pending(pending);

    // ── Registry ──────────────────────────────────────────────────────────
    // TD-493, as in registry_server's handle_trained(): with a run_id, an entry is recorded only
    // when its unit is pending and claimed by that run (even if an earlier pass recorded it too).
    // Without one, entries are recorded as given (DatasetRegistry already skipped recorded units).
    std::vector<DataVersion> registry;
    load_registry(registry);
    int rejected = 0;
    for (const auto& e : new_entries) {
        const bool claimed =
            run_id.empty() ||
            std::any_of(pending.begin(), pending.end(), [&](const PendingEntry& p) {
                return p.run_id == run_id &&
                       (e.segment_start < 0 ? p.path == e.data_file
                                            : segment_matches(p, e.data_file, e.segment_start,
                                                              e.segment_count));
            });
        if (claimed)
            registry.push_back(e);
        else
            ++rejected;
    }
    if (!save_registry(registry)) {  // TD-495
        Logger::error("LocalTransport::commit_trained — the trained records weren't saved");
    }
    if (rejected > 0)
        Logger::warn("LocalTransport::commit_trained — run '{}' doesn't hold the claim on {} "
                     "unit(s); not recorded",
                     run_id, rejected);

    // ── Pending ───────────────────────────────────────────────────────────

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

    if (!save_pending(pending)) {  // TD-495
        Logger::error(
            "LocalTransport::commit_trained — trained entries weren't removed from the queue");
    }
    unlock_pending(lock_fd);
}

bool LocalTransport::add_pending(const std::string& path, int segment_start, int segment_count) {
    if (const std::string problem = registry_path_problem(path); !problem.empty()) {
        Logger::error("LocalTransport::add_pending — refusing '{}': {}", path, problem);
        return false;
    }
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

    // TD-495: the whole queue is rewritten atomically; appending in place could leave a cut-off
    // line (a wrong path) when the write failed part-way.
    PendingEntry entry;
    entry.path = path;
    entry.segment_start = segment_start;
    entry.segment_count = segment_count;
    entries.push_back(std::move(entry));
    const bool ok = save_pending(entries);
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
            session_store_ = std::make_unique<adai::SessionCounterStore>(db_path.string());
        } catch (const std::exception& e) {
            Logger::warn(
                "LocalTransport::next_session — session_counters.db unavailable ({}); session "
                "numbers won't persist across restarts",
                e.what());
            return "session-01";
        }
    }

    // TD-304: one transaction, so two standalone trainers sharing session_counters.db never get the
    // same number (a load_all() + set() read-modify-write could, and a concurrent write failed at once).
    // A throw reaches IncrementalTrainer::begin_run(), which continues without a session id.
    // TD-308: the counters have their own store and table rather than DaemonConfigStore's.
    const int next = session_store_->next(model_name + "|" + run_id);
    return "session-" + zero_pad2(next);
}

// ── Phase 16: assign-by-count / unassign / delete ──────────────────────────

AssignResult LocalTransport::assign(const std::string& model_name,
                                    const std::vector<std::string>& paths, int count,
                                    const std::vector<SegmentTarget>& segments) {
    AssignResult result;
    // TD-321: a negative count used to mean "every pending entry", as 0 still does.
    if (count < 0) {
        Logger::error("LocalTransport::assign — count {} refused (0 = all, else a positive number)",
                      count);
        return result;
    }
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

    if (result.assigned > 0 && !save_pending(entries)) {  // TD-495: nothing was assigned
        result.assigned = 0;
        result.paths.clear();
        result.segments.clear();
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

    if (result.unassigned > 0 && !save_pending(entries)) {  // TD-495: nothing was unassigned
        result.unassigned = 0;
        result.paths.clear();
        result.segments.clear();
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

    if (pending_changed && !save_pending(pending)) {  // TD-495
        Logger::error("LocalTransport::delete_paths — the pending queue wasn't saved");
    }
    unlock_pending(lock_fd);
    if (registry_changed && !save_registry(registry)) {  // TD-495
        Logger::error("LocalTransport::delete_paths — the trained registry wasn't saved");
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

    // TD-495: report a save that failed; pending is written first, so a failed registry save leaves
    // the file both pending and trained (it trains again, nothing is lost).
    if (changed && !save_pending(pending)) {
        result.ok = false;
    }
    unlock_pending(lock_fd);
    if (changed && result.ok && !save_registry(registry)) {
        result.ok = false;
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
// JSON helpers shared between RemoteTransport methods -------------------------
//
// TD-497: requests are built and replies parsed with the vendored nlohmann/json. The old
// helpers sliced each object at the next '}' and read strings up to the next '"' with no
// unescaping, so a path containing '}' vanished from load_pending(), a '"' cut a path short,
// and each backslash came back quadrupled ('C:\data' as 'C:\\\\data').

using Json = nlohmann::json;

// A request body. Invalid UTF-8 is sent as U+FFFD rather than making dump() throw.
std::string to_body(const Json& j) {
    return dump_line(j);
}

// A reply's top-level object, or a discarded value (logged) when the reply isn't one.
Json parse_reply(const std::string& body, const char* method) {
    auto j = Json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) {
        Logger::error("RemoteTransport::{} — unreadable reply: {}", method, body.substr(0, 200));
        return Json(Json::value_t::discarded);
    }
    return j;
}

bool json_flag(const Json& j, const char* key, bool fallback = false) {
    const auto it = j.find(key);
    return (it != j.end() && it->is_boolean()) ? it->get<bool>() : fallback;
}

// body[key] when it is an array, else an empty array.
const Json& json_list(const Json& j, const char* key) {
    static const Json empty = Json::array();
    const auto it = j.find(key);
    return (it != j.end() && it->is_array()) ? *it : empty;
}

std::vector<std::string> json_strings(const Json& j, const char* key) {
    std::vector<std::string> out;
    for (const auto& v : json_list(j, key))
        if (v.is_string())
            out.push_back(v.get<std::string>());
    return out;
}

// TD-205: segment targets as [{"path":"...","segment_start":N,"segment_count":N},...]
Json segments_json(const std::vector<SegmentTarget>& v) {
    Json out = Json::array();
    for (const auto& s : v)
        out.push_back({{"path", s.path},
                       {"segment_start", s.segment_start},
                       {"segment_count", s.segment_count}});
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

    // {"entries":[{"data_file":"...","checksum":"...","num_samples":N,"trained":T,...},...]}
    const Json reply = parse_reply(res->body, "load_registry");
    if (reply.is_discarded() || !reply.contains("entries"))
        return false;
    out.clear();
    for (const auto& obj : json_list(reply, "entries")) {
        if (!obj.is_object())
            continue;
        DataVersion dv;
        dv.data_file = json_str(obj, "data_file");
        dv.checksum = json_str(obj, "checksum");
        dv.num_samples = json_num(obj, "num_samples", 0);
        dv.trained = json_flag(obj, "trained");
        dv.model_id = json_str(obj, "model_id");
        dv.added_utc = json_str(obj, "added_utc");
        dv.source = json_str(obj, "source");
        dv.segment_start = json_num(obj, "segment_start", -1);  // TD-333; absent = whole file
        dv.segment_count = json_num(obj, "segment_count", -1);
        if (!dv.data_file.empty())
            out.push_back(std::move(dv));
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

    // {"entries":[{"path":"...","run_id":"...",...},...]}
    const Json reply = parse_reply(res->body, "load_pending");
    if (reply.is_discarded() || !reply.contains("entries"))
        return false;
    out.clear();
    for (const auto& obj : json_list(reply, "entries")) {
        if (!obj.is_object())
            continue;
        PendingEntry e;
        e.path = json_str(obj, "path");
        if (e.path.empty())
            continue;
        e.run_id = json_str(obj, "run_id");
        e.model_name = json_str(obj, "model_name");
        e.source = json_str(obj, "source");
        e.added_utc = json_str(obj, "added_utc");
        e.checksum = json_str(obj, "checksum");
        e.size_bytes = json_num<std::size_t>(obj, "size_bytes", 0);
        e.num_entries = json_num(obj, "num_entries", -1);
        e.segment_start = json_num(obj, "segment_start", -1);
        e.segment_count = json_num(obj, "segment_count", -1);
        e.claimed_unix = json_num<long long>(obj, "claimed_unix", 0);  // TD-494
        out.push_back(std::move(e));
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

    const std::string body =
        to_body({{"run_id", run_id}, {"max_files", max_files}, {"model_name", model_name}});
    const auto res = cli.Post((group_prefix_ + "/acquire").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::acquire — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/acquire");
        resp.failed = true;
        return resp;
    }
    const Json reply = parse_reply(res->body, "acquire");
    if (reply.is_discarded()) {
        // The server may hold claims the client can't see; they expire with the lease (TD-494).
        resp.failed = true;
        return resp;
    }

    // TD-494: the claim lease (both response shapes); DatasetRegistry renews within it. An older
    // server omits it (0: claims don't expire there).
    resp.claim_lease_seconds = std::max(0, json_num(reply, "claim_lease_seconds", 0));

    if (reply.contains("files")) {
        // Structured (FTP) response: {"ftp_server_host":...,"files":[{...},...]}
        resp.ftp_server_host = json_str(reply, "ftp_server_host");
        resp.ftp_server_port = json_num(reply, "ftp_server_port", resp.ftp_server_port);
        resp.ftps_enabled = json_flag(reply, "ftps_enabled");  // Phase 3
        for (const auto& obj : json_list(reply, "files")) {
            if (!obj.is_object())
                continue;
            FileToken tok;
            tok.registry_path = json_str(obj, "registry_path");
            tok.ftp_path = json_str(obj, "ftp_path");
            tok.ftp_username = json_str(obj, "ftp_username");
            tok.ftp_password = json_str(obj, "ftp_password");
            tok.checksum = json_str(obj, "checksum");
            tok.token_expires_utc = json_str(obj, "token_expires_utc");
            tok.size_bytes = json_num<std::size_t>(obj, "size_bytes", 0);
            tok.segment_start = json_num(obj, "segment_start", -1);  // TD-205
            tok.segment_count = json_num(obj, "segment_count", -1);
            if (!tok.registry_path.empty())
                resp.files.push_back(std::move(tok));
        }
    } else {
        // Legacy format: {"acquired":["path1","path2",...]}
        // Wrap each path in a minimal FileToken with ftp fields empty.
        // TD-205: "segment_starts"/"segment_counts" are purely-additive parallel arrays (same
        // order/length as "acquired"); an older server omits them, so every entry defaults to -1
        // (whole file).
        const auto& paths = json_list(reply, "acquired");
        const auto& starts = json_list(reply, "segment_starts");
        const auto& counts = json_list(reply, "segment_counts");
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (!paths[i].is_string())
                continue;
            FileToken tok;
            tok.registry_path = paths[i].get<std::string>();
            if (i < starts.size() && starts[i].is_number_integer())
                tok.segment_start = starts[i].get<int>();
            if (i < counts.size() && counts[i].is_number_integer())
                tok.segment_count = counts[i].get<int>();
            resp.files.push_back(std::move(tok));
        }
    }

    Logger::info("RemoteTransport: acquired {} files for run '{}'", resp.files.size(), run_id);
#else
    Logger::error("RemoteTransport::acquire — not compiled");
    resp.failed = true;
#endif
    return resp;
}

void RemoteTransport::release(const std::string& run_id, const std::vector<std::string>& paths,
                              const std::vector<SegmentTarget>& segments) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    const std::string body =
        to_body({{"run_id", run_id}, {"files", paths}, {"segments", segments_json(segments)}});
    const auto res = cli.Post((group_prefix_ + "/release").c_str(), body, "application/json");
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

    Json body = {{"run_id", run_id}, {"files", files}, {"samples", samples}};
    if (!trained_segments.empty()) {
        // Exact mode: the server removes exactly these pending entries. Without the arrays it
        // keeps the pre-TD-333 behaviour (every entry sharing a path), for older clients.
        body["segment_starts"] = seg_starts;
        body["segment_counts"] = seg_counts;
    }

    const auto res =
        cli.Post((group_prefix_ + "/trained").c_str(), to_body(body), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::commit_trained — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/trained");
        return;
    }
    // TD-493: units this run doesn't hold a claim on aren't recorded.
    const Json reply = parse_reply(res->body, "commit_trained");
    if (!reply.is_discarded() && !json_list(reply, "rejected").empty()) {
        Logger::warn("RemoteTransport::commit_trained — registry didn't record some units: {}",
                     res->body);
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

    const std::string body = to_body(
        {{"path", path}, {"segment_start", segment_start}, {"segment_count", segment_count}});
    const auto res = cli.Post((group_prefix_ + "/pending/add").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::add_pending — HTTP {} from {}:{}{}: {}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/pending/add",
                      res ? res->body : "");
        return false;
    }
    const Json reply = parse_reply(res->body, "add_pending");
    if (reply.is_discarded())
        return false;
    if (!json_flag(reply, "added", true)) {
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

    const std::string body = to_body({{"model_name", model_name}, {"run_id", run_id}});
    const auto res = cli.Post((group_prefix_ + "/session/next").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::next_session — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/session/next");
        return "session-01";
    }
    const Json reply = parse_reply(res->body, "next_session");
    return reply.is_discarded() ? std::string() : json_str(reply, "session_id");
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

    const std::string body = to_body({{"model_name", model_name},
                                      {"paths", paths},
                                      {"count", count},
                                      {"segments", segments_json(segments)}});
    const auto res = cli.Post((group_prefix_ + "/assign").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::assign — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/assign");
        return result;
    }
    const Json reply = parse_reply(res->body, "assign");
    if (reply.is_discarded())
        return result;
    // TODO: See TD-498 in TECHNICAL_DEBT.md - response "segments" not parsed; result.segments stays empty.
    result.assigned = json_num(reply, "assigned", 0);
    result.paths = json_strings(reply, "paths");
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

    const std::string body = to_body({{"model_name", model_name},
                                      {"paths", paths},
                                      {"force", force},
                                      {"segments", segments_json(segments)}});
    const auto res = cli.Post((group_prefix_ + "/unassign").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::unassign — HTTP {} from {}:{}{}", res ? res->status : -1,
                      host_, port_, group_prefix_ + "/unassign");
        return result;
    }
    const Json reply = parse_reply(res->body, "unassign");
    if (reply.is_discarded())
        return result;
    result.unassigned = json_num(reply, "unassigned", 0);
    result.skipped = json_num(reply, "skipped", 0);
    result.paths = json_strings(reply, "paths");
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

    const std::string body = to_body({{"paths", paths},
                                      {"force", force},
                                      {"delete_files", delete_files},
                                      {"segments", segments_json(segments)}});
    const auto res = cli.Post((group_prefix_ + "/delete").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::delete_paths — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/delete");
        return result;
    }
    const Json reply = parse_reply(res->body, "delete_paths");
    if (reply.is_discarded())
        return result;
    result.deleted = json_num(reply, "deleted", 0);
    result.skipped = json_num(reply, "skipped", 0);
    result.not_found = json_num(reply, "not_found", 0);
    // "details":[{"path":"...","status":"...","file_deleted":bool,...},...]
    for (const auto& obj : json_list(reply, "details")) {
        if (!obj.is_object())
            continue;
        DeleteResult::Detail detail;
        detail.path = json_str(obj, "path");
        detail.status = json_str(obj, "status");
        detail.file_deleted = json_flag(obj, "file_deleted");
        detail.segment_start = json_num(obj, "segment_start", -1);
        detail.segment_count = json_num(obj, "segment_count", -1);
        result.details.push_back(std::move(detail));
    }
    Logger::info("RemoteTransport: delete — {} deleted, {} skipped, {} not_found", result.deleted,
                result.skipped, result.not_found);
#else
    Logger::error("RemoteTransport::delete_paths — not compiled");
#endif
    return result;
}

// Phase 11: server-side dataset fetch ---------------------------------------

namespace {
// The queued path from a fetch or upload reply, or "" (logged) when the registry reports failure.
std::string added_path(const std::string& body, const char* method) {
    const Json reply = parse_reply(body, method);
    if (reply.is_discarded())
        return "";
    if (!json_flag(reply, "added")) {
        Logger::error("RemoteTransport::{} — registry reported failure: {}", method, body);
        return "";
    }
    return json_str(reply, "path");
}
}  // namespace

std::string RemoteTransport::fetch_gutenberg(int book_id, int num_pairs,
                                             const std::string& model_name) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    // Fetching a book + converting it can take a while; give the server room.
    cli.set_read_timeout(std::max(timeout_ms_, 120000) / 1000, 0);

    const std::string body =
        to_body({{"book_id", book_id}, {"num_pairs", num_pairs}, {"model_name", model_name}});
    const auto res =
        cli.Post((group_prefix_ + "/fetch/gutenberg").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::fetch_gutenberg — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/fetch/gutenberg");
        return "";
    }
    return added_path(res->body, "fetch_gutenberg");
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

    const std::string body = to_body({{"dataset_id", dataset_id},
                                      {"num_pairs", num_pairs},
                                      {"split", split},
                                      {"input_field", input_field},
                                      {"output_field", output_field},
                                      {"model_name", model_name}});
    const auto res =
        cli.Post((group_prefix_ + "/fetch/huggingface").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::fetch_huggingface — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/fetch/huggingface");
        return "";
    }
    return added_path(res->body, "fetch_huggingface");
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
        // TD-492: 409 = the name is already claimed, trained or queued ("reason" in the body).
        Logger::error("RemoteTransport::upload_file — HTTP {} from {}:{}{}: {}",
                      res ? res->status : -1, host_, port_, path, res ? res->body : "");
        return "";
    }
    return added_path(res->body, "upload_file");
#else
    Logger::error("RemoteTransport::upload_file — not compiled");
    return "";
#endif
}

int RemoteTransport::renew_claims(const std::string& run_id) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);
    const auto res = cli.Post((group_prefix_ + "/renew").c_str(), to_body({{"run_id", run_id}}),
                              "application/json");
    if (!res || res->status != 200) {
        Logger::warn("RemoteTransport::renew_claims — HTTP {} from {}:{}{}", res ? res->status : -1,
                     host_, port_, group_prefix_ + "/renew");
        return -1;
    }
    const Json reply = parse_reply(res->body, "renew_claims");
    return reply.is_discarded() ? -1 : json_num(reply, "renewed", -1);
#else
    (void)run_id;
    return -1;
#endif
}

RequeueResult RemoteTransport::requeue_trained(const std::vector<std::string>& paths, bool all,
                                               const std::string& model_name) {
    RequeueResult result;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    const std::string body = to_body({{"paths", paths}, {"all", all}, {"model_name", model_name}});
    const auto res = cli.Post((group_prefix_ + "/requeue").c_str(), body, "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::requeue_trained — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/requeue");
        result.ok = false;
        return result;
    }
    const Json reply = parse_reply(res->body, "requeue_trained");
    if (reply.is_discarded()) {
        result.ok = false;
        return result;
    }
    result.requeued = json_num(reply, "requeued", 0);
    result.already_pending = json_num(reply, "already_pending", 0);
    result.not_found = json_num(reply, "not_found", 0);
    for (const auto& obj : json_list(reply, "details")) {
        if (!obj.is_object())
            continue;
        RequeueResult::Detail detail;
        detail.path = json_str(obj, "path");
        detail.status = json_str(obj, "status");
        result.details.push_back(std::move(detail));
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
