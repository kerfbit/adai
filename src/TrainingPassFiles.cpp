// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-11-12

// TD-566: moved out of IncrementalTrainingTool.cpp unchanged, so the LeJEPA pass
// (LeJEPATrainingPass.cpp) can be built into a test binary with them.

#include "TrainingPassFiles.hpp"

#include <system_error>

#include "DataTransport.hpp"
#include "Logger.hpp"

namespace adai {

namespace fs = std::filesystem;

bool resolve_pass_files(const AcquireResponse& resp, const DatasetConfig& dcfg,
                        std::vector<PendingFileRange>& ranges, std::vector<fs::path>& downloaded) {
    std::vector<std::string> local_paths;
    // Remote first: whenever the registry serves files over FTP, download them. An unset
    // DOWNLOAD_DIR used to silently fall back to the registry host's paths, which don't exist
    // on a separate trainer host; use <SESSION_DIR>/downloads instead.
    const bool use_ftp = !resp.ftp_server_host.empty();
    if (use_ftp) {
        std::string download_dir = dcfg.download_dir;
        if (download_dir.empty()) {
            download_dir = (fs::path(dcfg.session_dir) / "downloads").string();
            adai::Logger::warn("DOWNLOAD_DIR is not set; downloading registry files to {}",
                               download_dir);
        }
        std::error_code mk_ec;
        fs::create_directories(download_dir, mk_ec);
        const std::size_t warn_bytes =
            static_cast<std::size_t>(dcfg.large_file_warn_threshold_mb) * 1024ULL * 1024ULL;
        try {
            DataTransport dt(FtpsVerification{dcfg.ftps_ca_file, dcfg.ftps_pinned_public_key});
            downloaded = dt.fetch_all(resp, download_dir, dcfg.max_parallel_downloads, warn_bytes);
            for (const auto& p : downloaded)
                local_paths.push_back(p.string());
        } catch (const std::exception& ex) {
            adai::Logger::error("[DataTransport] Download failed: {}", ex.what());
            return false;
        }
    } else {
        for (const auto& f : resp.files)
            local_paths.push_back(f.registry_path);
    }
    // local_paths[i] corresponds 1:1 with resp.files[i] in both branches (fetch_all() fills its
    // result by index) -- zip each with its entry's segment range.
    ranges.clear();
    ranges.reserve(local_paths.size());
    for (std::size_t i = 0; i < local_paths.size(); ++i) {
        PendingFileRange r;
        r.path = local_paths[i];
        if (i < resp.files.size()) {
            r.segment_start = resp.files[i].segment_start;
            r.segment_count = resp.files[i].segment_count;
        }
        ranges.push_back(std::move(r));
    }
    return true;
}

void cleanup_downloads(const std::vector<fs::path>& local_paths) {
    for (const auto& p : local_paths) {
        // TD-337: segments of one file share a download, so a path can appear more than once.
        if (!p.empty() && fs::exists(p)) {
            std::error_code ec;
            fs::remove(p, ec);
            if (ec) {
                adai::Logger::warn("[cleanup] Failed to remove '{}': {}", p.string(), ec.message());
            }
        }
        if (!p.empty()) {
            std::error_code ec;
            fs::remove(DataTransport::checksum_path_for(p), ec);  // TD-338 sidecar
        }
    }
}

}  // namespace adai
