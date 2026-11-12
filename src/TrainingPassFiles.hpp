#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-11-12

// The "claimed files -> local files" step every training pass shares (train, retrain, resume and
// the LeJEPA pass), and the clean-up after it. Moved out of IncrementalTrainingTool.cpp unchanged
// (TD-566), so the LeJEPA pass can be tested.

#include <filesystem>
#include <vector>

#include "DatasetRegistry.hpp"
#include "RegistryTransport.hpp"

namespace adai {

/// TD-411: when the registry hands out an FTP host (a remote registry_server), downloads the
/// claimed files into `dcfg.download_dir` (or `<session_dir>/downloads`, with a warning); otherwise
/// uses the registry's own paths. `ranges` gets one entry per `resp.files` entry, in order, with
/// its TD-205 segment range; `downloaded` gets the files to delete after the pass. Returns false
/// (logged) if a download failed; the caller then releases the files.
bool resolve_pass_files(const AcquireResponse& resp, const DatasetConfig& dcfg,
                        std::vector<PendingFileRange>& ranges,
                        std::vector<std::filesystem::path>& downloaded);

/// Removes each download and its TD-338 checksum sidecar; a path listed twice (TD-337 segments) is
/// fine.
void cleanup_downloads(const std::vector<std::filesystem::path>& local_paths);

}  // namespace adai
