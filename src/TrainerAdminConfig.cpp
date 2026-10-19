// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-19

#include "TrainerAdminConfig.hpp"
#include <climits>
#include "nlohmann/json.hpp"

namespace adai {

namespace {

// Reads an integer key into @p out; returns an error message, or "" on success / absence.
std::string read_int(const nlohmann::json& j, const char* key, int min_value,
                     const char* range_note, std::optional<int>& out) {
    const auto it = j.find(key);
    if (it == j.end()) {
        return "";
    }
    if (!it->is_number_integer()) {
        return std::string("'") + key + "' must be an integer";
    }
    const auto v = it->get<long long>();
    if (v < min_value || v > INT_MAX) {
        return std::string("'") + key + "' must be >= " + std::to_string(min_value) + range_note;
    }
    out = static_cast<int>(v);
    return "";
}

}  // namespace

TrainerAdminConfigParse parse_trainer_admin_config(const std::string& body) {
    TrainerAdminConfigParse result;
    const auto j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        result.error = "body must be a JSON object";
        return result;
    }

    // port/host/dir are baked into the already-bound listener socket and the already-opened
    // daemon_config.db handle, so they stay file/CLI-only (TRAINER_ADMIN_* in
    // config.trainer.conf).
    for (const char* key : {"port", "host", "dir", "enabled"}) {
        if (j.contains(key)) {
            result.error = std::string("'") + key +
                           "' is immutable at runtime; set it via config.trainer.conf "
                           "(TRAINER_ADMIN_*) and restart";
            return result;
        }
    }

    auto& u = result.update;
    if (const auto it = j.find("auto_save_enabled"); it != j.end()) {
        if (!it->is_boolean()) {
            result.error = "'auto_save_enabled' must be true or false";
            return result;
        }
        u.auto_save_enabled = it->get<bool>();
    }

    for (const auto& error :
         {read_int(j, "auto_save_every_samples", 0, " (0 disables the sample-count trigger)",
                   u.auto_save_every_samples),
          read_int(j, "auto_save_every_minutes", 0, " (0 disables the time-based trigger)",
                   u.auto_save_every_minutes),
          read_int(j, "max_sessions_to_keep", 1, "", u.max_sessions_to_keep)}) {
        if (!error.empty()) {
            result.error = error;
            return result;
        }
    }

    if (!u.auto_save_enabled && !u.auto_save_every_samples && !u.auto_save_every_minutes &&
        !u.max_sessions_to_keep) {
        result.error =
            "no recognized mutable keys in body (auto_save_enabled, auto_save_every_samples, "
            "auto_save_every_minutes, max_sessions_to_keep)";
    }
    return result;
}

}  // namespace adai
