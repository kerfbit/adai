// @adai-status: experimental
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-03

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace adai {

/// TD-521: the longest `POST /admin/checkpoint?wait_ms=` a pass's admin API (and trainer_service's proxy
/// in front of it) accepts; each wait holds an HTTP worker thread.
constexpr int kMaxCheckpointWaitMs = 60000;

/// TD-522: the most log entries a pass keeps for `GET /admin/logs`, and the largest `?limit=` accepted.
constexpr int kMaxAdminLogEntries = 200;

/**
 * @brief TD-519: one validated `PUT /admin/config` body for the trainer's four mutable keys.
 *
 * Shared by TrainerAdminAPI (a live pass) and TrainerServiceProxy (idle, no child) so both
 * accept and reject exactly the same bodies. Each field is set only when the body sent it.
 */
struct TrainerAdminConfigUpdate {
    std::optional<bool> auto_save_enabled;
    std::optional<int> auto_save_every_samples;
    std::optional<int> auto_save_every_minutes;
    std::optional<int> max_sessions_to_keep;
};

/** @brief Result of parse_trainer_admin_config(): @c error is empty on success. */
struct TrainerAdminConfigParse {
    TrainerAdminConfigUpdate update;
    std::string error;  ///< human-readable reason for a 400; empty when the body is valid
};

/**
 * @brief Parse and validate a `PUT /admin/config` body as JSON.
 *
 * The whole body is checked before anything is returned, so a caller applying the result
 * never applies part of a rejected request. Rejected (non-empty @c error): a body that isn't
 * a JSON object; any of the immutable keys `port`/`host`/`dir`/`enabled`; a non-boolean
 * `auto_save_enabled`; a non-integer or out-of-range number (`auto_save_every_samples` and
 * `auto_save_every_minutes` >= 0, `max_sessions_to_keep` >= 1); a body with none of the four
 * mutable keys. Other keys are ignored.
 */
TrainerAdminConfigParse parse_trainer_admin_config(const std::string& body);

/**
 * @brief TD-521: the daemon_config.db overrides (DaemonConfigStore::load_all()) that pass the same checks
 * as a PUT body, key by key.
 *
 * `auto_save_enabled` must be `true`/`false` (or `1`/`0`), the numbers whole decimal values in the PUT
 * ranges. Each other value of the four keys is left out and described in @p rejected (when given), so
 * the caller keeps its config-file value and can log why. Other keys are ignored.
 */
TrainerAdminConfigUpdate parse_trainer_admin_overrides(const std::map<std::string, std::string>& overrides,
                                                       std::vector<std::string>* rejected = nullptr);

}  // namespace adai
