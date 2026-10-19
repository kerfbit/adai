// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-19

#pragma once

#include <optional>
#include <string>

namespace adai {

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

}  // namespace adai
