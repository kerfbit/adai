package com.adai.ops.data.wearsync

import com.adai.ops.network.dto.SessionSummaryDto

/**
 * Picks which session the watch face relay should track, since `metrics_api_server` has no
 * "current session" concept of its own (per CLAUDE.md, clients always pick a key from
 * `/api/sessions`). Pure logic so it's unit-testable without a real API client.
 *
 * Policy: prefer a session that's actively training (highest `session_id` breaks ties, i.e.
 * the most recently started one); if none are training, fall back to the one with the most
 * recent `last_update_time` so the watch still shows the last run's shape instead of going
 * blank between sessions.
 */
object ActiveSessionSelector {

    fun select(sessions: List<SessionSummaryDto>): SessionSummaryDto? {
        if (sessions.isEmpty()) return null

        val training = sessions.filter { it.is_training }
        if (training.isNotEmpty()) {
            return training.maxByOrNull { it.session_id }
        }

        return sessions.maxByOrNull { it.last_update_time }
    }
}
