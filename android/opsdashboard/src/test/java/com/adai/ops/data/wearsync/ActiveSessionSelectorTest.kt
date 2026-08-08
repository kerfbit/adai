package com.adai.ops.data.wearsync

import com.adai.ops.network.dto.SessionSummaryDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ActiveSessionSelectorTest {

    @Test
    fun `empty list returns null`() {
        assertNull(ActiveSessionSelector.select(emptyList()))
    }

    @Test
    fun `single training session is selected`() {
        val session = session(key = "a", sessionId = 1, isTraining = true)
        assertEquals(session, ActiveSessionSelector.select(listOf(session)))
    }

    @Test
    fun `prefers training session over idle ones regardless of update time`() {
        val idleButRecent = session(key = "idle", sessionId = 1, isTraining = false, lastUpdate = 1000L)
        val trainingButOlder = session(key = "training", sessionId = 2, isTraining = true, lastUpdate = 500L)
        val result = ActiveSessionSelector.select(listOf(idleButRecent, trainingButOlder))
        assertEquals(trainingButOlder, result)
    }

    @Test
    fun `ties among training sessions broken by highest session id`() {
        val older = session(key = "old", sessionId = 1, isTraining = true)
        val newer = session(key = "new", sessionId = 5, isTraining = true)
        val result = ActiveSessionSelector.select(listOf(older, newer))
        assertEquals(newer, result)
    }

    @Test
    fun `no training sessions falls back to most recently updated`() {
        val stale = session(key = "stale", sessionId = 1, isTraining = false, lastUpdate = 100L)
        val recent = session(key = "recent", sessionId = 2, isTraining = false, lastUpdate = 9999L)
        val result = ActiveSessionSelector.select(listOf(stale, recent))
        assertEquals(recent, result)
    }

    private fun session(
        key: String,
        sessionId: Int,
        isTraining: Boolean,
        lastUpdate: Long = 0L,
    ) = SessionSummaryDto(
        key = key,
        session_id = sessionId,
        is_training = isTraining,
        last_update_time = lastUpdate,
    )
}
