package com.adai.ops.testutil

// @adai-status: beta        (in-memory fake backing both plain-JVM and instrumented tests, TD-048)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import com.adai.ops.data.wearsync.WatchFacePushRepository
import com.adai.ops.data.wearsync.WatchFacePushResult

class FakeWatchFacePushRepository(
    private val supported: Boolean = true,
    private val pushResult: () -> WatchFacePushResult = { WatchFacePushResult.Success("slot-1") },
    private val setActiveResult: (String) -> Result<Unit> = { Result.success(Unit) },
) : WatchFacePushRepository {

    var pushCallCount = 0
        private set
    val setActiveCalls = mutableListOf<String>()

    override fun isSupported(): Boolean = supported

    override suspend fun pushWatchFace(): WatchFacePushResult {
        pushCallCount++
        return pushResult()
    }

    override suspend fun setActive(slotId: String): Result<Unit> {
        setActiveCalls.add(slotId)
        return setActiveResult(slotId)
    }
}
