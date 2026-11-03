package com.adai.ops.network.dto

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// TD-520: /admin/status sends null losses (none yet, or a diverged NaN/inf loss). Same settings as
// ApiClientProvider's Json.
class TrainerStatusDtoTest {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        explicitNulls = false
    }

    @Test
    fun `null losses decode as null`() {
        val status = json.decodeFromString<TrainerStatusDto>(
            """{"phase":"idle","paused":false,"last_loss":null,"best_loss":null,"loss_diverged":false}""",
        )
        assertNull(status.last_loss)
        assertNull(status.best_loss)
        assertFalse(status.loss_diverged)
    }

    @Test
    fun `a diverged pass decodes with its best loss`() {
        val status = json.decodeFromString<TrainerStatusDto>(
            """{"phase":"training","last_loss":null,"best_loss":0.5,"loss_diverged":true}""",
        )
        assertNull(status.last_loss)
        assertEquals(0.5, status.best_loss!!, 0.0)
        assertTrue(status.loss_diverged)
    }

    @Test
    fun `an older server's numeric losses still decode`() {
        val status = json.decodeFromString<TrainerStatusDto>("""{"last_loss":0.25,"best_loss":0.2}""")
        assertEquals(0.25, status.last_loss!!, 0.0)
        assertEquals(0.2, status.best_loss!!, 0.0)
        assertFalse(status.loss_diverged)
    }
}
