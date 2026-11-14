package com.adai.ops.ui.metrics

import com.adai.ops.network.dto.CurrentMetricsDto
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// TD-341: the session screen's per-layer gate rows, and the current-metrics JSON they come from
// (null where a layer has no gated path). Same Json settings as ApiClientProvider's.
class GateRowsTest {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        explicitNulls = false
    }

    @Test
    fun `gate arrays with nulls decode`() {
        val current = json.decodeFromString<CurrentMetricsDto>(
            """{"current_epoch":2,"world_model_gates":[0.25,null],"memory_gates":[null,-0.5]}""",
        )
        assertEquals(listOf(0.25, null), current.world_model_gates)
        assertEquals(listOf(null, -0.5), current.memory_gates)
    }

    @Test
    fun `an older server without gate fields decodes with none`() {
        val current = json.decodeFromString<CurrentMetricsDto>("""{"current_epoch":2}""")
        assertTrue(current.world_model_gates.isEmpty())
        assertTrue(gateRows(current.world_model_gates, current.memory_gates).isEmpty())
    }

    @Test
    fun `one row per layer with a gated path, a missing path shown as a dash`() {
        val rows = gateRows(listOf(0.25, null, null), listOf(null, -0.5, null))
        assertEquals(
            listOf(
                "Gate L0 tanh" to "world 0.2500 · memory —",
                "Gate L1 tanh" to "world — · memory -0.5000",
            ),
            rows,
        )
    }

    @Test
    fun `arrays of different lengths are read to the longer one`() {
        assertEquals(listOf("Gate L1 tanh" to "world — · memory 0.1000"), gateRows(emptyList(), listOf(null, 0.1)))
    }
}
