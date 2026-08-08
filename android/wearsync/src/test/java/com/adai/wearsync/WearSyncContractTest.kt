package com.adai.wearsync

import org.junit.Assert.assertEquals
import org.junit.Test

class WearSyncContractTest {

    @Test
    fun `empty input returns zero min and max`() {
        val result = minMax(emptyList())
        assertEquals(0.0, result.min, 0.0)
        assertEquals(0.0, result.max, 0.0)
    }

    @Test
    fun `all non-finite input returns zero min and max`() {
        val result = minMax(listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY))
        assertEquals(0.0, result.min, 0.0)
        assertEquals(0.0, result.max, 0.0)
    }

    @Test
    fun `non-finite values are dropped before computing bounds`() {
        val result = minMax(listOf(1.0, Double.NaN, 5.0, Double.POSITIVE_INFINITY, 3.0))
        assertEquals(1.0, result.min, 0.0)
        assertEquals(5.0, result.max, 0.0)
    }

    @Test
    fun `single value is both min and max`() {
        val result = minMax(listOf(2.5))
        assertEquals(2.5, result.min, 0.0)
        assertEquals(2.5, result.max, 0.0)
    }
}
