package com.adai.ops.settings

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * TD-131: encodeGroups/decodeGroups used to be a naive comma-join/split, which silently
 * corrupted any group name containing a literal comma. These test the pure encode/decode
 * pair directly (no Context/DataStore needed).
 */
class OpsSettingsDataStoreTest {

    @Test
    fun `a group name containing a comma round-trips intact`() {
        val groups = listOf("a,b", "normal-group")

        val encoded = OpsSettingsDataStore.encodeGroups(groups)
        val decoded = OpsSettingsDataStore.decodeGroups(encoded)

        assertEquals(groups, decoded)
    }

    @Test
    fun `old comma-joined values from before this fix still decode correctly`() {
        // Pre-fix encodeGroups() wrote exactly this shape for names with no embedded comma
        // (the only shape the old scheme could round-trip correctly) — decodeGroups() must
        // keep reading it via its comma-split fallback now that encoding has changed.
        val decoded = OpsSettingsDataStore.decodeGroups("group1,group2")

        assertEquals(listOf("group1", "group2"), decoded)
    }

    @Test
    fun `blank or null raw value decodes to an empty list`() {
        assertEquals(emptyList<String>(), OpsSettingsDataStore.decodeGroups(null))
        assertEquals(emptyList<String>(), OpsSettingsDataStore.decodeGroups(""))
    }
}
