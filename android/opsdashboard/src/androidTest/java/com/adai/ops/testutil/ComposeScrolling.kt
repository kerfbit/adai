package com.adai.ops.testutil

import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.SemanticsNodeInteractionsProvider
import androidx.compose.ui.test.hasContentDescription
import androidx.compose.ui.test.hasScrollToNodeAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.ComposeTestRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performScrollToNode

/**
 * Helpers for screens built on a LazyColumn: an item below the fold is never composed, so
 * `onNodeWithText(...)` can't find it until the list has been scrolled to it. Written when these
 * tests first ran on a device (November 21, 2026; before that they were only compiled) and most
 * failures were exactly that.
 */

/**
 * Scrolls whichever scrollable container (LazyColumn, LazyRow) holds a node matching [matcher]
 * until it is composed and on screen. Returns false when no container has one (yet).
 */
fun SemanticsNodeInteractionsProvider.scrollToNode(matcher: SemanticsMatcher): Boolean {
    val containers = onAllNodes(hasScrollToNodeAction())
    val count = containers.fetchSemanticsNodes().size
    for (i in 0 until count) {
        if (runCatching { containers[i].performScrollToNode(matcher) }.isSuccess) return true
    }
    return false
}

/** `onNodeWithText`, after scrolling the node into view. */
fun SemanticsNodeInteractionsProvider.onNodeWithTextScrolled(
    text: String,
    substring: Boolean = false,
): SemanticsNodeInteraction {
    scrollToNode(hasText(text, substring = substring))
    return onNodeWithText(text, substring = substring)
}

/** `onNodeWithContentDescription`, after scrolling the node into view. */
fun SemanticsNodeInteractionsProvider.onNodeWithContentDescriptionScrolled(
    description: String,
): SemanticsNodeInteraction {
    scrollToNode(hasContentDescription(description))
    return onNodeWithContentDescription(description)
}

/**
 * Waits until a node containing [text] exists, scrolling to it on each check, so content that
 * loads asynchronously below the fold is found too.
 */
fun ComposeTestRule.waitUntilTextShown(text: String, substring: Boolean = true, timeoutMillis: Long = 5_000) {
    waitUntil(timeoutMillis) {
        scrollToNode(hasText(text, substring = substring)) ||
            onAllNodes(hasText(text, substring = substring)).fetchSemanticsNodes().isNotEmpty()
    }
}
