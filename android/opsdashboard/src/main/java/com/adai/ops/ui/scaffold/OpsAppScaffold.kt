package com.adai.ops.ui.scaffold

import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.adaptive.navigationsuite.ExperimentalMaterial3AdaptiveNavigationSuiteApi
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteScaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.adai.ops.OpsApp
import com.adai.ops.ui.navigation.OpsDestination
import com.adai.ops.ui.navigation.OpsNavHost

/**
 * Top-level chrome: NavigationSuiteScaffold renders bottom nav (compact width) / nav
 * rail (medium) / permanent drawer (expanded) automatically from the window's adaptive
 * info, so no manual per-screen WindowSizeClass branching is needed at this layer.
 * Section state is preserved across tab switches via save/restoreState, matching the
 * standard bottom-nav pattern.
 */
@OptIn(ExperimentalMaterial3AdaptiveNavigationSuiteApi::class)
@Composable
fun OpsAppScaffold(app: OpsApp) {
    val navController = rememberNavController()
    val currentBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = currentBackStackEntry?.destination

    NavigationSuiteScaffold(
        navigationSuiteItems = {
            OpsDestination.entries.forEach { destination ->
                item(
                    selected = currentRoute?.hierarchy?.any { it.route == destination.route } == true,
                    onClick = {
                        navController.navigate(destination.route) {
                            popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    icon = { Icon(destination.icon, contentDescription = destination.label) },
                    label = { Text(destination.label) },
                )
            }
        },
    ) {
        OpsNavHost(app, navController)
    }
}
