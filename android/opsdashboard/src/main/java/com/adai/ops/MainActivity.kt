package com.adai.ops

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.fragment.app.FragmentActivity
import com.adai.ops.ui.common.LocalAdminAuthGate
import com.adai.ops.ui.scaffold.OpsAppScaffold
import com.adai.ops.ui.theme.AdaiOpsTheme

// FragmentActivity (not ComponentActivity) is required as the host for BiometricPrompt,
// which every admin action's device-credential check goes through — see
// ui/common/BiometricAdminAuthGate.kt.
class MainActivity : FragmentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides (application as OpsApp).container.adminAuthGate) {
                AdaiOpsTheme {
                    Surface(modifier = Modifier.fillMaxSize()) {
                        OpsAppScaffold(app = application as OpsApp)
                    }
                }
            }
        }
    }
}
