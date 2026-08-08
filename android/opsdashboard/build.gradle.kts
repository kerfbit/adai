plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.adai.ops"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.adai.ops"
        // androidx.wear.watchfacepush requires 33+ on the phone side.
        minSdk = 33
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(project(":wearsync"))

    implementation(libs.core.ktx)
    implementation(libs.lifecycle.runtime.ktx)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.activity.compose)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.extended)
    debugImplementation(libs.compose.ui.tooling)

    implementation(libs.compose.material3.adaptive.navigation.suite)
    implementation(libs.compose.material3.adaptive)
    implementation(libs.compose.material3.adaptive.layout)
    implementation(libs.compose.material3.adaptive.navigation)

    implementation(libs.navigation.compose)

    implementation(libs.retrofit.core)
    implementation(libs.retrofit.kotlinx.serialization.converter)
    implementation(libs.okhttp.core)
    implementation(libs.okhttp.logging.interceptor)
    implementation(libs.kotlinx.serialization.json)

    implementation(libs.datastore.preferences)

    // Device-credential (PIN/pattern/password/biometric) verification gate in front of
    // every admin action — see ui/common/BiometricAdminAuthGate.kt. Brings in
    // androidx.fragment transitively, required by BiometricPrompt's FragmentActivity host.
    implementation(libs.biometric)

    implementation(libs.play.services.wearable)
    implementation(libs.work.runtime.ktx)
    implementation(libs.wear.watchfacepush)
    implementation(libs.watchface.validator.android)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.androidx.test.ext.junit)
}

// Bundles the compiled :wearface Watch Face Format APK as a raw asset so
// WatchFacePushRepository can push it to the watch at runtime (validated + tokenized on
// the fly via the DwfValidator library, no separate offline tool step needed). Re-run
// whenever watchface.xml changes — a plain `./gradlew :opsdashboard:assembleDebug` re-triggers
// this automatically via the preBuild dependency below.
val copyWearFaceApk by tasks.registering(Copy::class) {
    dependsOn(":wearface:resignWffApk")
    from(project(":wearface").layout.buildDirectory.dir("wff")) {
        include("wearface-wff.apk")
        rename { "wearface.apk" }
    }
    into("src/main/assets")
}

tasks.named("preBuild") {
    dependsOn(copyWearFaceApk)
}
