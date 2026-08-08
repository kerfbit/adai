plugins {
    alias(libs.plugins.android.application)
}

// Watch Face Format bundle — resource-only APK (android:hasCode="false" in the manifest),
// so no Kotlin plugin and no code dependencies belong here; everything lives in res/raw and
// res/xml. Pushed to the watch by opsdashboard via Watch Face Push, not installed/updated
// via `adb install` directly (see android/opsdashboard's WatchFacePushRepository).
//
// Package name follows the format Watch Face Push's validator requires: the pushing client
// app's own package ("com.adai.ops"), then literally ".watchfacepush.", then an identifier.
android {
    namespace = "com.adai.ops.watchfacepush.training"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.adai.ops.watchfacepush.training"
        // Watch Face Format version="4" (declared in the manifest) requires Wear OS 4+;
        // in practice this bundle is only ever delivered via Watch Face Push, which itself
        // requires Wear OS 6+.
        minSdk = 33
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }
}

// AGP still emits an (empty) classes.dex and a META-INF/com/android/build/** metadata file
// for a codeless application module even with android:hasCode="false" — the Watch Face Push
// validator rejects an APK containing either. There's no supported AGP 8.x DSL to suppress
// this at the source (AGP 9's `android.enableKotlin = false` reportedly does, but that's a
// bigger toolchain jump than justified for one module), so strip them from the built APK
// directly.
//
// Stripping files from an already-signed APK invalidates its signature (the validator
// reports "JAR_SIG_NO_MANIFEST"), so this also re-signs with the same debug keystore AGP
// itself uses — `:opsdashboard`'s copyWearFaceApk task consumes this resigned output, not
// the raw assembleDebug one.
val wffApkFileName = "wearface-wff.apk"

// A separate directory (not AGP's own outputs/apk/debug) — writing into AGP's output dir
// trips Gradle's task-output validation (":wearface:createDebugApkListingFileRedirect" also
// reads/writes there without an explicit dependency on this task).
val wffOutputDir = layout.buildDirectory.dir("wff")

val prepareWffApk by tasks.registering(Copy::class) {
    dependsOn("assembleDebug")
    from(layout.buildDirectory.file("outputs/apk/debug/wearface-debug.apk"))
    into(wffOutputDir)
    rename { wffApkFileName }
}

val stripDisallowedFiles by tasks.registering(Exec::class) {
    dependsOn(prepareWffApk)
    workingDir = wffOutputDir.get().asFile
    commandLine("zip", "-d", wffApkFileName, "classes.dex", "META-INF/com/android/build/*")
    // `zip -d` exits non-zero if a build was already stripped and re-run with nothing new to
    // delete (e.g. after a clean rebuild it's freshly copied so this shouldn't normally
    // trigger, but a stale up-to-date Copy could) — treat that as success, not a failure.
    isIgnoreExitValue = true
}

val debugKeystore = File(System.getProperty("user.home"), ".android/debug.keystore")

val resignWffApk by tasks.registering(Exec::class) {
    dependsOn(stripDisallowedFiles)
    workingDir = wffOutputDir.get().asFile
    val apksigner = android.sdkDirectory.resolve("build-tools/36.0.0/apksigner").absolutePath
    commandLine(
        apksigner, "sign",
        "--ks", debugKeystore.absolutePath,
        "--ks-pass", "pass:android",
        "--key-pass", "pass:android",
        "--ks-key-alias", "androiddebugkey",
        wffApkFileName,
    )
}

tasks.named("assemble") {
    finalizedBy(resignWffApk)
}
