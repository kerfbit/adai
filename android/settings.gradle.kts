pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Only for a transitive dependency of com.google.android.wearable.watchface.validator
        // (the offline Watch Face Push validator/token-generation library) — not published to
        // Google/Maven Central.
        exclusiveContent {
            forRepository { maven(url = "https://jitpack.io") }
            filter { includeGroup("com.github.xgouchet") }
        }
    }
}

rootProject.name = "adai-chatbot"
include(":app", ":opsdashboard", ":wearsync", ":wearface", ":wearcomplications")
