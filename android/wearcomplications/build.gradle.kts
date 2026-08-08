plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.adai.wearcomplications"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.adai.wearcomplications"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
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
}

dependencies {
    implementation(project(":wearsync"))

    implementation(libs.core.ktx)
    implementation(libs.wear.watchface.complications.data.source)
    implementation(libs.play.services.wearable)

    testImplementation(libs.junit)
}
