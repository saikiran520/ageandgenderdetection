plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.ebani.ageandgenderdetection"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.ebani.ageandgenderdetection"
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // ONNX Runtime ships native libraries for four ABIs, ~125 MB in
            // total. Keeping arm64-v8a (every modern phone) and x86_64 (the
            // standard emulator) gives one APK that installs everywhere we care
            // about and drops ~90 MB. Add "armeabi-v7a" back for 32-bit devices.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }

    androidResources {
        // The ONNX graphs are already incompressible. Storing them uncompressed lets
        // ModelManager copy them out of the APK by a straight stream copy and lets
        // ONNX Runtime memory-map the 100 MB MiVOLO weights instead of holding them
        // on the Java heap.
        noCompress += listOf("onnx", "json")
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    // CameraX: preview + single still capture.
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    // Fully local inference runtime. The AAR bundles arm64-v8a, armeabi-v7a,
    // x86 and x86_64 native libraries; nothing is fetched at runtime.
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
