plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.ouor.rvcandroid"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    ndkVersion = "27.1.12297006"

    defaultConfig {
        applicationId = "com.ouor.rvcandroid"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // RVC inference is only feasible on modern 64-bit ARM devices.
            // Restricting ABIs avoids shipping ~150 MB of unused ORT native libs.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared")
                cppFlags += "-std=c++17"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            // onnxruntime-android-qnn:1.22 ships QNN SDK 2.25 libs inside its
            // AAR; tools/setup_qnn_libs.sh drops 2.45 versions into
            // src/main/jniLibs. pickFirsts makes our copies win the merge.
            // Without this AGP fails the build with a duplicate-file error.
            pickFirsts += listOf(
                "lib/arm64-v8a/libQnnHtp.so",
                "lib/arm64-v8a/libQnnSystem.so",
                "lib/arm64-v8a/libQnnHtpV79Stub.so",
                "lib/arm64-v8a/libQnnHtpPrepare.so",
            )
            // librvc_synth_runner.so is a PIE executable, not a real
            // library — we spawn it via ProcessBuilder. Android's
            // packager will only extract it onto the filesystem (as
            // required for execve) when legacy packaging is on.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    debugImplementation(libs.androidx.compose.ui.tooling)

    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.onnxruntime.android)

    // Kept for the XML <application> theme parent (Theme.Material3.*).
    implementation(libs.material)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
}
