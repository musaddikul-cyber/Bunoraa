plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.bunoraa.admin"
    compileSdk = 34
    val googleClientId: String = (project.findProperty("OIDC_GOOGLE_CLIENT_ID") as String?) ?: ""
    val microsoftClientId: String = (project.findProperty("OIDC_MICROSOFT_CLIENT_ID") as String?) ?: ""
    val microsoftTenant: String = (project.findProperty("OIDC_MICROSOFT_TENANT") as String?) ?: "common"
    val redirectScheme: String = (project.findProperty("OIDC_REDIRECT_SCHEME") as String?) ?: "com.bunoraa.admin"

    defaultConfig {
        applicationId = "com.bunoraa.admin"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("String", "API_BASE_URL", "\"https://api.bunoraa.com/api/v1/\"")
        buildConfigField("String", "OIDC_GOOGLE_CLIENT_ID", "\"$googleClientId\"")
        buildConfigField("String", "OIDC_MICROSOFT_CLIENT_ID", "\"$microsoftClientId\"")
        buildConfigField("String", "OIDC_MICROSOFT_TENANT", "\"$microsoftTenant\"")
        buildConfigField("String", "OIDC_REDIRECT_SCHEME", "\"$redirectScheme\"")
        manifestPlaceholders["appAuthRedirectScheme"] = redirectScheme
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = libs.versions.composeCompiler.get()
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    implementation(projects.core.common)
    implementation(projects.core.network)
    implementation(projects.core.database)
    implementation(projects.core.datastore)
    implementation(projects.core.designsystem)
    implementation(projects.feature.auth)
    implementation(projects.feature.dashboard)

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime)
    implementation(libs.androidx.lifecycle.viewmodel)
    implementation(libs.androidx.activity.compose)
    implementation(libs.kotlinx.coroutines)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.navigation.compose)

    implementation(libs.androidx.work.runtime)
    implementation(libs.appauth)

    implementation(platform(libs.firebase.bom))
    implementation(libs.firebase.messaging)

    debugImplementation(libs.androidx.compose.ui.tooling)
}
