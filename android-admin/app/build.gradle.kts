plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.androidx.baselineprofile)
}

android {
    namespace = "com.bunoraa.admin"
    compileSdk = 34
    val dotEnv = mutableMapOf<String, String>()
    val envFile = rootProject.projectDir.parentFile.resolve(".env")
    if (envFile.exists()) {
        envFile.forEachLine { line ->
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#") || !trimmed.contains("=")) return@forEachLine
            val idx = trimmed.indexOf("=")
            val key = trimmed.substring(0, idx).trim()
            val value = trimmed.substring(idx + 1).trim().removeSurrounding("\"").removeSurrounding("'")
            if (key.isNotBlank()) {
                dotEnv[key] = value
            }
        }
    }

    fun resolveConfig(keys: List<String>, defaultValue: String = ""): String {
        for (key in keys) {
            val fromProperty = (project.findProperty(key) as String?)?.trim().orEmpty()
            if (fromProperty.isNotBlank()) return fromProperty
            val fromEnvFile = dotEnv[key]?.trim().orEmpty()
            if (fromEnvFile.isNotBlank()) return fromEnvFile
        }
        return defaultValue
    }

    val googleClientId = resolveConfig(listOf("OIDC_GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_ID"))
    val microsoftClientId = resolveConfig(listOf("OIDC_MICROSOFT_CLIENT_ID"))
    val microsoftTenant = resolveConfig(listOf("OIDC_MICROSOFT_TENANT"), "common")
    val redirectScheme = resolveConfig(listOf("OIDC_REDIRECT_SCHEME"), "com.bunoraa.admin")
    val configuredRedirectUri = resolveConfig(listOf("OIDC_REDIRECT_URI", "GOOGLE_REDIRECT_URI"))
    val redirectUri = when {
        configuredRedirectUri.isBlank() -> "$redirectScheme:/oauth2redirect"
        configuredRedirectUri.startsWith("http://") || configuredRedirectUri.startsWith("https://") -> {
            "$redirectScheme:/oauth2redirect"
        }
        else -> configuredRedirectUri
    }
    val resolvedRedirectPath = runCatching { java.net.URI(redirectUri).path }
        .getOrNull()
        .takeIf { !it.isNullOrBlank() } ?: "/oauth2redirect"
    val resolvedRedirectScheme = redirectUri.substringBefore(":").ifBlank { redirectScheme }

    defaultConfig {
        applicationId = "com.bunoraa.admin"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("String", "API_BASE_URL", "\"https://api.bunoraa.com/api/v1/\"")
        buildConfigField("String", "WS_BASE_URL", "\"wss://api.bunoraa.com\"")
        buildConfigField("String", "OIDC_GOOGLE_CLIENT_ID", "\"$googleClientId\"")
        buildConfigField("String", "OIDC_MICROSOFT_CLIENT_ID", "\"$microsoftClientId\"")
        buildConfigField("String", "OIDC_MICROSOFT_TENANT", "\"$microsoftTenant\"")
        buildConfigField("String", "OIDC_REDIRECT_SCHEME", "\"$resolvedRedirectScheme\"")
        buildConfigField("String", "OIDC_REDIRECT_URI", "\"$redirectUri\"")
        manifestPlaceholders["appAuthRedirectScheme"] = resolvedRedirectScheme
        manifestPlaceholders["appAuthRedirectPath"] = resolvedRedirectPath
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

    baselineProfile {
        mergeIntoMain = true
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
    implementation(libs.androidx.profileinstaller)

    baselineProfile(project(":baselineprofile"))

    debugImplementation(libs.androidx.compose.ui.tooling)
}
