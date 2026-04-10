pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BunoraaAdmin"
val includePerfModules = providers.gradleProperty("includePerfModules").orNull == "true"

include(
    ":app",
    ":core:common",
    ":core:network",
    ":core:database",
    ":core:datastore",
    ":core:designsystem",
    ":feature:auth",
    ":feature:dashboard",
)

if (includePerfModules) {
    include(
        ":baselineprofile",
        ":benchmark",
    )
}
