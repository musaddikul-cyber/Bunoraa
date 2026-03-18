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
    }
}

rootProject.name = "BunoraaAdmin"

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
