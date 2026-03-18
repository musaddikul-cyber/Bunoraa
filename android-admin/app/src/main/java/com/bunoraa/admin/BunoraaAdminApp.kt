package com.bunoraa.admin

import android.app.Application

class BunoraaAdminApp : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        SyncScheduler.schedule(this)
    }
}
