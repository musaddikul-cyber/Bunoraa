package com.bunoraa.admin

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

class DashboardSyncWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val app = applicationContext as? BunoraaAdminApp
            ?: return Result.failure()
        return try {
            app.container.dashboardRepository.refresh()
            Result.success()
        } catch (exc: Exception) {
            Result.retry()
        }
    }
}
