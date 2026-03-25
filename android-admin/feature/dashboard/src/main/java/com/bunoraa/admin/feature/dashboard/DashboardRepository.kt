package com.bunoraa.admin.feature.dashboard

import com.bunoraa.admin.core.database.DashboardDao
import com.bunoraa.admin.core.database.DashboardEntity
import com.bunoraa.admin.core.network.AdminApiService
import com.bunoraa.admin.core.network.DashboardPayload
import com.bunoraa.admin.core.network.RealtimeClient
import com.bunoraa.admin.core.network.RealtimeEvent
import com.bunoraa.admin.core.network.RealtimePollingPayload
import com.bunoraa.admin.core.network.RealtimeStatus
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class DashboardRepository(
    private val api: AdminApiService,
    private val dao: DashboardDao,
    private val realtimeClient: RealtimeClient,
    private val wsBaseUrl: String,
) {
    fun observeDashboard(): Flow<DashboardUiModel?> {
        return dao.observeDashboard().map { entity ->
            entity?.toUiModel()
        }
    }

    val realtimeEvents: Flow<RealtimeEvent> = realtimeClient.events
    val realtimeStatus: Flow<RealtimeStatus> = realtimeClient.status

    suspend fun refresh(): Result<DashboardUiModel> {
        return try {
            val response = api.getDashboard()
            val payload = response.data
            if (response.success && payload != null) {
                val entity = payload.toEntity()
                dao.upsert(entity)
                Result.success(entity.toUiModel())
            } else {
                Result.failure(RuntimeException(response.message ?: "Dashboard refresh failed"))
            }
        } catch (exc: Exception) {
            Result.failure(exc)
        }
    }

    suspend fun pollRealtimeEvents(since: String?): Result<RealtimePollingPayload> {
        return try {
            val response = api.pollRealtimeEvents(since = since, limit = 50)
            val payload = response.data
            if (response.success && payload != null) {
                Result.success(payload)
            } else {
                Result.failure(RuntimeException(response.message ?: "Realtime polling failed"))
            }
        } catch (exc: Exception) {
            Result.failure(exc)
        }
    }

    fun connectRealtime() {
        realtimeClient.connect(wsBaseUrl, "/ws/admin/updates/")
    }

    fun disconnectRealtime() {
        realtimeClient.disconnect()
    }

    private fun DashboardPayload.toEntity(): DashboardEntity {
        return DashboardEntity(
            id = 0,
            generatedAt = generatedAt,
            windowDays = windowDays,
            users = totals.users,
            products = totals.products,
            orders = totals.orders,
            ordersPending = totals.ordersPending,
            revenue30d = totals.revenue30d,
        )
    }

    private fun DashboardEntity.toUiModel(): DashboardUiModel {
        return DashboardUiModel(
            generatedAt = generatedAt,
            windowDays = windowDays,
            users = users,
            products = products,
            orders = orders,
            ordersPending = ordersPending,
            revenue30d = revenue30d,
        )
    }
}

data class DashboardUiModel(
    val generatedAt: String,
    val windowDays: Int,
    val users: Long,
    val products: Long,
    val orders: Long,
    val ordersPending: Long,
    val revenue30d: String,
)
