package com.bunoraa.admin.feature.dashboard

import com.bunoraa.admin.core.database.DashboardDao
import com.bunoraa.admin.core.database.DashboardEntity
import com.bunoraa.admin.core.network.AdminApiService
import com.bunoraa.admin.core.network.DashboardPayload
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class DashboardRepository(
    private val api: AdminApiService,
    private val dao: DashboardDao,
) {
    fun observeDashboard(): Flow<DashboardUiModel?> {
        return dao.observeDashboard().map { entity ->
            entity?.toUiModel()
        }
    }

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
