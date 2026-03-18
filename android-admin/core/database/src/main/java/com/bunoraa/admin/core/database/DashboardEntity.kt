package com.bunoraa.admin.core.database

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "dashboard_cache")
data class DashboardEntity(
    @PrimaryKey val id: Int = 0,
    @ColumnInfo(name = "generated_at") val generatedAt: String,
    @ColumnInfo(name = "window_days") val windowDays: Int,
    @ColumnInfo(name = "users") val users: Long,
    @ColumnInfo(name = "products") val products: Long,
    @ColumnInfo(name = "orders") val orders: Long,
    @ColumnInfo(name = "orders_pending") val ordersPending: Long,
    @ColumnInfo(name = "revenue_30d") val revenue30d: String,
)
