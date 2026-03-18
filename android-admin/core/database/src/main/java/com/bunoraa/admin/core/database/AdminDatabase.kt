package com.bunoraa.admin.core.database

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(
    entities = [DashboardEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class AdminDatabase : RoomDatabase() {
    abstract fun dashboardDao(): DashboardDao
}
