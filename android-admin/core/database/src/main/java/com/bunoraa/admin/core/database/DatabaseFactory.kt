package com.bunoraa.admin.core.database

import android.content.Context
import androidx.room.Room

fun createAdminDatabase(context: Context): AdminDatabase {
    return Room.databaseBuilder(
        context,
        AdminDatabase::class.java,
        "bunoraa_admin.db",
    ).fallbackToDestructiveMigration().build()
}
