package com.bunoraa.admin

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import kotlin.math.abs

object AdminNotificationManager {
    private const val CHANNEL_ID = "admin_updates"

    fun showNotification(
        context: Context,
        deepLink: AdminDeepLink,
    ) {
        ensureChannel(context)
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        AdminDeepLinkParser.toIntent(intent, deepLink)

        val pendingIntent = PendingIntent.getActivity(
            context,
            abs(deepLink.hashCode()),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_more)
            .setContentTitle(deepLink.title)
            .setContentText(deepLink.message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(deepLink.message))
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()

        NotificationManagerCompat.from(context).notify(abs(deepLink.hashCode()), notification)
    }

    private fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Admin Updates",
            NotificationManager.IMPORTANCE_HIGH,
        ).apply {
            description = "Realtime updates and alerts from Bunoraa Admin"
        }
        manager.createNotificationChannel(channel)
    }
}
