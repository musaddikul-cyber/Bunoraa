package com.bunoraa.admin

import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class AdminMessagingService : FirebaseMessagingService() {
    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        val title = message.notification?.title
            ?: message.data["title"]
            ?: "Bunoraa Admin"
        val body = message.notification?.body
            ?: message.data["message"]
            ?: "You have a new update."
        val deepLink = AdminDeepLinkParser.fromPayload(message.data, title, body)
        AdminNotificationManager.showNotification(this, deepLink)
    }

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        val app = application as? BunoraaAdminApp ?: return
        app.container.pushTokenRegistrar.savePending(token)
        CoroutineScope(Dispatchers.IO).launch {
            app.container.pushTokenRegistrar.registerIfPossible()
        }
    }
}
