package com.bunoraa.admin

import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class AdminMessagingService : FirebaseMessagingService() {
    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        // TODO: show high-priority notification or update local cache.
    }

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        // TODO: register push token with /api/v1/notifications/push-tokens/.
    }
}
