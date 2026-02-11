self.addEventListener("push", (event) => {
  let payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch (error) {
    payload = { body: event.data ? event.data.text() : "" };
  }

  const title = payload.title || "Bunoraa";
  const options = {
    body: payload.body || "",
    data: payload.data || payload,
    icon: payload.icon || "/favicon.ico",
    badge: payload.badge || "/favicon.ico",
    actions: payload.actions || [],
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl =
    (event.notification && event.notification.data && event.notification.data.url) ||
    "/";
  event.waitUntil(clients.openWindow(targetUrl));
});

self.addEventListener("pushsubscriptionchange", (event) => {
  event.waitUntil(self.registration.pushManager.subscribe({ userVisibleOnly: true }));
});
