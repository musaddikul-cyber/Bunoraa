from __future__ import annotations

from django.utils import timezone

from core.utils.axes import get_client_ip

from .models import AdminAuditLog


class AdminAuditMiddleware:
    """Record audit logs for admin API requests."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        try:
            if not request.path.startswith("/api/v1/admin/"):
                return response
            if request.method == "OPTIONS":
                return response

            match = getattr(request, "resolver_match", None)
            action = (match.view_name if match else "") or request.path

            AdminAuditLog.objects.create(
                actor=request.user if getattr(request, "user", None) and request.user.is_authenticated else None,
                action=action,
                resource_type=match.app_name if match else "",
                resource_id=str(match.kwargs.get("pk")) if match and match.kwargs else "",
                path=request.path,
                method=request.method,
                status_code=getattr(response, "status_code", 0) or 0,
                ip_address=get_client_ip(request) or None,
                user_agent=request.META.get("HTTP_USER_AGENT", "")[:512],
                metadata={
                    "query": request.META.get("QUERY_STRING", ""),
                    "timestamp": timezone.now().isoformat(),
                },
            )
        except Exception:
            # Never block responses on audit logging failures.
            pass

        return response
