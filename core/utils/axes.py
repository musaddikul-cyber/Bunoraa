def get_client_ip(request):
    """Resolve client IP with Cloudflare/Render headers and fallbacks."""
    meta = request.META

    cf_ip = meta.get("HTTP_CF_CONNECTING_IP")
    if cf_ip:
        return cf_ip

    xff = meta.get("HTTP_X_FORWARDED_FOR")
    if xff:
        # XFF can contain multiple IPs, take the first one (client)
        return xff.split(",")[0].strip()

    real_ip = meta.get("HTTP_X_REAL_IP")
    if real_ip:
        return real_ip

    return meta.get("REMOTE_ADDR", "")
