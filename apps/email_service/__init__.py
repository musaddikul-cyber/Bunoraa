"""
Bunoraa Email Service Provider
==============================

A comprehensive self-hosted email service provider similar to SendGrid.
Provides email delivery infrastructure, tracking, templates, and APIs.

Features:
- Multi-tenant email delivery
- Domain verification (SPF, DKIM, DMARC)
- Email tracking (opens, clicks, bounces, complaints)
- Template management with versioning
- Suppression lists (unsubscribes, bounces, spam reports)
- Webhooks for delivery events
- Rate limiting and throttling
- Analytics and reporting
- API key authentication
- Bulk sending with queue management
"""

default_app_config = 'apps.email_service.apps.EmailServiceConfig'
