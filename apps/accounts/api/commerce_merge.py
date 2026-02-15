"""Helpers for merging guest commerce session state after authentication."""
from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def merge_guest_commerce_state(request, user) -> None:
    """Best-effort merge of guest cart/wishlist into authenticated user data."""
    if not user or not getattr(user, 'is_authenticated', False):
        return

    session = getattr(request, 'session', None)
    session_key = getattr(session, 'session_key', None) if session else None
    if not session:
        return

    if not session_key:
        try:
            session.cycle_key()
        except Exception:
            logger.exception(
                "Failed to rotate session key after authentication for user_id=%s",
                getattr(user, 'id', None),
            )
        return

    try:
        from apps.commerce.services import SessionMergeService

        SessionMergeService.merge_guest_state_to_user(user=user, session_key=session_key)
    except Exception:
        logger.exception(
            "Failed to merge guest commerce state for user_id=%s session_key=%s",
            getattr(user, 'id', None),
            session_key,
        )
    finally:
        try:
            session.cycle_key()
        except Exception:
            logger.exception(
                "Failed to rotate session key after authentication for user_id=%s",
                getattr(user, 'id', None),
            )
