"""
Custom logging filters for suppressing expected errors.
"""
import logging
import asyncio
import concurrent.futures

try:
    from core.request_context import get_request_id
except Exception:  # pragma: no cover
    def get_request_id() -> str:  # type: ignore[no-redef]
        return "-"


class IgnoreCancelledErrorFilter(logging.Filter):
    """
    Filter to suppress CancelledError tracebacks from asgiref.
    
    These errors occur when clients disconnect (close browser tab, cancel request, etc)
    and are expected behavior, not actual errors.
    """
    
    def filter(self, record):
        """
        Return False to suppress the log record if it's a CancelledError.
        """
        # Check if this is an ERROR level log with CancelledError
        if record.levelno >= logging.ERROR:
            # Check the exception info
            if record.exc_info:
                exc_type = record.exc_info[0]
                if exc_type in (asyncio.CancelledError, concurrent.futures.CancelledError):
                    return False
            
            # Also check if CancelledError is in the message
            message = record.getMessage()
            if (
                'CancelledError' in message
                or 'exception in shielded future' in message
            ):
                return False
        
        return True


class RequestIdFilter(logging.Filter):
    """Attach `request_id` to log records for correlation."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            value = get_request_id() or "-"
        except Exception:
            value = "-"
        record.request_id = value
        return True
