"""
Custom logging filters for suppressing expected errors.
"""
import logging
import asyncio


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
                if exc_type is asyncio.CancelledError:
                    return False
            
            # Also check if CancelledError is in the message
            if 'CancelledError' in record.getMessage():
                return False
        
        return True
