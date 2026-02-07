"""
Storage backends for Bunoraa.
"""
from .r2_storage import (
    BunoraR2Storage,
    BunoraR2MediaStorage,
    BunoraR2StaticStorage,
    BunoraR2BackupStorage,
)

__all__ = [
    'BunoraR2Storage',
    'BunoraR2MediaStorage',
    'BunoraR2StaticStorage',
    'BunoraR2BackupStorage',
]
