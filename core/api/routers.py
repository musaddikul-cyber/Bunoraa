from rest_framework.routers import DefaultRouter as DRFDefaultRouter
from rest_framework.routers import SimpleRouter as DRFSimpleRouter


class DefaultRouter(DRFDefaultRouter):
    """DefaultRouter that accepts optional trailing slashes."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trailing_slash", "/?")
        super().__init__(*args, **kwargs)


class SimpleRouter(DRFSimpleRouter):
    """SimpleRouter that accepts optional trailing slashes."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trailing_slash", "/?")
        super().__init__(*args, **kwargs)
