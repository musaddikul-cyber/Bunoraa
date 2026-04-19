import builtins
import importlib
import sys


def test_monitoring_imports_without_resource_module(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'resource':
            raise ModuleNotFoundError("No module named 'resource'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    sys.modules.pop('core.monitoring', None)

    try:
        module = importlib.import_module('core.monitoring')
        assert module.resource is None
        assert module._get_resource_memory_usage_mb() == 0.0
    finally:
        sys.modules.pop('core.monitoring', None)


def test_get_load_average_returns_zeroes_when_unsupported(monkeypatch):
    from core import monitoring

    def raise_unsupported():
        raise OSError('unsupported')

    monkeypatch.setattr(monitoring.os, 'getloadavg', raise_unsupported, raising=True)

    assert monitoring._get_load_average() == (0.0, 0.0, 0.0)
