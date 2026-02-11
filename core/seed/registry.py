from __future__ import annotations

import importlib
from typing import Dict, List

from django.apps import apps

from .base import SeedSpec

_SEED_REGISTRY: Dict[str, SeedSpec] = {}


def register_seed(spec: SeedSpec) -> SeedSpec:
    if not spec.name:
        raise ValueError("SeedSpec.name is required")
    if spec.name in _SEED_REGISTRY:
        raise ValueError(f"Duplicate SeedSpec name: {spec.name}")
    _SEED_REGISTRY[spec.name] = spec
    return spec


def get_seed_specs() -> Dict[str, SeedSpec]:
    return dict(_SEED_REGISTRY)


def autodiscover_seeders() -> List[str]:
    loaded = []
    for app_config in apps.get_app_configs():
        module_name = f"{app_config.name}.seeders"
        try:
            importlib.import_module(module_name)
            loaded.append(module_name)
        except ModuleNotFoundError:
            continue
        except Exception:
            # Ignore failures to avoid blocking runtime, but caller can inspect logs
            continue
    return loaded
