from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterable

from django.core.management.base import BaseCommand, CommandError

from core.seed.registry import autodiscover_seeders, get_seed_specs
from core.seed.runner import SeedRunner


def split_csv_option(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def resolve_confirm_prune(options: dict, *legacy_keys: str) -> bool:
    if options.get("confirm_prune"):
        return True
    return any(bool(options.get(key)) for key in legacy_keys)


def validate_seed_specs(only: Iterable[str] | None, *, kind: str) -> None:
    if not only:
        return

    autodiscover_seeders()
    specs = get_seed_specs()
    available = set(specs.keys())

    selected = [name.strip() for name in only if name and name.strip()]
    missing = sorted(name for name in selected if name not in available)
    wrong_kind = sorted(
        name
        for name in selected
        if name in specs and specs[name].kind != kind
    )

    errors = []
    if missing:
        errors.append(f"Unknown seed specs: {', '.join(missing)}")
    if wrong_kind:
        errors.append(
            f"Seed specs with wrong kind (expected '{kind}'): {', '.join(wrong_kind)}"
        )
    if errors:
        raise CommandError(" | ".join(errors))


@contextmanager
def temporary_environ(overrides: dict[str, str] | None):
    if not overrides:
        yield
        return

    sentinel = object()
    previous: dict[str, str | object] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key, sentinel)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def run_seed_command(
    command: BaseCommand,
    *,
    kind: str,
    success_label: str,
    only: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    dry_run: bool = False,
    no_prune: bool = False,
    confirm_prune: bool = False,
    seed: int | None = None,
    env_overrides: dict[str, str] | None = None,
    fail_on_errors: bool = True,
):
    only_list = [item.strip() for item in only if item and item.strip()] if only else None
    exclude_list = [item.strip() for item in exclude if item and item.strip()] if exclude else None

    validate_seed_specs(only_list, kind=kind)
    validate_seed_specs(exclude_list, kind=kind)

    if seed is not None:
        random.seed(seed)

    runner = SeedRunner(
        dry_run=dry_run,
        prune=not no_prune,
        confirm_prune=confirm_prune,
        logger=command.stdout.write,
    )

    with temporary_environ(env_overrides):
        result = runner.run(
            only=only_list,
            exclude=exclude_list,
            kind=kind,
        )

    command.stdout.write("")
    command.stdout.write(command.style.SUCCESS(success_label))
    command.stdout.write(f"  Created: {result.created}")
    command.stdout.write(f"  Updated: {result.updated}")
    command.stdout.write(f"  Pruned:  {result.pruned}")
    command.stdout.write(f"  Skipped: {result.skipped}")
    command.stdout.write(f"  Errors:  {result.errors}")

    if fail_on_errors and result.errors:
        raise CommandError(
            f"{success_label.rstrip('.')} failed with {result.errors} error(s)."
        )

    return result
