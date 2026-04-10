"""
Import a site backup archive produced by `backup_site`.

Usage examples:
  python manage.py import_site_backup
  python manage.py import_site_backup --input=backups/site_backup_20260404T032035Z.tar.gz
  python manage.py import_site_backup --apps=accounts,catalog --ignorenonexistent
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from django.apps import apps as django_apps
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import connections


DEFAULT_EXCLUDED_APPS = {"auth", "contenttypes"}


def _resolve_latest_backup(backups_dir: Path) -> Path:
    candidates = [p for p in backups_dir.glob("site_backup_*.tar*") if p.is_file()]
    if not candidates:
        raise CommandError(f"No backup archives found in {backups_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if root not in member_path.parents and member_path != root:
            raise CommandError(f"Unsafe archive member path detected: {member.name}")
    archive.extractall(destination)


@contextmanager
def _fixture_import_flag():
    original = os.environ.get("BUNORAA_IMPORTING_FIXTURES")
    os.environ["BUNORAA_IMPORTING_FIXTURES"] = "1"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("BUNORAA_IMPORTING_FIXTURES", None)
        else:
            os.environ["BUNORAA_IMPORTING_FIXTURES"] = original


def _count_other_db_connections(database: str) -> int:
    connection = connections[database]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND pid <> pg_backend_pid()
              AND state <> 'idle'
            """
        )
        result = cursor.fetchone()
    return int(result[0]) if result else 0


def _app_dependencies(app_label: str) -> set[str]:
    """Return direct inter-app model relation dependencies."""
    try:
        app_config = django_apps.get_app_config(app_label)
    except LookupError:
        return set()

    deps: set[str] = set()
    for model in app_config.get_models():
        for field in model._meta.get_fields():
            if not getattr(field, "is_relation", False):
                continue
            related_model = getattr(field, "related_model", None)
            if related_model is None:
                continue
            related_app = related_model._meta.app_label
            if related_app != app_label:
                deps.add(related_app)
    return deps


def _order_fixtures_by_dependencies(fixtures: list[Path]) -> list[Path]:
    """
    Topologically sort fixtures by app-level model dependencies.
    Falls back to lexical ordering for unresolved cycles.
    """
    app_to_fixture = {fixture.stem: fixture for fixture in fixtures}
    selected_apps = set(app_to_fixture.keys())

    graph: dict[str, set[str]] = defaultdict(set)
    in_degree: dict[str, int] = {app: 0 for app in selected_apps}

    for app in selected_apps:
        deps = _app_dependencies(app) & selected_apps
        for dep in deps:
            if app in graph[dep]:
                continue
            graph[dep].add(app)
            in_degree[app] += 1

    queue = deque(sorted([app for app in selected_apps if in_degree[app] == 0]))
    ordered_apps: list[str] = []

    while queue:
        app = queue.popleft()
        ordered_apps.append(app)
        for dependent in sorted(graph.get(app, set())):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered_apps) < len(selected_apps):
        remaining = sorted(selected_apps - set(ordered_apps))
        ordered_apps.extend(remaining)

    return [app_to_fixture[app] for app in ordered_apps]


def _group_fixtures_by_dependency(fixtures: list[Path]) -> list[list[Path]]:
    """
    Group fixtures by strongly connected app components so cyclic dependencies
    can be loaded in a single loaddata call.
    """
    if not fixtures:
        return []

    app_to_fixture = {fixture.stem: fixture for fixture in fixtures}
    selected_apps = set(app_to_fixture.keys())
    deps_map: dict[str, set[str]] = {
        app: (_app_dependencies(app) & selected_apps) for app in selected_apps
    }

    # Tarjan SCC
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for dep in sorted(deps_map.get(node, set())):
            if dep not in indices:
                strongconnect(dep)
                lowlinks[node] = min(lowlinks[node], lowlinks[dep])
            elif dep in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[dep])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                current = stack.pop()
                on_stack.remove(current)
                component.append(current)
                if current == node:
                    break
            sccs.append(sorted(component))

    for app in sorted(selected_apps):
        if app not in indices:
            strongconnect(app)

    app_to_component: dict[str, int] = {}
    for component_id, component_apps in enumerate(sccs):
        for app in component_apps:
            app_to_component[app] = component_id

    component_graph: dict[int, set[int]] = defaultdict(set)
    component_in_degree: dict[int, int] = {i: 0 for i in range(len(sccs))}

    # app depends on dep => dep component must be loaded before app component
    for app, deps in deps_map.items():
        source = app_to_component[app]
        for dep in deps:
            target = app_to_component[dep]
            if source == target:
                continue
            if source in component_graph[target]:
                continue
            component_graph[target].add(source)
            component_in_degree[source] += 1

    queue = deque(sorted([cid for cid, deg in component_in_degree.items() if deg == 0]))
    ordered_components: list[int] = []
    while queue:
        cid = queue.popleft()
        ordered_components.append(cid)
        for dependent in sorted(component_graph.get(cid, set())):
            component_in_degree[dependent] -= 1
            if component_in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered_components) < len(sccs):
        remaining = sorted(set(range(len(sccs))) - set(ordered_components))
        ordered_components.extend(remaining)

    groups: list[list[Path]] = []
    for component_id in ordered_components:
        component_apps = sccs[component_id]
        groups.append([app_to_fixture[app] for app in component_apps])

    return groups


class Command(BaseCommand):
    help = "Import fixtures from a site backup archive into the configured database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            help="Backup archive path (defaults to latest file in ./backups).",
        )
        parser.add_argument(
            "--apps",
            type=str,
            help="Comma-separated app labels to import (based on fixtures/<app>.json filenames).",
        )
        parser.add_argument(
            "--exclude-apps",
            type=str,
            help="Comma-separated app labels to skip.",
        )
        parser.add_argument(
            "--include-framework-apps",
            action="store_true",
            help=(
                "Include framework-managed apps (auth, contenttypes). "
                "By default these are skipped to avoid duplicate permission/content-type records."
            ),
        )
        parser.add_argument(
            "--database",
            type=str,
            default="default",
            help="Database alias (default: default).",
        )
        parser.add_argument(
            "--ignorenonexistent",
            action="store_true",
            help="Ignore fields/models missing from current schema during loaddata.",
        )
        parser.add_argument(
            "--list-only",
            action="store_true",
            help="List selected fixture files and exit without importing.",
        )
        parser.add_argument(
            "--require-exclusive-db",
            action="store_true",
            help="Abort if other active connections are using this database.",
        )

    def _parse_csv(self, value: str | None) -> set[str]:
        if not value:
            return set()
        return {item.strip() for item in value.split(",") if item.strip()}

    def _select_fixtures(
        self,
        fixtures: Iterable[Path],
        include_apps: set[str],
        exclude_apps: set[str],
    ) -> list[Path]:
        selected: list[Path] = []
        for fixture in sorted(fixtures, key=lambda p: p.name.lower()):
            app_label = fixture.stem
            if include_apps and app_label not in include_apps:
                continue
            if app_label in exclude_apps:
                continue
            selected.append(fixture)
        return selected

    def handle(self, *args, **options):
        database = options["database"]
        include_apps = self._parse_csv(options.get("apps"))
        exclude_apps = self._parse_csv(options.get("exclude_apps"))
        if not options.get("include_framework_apps"):
            exclude_apps |= DEFAULT_EXCLUDED_APPS

        if options.get("input"):
            archive_path = Path(options["input"]).expanduser().resolve()
            if not archive_path.exists():
                raise CommandError(f"Backup archive not found: {archive_path}")
        else:
            archive_path = _resolve_latest_backup(Path("backups").resolve())

        self.stdout.write(self.style.NOTICE(f"Using backup archive: {archive_path}"))

        tmpdir = Path(tempfile.mkdtemp(prefix="site_restore_"))
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                _safe_extract_tar(tar, tmpdir)

            fixtures_dir = tmpdir / "fixtures"
            if not fixtures_dir.exists():
                raise CommandError(
                    "Backup archive does not contain a fixtures/ directory. "
                    "This command expects archives produced by backup_site."
                )

            fixtures = self._select_fixtures(
                fixtures=fixtures_dir.glob("*.json"),
                include_apps=include_apps,
                exclude_apps=exclude_apps,
            )
            fixtures = _order_fixtures_by_dependencies(fixtures)
            fixture_groups = _group_fixtures_by_dependency(fixtures)

            if not fixtures:
                raise CommandError("No fixture files selected for import.")

            self.stdout.write(
                self.style.NOTICE(f"Selected {len(fixtures)} fixture files for import.")
            )
            if not options.get("include_framework_apps"):
                self.stdout.write(
                    self.style.NOTICE(
                        "Auto-excluding framework apps: "
                        + ", ".join(sorted(DEFAULT_EXCLUDED_APPS))
                    )
                )
            for fixture in fixtures:
                self.stdout.write(f"  - {fixture.name}")

            if options.get("list_only"):
                self.stdout.write(self.style.SUCCESS("List-only mode: no data imported."))
                return

            try:
                other_connections = _count_other_db_connections(database)
            except Exception:
                other_connections = 0

            if options.get("require_exclusive_db") and other_connections > 0:
                raise CommandError(
                    "Detected active database traffic while importing fixtures. "
                    "Stop web/worker processes and retry."
                )
            if other_connections > 0:
                self.stdout.write(
                    self.style.WARNING(
                        f"Detected {other_connections} other active DB connection(s). "
                        "Imports can fail under live traffic."
                    )
                )

            loaddata_kwargs = {
                "database": database,
                "verbosity": options.get("verbosity", 1),
            }
            if options.get("ignorenonexistent"):
                loaddata_kwargs["ignorenonexistent"] = True

            self.stdout.write(
                self.style.NOTICE(
                    f"Importing fixtures into database '{database}' via loaddata (one file at a time)..."
                )
            )
            total_started_at = time.monotonic()
            total_groups = len(fixture_groups)
            with _fixture_import_flag():
                for index, group in enumerate(fixture_groups, start=1):
                    group_started_at = time.monotonic()
                    if len(group) == 1:
                        group_label = group[0].name
                    else:
                        group_label = ", ".join(path.name for path in group)
                    self.stdout.write(
                        self.style.NOTICE(
                            f"[{index}/{total_groups}] Loading {group_label} ..."
                        )
                    )
                    call_command("loaddata", *(str(path) for path in group), **loaddata_kwargs)
                    group_elapsed = time.monotonic() - group_started_at
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"[{index}/{total_groups}] Loaded {group_label} in {group_elapsed:.1f}s"
                        )
                    )

            total_elapsed = time.monotonic() - total_started_at
            self.stdout.write(
                self.style.SUCCESS(
                    f"Backup import completed successfully in {total_elapsed:.1f}s."
                )
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
