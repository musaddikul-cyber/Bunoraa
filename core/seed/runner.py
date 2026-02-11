from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from django.conf import settings

from .base import SeedContext, SeedResult, SeedSpec
from .registry import autodiscover_seeders, get_seed_specs


class SeedRunner:
    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        dry_run: bool = False,
        prune: bool = True,
        confirm_prune: bool = False,
        environment: str | None = None,
        logger=None,
    ) -> None:
        self.base_dir = base_dir or Path(settings.BASE_DIR)
        self.dry_run = dry_run
        self.prune = prune
        self.confirm_prune = confirm_prune
        self.environment = environment or getattr(settings, "ENVIRONMENT", "development")
        self.logger = logger

    def run(
        self,
        *,
        only: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        kind: str = "prod",
    ) -> SeedResult:
        autodiscover_seeders()
        specs = get_seed_specs()

        selected = self._select_specs(specs, only=only, exclude=exclude, kind=kind)
        ordered = self._resolve_order(selected, specs)

        ctx = SeedContext(
            base_dir=self.base_dir,
            dry_run=self.dry_run,
            prune=self._prune_enabled(),
            confirm_prune=self.confirm_prune,
            environment=self.environment,
            logger=self.logger,
        )

        result = SeedResult()
        for name in ordered:
            spec = specs[name]
            ctx.log(f"[seed] Running {name}")
            try:
                spec_result = spec.apply(ctx)
                result.merge(spec_result)
            except Exception as exc:
                ctx.log(f"[seed:{name}] ERROR: {exc}")
                result.errors += 1
        return result

    def _prune_enabled(self) -> bool:
        if not self.prune:
            return False
        if self.environment == "production" and not self.confirm_prune:
            if self.logger:
                self.logger("[seed] Prune blocked in production (confirm required).")
            return False
        return True

    def _select_specs(
        self,
        specs: dict[str, SeedSpec],
        *,
        only: Iterable[str] | None,
        exclude: Iterable[str] | None,
        kind: str,
    ) -> list[str]:
        names = [name for name, spec in specs.items() if spec.kind == kind]

        if only:
            only_set = {n.strip() for n in only}
            names = [n for n in names if n in only_set]

        if exclude:
            exclude_set = {n.strip() for n in exclude}
            names = [n for n in names if n not in exclude_set]

        return names

    def _resolve_order(self, selected: list[str], specs: dict[str, SeedSpec]) -> list[str]:
        graph = defaultdict(list)
        indegree = defaultdict(int)

        selected_set = set(selected)
        for name in selected:
            spec = specs[name]
            for dep in spec.dependencies:
                if dep not in selected_set:
                    continue
                graph[dep].append(name)
                indegree[name] += 1

        queue = deque([name for name in selected if indegree[name] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in graph[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        # Fallback if cycle or missing dependencies
        if len(order) != len(selected):
            return selected
        return order
