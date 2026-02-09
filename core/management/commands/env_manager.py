from django.core.management.base import BaseCommand, CommandError

from apps.env_registry import env_manager


class Command(BaseCommand):
    help = "Manage env targets from config/env.schema.yml (sync/validate/prune)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--schema",
            default=str(env_manager.DEFAULT_SCHEMA_PATH),
            help="Path to env schema YAML (default: config/env.schema.yml)",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        sync_parser = subparsers.add_parser("sync", help="Sync env targets from schema.")
        sync_parser.add_argument("--env", required=True, choices=["development", "production"])
        sync_parser.add_argument("--targets", action="append", help="Comma-separated target names.")
        sync_parser.add_argument("--set", action="append", help="Override KEY=VALUE.")
        sync_parser.add_argument("--prune", action="store_true", help="Remove unknown keys.")
        sync_parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")

        validate_parser = subparsers.add_parser("validate", help="Validate env targets.")
        validate_parser.add_argument("--env", required=True, choices=["development", "production"])
        validate_parser.add_argument("--targets", action="append", help="Comma-separated target names.")

        prune_parser = subparsers.add_parser("prune", help="Remove unknown keys from targets.")
        prune_parser.add_argument("--env", required=True, choices=["development", "production"])
        prune_parser.add_argument("--targets", action="append", help="Comma-separated target names.")
        prune_parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")

    def handle(self, *args, **options):
        argv = ["env_manager"]
        schema = options.get("schema")
        if schema:
            argv.extend(["--schema", schema])

        command = options.get("command")
        if not command:
            raise SystemExit(1)
        argv.append(command)

        if options.get("env"):
            argv.extend(["--env", options["env"]])

        targets = options.get("targets") or []
        for target in targets:
            argv.extend(["--targets", target])

        overrides = options.get("set") or []
        for item in overrides:
            argv.extend(["--set", item])

        if options.get("prune"):
            argv.append("--prune")
        if options.get("dry_run"):
            argv.append("--dry-run")

        exit_code = env_manager.main_with_argv(argv)
        if exit_code:
            raise CommandError(f"env_manager failed with exit code {exit_code}")
