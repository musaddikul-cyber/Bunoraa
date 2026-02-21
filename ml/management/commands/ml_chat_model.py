"""
ML Chat Model command.

Usage examples:
  python manage.py ml_chat_model --status
  python manage.py ml_chat_model --warmup
  python manage.py ml_chat_model --warmup --model Qwen/Qwen2.5-1.5B-Instruct --strict
  python manage.py ml_chat_model --prompt "Track my order"
"""

from __future__ import annotations

import json
from typing import Any, Dict

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Inspect, warmup, and smoke-test the local ML chat model service"

    def add_arguments(self, parser):
        parser.add_argument(
            "--status",
            action="store_true",
            help="Show chat model service status",
        )
        parser.add_argument(
            "--warmup",
            action="store_true",
            help="Preload tokenizer/model into memory",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="",
            help="Optional model ID override for warmup/test",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default="",
            help="Run a local generation smoke test with this prompt",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Return non-zero exit code if warmup/test fails",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output result as JSON",
        )

    def handle(self, *args, **options):
        from ml.services.chat_model_service import ChatModelService

        model_override = (options.get("model") or "").strip() or None
        run_status = bool(options.get("status"))
        run_warmup = bool(options.get("warmup"))
        prompt = (options.get("prompt") or "").strip()
        strict = bool(options.get("strict"))
        as_json = bool(options.get("json"))

        if not (run_status or run_warmup or prompt):
            run_status = True

        result: Dict[str, Any] = {
            "enabled": ChatModelService.is_enabled(),
            "model_override": model_override,
        }

        if run_warmup:
            result["warmup_ok"] = ChatModelService.warmup(model_id_override=model_override)

        if prompt:
            reply = ChatModelService.generate_reply(
                system_prompt="You are Bunoraa's local support assistant.",
                history=[{"role": "user", "content": prompt}],
                personalization={},
                model_id_override=model_override,
            )
            result["smoke_test_prompt"] = prompt
            result["smoke_test_reply"] = reply
            result["smoke_test_ok"] = bool(reply)

        if run_status or run_warmup or prompt:
            result["status"] = ChatModelService.get_status()

        if as_json:
            self.stdout.write(json.dumps(result, indent=2, default=str))
        else:
            self.stdout.write("")
            self.stdout.write(self.style.NOTICE("Local Chat Model"))
            self.stdout.write(f"Enabled: {result['enabled']}")
            if model_override:
                self.stdout.write(f"Override: {model_override}")

            if run_warmup:
                warmup_ok = bool(result.get("warmup_ok"))
                style = self.style.SUCCESS if warmup_ok else self.style.ERROR
                self.stdout.write(style(f"Warmup: {'OK' if warmup_ok else 'FAILED'}"))

            if prompt:
                smoke_ok = bool(result.get("smoke_test_ok"))
                style = self.style.SUCCESS if smoke_ok else self.style.ERROR
                self.stdout.write(style(f"Smoke Test: {'OK' if smoke_ok else 'FAILED'}"))
                if result.get("smoke_test_reply"):
                    self.stdout.write(f"Reply: {result['smoke_test_reply']}")

            status = result.get("status") or {}
            self.stdout.write(f"Loaded Model: {status.get('loaded_model_id') or '-'}")
            self.stdout.write(f"Device: {status.get('compute_device') or '-'}")
            if status.get("load_error"):
                self.stdout.write(self.style.WARNING(f"Load Error: {status['load_error']}"))

        if strict:
            if run_warmup and not result.get("warmup_ok"):
                raise CommandError("Local chat model warmup failed")
            if prompt and not result.get("smoke_test_ok"):
                raise CommandError("Local chat model smoke test failed")
