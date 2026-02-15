"""
Local pretrained chat model service.

This service runs inference with a Hugging Face chat model using local/runtime
configuration from Django settings. It does not require external API keys.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from django.conf import settings

logger = logging.getLogger("bunoraa.ml")


class ChatModelService:
    """Thread-safe lazy loader and inference wrapper for local chat models."""

    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    _loaded_model_id: Optional[str] = None
    _compute_device: str = "cpu"
    _load_error: Optional[str] = None
    _last_load_seconds: float = 0.0

    @classmethod
    def is_enabled(cls) -> bool:
        return bool(
            getattr(settings, "ML_ENABLED", False)
            and getattr(settings, "ML_CHAT_ASSISTANT_ENABLED", True)
            and getattr(settings, "CHAT_AI_LOCAL_MODEL_ENABLED", True)
        )

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        return {
            "enabled": cls.is_enabled(),
            "loaded_model_id": cls._loaded_model_id,
            "compute_device": cls._compute_device,
            "load_error": cls._load_error,
            "last_load_seconds": cls._last_load_seconds,
        }

    @classmethod
    def generate_reply(
        cls,
        *,
        system_prompt: str,
        history: List[Dict[str, str]],
        personalization: Optional[Dict[str, Any]] = None,
        model_id_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Generate a chat reply from a local pretrained model."""
        if not cls.is_enabled():
            return None

        if not cls._ensure_loaded(model_id_override=model_id_override):
            return None

        try:
            import torch
        except Exception as exc:
            cls._load_error = f"PyTorch import failed: {exc}"
            logger.warning("Local chat model disabled: %s", cls._load_error)
            return None

        messages = cls._build_messages(
            system_prompt=system_prompt,
            history=history,
            personalization=personalization or {},
        )
        prompt = cls._format_messages(messages)
        if not prompt:
            return None

        max_input_tokens = max(256, int(getattr(settings, "CHAT_AI_MAX_INPUT_TOKENS", 2048) or 2048))
        tokenizer = cls._tokenizer
        model = cls._model

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        compute_device = cls._get_model_device(model)
        inputs = {name: tensor.to(compute_device) for name, tensor in inputs.items()}

        generation_config = cls._build_generation_config(
            tokenizer=tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        try:
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_config)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning("Local chat generation runtime failure: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Local chat generation failed: %s", exc)
            return None

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated = outputs[0][prompt_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return cls._sanitize_response(text)

    @classmethod
    def _ensure_loaded(cls, *, model_id_override: Optional[str] = None) -> bool:
        model_id = cls._resolve_model_id(model_id_override)
        if not model_id:
            cls._load_error = "CHAT_AI_DEFAULT_MODEL is empty"
            return False

        if cls._model is not None and cls._tokenizer is not None and cls._loaded_model_id == model_id:
            return True

        with cls._lock:
            if cls._model is not None and cls._tokenizer is not None and cls._loaded_model_id == model_id:
                return True

            start = time.perf_counter()
            cls._load_error = None

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:
                cls._load_error = f"Missing ML dependencies: {exc}"
                logger.warning("Failed to import local chat dependencies: %s", exc)
                return False

            local_files_only = bool(
                getattr(settings, "CHAT_AI_MODEL_LOCAL_FILES_ONLY", False)
                or not getattr(settings, "CHAT_AI_MODEL_ALLOW_DOWNLOAD", True)
            )
            trust_remote_code = bool(getattr(settings, "CHAT_AI_MODEL_TRUST_REMOTE_CODE", False))
            cache_dir = (getattr(settings, "CHAT_AI_MODEL_CACHE_DIR", "") or "").strip() or None
            revision = (getattr(settings, "CHAT_AI_MODEL_REVISION", "") or "").strip() or None
            use_fast_tokenizer = bool(getattr(settings, "CHAT_AI_MODEL_USE_FAST_TOKENIZER", True))

            tokenizer_kwargs: Dict[str, Any] = {
                "local_files_only": local_files_only,
                "trust_remote_code": trust_remote_code,
                "use_fast": use_fast_tokenizer,
            }
            model_kwargs: Dict[str, Any] = {
                "local_files_only": local_files_only,
                "trust_remote_code": trust_remote_code,
            }
            if cache_dir:
                tokenizer_kwargs["cache_dir"] = cache_dir
                model_kwargs["cache_dir"] = cache_dir
            if revision:
                tokenizer_kwargs["revision"] = revision
                model_kwargs["revision"] = revision

            compute_device = cls._resolve_device(torch)
            torch_dtype = cls._resolve_dtype(torch, compute_device)
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            quant_mode = (getattr(settings, "CHAT_AI_MODEL_QUANTIZATION", "none") or "none").strip().lower()
            if quant_mode in {"8bit", "4bit"}:
                try:
                    import bitsandbytes  # noqa: F401
                    model_kwargs[f"load_in_{quant_mode}"] = True
                    model_kwargs["device_map"] = "auto"
                except Exception:
                    logger.warning(
                        "CHAT_AI_MODEL_QUANTIZATION=%s requested but bitsandbytes is unavailable; loading full precision",
                        quant_mode,
                    )
            elif compute_device == "cuda":
                model_kwargs["device_map"] = "auto"

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
                if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                if compute_device == "mps":
                    model = model.to("mps")
                elif compute_device == "cpu":
                    model = model.to("cpu")
                model.eval()
            except Exception as exc:
                cls._load_error = f"Failed to load model '{model_id}': {exc}"
                logger.warning(cls._load_error)
                return False

            cls._model = model
            cls._tokenizer = tokenizer
            cls._loaded_model_id = model_id
            cls._compute_device = compute_device
            cls._last_load_seconds = time.perf_counter() - start
            logger.info(
                "Loaded local chat model '%s' on %s in %.2fs",
                model_id,
                compute_device,
                cls._last_load_seconds,
            )
            return True

    @classmethod
    def _resolve_model_id(cls, override: Optional[str]) -> str:
        if override and override.strip():
            return override.strip()
        return (getattr(settings, "CHAT_AI_DEFAULT_MODEL", "") or "").strip()

    @classmethod
    def _resolve_device(cls, torch_module) -> str:
        configured = (getattr(settings, "CHAT_AI_MODEL_DEVICE", "auto") or "auto").strip().lower()
        if configured and configured != "auto":
            return configured
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def _resolve_dtype(cls, torch_module, compute_device: str):
        configured = (getattr(settings, "CHAT_AI_MODEL_DTYPE", "auto") or "auto").strip().lower()
        mapping = {
            "float16": getattr(torch_module, "float16", None),
            "fp16": getattr(torch_module, "float16", None),
            "bfloat16": getattr(torch_module, "bfloat16", None),
            "bf16": getattr(torch_module, "bfloat16", None),
            "float32": getattr(torch_module, "float32", None),
            "fp32": getattr(torch_module, "float32", None),
        }
        if configured in mapping and mapping[configured] is not None:
            return mapping[configured]
        if compute_device in {"cuda", "mps"}:
            return getattr(torch_module, "float16", None)
        return getattr(torch_module, "float32", None)

    @classmethod
    def _get_model_device(cls, model) -> str:
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return cls._compute_device or "cpu"

    @classmethod
    def _build_messages(
        cls,
        *,
        system_prompt: str,
        history: List[Dict[str, str]],
        personalization: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        system_parts = [system_prompt.strip() if system_prompt else "You are a helpful support assistant."]

        personalization_block = cls._format_personalization(personalization)
        if personalization_block:
            system_parts.append(
                "Use this customer context to personalize answers while protecting privacy:\n"
                f"{personalization_block}"
            )
        messages.append({"role": "system", "content": "\n\n".join(part for part in system_parts if part)})

        history_limit = max(2, int(getattr(settings, "CHAT_AI_CONTEXT_HISTORY_LIMIT", 10) or 10))
        for item in history[-history_limit:]:
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": content})
        return messages

    @classmethod
    def _format_personalization(cls, context: Dict[str, Any]) -> str:
        if not context or not getattr(settings, "CHAT_AI_PERSONALIZATION_ENABLED", True):
            return ""
        lines: List[str] = []
        for key, value in context.items():
            if value in (None, "", [], {}, ()):
                continue
            label = key.replace("_", " ").strip().title()
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    @classmethod
    def _format_messages(cls, messages: List[Dict[str, str]]) -> str:
        tokenizer = cls._tokenizer
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            lines: List[str] = []
            for message in messages:
                role = message.get("role", "user").upper()
                content = (message.get("content") or "").strip()
                if content:
                    lines.append(f"{role}: {content}")
            lines.append("ASSISTANT:")
            return "\n\n".join(lines)

    @classmethod
    def _build_generation_config(
        cls,
        *,
        tokenizer,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
    ) -> Dict[str, Any]:
        temp = float(
            temperature
            if temperature is not None
            else getattr(settings, "CHAT_AI_TEMPERATURE_DEFAULT", 0.7)
        )
        temp = max(0.0, min(temp, 2.0))
        do_sample = bool(getattr(settings, "CHAT_AI_DO_SAMPLE", True))
        if temp == 0:
            do_sample = False

        max_tokens = int(
            max_new_tokens
            if max_new_tokens is not None
            else getattr(settings, "CHAT_AI_MAX_NEW_TOKENS", 256)
        )
        max_tokens = max(32, min(max_tokens, 1024))

        config: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_p": float(getattr(settings, "CHAT_AI_TOP_P", 0.92)),
            "top_k": int(getattr(settings, "CHAT_AI_TOP_K", 50)),
            "repetition_penalty": float(getattr(settings, "CHAT_AI_REPETITION_PENALTY", 1.08)),
            "no_repeat_ngram_size": int(getattr(settings, "CHAT_AI_NO_REPEAT_NGRAM_SIZE", 3)),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            config["temperature"] = max(0.05, temp)
        return config

    @classmethod
    def _sanitize_response(cls, text: str) -> Optional[str]:
        if not text:
            return None

        cleaned = text.strip()
        leading_tokens = (
            "assistant:",
            "Assistant:",
            "<|assistant|>",
            "<assistant>",
            "[/INST]",
        )
        for token in leading_tokens:
            if cleaned.startswith(token):
                cleaned = cleaned[len(token):].strip()

        if not cleaned:
            return None

        max_chars = max(200, int(getattr(settings, "CHAT_AI_MAX_RESPONSE_CHARS", 4000) or 4000))
        return cleaned[:max_chars].strip()
