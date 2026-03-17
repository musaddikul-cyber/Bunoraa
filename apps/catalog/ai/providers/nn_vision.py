from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

class NNVisionLoadError(RuntimeError):
    """Raised when NN vision runtime/model loading fails."""


class NNVisionInferenceError(RuntimeError):
    """Raised when NN vision inference fails."""


_PROMPT_BANK: list[dict[str, Any]] = [
    {
        "label": "women apparel set",
        "prompt": "a product photo of a women's apparel set",
        "tokens": ["women", "apparel", "set", "fashion"],
    },
    {
        "label": "embroidered kurti",
        "prompt": "a product photo of an embroidered kurti",
        "tokens": ["embroidered", "kurti", "apparel", "fashion"],
    },
    {
        "label": "dress",
        "prompt": "a product photo of a dress",
        "tokens": ["dress", "apparel", "fashion"],
    },
    {
        "label": "shirt",
        "prompt": "a product photo of a shirt",
        "tokens": ["shirt", "apparel", "fashion"],
    },
    {
        "label": "handmade home decor",
        "prompt": "a product photo of handmade home decor",
        "tokens": ["handmade", "home", "decor", "artisan"],
    },
    {
        "label": "bag",
        "prompt": "a product photo of a bag",
        "tokens": ["bag", "accessory", "fashion"],
    },
    {
        "label": "shoe",
        "prompt": "a product photo of a shoe",
        "tokens": ["shoe", "footwear", "fashion"],
    },
    {
        "label": "jewelry",
        "prompt": "a product photo of jewelry",
        "tokens": ["jewelry", "accessory", "fashion"],
    },
    {
        "label": "bottle",
        "prompt": "a product photo of a reusable bottle",
        "tokens": ["bottle", "home", "utility"],
    },
    {
        "label": "tableware",
        "prompt": "a product photo of tableware",
        "tokens": ["tableware", "home", "kitchen"],
    },
]


def _dedupe_tokens(values: list[str]) -> list[str]:
    return list(dict.fromkeys(token.strip().lower() for token in values if token and token.strip()))


class NNVisionProvider:
    """
    CLIP-based lightweight NN vision inference for ProductAI.
    Loads lazily and caches runtime in-process.
    """

    _runtime_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        timeout_seconds: float = 8.0,
        cache_dir: str,
    ):
        self.model_id = str(model_id or "openai/clip-vit-base-patch32").strip()
        self.device = str(device or "cpu").strip().lower()
        self.timeout_seconds = float(timeout_seconds or 8.0)
        self.cache_dir = str(cache_dir or "").strip()

    def analyze(self, image_paths: list[str]) -> dict[str, Any]:
        result = {
            "nn_enabled": True,
            "nn_model_id": self.model_id,
            "nn_inference_status": "disabled",
            "nn_confidence": 0.0,
            "nn_labels": [],
            "nn_caption_like_summary": "",
            "nn_tokens": [],
        }

        if not image_paths:
            result["nn_inference_status"] = "skipped_no_images"
            return result

        runtime = self._get_runtime()
        started = time.monotonic()
        try:
            labels = self._predict_labels(image_paths=image_paths, runtime=runtime)
        except NNVisionLoadError:
            raise
        except Exception as exc:
            raise NNVisionInferenceError(
                f"NN inference failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        elapsed = float(time.monotonic() - started)
        if elapsed > self.timeout_seconds:
            raise NNVisionInferenceError(
                f"NN inference timed out after {elapsed:.2f}s (budget {self.timeout_seconds:.2f}s)."
            )

        if not labels:
            result["nn_inference_status"] = "no_predictions"
            return result

        top_label = labels[0]
        caption = self._build_caption(labels)
        nn_tokens = _dedupe_tokens(
            [token for item in labels for token in item.get("tokens") or []]
            + [item.get("label", "") for item in labels]
        )[:16]

        result.update(
            {
                "nn_inference_status": "ok",
                "nn_confidence": float(top_label.get("score", 0.0) or 0.0),
                "nn_labels": labels,
                "nn_caption_like_summary": caption,
                "nn_tokens": nn_tokens,
            }
        )
        return result

    def _runtime_key(self) -> tuple[str, str, str]:
        return (self.model_id, self.device, self.cache_dir)

    def _get_runtime(self) -> dict[str, Any]:
        key = self._runtime_key()
        cached = self._runtime_cache.get(key)
        if cached:
            return cached
        runtime = self._load_runtime()
        self._runtime_cache[key] = runtime
        return runtime

    def _load_runtime(self) -> dict[str, Any]:
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except Exception as exc:
            raise NNVisionLoadError(
                "NN runtime dependencies missing (torch/transformers)."
            ) from exc

        cache_dir_path = Path(self.cache_dir)
        try:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise NNVisionLoadError(
                f"NN model cache directory is not writable: {cache_dir_path}"
            ) from exc

        try:
            processor = CLIPProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(cache_dir_path),
            )
            model = CLIPModel.from_pretrained(
                self.model_id,
                cache_dir=str(cache_dir_path),
            )
        except Exception as exc:
            raise NNVisionLoadError(
                f"Unable to load pretrained NN model '{self.model_id}': {exc.__class__.__name__}: {exc}"
            ) from exc

        if self.device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = "cpu"

        try:
            model = model.to(resolved_device)
            model.eval()
        except Exception as exc:
            raise NNVisionLoadError(
                f"Unable to initialize NN model on device '{resolved_device}': {exc.__class__.__name__}: {exc}"
            ) from exc

        return {
            "torch": torch,
            "processor": processor,
            "model": model,
            "device": resolved_device,
        }

    def _predict_labels(self, *, image_paths: list[str], runtime: dict[str, Any]) -> list[dict[str, Any]]:
        torch = runtime["torch"]
        processor = runtime["processor"]
        model = runtime["model"]
        device = runtime["device"]

        images = []
        for path in image_paths[:3]:
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            except Exception:
                continue

        if not images:
            return []

        prompts = [item["prompt"] for item in _PROMPT_BANK]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        try:
            inputs = inputs.to(device)
        except Exception:
            # Fallback for environments where BatchEncoding.to() is unavailable.
            converted = {}
            for key, value in inputs.items():
                converted[key] = value.to(device) if hasattr(value, "to") else value
            inputs = converted

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=-1)
            mean_probs = probs.mean(dim=0)

        top_k = min(5, len(_PROMPT_BANK))
        values, indexes = mean_probs.topk(top_k)
        labels: list[dict[str, Any]] = []
        for score, index in zip(values.tolist(), indexes.tolist()):
            prompt_item = _PROMPT_BANK[int(index)]
            labels.append(
                {
                    "label": prompt_item["label"],
                    "score": round(float(score), 4),
                    "tokens": list(prompt_item.get("tokens") or []),
                }
            )
        return labels

    @staticmethod
    def _build_caption(labels: list[dict[str, Any]]) -> str:
        top = [str(item.get("label") or "").strip() for item in labels[:3] if str(item.get("label") or "").strip()]
        if not top:
            return ""
        if len(top) == 1:
            return f"NN vision suggests: {top[0]}."
        return "NN vision suggests: " + ", ".join(top[:-1]) + f", and {top[-1]}."
