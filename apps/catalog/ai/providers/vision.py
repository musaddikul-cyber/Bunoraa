from __future__ import annotations

from statistics import mean
from typing import Any

from PIL import Image

from apps.catalog.models import (
    get_active_aspect_ratio_codes,
    get_default_aspect_ratio_code,
    parse_aspect_ratio_code,
)


def _aspect_ratio_choice(width: int, height: int, allowed_codes: set[str]) -> str:
    default_code = get_default_aspect_ratio_code()
    if not width or not height:
        return default_code
    ratio = width / height
    ratio_candidates: dict[str, float] = {}
    for code in allowed_codes:
        parsed = parse_aspect_ratio_code(code)
        if parsed is not None:
            ratio_candidates[code] = float(parsed[0] / parsed[1])
    if not ratio_candidates:
        return default_code
    return min(ratio_candidates, key=lambda key: abs(ratio_candidates[key] - ratio))


class VisionProvider:
    """
    Local lightweight vision analyzer for production-safe defaults.
    """

    def analyze(self, image_paths: list[str]) -> dict[str, Any]:
        widths = []
        heights = []

        for path in image_paths:
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    widths.append(width)
                    heights.append(height)
            except Exception:
                continue

        avg_width = int(mean(widths)) if widths else 0
        avg_height = int(mean(heights)) if heights else 0
        allowed_codes = get_active_aspect_ratio_codes()
        aspect_ratio = _aspect_ratio_choice(avg_width, avg_height, allowed_codes)

        return {
            "aspect_ratio": aspect_ratio,
            "avg_width": avg_width,
            "avg_height": avg_height,
            # Do not infer product names from local temp/upload filenames.
            "candidate_name": "",
            "image_count": len(image_paths),
            "tokens": [],
        }
