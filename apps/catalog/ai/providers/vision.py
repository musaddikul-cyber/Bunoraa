from __future__ import annotations

import re
from statistics import mean
from typing import Any

from PIL import Image

from apps.catalog.models import (
    get_active_aspect_ratio_codes,
    get_default_aspect_ratio_code,
    parse_aspect_ratio_code,
)


COLOR_REFERENCE = {
    "red": (200, 60, 60),
    "pink": (232, 122, 170),
    "orange": (228, 140, 72),
    "yellow": (225, 200, 90),
    "green": (95, 155, 95),
    "teal": (72, 150, 150),
    "blue": (85, 120, 180),
    "purple": (150, 110, 180),
    "brown": (145, 110, 85),
    "black": (50, 50, 50),
    "gray": (140, 140, 140),
    "white": (220, 220, 220),
    "beige": (210, 190, 160),
}

APPAREL_HINT_TOKENS = ("women", "apparel", "fashion", "outfit", "set")


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


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    best_name = "colorful"
    best_distance = float("inf")
    for name, reference in COLOR_REFERENCE.items():
        r2, g2, b2 = reference
        distance = ((red - r2) ** 2) + ((green - g2) ** 2) + ((blue - b2) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


class VisionProvider:
    """
    Lightweight vision analyzer for production-safe defaults.
    Uses deterministic heuristics (aspect, dominant color, person-presence hints)
    to produce better search anchors when no OCR/context is available.
    """

    _face_detector = None
    _face_detector_ready = False

    @classmethod
    def _get_face_detector(cls):
        if cls._face_detector_ready:
            return cls._face_detector
        cls._face_detector_ready = True
        try:
            import cv2  # type: ignore

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if detector is not None and not detector.empty():
                cls._face_detector = detector
            else:
                cls._face_detector = None
        except Exception:
            cls._face_detector = None
        return cls._face_detector

    def _detect_people_present(self, image_paths: list[str]) -> bool:
        detector = self._get_face_detector()
        if detector is None:
            return False
        try:
            import cv2  # type: ignore
        except Exception:
            return False

        for path in image_paths[:2]:
            try:
                image = cv2.imread(path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
                if len(faces) > 0:
                    return True
            except Exception:
                continue
        return False

    def _dominant_color_terms(self, image_paths: list[str], *, limit: int = 2) -> list[str]:
        scores: dict[str, float] = {}
        for path in image_paths[:3]:
            try:
                with Image.open(path) as image:
                    rgb_image = image.convert("RGB").resize((72, 72))
                    quantized = rgb_image.quantize(colors=6, method=Image.MEDIANCUT).convert("RGB")
                    colors = quantized.getcolors(maxcolors=72 * 72) or []
                    if not colors:
                        continue
                    total = float(sum(count for count, _ in colors) or 1.0)
                    for count, color in colors:
                        color_name = _nearest_color_name(color)
                        scores[color_name] = scores.get(color_name, 0.0) + (float(count) / total)
            except Exception:
                continue

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        out: list[str] = []
        for color_name, _ in ranked:
            if color_name in {"black", "white", "gray"} and out:
                continue
            out.append(color_name)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _build_candidate_name(*, dominant_colors: list[str], people_present: bool) -> str:
        if people_present and dominant_colors:
            return f"{dominant_colors[0].title()} Women's Apparel Set"
        if people_present:
            return "Women's Apparel Set"
        if dominant_colors:
            return f"{dominant_colors[0].title()} Product"
        return ""

    @staticmethod
    def _scene_summary(*, dominant_colors: list[str], people_present: bool) -> str:
        if people_present and dominant_colors:
            return f"Model wearing a {dominant_colors[0]} apparel outfit in a product-style photo."
        if people_present:
            return "Model wearing an apparel outfit in a product-style photo."
        if dominant_colors:
            return f"Product photo with dominant {dominant_colors[0]} color."
        return ""

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'\-]{1,30}", text or "")]

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
        dominant_colors = self._dominant_color_terms(image_paths)
        people_present = self._detect_people_present(image_paths)
        candidate_name = self._build_candidate_name(
            dominant_colors=dominant_colors,
            people_present=people_present,
        )
        scene_summary = self._scene_summary(
            dominant_colors=dominant_colors,
            people_present=people_present,
        )

        tokens: list[str] = []
        tokens.extend(dominant_colors)
        if people_present:
            tokens.extend(APPAREL_HINT_TOKENS)
        tokens.extend(self._tokenize_text(candidate_name))
        tokens.extend(self._tokenize_text(scene_summary))
        deduped_tokens = list(dict.fromkeys(token for token in tokens if len(token) >= 3))[:16]

        return {
            "aspect_ratio": aspect_ratio,
            "avg_width": avg_width,
            "avg_height": avg_height,
            "candidate_name": candidate_name,
            "scene_summary": scene_summary,
            "dominant_colors": dominant_colors,
            "people_present": people_present,
            "image_count": len(image_paths),
            "tokens": deduped_tokens,
        }
