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
NEUTRAL_COLOR_NAMES = {"black", "white", "gray"}
FLORAL_ACCENT_COLORS = {"pink", "purple", "green", "teal", "red", "yellow"}


def _color_distance_sq(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return ((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2)


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

    @staticmethod
    def _estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = rgb.load()
        band_x = max(1, min(8, width // 6))
        band_y = max(1, min(8, height // 6))
        samples: list[tuple[int, int, int]] = []

        for x in range(band_x):
            for y in range(band_y):
                samples.append(pixels[x, y])
                samples.append(pixels[width - 1 - x, y])
                samples.append(pixels[x, height - 1 - y])
                samples.append(pixels[width - 1 - x, height - 1 - y])

        if not samples:
            return (200, 200, 200)
        red = int(mean(sample[0] for sample in samples))
        green = int(mean(sample[1] for sample in samples))
        blue = int(mean(sample[2] for sample in samples))
        return (red, green, blue)

    def _extract_foreground(
        self,
        image: Image.Image,
        *,
        target_size: int,
        threshold: int = 34,
    ) -> tuple[int, int, list[bool], list[tuple[int, int, tuple[int, int, int]]], str]:
        resized = image.convert("RGBA").resize((target_size, target_size))
        rgb = resized.convert("RGB")
        alpha = resized.getchannel("A")
        width, height = rgb.size
        pixels = rgb.load()
        alpha_pixels = alpha.load()
        bg_rgb = self._estimate_background_color(rgb)
        bg_name = _nearest_color_name(bg_rgb)
        threshold_sq = threshold * threshold

        mask: list[bool] = [False] * (width * height)
        foreground: list[tuple[int, int, tuple[int, int, int]]] = []

        for y in range(height):
            row = y * width
            for x in range(width):
                rgba_alpha = int(alpha_pixels[x, y])
                if rgba_alpha <= 12:
                    continue
                pixel = pixels[x, y]
                if _color_distance_sq(pixel, bg_rgb) >= threshold_sq:
                    mask[row + x] = True
                    foreground.append((x, y, pixel))

        if len(foreground) < int(width * height * 0.04):
            mask = [False] * (width * height)
            foreground = []
            relaxed_threshold_sq = max(18, threshold - 10) ** 2
            for y in range(height):
                row = y * width
                for x in range(width):
                    rgba_alpha = int(alpha_pixels[x, y])
                    if rgba_alpha <= 6:
                        continue
                    pixel = pixels[x, y]
                    if _color_distance_sq(pixel, bg_rgb) >= relaxed_threshold_sq:
                        mask[row + x] = True
                        foreground.append((x, y, pixel))

        if not foreground:
            mask = [False] * (width * height)
            foreground = []
            for y in range(height):
                row = y * width
                for x in range(width):
                    rgba_alpha = int(alpha_pixels[x, y])
                    if rgba_alpha <= 6:
                        continue
                    pixel = pixels[x, y]
                    mask[row + x] = True
                    foreground.append((x, y, pixel))

        return width, height, mask, foreground, bg_name

    @staticmethod
    def _mask_region_coverage(
        mask: list[bool],
        width: int,
        height: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> float:
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        total = (x1 - x0) * (y1 - y0)
        if total <= 0:
            return 0.0
        selected = 0
        for y in range(y0, y1):
            row = y * width
            for x in range(x0, x1):
                if mask[row + x]:
                    selected += 1
        return float(selected) / float(total)

    @staticmethod
    def _component_count(mask: list[bool], width: int, height: int, *, min_pixels: int = 18) -> int:
        if not mask:
            return 0
        visited = [False] * len(mask)
        components = 0

        for idx, is_foreground in enumerate(mask):
            if not is_foreground or visited[idx]:
                continue
            stack = [idx]
            visited[idx] = True
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                x = current % width
                y = current // width
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    nidx = ny * width + nx
                    if visited[nidx] or not mask[nidx]:
                        continue
                    visited[nidx] = True
                    stack.append(nidx)
            if size >= min_pixels:
                components += 1
                if components >= 6:
                    return components
        return components

    def _shape_signals(self, image_paths: list[str]) -> dict[str, Any]:
        foreground_ratios: list[float] = []
        bbox_ratios: list[float] = []
        top_coverages: list[float] = []
        mid_coverages: list[float] = []
        bottom_coverages: list[float] = []
        component_counts: list[int] = []
        neutral_background_votes = 0

        for path in image_paths[:3]:
            try:
                with Image.open(path) as image:
                    width, height, mask, foreground, bg_name = self._extract_foreground(
                        image,
                        target_size=88,
                        threshold=34,
                    )
            except Exception:
                continue

            if not foreground:
                continue

            if bg_name in NEUTRAL_COLOR_NAMES:
                neutral_background_votes += 1

            total_pixels = max(1, width * height)
            foreground_ratio = float(len(foreground)) / float(total_pixels)
            foreground_ratios.append(foreground_ratio)

            xs = [item[0] for item in foreground]
            ys = [item[1] for item in foreground]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            bbox_width = max(1, max_x - min_x + 1)
            bbox_height = max(1, max_y - min_y + 1)
            bbox_ratios.append(float(bbox_height) / float(bbox_width))

            band = max(2, int(bbox_height * 0.18))
            top_coverages.append(
                self._mask_region_coverage(mask, width, height, min_x, min_y, max_x + 1, min_y + band)
            )
            mid_y0 = min_y + int(bbox_height * 0.35)
            mid_y1 = min_y + int(bbox_height * 0.65)
            bottom_y0 = max(min_y, max_y - band + 1)
            bottom_y1 = max_y + 1
            mid_coverages.append(
                self._mask_region_coverage(mask, width, height, min_x, mid_y0, max_x + 1, mid_y1)
            )
            bottom_coverages.append(
                self._mask_region_coverage(mask, width, height, min_x, bottom_y0, max_x + 1, bottom_y1)
            )
            component_counts.append(
                self._component_count(mask, width, height, min_pixels=max(18, int(total_pixels * 0.008)))
            )

        avg_foreground_ratio = float(mean(foreground_ratios)) if foreground_ratios else 0.0
        avg_bbox_ratio = float(mean(bbox_ratios)) if bbox_ratios else 0.0
        avg_top = float(mean(top_coverages)) if top_coverages else 0.0
        avg_mid = float(mean(mid_coverages)) if mid_coverages else 0.0
        avg_bottom = float(mean(bottom_coverages)) if bottom_coverages else 0.0
        multi_piece = any(count >= 2 for count in component_counts)
        garment_silhouette = (
            avg_bbox_ratio >= 1.0
            and avg_bbox_ratio <= 3.0
            and avg_mid >= (avg_top + 0.08)
            and avg_bottom >= (avg_top * 0.75)
        )
        apparel_item = (
            avg_foreground_ratio >= 0.10
            and avg_foreground_ratio <= 0.86
            and avg_bbox_ratio >= 0.95
            and avg_bbox_ratio <= 3.2
            and (garment_silhouette or multi_piece)
        )
        background_neutral = neutral_background_votes >= max(1, len(foreground_ratios) // 2)

        return {
            "apparel_item": apparel_item,
            "multi_piece": multi_piece,
            "background_neutral": background_neutral,
            "foreground_ratio": round(avg_foreground_ratio, 4) if avg_foreground_ratio else 0.0,
            "bbox_ratio": round(avg_bbox_ratio, 4) if avg_bbox_ratio else 0.0,
        }

    def _dominant_color_terms(self, image_paths: list[str], *, limit: int = 2) -> list[str]:
        scores: dict[str, float] = {}
        for path in image_paths[:3]:
            try:
                with Image.open(path) as image:
                    width, height, _mask, foreground, _bg_name = self._extract_foreground(
                        image,
                        target_size=72,
                        threshold=30,
                    )
                    if not foreground:
                        rgb_image = image.convert("RGB").resize((72, 72))
                        foreground = [
                            (index % 72, index // 72, color)
                            for index, color in enumerate(list(rgb_image.getdata()))
                        ]
                    sample_step = max(1, len(foreground) // 1800)
                    for _, _, color in foreground[::sample_step]:
                        color_name = _nearest_color_name(color)
                        weight = 1.0
                        if color_name in NEUTRAL_COLOR_NAMES and len(foreground) > (width * height * 0.25):
                            weight = 0.35
                        scores[color_name] = scores.get(color_name, 0.0) + weight
            except Exception:
                continue

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        has_non_neutral = any(color_name not in NEUTRAL_COLOR_NAMES for color_name, _ in ranked)
        out: list[str] = []
        for color_name, _ in ranked:
            if has_non_neutral and color_name in NEUTRAL_COLOR_NAMES and out:
                continue
            out.append(color_name)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _build_candidate_name(
        *,
        dominant_colors: list[str],
        people_present: bool,
        apparel_item: bool,
        multi_piece: bool,
        floral_hint: bool,
    ) -> str:
        if people_present and dominant_colors:
            return f"{dominant_colors[0].title()} Women's Apparel Set"
        if people_present:
            return "Women's Apparel Set"
        if apparel_item:
            prefix = f"{dominant_colors[0].title()} " if dominant_colors else ""
            if floral_hint:
                suffix = "Embroidered Apparel Set" if multi_piece else "Embroidered Apparel Piece"
            else:
                suffix = "Apparel Set" if multi_piece else "Apparel Piece"
            return f"{prefix}{suffix}".strip()
        non_neutral = [color for color in dominant_colors if color not in NEUTRAL_COLOR_NAMES]
        if non_neutral:
            return f"{non_neutral[0].title()} Product"
        return ""

    @staticmethod
    def _scene_summary(
        *,
        dominant_colors: list[str],
        people_present: bool,
        apparel_item: bool,
        multi_piece: bool,
        floral_hint: bool,
    ) -> str:
        if people_present and dominant_colors:
            return f"Model wearing a {dominant_colors[0]} apparel outfit in a product-style photo."
        if people_present:
            return "Model wearing an apparel outfit in a product-style photo."
        if apparel_item:
            base = "Two-piece apparel set" if multi_piece else "Apparel piece"
            if floral_hint:
                base = f"Embroidered {base.lower()}"
            if dominant_colors:
                return f"{base} in {dominant_colors[0]} tone on a plain background."
            return f"{base} on a plain background."
        non_neutral = [color for color in dominant_colors if color not in NEUTRAL_COLOR_NAMES]
        if non_neutral:
            return f"Product photo with dominant {non_neutral[0]} color."
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
        shape_signals = self._shape_signals(image_paths)
        people_present = self._detect_people_present(image_paths)
        apparel_item = people_present or bool(shape_signals.get("apparel_item"))
        multi_piece = bool(shape_signals.get("multi_piece"))
        floral_hint = (
            apparel_item
            and len(set(dominant_colors).intersection(FLORAL_ACCENT_COLORS)) >= 2
        )
        candidate_name = self._build_candidate_name(
            dominant_colors=dominant_colors,
            people_present=people_present,
            apparel_item=apparel_item,
            multi_piece=multi_piece,
            floral_hint=floral_hint,
        )
        scene_summary = self._scene_summary(
            dominant_colors=dominant_colors,
            people_present=people_present,
            apparel_item=apparel_item,
            multi_piece=multi_piece,
            floral_hint=floral_hint,
        )

        tokens: list[str] = []
        tokens.extend(dominant_colors)
        if people_present or apparel_item:
            tokens.extend(APPAREL_HINT_TOKENS)
        if floral_hint:
            tokens.extend(["embroidered", "floral"])
        if multi_piece:
            tokens.extend(["set", "two-piece"])
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
            "apparel_item": apparel_item,
            "multi_piece_layout": multi_piece,
            "floral_hint": floral_hint,
            "background_neutral": bool(shape_signals.get("background_neutral")),
            "foreground_ratio": shape_signals.get("foreground_ratio", 0.0),
            "bbox_ratio": shape_signals.get("bbox_ratio", 0.0),
            "image_count": len(image_paths),
            "tokens": deduped_tokens,
        }
