from __future__ import annotations

import re
from typing import Any


class OCRProvider:
    """
    OCR provider with graceful fallback when OCR dependencies are unavailable.
    """

    def __init__(self):
        self._pytesseract = None
        self._cv2 = None
        self._load_optional_backends()

    def _load_optional_backends(self):
        try:
            import pytesseract  # type: ignore

            self._pytesseract = pytesseract
        except Exception:
            self._pytesseract = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception:
            self._cv2 = None

    def extract(self, image_paths: list[str]) -> dict[str, Any]:
        texts: list[str] = []
        for path in image_paths:
            text = self._extract_from_path(path)
            if text:
                texts.append(text)

        combined = "\n".join(texts).strip()
        sku_candidates = self._extract_sku_candidates(combined)
        return {
            "text": combined,
            "lines": [line.strip() for line in combined.splitlines() if line.strip()],
            "sku_candidates": sku_candidates,
        }

    def _extract_from_path(self, path: str) -> str:
        if self._pytesseract and self._cv2:
            try:
                image = self._cv2.imread(path)
                if image is not None:
                    text = self._pytesseract.image_to_string(image)
                    if text and text.strip():
                        return text
            except Exception:
                pass

        # No OCR backend available: return empty text rather than noisy
        # tempfile/upload filename tokens.
        return ""

    @staticmethod
    def _extract_sku_candidates(text: str) -> list[str]:
        pattern = re.compile(r"\b[A-Z0-9]{2,6}[-_][A-Z0-9]{2,12}\b")
        return list(dict.fromkeys(pattern.findall((text or "").upper())))
