import re
from typing import Any

EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"\b(\+?\d[\d\s\-().]{6,}\d)\b")


def redact_pii(text: str) -> str:
    if not text:
        return text
    masked = EMAIL_RE.sub(r"***@\2", text)
    masked = PHONE_RE.sub("***", masked)
    return masked


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: redact_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_payload(v) for v in value]
    if isinstance(value, str):
        return redact_pii(value)
    return value
