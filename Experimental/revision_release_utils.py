"""Small release-safety helpers shared by the revision runners."""

import os
import re


def safe_error(error: Exception, *secrets: str) -> str:
    """Keep diagnostic text without echoing credentials from provider errors."""
    text = f"{type(error).__name__}: {error}"
    values = list(secrets) + [
        value for name, value in os.environ.items()
        if any(word in name.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    ]
    for value in sorted({v for v in values if v}, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text
