from __future__ import annotations

import re
from urllib.parse import urlsplit

from jsonschema import FormatChecker


_URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*\Z")


def _absolute_uri(value: object) -> bool:
    """Validate the contract's URI strings without optional jsonschema extras.

    jsonschema delegates RFC format support to optional packages.  Minimal
    runtime images may therefore silently accept every ``format: uri`` value.
    The middleware contracts use URIs as transport endpoints, so require an
    absolute URI with a syntactically valid scheme and no whitespace/control
    characters regardless of which extras happen to be installed.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    try:
        parsed = urlsplit(value)
    except (TypeError, ValueError):
        return False
    if not _URI_SCHEME.fullmatch(parsed.scheme):
        return False
    # Network transports need an authority; non-network absolute URI schemes
    # such as urn: remain valid contract values.
    if parsed.scheme.lower() in {"http", "https", "ws", "wss"}:
        try:
            parsed.port
            return bool(parsed.hostname)
        except ValueError:
            return False
    return bool(parsed.path or parsed.netloc)


def contract_format_checker() -> FormatChecker:
    checker = FormatChecker()

    @checker.checks("uri")
    def check_uri(value: object) -> bool:
        return _absolute_uri(value)

    return checker
